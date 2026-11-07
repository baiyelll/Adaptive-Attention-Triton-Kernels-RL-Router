"""
Fused causal FlashAttention + sliding-window attention forward kernel in Triton.

One kernel handles both modes via the WINDOW argument:
  - full causal  → WINDOW >= N
  - sliding      → WINDOW = W (tokens, inclusive of the current token)

Designed for Turing (sm_75 / T4): fp16 inputs, fp32 accumulators, small tiles.
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=4, num_stages=1),
    ],
    key=["N", "WINDOW", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q, K, V, O, sm_scale,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sob, soh, som, sod,
    H, N, WINDOW,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (cdiv(N, BLOCK_M), B * H)
    Each program owns one Q-block of one (batch, head).
    """
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    q_base = Q + off_b * sqb + off_h * sqh
    k_base = K + off_b * skb + off_h * skh
    v_base = V + off_b * svb + off_h * svh
    o_base = O + off_b * sob + off_h * soh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block once (fp16 → fp32 for the rest of the math)
    q = tl.load(
        q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_m[:, None] < N,
        other=0.0,
    )

    # Online-softmax state
    m_i = tl.full([BLOCK_M], -1.0e6, dtype=tl.float32)  # NOT -inf (NaN trap)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Fold log2(e) into the scale so we can use exp2
    qk_scale = sm_scale * 1.4426950408889634

    # KV range this Q-block can see (the whole sliding-window speedup)
    # WINDOW includes the current token → lo = max(0, start_m*BM - WINDOW + 1)
    lo = tl.maximum(0, start_m * BLOCK_M - WINDOW + 1)
    lo = (lo // BLOCK_N) * BLOCK_N  # align down to a KV-block boundary
    hi = tl.minimum(N, (start_m + 1) * BLOCK_M)  # causal upper bound

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(
            k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=offs_n[:, None] < N,
            other=0.0,
        )  # (BN, D)

        qk = tl.dot(q, tl.trans(k)) * qk_scale  # (BM, BN) fp32

        # Causal + window + bounds mask
        keep = (
            (offs_m[:, None] >= offs_n[None, :])
            & ((offs_m[:, None] - offs_n[None, :]) < WINDOW)
            & (offs_n[None, :] < N)
        )
        qk = tl.where(keep, qk, -1.0e6)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
            mask=offs_n[:, None] < N,
            other=0.0,
        )  # (BN, D)

        # Accumulate in fp32; cast P to fp16 for the tensor-core matmul
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    # Final normalisation
    acc = acc / l_i[:, None]
    tl.store(
        o_base + offs_m[:, None] * som + offs_d[None, :] * sod,
        acc.to(O.dtype.element_ty),
        mask=offs_m[:, None] < N,
    )
