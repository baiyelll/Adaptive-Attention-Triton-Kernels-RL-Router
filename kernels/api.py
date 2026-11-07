"""
Python wrappers around the Triton attention kernel.
"""

from __future__ import annotations

import torch
import triton

from .attention_fwd import _attn_fwd


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """
    Fused causal / sliding-window attention.

    Args:
        q, k, v: (B, H, N, D) fp16 CUDA tensors, last dim contiguous.
        window:  None → full causal; int → sliding window of that size
                 (window includes the current token, same semantics as the
                 reference implementation).
        sm_scale: softmax scale; defaults to 1/sqrt(D).

    Returns:
        o: (B, H, N, D) same dtype/device as q.
    """
    assert q.is_cuda and q.dtype == torch.float16, "T4 / this kernel: fp16 CUDA only"
    assert q.shape == k.shape == v.shape, "q, k, v must have identical shape"
    B, H, N, D = q.shape
    assert D in (32, 64, 128), f"HEAD_DIM must be power-of-two in {{32,64,128}}, got {D}"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (
        "last dimension must be contiguous (stride == 1)"
    )

    window = N if window is None else int(window)
    sm_scale = float(sm_scale) if sm_scale is not None else D ** -0.5

    o = torch.empty_like(q)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)

    _attn_fwd[grid](
        q, k, v, o, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, N, window,
        HEAD_DIM=D,
    )
    return o


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full causal attention (WINDOW = N)."""
    return attention(q, k, v, window=None)


def sliding_window_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    """Sliding-window attention of size `window` (inclusive of current token)."""
    return attention(q, k, v, window=window)
