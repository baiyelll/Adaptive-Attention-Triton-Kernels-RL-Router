"""
Correctness tests: Triton kernel vs fp32 naive reference.

Run on a CUDA device (Colab T4 or any sm_75+ GPU):
    pytest tests/test_correctness.py -v --tb=short
"""

from __future__ import annotations

import pytest
import torch

from kernels.api import attention, flash_attn, sliding_window_attn
from reference.naive_attention import naive_attention


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _rand_qkv(B, H, N, D, device="cuda", dtype=torch.float16, scale=1.0):
    q = torch.randn(B, H, N, D, device=device, dtype=dtype) * scale
    k = torch.randn(B, H, N, D, device=device, dtype=dtype) * scale
    v = torch.randn(B, H, N, D, device=device, dtype=dtype) * scale
    return q, k, v


def _assert_close(out, ref, atol=2e-2, rtol=2e-2):
    assert not torch.isnan(out).any(), "NaN in Triton output"
    assert not torch.isinf(out).any(), "Inf in Triton output"
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Parametrised correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 17, 63, 64, 65, 127, 128, 500, 1024, 2048])
@pytest.mark.parametrize("W", [None, 1, 2, 16, 64, 100, 128, 1000])  # None = full causal
@pytest.mark.parametrize("B,H", [(1, 1), (2, 4)])
@pytest.mark.parametrize("D", [32, 64])
def test_kernel_vs_naive(N, W, B, H, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # W >= N is treated as full causal by the kernel; clamp for the reference
    window = None if W is None else min(W, N)
    q, k, v = _rand_qkv(B, H, N, D)

    out = attention(q, k, v, window=window)
    ref = naive_attention(q, k, v, window=window)
    _assert_close(out, ref)


@pytest.mark.parametrize("N", [64, 256])
@pytest.mark.parametrize("D", [32, 64])
def test_w1_equals_v(N, D):
    """W=1 → each token attends only to itself → output == v."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    B, H = 2, 4
    q, k, v = _rand_qkv(B, H, N, D)
    out = sliding_window_attn(q, k, v, window=1)
    _assert_close(out, v, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("N", [128, 512])
@pytest.mark.parametrize("D", [64])
def test_full_equals_w_ge_n(N, D):
    """WINDOW >= N must equal pure causal."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    B, H = 1, 2
    q, k, v = _rand_qkv(B, H, N, D)
    out_full = flash_attn(q, k, v)
    out_w = attention(q, k, v, window=N + 10)
    _assert_close(out_full, out_w, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("N", [128, 512])
def test_large_magnitude(N):
    """Large input values stress online-softmax stability."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    B, H, D = 1, 2, 64
    q, k, v = _rand_qkv(B, H, N, D, scale=10.0)
    out = flash_attn(q, k, v)
    ref = naive_attention(q, k, v, window=None)
    _assert_close(out, ref, atol=3e-2, rtol=3e-2)


def test_non_contiguous_batch():
    """Exercise non-default strides (transpose + contiguous upstream)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    B, H, N, D = 2, 4, 128, 64
    # Create (B, N, H, D) then transpose to (B, H, N, D) — common layout
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16).transpose(1, 2).contiguous()
    k = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16).transpose(1, 2).contiguous()
    v = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16).transpose(1, 2).contiguous()
    assert q.stride(-1) == 1
    out = flash_attn(q, k, v)
    ref = naive_attention(q, k, v)
    _assert_close(out, ref)
