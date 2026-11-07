"""
Parity: original PyTorch attention vs Triton-backed modules on the same inputs.

Run on CUDA:
    pytest tests/test_router_parity.py -v --tb=short
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router_integration.triton_layers import (
    TritonFullAttention,
    TritonSlidingWindowAttention,
)


# ---------------------------------------------------------------------------
# Minimal copies of the original pure-PyTorch attentions (for reference)
# ---------------------------------------------------------------------------
class OrigFull(nn.Module):
    def __init__(self, n_head=4, n_embd=128):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        Q = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K, V = Q, Q
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous().view(B, T, C)
        return out


class OrigSliding(nn.Module):
    def __init__(self, n_head=4, n_embd=128, window_size=64):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = window_size

    def forward(self, x):
        B, T, C = x.shape
        Q = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K, V = Q, Q
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.zeros(T, T, device=x.device, dtype=torch.bool)
        for i in range(T):
            mask[i, max(0, i - self.window_size) : i + 1] = True
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous().view(B, T, C)
        return out


class _Cfg:
    n_head = 4
    n_embd = 128
    dropout = 0.0
    block_size = 256
    attention_variants = {"window_size": 64}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("T", [32, 64, 128, 256])
@pytest.mark.parametrize("B", [1, 2])
def test_full_parity(T, B):
    cfg = _Cfg()
    x = torch.randn(B, T, cfg.n_embd, device="cuda", dtype=torch.float32)
    orig = OrigFull().cuda().eval()
    trit = TritonFullAttention(cfg).cuda().eval()
    with torch.no_grad():
        ref = orig(x)
        out = trit(x)
    # fp16 kernel vs fp32 reference → looser tol
    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("T", [32, 64, 128, 256])
@pytest.mark.parametrize("W", [16, 64])
def test_sliding_parity(T, W):
    cfg = _Cfg()
    cfg.attention_variants = {"window_size": W}
    B = 2
    x = torch.randn(B, T, cfg.n_embd, device="cuda", dtype=torch.float32)
    orig = OrigSliding(window_size=W).cuda().eval()
    trit = TritonSlidingWindowAttention(cfg, window_size=W).cuda().eval()
    with torch.no_grad():
        ref = orig(x)
        out = trit(x)
    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_w1_matches_identity_path():
    """With window_size=0 → kernel_window=1 → each token only sees itself → out ≈ x (up to scale)."""
    cfg = _Cfg()
    # window_size=0 → kernel_window=1 → attend only to self
    # For self-only attention with Q=K=V=x the softmax is 1 and out = V = x (per head reshape).
    trit = TritonSlidingWindowAttention(cfg, window_size=0).cuda().eval()
    x = torch.randn(1, 64, 128, device="cuda")
    with torch.no_grad():
        out = trit(x)
    # Should be very close to x
    torch.testing.assert_close(out.float(), x.float(), atol=3e-2, rtol=3e-2)
