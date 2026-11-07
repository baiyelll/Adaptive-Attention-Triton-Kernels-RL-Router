"""
fp32 reference attention used as the correctness oracle.
"""

from __future__ import annotations

import torch


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int | None = None,
) -> torch.Tensor:
    """
    Standard scaled-dot-product attention with causal (+ optional sliding-window) mask.

    Args:
        q, k, v: (B, H, N, D) any floating dtype (will be cast to fp32 internally).
        window:  None → pure causal; int W → each token attends to the previous
                 (W-1) tokens + itself, i.e. (i - j) < W.

    Returns:
        (B, H, N, D) same dtype as the input q.
    """
    B, H, N, D = q.shape
    scale = D ** -0.5

    # fp32 scores
    s = (q.float() @ k.float().transpose(-1, -2)) * scale  # (B,H,N,N)

    i = torch.arange(N, device=q.device)[:, None]
    j = torch.arange(N, device=q.device)[None, :]
    mask = j <= i  # causal
    if window is not None:
        mask = mask & ((i - j) < window)  # sliding window (includes self)

    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = p @ v.float()
    return out.to(q.dtype)
