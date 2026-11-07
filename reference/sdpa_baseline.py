"""
PyTorch scaled_dot_product_attention baselines.

"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_full_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full causal attention via F.scaled_dot_product_attention."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def sdpa_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """
    Sliding-window attention by materialising a boolean attn_mask.
    This is the honest non-fused baseline on a T4.
    """
    B, H, N, D = q.shape
    i = torch.arange(N, device=q.device)[:, None]
    j = torch.arange(N, device=q.device)[None, :]
    # True = keep, False = mask out (SDPA expects additive mask or bool)
    keep = (j <= i) & ((i - j) < window)
    # SDPA bool mask: True means "attend", False means "-inf"
    return F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
