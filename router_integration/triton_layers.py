"""
Drop-in Triton-backed attention modules that match the Hybrid Router interfaces.

Important matching details from the original code:
  - No separate QKV projections: input x is reshaped directly to Q=K=V.
  - Sliding window mask is inclusive: positions [i-W, i] → length W+1.
    We therefore call the kernel with window = config_window_size + 1.
  - head_dim = 32, n_embd = 128, n_head = 4 (from default_config).
  - Dropout is applied only in training; for parity / latency we run eval mode.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton kernels live in the sibling package
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernels.api import flash_attn, sliding_window_attn


class TritonFullAttention(nn.Module):
    """Full causal attention via the fused Triton kernel. Interface matches FullAttention."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = getattr(config, "dropout", 0.0)
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            [B, T, C]
        """
        B, T, C = x.shape
        # Same reshape as original: Q = K = V = x
        q = (
            x.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        )  # [B,H,T,D]
        k = q
        v = q

        if q.dtype != torch.float16:
            q = q.half()
            k = k.half()
            v = v.half()

        out = flash_attn(q, k, v)  # [B,H,T,D] fp16
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        if self.training and self.dropout > 0:
            out = self.dropout_layer(out)
        return out.to(x.dtype)


class TritonSlidingWindowAttention(nn.Module):
    """
    Sliding-window attention via the fused Triton kernel.
    Matches original band mask: positions [i-W, i] inclusive → kernel WINDOW = W+1.
    """

    def __init__(self, config, window_size: int = 64):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = getattr(config, "dropout", 0.0)
        self.window_size = window_size
        # Original mask length = window_size + 1
        self.kernel_window = window_size + 1
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = q
        v = q

        if q.dtype != torch.float16:
            q = q.half()
            k = k.half()
            v = v.half()

        out = sliding_window_attn(q, k, v, window=self.kernel_window)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        if self.training and self.dropout > 0:
            out = self.dropout_layer(out)
        return out.to(x.dtype)


class TritonHybridAttentionBlock(nn.Module):
    """
    Drop-in replacement for HybridAttentionBlock.
    Full and Sliding use Triton; Linear stays pure PyTorch (identical to original).
    """

    def __init__(self, layer_idx: int, config, linear_attn_module: nn.Module):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.dropout = getattr(config, "dropout", 0.0)

        window_size = 64
        if hasattr(config, "attention_variants"):
            window_size = config.attention_variants.get("window_size", 64)
        elif isinstance(getattr(config, "attention_variants", None), dict):
            window_size = config.attention_variants.get("window_size", 64)

        self.full_attn = TritonFullAttention(config)
        self.sliding_attn = TritonSlidingWindowAttention(
            config, window_size=window_size
        )
        # Linear is passed in (or created) so weights can be shared / loaded
        self.linear_attn = linear_attn_module

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(self.dropout),
        )
        self.attn_map = {
            0: self.full_attn,
            1: self.sliding_attn,
            2: self.linear_attn,
        }

    def forward(
        self, x: torch.Tensor, action: int = 0, return_routing_info: bool = False
    ):
        attn_out = self.attn_map[action](self.ln1(x))
        x = x + attn_out
        ffn_out = self.mlp(self.ln2(x))
        x = x + ffn_out
        if return_routing_info:
            return x, {"action": action, "layer_idx": self.layer_idx}
        return x


def swap_to_triton_attention(model: nn.Module, config) -> nn.Module:
    """
    In-place (or returned) conversion of a HybridTransformer:
    replace Full and Sliding modules with Triton versions; keep Linear weights.
    """
    from copy import deepcopy

    for i, block in enumerate(model.blocks):
        # Preserve the existing LinearAttention (weights + feature_proj)
        linear = block.linear_attn
        new_block = TritonHybridAttentionBlock(i, config, linear_attn_module=linear)
        # Copy LayerNorm and MLP weights so the frozen transformer stays identical
        new_block.ln1.load_state_dict(block.ln1.state_dict())
        new_block.ln2.load_state_dict(block.ln2.state_dict())
        new_block.mlp.load_state_dict(block.mlp.state_dict())
        model.blocks[i] = new_block
    return model
