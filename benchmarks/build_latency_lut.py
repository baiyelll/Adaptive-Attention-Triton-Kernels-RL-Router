"""
Build a real-latency lookup table for the Hybrid Router on a CUDA GPU (T4).

Times one full transformer *block* (attn + MLP + norms) for each variant
{full, sliding, linear} at several (B, N) points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.bench_utils import log_env
from router_integration.latency_lut import (
    LatencyLUT,
    measure_block_latency,
    VARIANT_NAMES,
)
from router_integration.triton_layers import (
    TritonFullAttention,
    TritonSlidingWindowAttention,
    TritonHybridAttentionBlock,
)


# ---------------------------------------------------------------------------
# Minimal config + Linear attention (mirrors the original)
# ---------------------------------------------------------------------------
class _Cfg:
    n_head = 4
    n_embd = 128
    dropout = 0.0
    block_size = 256
    attention_variants = {"window_size": 64, "linear_dim": 64}


class _LinearAttention(nn.Module):
    """Copy of the original Performer-style linear attention (no Triton)."""

    def __init__(self, config, feature_dim: int = 64):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.feature_proj = nn.Linear(self.head_dim, feature_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = Q
        V = Q
        Q_feat = torch.nn.functional.elu(self.feature_proj(Q)) + 1
        K_feat = torch.nn.functional.elu(self.feature_proj(K)) + 1
        KV = K_feat.transpose(-2, -1) @ V
        out = Q_feat @ KV
        K_sum = K_feat.sum(dim=2, keepdim=True)
        normalizer = (Q_feat @ K_sum.transpose(-2, -1)) + 1e-6
        out = out / normalizer
        return out.transpose(1, 2).contiguous().view(B, T, C)


def build_block(device="cuda"):
    cfg = _Cfg()
    linear = _LinearAttention(cfg, feature_dim=64).to(device)
    block = TritonHybridAttentionBlock(0, cfg, linear_attn_module=linear).to(device)
    block.eval()
    return block, cfg


def main():
    assert torch.cuda.is_available(), "CUDA required"
    log_env("results/env_lut.json")

    block, cfg = build_block()
    lut = LatencyLUT()

    # Points the Router actually uses + neighbours for safe interpolation
    B_LIST = [1, 4]
    N_LIST = [64, 128, 256, 512]

    print("Measuring per-block latency (attn + MLP + norms)…")
    for B in B_LIST:
        for N in N_LIST:
            x = torch.randn(B, N, cfg.n_embd, device="cuda", dtype=torch.float32)
            for action, name in VARIANT_NAMES.items():
                stats = measure_block_latency(
                    block, x, action=action, warmup=15, rep=40
                )
                lut.add(name, B, N, stats["median"])
                print(
                    f"  B={B} N={N} {name:8s}: "
                    f"{stats['median']:.4f} ms  "
                    f"(IQR {stats['p25']:.4f}-{stats['p75']:.4f})"
                )

    # Overhead: embedding + final LN + head (approximate with a short run)
    # For the small model this is tiny; measure once at B=4, N=256.
    emb = nn.Embedding(50257, cfg.n_embd).cuda()
    pos = nn.Embedding(cfg.block_size, cfg.n_embd).cuda()
    ln = nn.LayerNorm(cfg.n_embd).cuda()
    head = nn.Linear(cfg.n_embd, 50257, bias=False).cuda()
    idx = torch.randint(0, 50257, (4, 256), device="cuda")

    def overhead_fn():
        with torch.no_grad():
            t = emb(idx) + pos(torch.arange(256, device="cuda"))
            t = ln(t)
            _ = head(t)

    from benchmarks.bench_utils import bench

    oh = bench(overhead_fn, warmup=10, rep=30)
    lut.overhead_ms = oh["median"]
    print(f"\nEmbedding+head overhead: {lut.overhead_ms:.4f} ms")

    out = Path("results/latency_lut.csv")
    lut.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
