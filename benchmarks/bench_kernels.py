"""
Kernel benchmark sweep.

Produces results/kernel_bench.csv with median latency, IQR, peak memory and
achieved TFLOPS for:
  - naive fp32 (skipped when it would OOM)
  - SDPA full causal
  - SDPA + bool mask (sliding)
  - Triton full causal
  - Triton sliding-window

Usage (on Colab T4):
    python -m benchmarks.bench_kernels
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from benchmarks.bench_utils import bench, log_env, peak_memory_mb, check_sdpa_backends
from kernels.api import flash_attn, sliding_window_attn
from reference.naive_attention import naive_attention
from reference.sdpa_baseline import sdpa_full_causal, sdpa_sliding_window


# ---------------------------------------------------------------------------
# Config — edit to match your Router if needed
# ---------------------------------------------------------------------------
B = 4
H = 8          # typical small Router head count
D = 64
N_LIST = [256, 512, 1024, 2048, 4096, 8192]
W_LIST = [64, 128, 256, 512, 1024]

OUT_CSV = Path("results/kernel_bench.csv")
FIELDNAMES = [
    "impl", "B", "H", "N", "D", "W",
    "median_ms", "p25_ms", "p75_ms",
    "peak_mem_mb", "tflops",
]


def flops_full(B, H, N, D):
    # 2 * B * H * N * N * D  (QK + PV, halved by causality ≈ N²/2 each)
    return 2.0 * B * H * N * N * D


def flops_window(B, H, N, D, W):
    return 4.0 * B * H * N * min(W, N) * D


def run_one(impl_name, fn, B, H, N, D, W, flops):
    torch.cuda.reset_peak_memory_stats()
    # allocate outside the timed region
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    def call():
        fn(q, k, v)

    stats = bench(call, warmup=15, rep=50)
    mem = peak_memory_mb()
    tflops = (flops / 1e12) / (stats["median"] / 1e3) if stats["median"] > 0 else 0.0
    return {
        "impl": impl_name,
        "B": B, "H": H, "N": N, "D": D, "W": W if W is not None else -1,
        "median_ms": round(stats["median"], 4),
        "p25_ms": round(stats["p25"], 4),
        "p75_ms": round(stats["p75"], 4),
        "peak_mem_mb": round(mem, 2),
        "tflops": round(tflops, 3),
    }


def main():
    assert torch.cuda.is_available(), "CUDA required"
    env = log_env()
    print("Environment:", env)
    print("SDPA backends:", check_sdpa_backends())

    rows = []
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    for N in N_LIST:
        print(f"\n=== N = {N} ===")

        # --- full causal ---
        # naive (skip large N)
        if N <= 2048:
            try:
                r = run_one(
                    "naive_fp32",
                    lambda q, k, v: naive_attention(q, k, v, window=None),
                    B, H, N, D, None, flops_full(B, H, N, D),
                )
                rows.append(r)
                print(f"  naive_fp32: {r['median_ms']:.3f} ms")
            except RuntimeError as e:
                print(f"  naive_fp32 skipped: {e}")

        # SDPA full
        r = run_one(
            "sdpa_full",
            sdpa_full_causal,
            B, H, N, D, None, flops_full(B, H, N, D),
        )
        rows.append(r)
        print(f"  sdpa_full:  {r['median_ms']:.3f} ms")

        # Triton full
        r = run_one(
            "triton_full",
            flash_attn,
            B, H, N, D, None, flops_full(B, H, N, D),
        )
        rows.append(r)
        print(f"  triton_full:{r['median_ms']:.3f} ms")

        # --- sliding window ---
        for W in W_LIST:
            if W > N:
                continue
            # SDPA + mask
            try:
                r = run_one(
                    "sdpa_mask",
                    lambda q, k, v, w=W: sdpa_sliding_window(q, k, v, w),
                    B, H, N, D, W, flops_window(B, H, N, D, W),
                )
                rows.append(r)
                print(f"  sdpa_mask W={W}: {r['median_ms']:.3f} ms")
            except RuntimeError as e:
                print(f"  sdpa_mask W={W} skipped: {e}")

            # Triton window
            r = run_one(
                "triton_window",
                lambda q, k, v, w=W: sliding_window_attn(q, k, v, w),
                B, H, N, D, W, flops_window(B, H, N, D, W),
            )
            rows.append(r)
            print(f"  triton_win W={W}: {r['median_ms']:.3f} ms")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
