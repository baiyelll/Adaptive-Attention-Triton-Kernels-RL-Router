"""
Plot kernel benchmark results.

Produces:
  results/latency_vs_n.png
  results/speedup_vs_w.png
  results/memory_vs_n.png

Usage:
    python -m benchmarks.plot_results
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV = Path("results/kernel_bench.csv")
OUT_DIR = Path("results")


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"{CSV} not found — run bench_kernels.py first")

    df = pd.read_csv(CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- latency vs N (log-log) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for impl in ["naive_fp32", "sdpa_full", "triton_full"]:
        sub = df[(df.impl == impl) & (df.W == -1)]
        if sub.empty:
            continue
        ax.plot(sub.N, sub.median_ms, marker="o", label=impl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Full causal attention latency vs N")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_vs_n.png", dpi=150)
    plt.close(fig)
    print("Wrote results/latency_vs_n.png")

    # ---- speedup of windowed kernel vs full at N=4096 ----
    target_n = 4096
    sub = df[df.N == target_n]
    if not sub.empty:
        full_ms = sub[sub.impl == "triton_full"]["median_ms"]
        if len(full_ms):
            full_ms = full_ms.values[0]
            win = sub[sub.impl == "triton_window"].copy()
            win["speedup"] = full_ms / win["median_ms"]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(win.W, win.speedup, marker="s", color="C1")
            ax.axhline(1.0, color="gray", ls="--")
            ax.set_xlabel("Window size W")
            ax.set_ylabel(f"Speedup vs Triton full (N={target_n})")
            ax.set_title("Sliding-window speedup")
            ax.grid(True, ls="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "speedup_vs_w.png", dpi=150)
            plt.close(fig)
            print("Wrote results/speedup_vs_w.png")

    # ---- memory vs N ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for impl in ["sdpa_full", "triton_full", "sdpa_mask", "triton_window"]:
        sub = df[df.impl == impl]
        if sub.empty:
            continue
        # for window impls take the largest W shown
        if "window" in impl or "mask" in impl:
            sub = sub.sort_values("W").groupby("N").last().reset_index()
        else:
            sub = sub[sub.W == -1]
        ax.plot(sub.N, sub.peak_mem_mb, marker="o", label=impl)
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_title("Peak GPU memory vs N")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "memory_vs_n.png", dpi=150)
    plt.close(fig)
    print("Wrote results/memory_vs_n.png")


if __name__ == "__main__":
    main()
