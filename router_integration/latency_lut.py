"""
Real-latency lookup table for the Hybrid Attention Router.

Measure once (CUDA events, full transformer block), then look up during PPO.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

VARIANT_NAMES = {0: "full", 1: "sliding", 2: "linear"}


class LatencyLUT:
    """
    Stores median latency (ms) for (variant, B, N) and supports linear
    interpolation over N.
    """

    def __init__(self, path: Optional[str | Path] = None):
        # key: (variant_name, B, N) -> median_ms
        self._table: Dict[Tuple[str, int, int], float] = {}
        self._by_variant_B: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
        self.overhead_ms: float = 0.0  # embedding + head (measured once)
        if path is not None:
            self.load(path)

    def add(self, variant: str, B: int, N: int, median_ms: float):
        self._table[(variant, B, N)] = median_ms
        key = (variant, B)
        self._by_variant_B.setdefault(key, []).append((N, median_ms))
        self._by_variant_B[key].sort()

    def layer_latency(self, variant: int | str, B: int, N: int) -> float:
        """Return latency (ms) for one transformer block of the given variant."""
        name = VARIANT_NAMES[variant] if isinstance(variant, int) else variant
        key = (name, B)
        if key not in self._by_variant_B or not self._by_variant_B[key]:
            raise KeyError(f"No LUT entries for variant={name}, B={B}")

        points = self._by_variant_B[key]
        # exact hit
        for n, ms in points:
            if n == N:
                return ms
        # clamp + linear interpolation
        if N <= points[0][0]:
            return points[0][1]
        if N >= points[-1][0]:
            return points[-1][1]
        for i in range(len(points) - 1):
            n0, m0 = points[i]
            n1, m1 = points[i + 1]
            if n0 <= N <= n1:
                t = (N - n0) / max(n1 - n0, 1)
                return m0 + t * (m1 - m0)
        return points[-1][1]

    def config_latency(self, actions: List[int], B: int, N: int) -> float:
        """Sum of per-layer latencies + constant overhead."""
        total = self.overhead_ms
        for a in actions:
            total += self.layer_latency(a, B, N)
        return total

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["variant", "B", "N", "median_ms", "overhead_ms"]
            )
            w.writeheader()
            for (variant, B, N), ms in sorted(self._table.items()):
                w.writerow(
                    {
                        "variant": variant,
                        "B": B,
                        "N": N,
                        "median_ms": round(ms, 6),
                        "overhead_ms": round(self.overhead_ms, 6),
                    }
                )

    def load(self, path: str | Path):
        path = Path(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add(
                    row["variant"],
                    int(row["B"]),
                    int(row["N"]),
                    float(row["median_ms"]),
                )
                if "overhead_ms" in row and row["overhead_ms"]:
                    self.overhead_ms = float(row["overhead_ms"])


def measure_block_latency(
    block: torch.nn.Module,
    x: torch.Tensor,
    action: int,
    warmup: int = 20,
    rep: int = 50,
) -> Dict[str, float]:
    """CUDA-event timing of a single HybridAttentionBlock forward."""
    assert x.is_cuda

    def fn():
        with torch.no_grad():
            block(x, action=action)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = len(times)
    return {
        "median": times[n // 2],
        "p25": times[n // 4],
        "p75": times[(3 * n) // 4],
    }
