"""
CUDA-event timing utilities and environment logging.
Never use time.time() for GPU kernels.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Callable, Dict, Any

import torch


def log_env(path: str | Path = "results/env.json") -> Dict[str, Any]:
    """Write torch / triton / CUDA / GPU info to JSON and return the dict."""
    import triton  # local import so the module can still be imported without triton

    env = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cc": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(env, f, indent=2)
    return env


def bench(
    fn: Callable[[], None],
    warmup: int = 25,
    rep: int = 100,
) -> Dict[str, float]:
    """
    Time `fn` with CUDA events.

    Returns dict with median, p25, p75 (milliseconds).
    Warm-up excludes Triton compile / autotune cost.
    """
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
        "mean": sum(times) / n,
        "min": times[0],
        "max": times[-1],
    }


def check_sdpa_backends(
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
) -> Dict[str, str]:
    """
    Probe which SDPBackend is available on the current GPU.
    On a T4 expect FLASH_ATTENTION to be unavailable.
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import torch.nn.functional as F

    if q is None:
        B, H, N, D = 1, 1, 64, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    results = {}
    for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            results[be.name] = "OK"
        except Exception as e:
            results[be.name] = f"unavailable: {type(e).__name__}"
    return results


def peak_memory_mb() -> float:
    """Current peak allocated memory in MiB (reset with torch.cuda.reset_peak_memory_stats)."""
    return torch.cuda.max_memory_allocated() / (1024 ** 2)
