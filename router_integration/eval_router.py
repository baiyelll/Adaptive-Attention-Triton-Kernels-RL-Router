"""
Final evaluation: compare policies under real end-to-end CUDA timing.

Reports perplexity (cross-entropy) and measured latency for:
  - all full / all sliding / all linear
  - random per-layer
  - a trained real-latency router (if checkpoint provided)

Usage:
    python -m router_integration.eval_router --lut results/latency_lut.csv \\
        --checkpoint results/real_latency_runs/router_seed0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.bench_utils import bench
from router_integration.latency_lut import LatencyLUT
from router_integration.train_ppo_real_latency import HybridTransformer, SimplePPO


def make_batch(B=4, T=256, vocab=50257, device="cuda"):
    idx = torch.randint(0, vocab, (B, T), device=device)
    return idx, idx


@torch.no_grad()
def eval_config(model, actions, n_batches=20, B=4, T=256):
    losses = []
    for _ in range(n_batches):
        idx, targets = make_batch(B, T)
        loss, _ = model(idx, targets=targets, actions=actions)
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.exp(np.mean(losses)))


def time_config(model, actions, B=4, T=256, warmup=10, rep=30):
    idx, _ = make_batch(B, T)

    def fn():
        with torch.no_grad():
            model(idx, actions=actions)

    return bench(fn, warmup=warmup, rep=rep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut", default="results/latency_lut.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="results/router_eval.csv")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    model = HybridTransformer().cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layer = model.n_layer
    rows = []

    configs = {
        "all_full": [0] * n_layer,
        "all_sliding": [1] * n_layer,
        "all_linear": [2] * n_layer,
    }
    # random
    rng = np.random.default_rng(0)
    for i in range(5):
        configs[f"random_{i}"] = rng.integers(0, 3, size=n_layer).tolist()

    print("Evaluating fixed configs…")
    for name, actions in configs.items():
        loss, ppl = eval_config(model, actions)
        timing = time_config(model, actions)
        rows.append(
            {
                "policy": name,
                "actions": str(actions),
                "loss": round(loss, 4),
                "perplexity": round(ppl, 2),
                "latency_ms": round(timing["median"], 4),
                "p25": round(timing["p25"], 4),
                "p75": round(timing["p75"], 4),
            }
        )
        print(
            f"  {name:16s}  loss={loss:.4f}  ppl={ppl:.1f}  lat={timing['median']:.3f} ms"
        )

    # Trained router (if provided)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading router from {args.checkpoint}")
        agent = SimplePPO(torch.device("cuda"))
        ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.value.load_state_dict(ckpt["value"])

        # Roll out one full set of layer decisions from a dummy obs trajectory
        from router_integration.routing_env import AttentionRoutingEnv

        class Cfg:
            n_layer = 6
            n_embd = 128
            n_head = 4
            block_size = 256
            attention_variants = {"window_size": 64}
            reward = {"alpha": 1.0, "beta": 0.1, "gamma": 0.05}
            rl = {"max_steps_per_episode": 256}

        lut = LatencyLUT(args.lut) if Path(args.lut).exists() else None

        def loader():
            while True:
                yield make_batch()

        env = AttentionRoutingEnv(
            model,
            loader(),
            Cfg(),
            torch.device("cuda"),
            lut,
            use_real_latency=lut is not None,
        )
        obs, _ = env.reset()
        actions = []
        for _ in range(n_layer):
            a, _, _ = agent.select(obs)
            obs, _, term, trunc, info = env.step(a)
            actions = info["actions"]
            if term or trunc:
                break

        loss, ppl = eval_config(model, actions)
        timing = time_config(model, actions)
        rows.append(
            {
                "policy": "trained_router",
                "actions": str(actions),
                "loss": round(loss, 4),
                "perplexity": round(ppl, 2),
                "latency_ms": round(timing["median"], 4),
                "p25": round(timing["p25"], 4),
                "p75": round(timing["p75"], 4),
            }
        )
        print(
            f"  trained_router   loss={loss:.4f}  ppl={ppl:.1f}  lat={timing['median']:.3f} ms  actions={actions}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
