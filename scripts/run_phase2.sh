#!/usr/bin/env bash
# Phase 2 on a CUDA T4 (Colab or local).
# Prerequisites: Phase 1 already green (kernels + correctness).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Phase 2: Kernel tests ==="
python -m pytest tests/test_router_parity.py -v --tb=short

echo ""
echo "=== Phase 7: Build real-latency LUT ==="
python -m benchmarks.build_latency_lut

echo ""
echo "=== Phase 8: Retrain PPO with real latency (1 seed, short) ==="
python -m router_integration.train_ppo_real_latency --seeds 0 --epochs 20 --steps-per-epoch 64

echo ""
echo "=== Phase 9: Final evaluation ==="
python -m router_integration.eval_router \
  --lut results/latency_lut.csv \
  --checkpoint results/real_latency_runs/router_seed0.pt

echo ""
echo "Done. Check results/latency_lut.csv, results/real_latency_runs/, results/router_eval.csv"
