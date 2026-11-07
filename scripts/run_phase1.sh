#!/usr/bin/env bash
# Run Phase 1 on a CUDA machine (Colab T4 recommended).
# Usage:  bash scripts/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 0. Environment ==="
python -c "
from benchmarks.bench_utils import log_env, check_sdpa_backends
print(log_env())
print('SDPA backends:', check_sdpa_backends())
"

echo ""
echo "=== 1-3. Correctness tests ==="
python -m pytest tests/test_correctness.py -v --tb=short

echo ""
echo "=== 4. Kernel benchmarks ==="
python -m benchmarks.bench_kernels

echo ""
echo "=== 4b. Plots ==="
python -m benchmarks.plot_results

echo ""
echo "Done. Check results/ for CSVs and PNGs."
