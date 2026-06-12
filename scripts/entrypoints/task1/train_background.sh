#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "${CONDA_ENV:-hw3t1}"
PYTHON="$CONDA_ENV_PYTHON"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/task1/final/environment/kitchen_2dgs}"
MAX_STEPS="${MAX_STEPS:-7000}"

export PYTHONPATH="$ROOT/external/2d-gaussian-splatting/submodules/simple-knn:$ROOT/external/2d-gaussian-splatting/submodules/diff-surfel-rasterization${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$ROOT/logs"
cd "$ROOT/external/2d-gaussian-splatting"

exec "$PYTHON" train.py \
  -s "$ROOT/data/background/kitchen" \
  -m "$RUN_DIR" \
  --eval \
  --iterations "$MAX_STEPS" \
  --test_iterations "$MAX_STEPS" \
  --save_iterations "$MAX_STEPS" \
  --quiet
