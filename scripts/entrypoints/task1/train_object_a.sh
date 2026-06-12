#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "${CONDA_ENV:-hw3t1}"
PYTHON="$CONDA_ENV_PYTHON"
SOURCE="${SOURCE:-$ROOT/data/task1/object_a/current}"
RUN_ID="${RUN_ID:-object_a_2dgs_white30k}"
OUTPUT="${OUTPUT:-$ROOT/outputs/task1/final/objects/object_a/model}"
ITERATIONS="${ITERATIONS:-${MAX_STEPS:-30000}}"
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.05}"
LAMBDA_DIST="${LAMBDA_DIST:-0}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-1}"

export PYTHONPATH="$ROOT/external/2d-gaussian-splatting/submodules/simple-knn:$ROOT/external/2d-gaussian-splatting/submodules/diff-surfel-rasterization${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi

cd "$ROOT/external/2d-gaussian-splatting"
EXTRA_ARGS=()
if [[ "$WHITE_BACKGROUND" == "1" ]]; then
  EXTRA_ARGS+=(--white_background)
fi
exec "$PYTHON" train.py \
  -s "$SOURCE" \
  -m "$OUTPUT" \
  --eval \
  --iterations "$ITERATIONS" \
  --test_iterations "$ITERATIONS" \
  --save_iterations "$ITERATIONS" \
  --checkpoint_iterations "$ITERATIONS" \
  --lambda_normal "$LAMBDA_NORMAL" \
  --lambda_dist "$LAMBDA_DIST" \
  "${EXTRA_ARGS[@]}"
