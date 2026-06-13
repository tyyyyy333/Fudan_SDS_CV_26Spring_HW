#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-hw3t2}"
DATASET_PATH="${DATASET_PATH:-data/calvin/task_ABC_D}"
CHECKPOINT="${CHECKPOINT:-outputs/task2/runs/task2_fair_abc_10k_cosine_b256_c10_s1000/abc_to_d_train/checkpoints/010000/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/task2/d_rollout}"
DEVICE="${DEVICE:-cuda}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DEBUG="${DEBUG:-0}"

source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "$CONDA_ENV"
PYTHON="$CONDA_ENV_PYTHON"

export PYTHONPATH="$ROOT/external/lerobot/src:$ROOT/external/calvin:$ROOT/external/calvin/calvin_models:$ROOT/external/calvin/calvin_env${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

args=(
  --dataset-path "$DATASET_PATH"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

if [[ "$CHECK_ONLY" == "1" ]]; then
  args+=(--check-only)
fi
if [[ "$DEBUG" == "1" ]]; then
  args+=(--debug)
fi

"$PYTHON" scripts/evaluate_task2_d_rollout.py "${args[@]}"
