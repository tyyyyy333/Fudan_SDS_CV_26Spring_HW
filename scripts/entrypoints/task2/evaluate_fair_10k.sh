#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-hw3t2}"
DATA_ROOT="${DATA_ROOT:-data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D}"
NUM_WORKERS="${NUM_WORKERS:-16}"

source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "$CONDA_ENV"
PYTHON="$CONDA_ENV_PYTHON"

export PYTHONPATH="${ROOT}/external/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

"$PYTHON" scripts/audit_task2_fairness.py

"$PYTHON" scripts/evaluate_task2_d_offline.py \
  --d-root "$DATA_ROOT/calvin_task_ABC_D_lerobot_3_4" \
  --output-dir outputs/task2/zero_shot_d_action_error_full \
  --batch-size 256 \
  --num-workers "$NUM_WORKERS" \
  --max-batches 0

"$PYTHON" scripts/analyze_task2_action_chunking.py \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/task2/action_chunking_robustness \
  --batch-size 128 \
  --num-workers "$NUM_WORKERS" \
  --max-batches 64

conda run -n hw3t1 python scripts/plot_task2_action_chunking.py \
  --metrics outputs/task2/action_chunking_robustness/metrics.json
