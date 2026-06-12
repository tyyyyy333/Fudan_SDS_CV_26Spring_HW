#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-hw3t2}"
DATA_ROOT="${DATA_ROOT:-data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D}"
RUN_ID="${RUN_ID:-task2_single_b_actual40g_b512_c10_w8_5k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/task2/runs/${RUN_ID}/single_b_train}"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"

source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "$CONDA_ENV"
PYTHON="$CONDA_ENV_PYTHON"
export PYTHONPATH="${ROOT}/external/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m lerobot.scripts.lerobot_train \
  --policy.type act \
  --policy.device cuda \
  --dataset.repo_id local/calvin_task_ABC_D_lerobot_1_4 \
  --dataset.root "$DATA_ROOT/calvin_task_ABC_D_lerobot_1_4" \
  --dataset.video_backend pyav \
  --tolerance_s 0.01 \
  --output_dir "$OUTPUT_DIR" \
  --job_name hw3_single_b_actual40g_b512_5k \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --policy.chunk_size "$CHUNK_SIZE" \
  --policy.n_action_steps "$CHUNK_SIZE" \
  --policy.push_to_hub false \
  --policy.repo_id hw3_single_b_actual40g_b512_5k_act \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor 2 \
  --persistent_workers true \
  --save_freq "$STEPS" \
  --log_freq 20 \
  --wandb.enable false \
  --use_policy_training_preset false \
  --optimizer.type adamw \
  --optimizer.lr 0.0001 \
  --optimizer.weight_decay 0.0001 \
  --optimizer.grad_clip_norm 10 \
  --optimizer.betas "[0.9,0.999]" \
  --optimizer.eps 1e-8
