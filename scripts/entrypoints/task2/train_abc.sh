#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-hw3t2}"
DATA_ROOT="${DATA_ROOT:-data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D}"
RUN_ID="${RUN_ID:-task2_fair_abc_10k_cosine_b256_c10_s1000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/task2/runs/${RUN_ID}/abc_to_d_train}"
LOCK_FILE="${LOCK_FILE:-/tmp/hw3_task2_fair_abc_10k.lock}"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "A+B+C fair 10k training is already running: $LOCK_FILE"
  exit 0
fi

STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-20}"
SEED="${SEED:-1000}"

LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DECAY_STEPS="${DECAY_STEPS:-$STEPS}"
DECAY_LR="${DECAY_LR:-0.00001}"
REPO_IDS="[local/calvin_task_ABC_D_lerobot_0_4,local/calvin_task_ABC_D_lerobot_1_4,local/calvin_task_ABC_D_lerobot_2_4]"
ROOTS="[${DATA_ROOT}/calvin_task_ABC_D_lerobot_0_4,${DATA_ROOT}/calvin_task_ABC_D_lerobot_1_4,${DATA_ROOT}/calvin_task_ABC_D_lerobot_2_4]"

mkdir -p logs "outputs/task2/runs/${RUN_ID}"

source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "$CONDA_ENV"
PYTHON="$CONDA_ENV_PYTHON"

export PYTHONPATH="${ROOT}/external/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m lerobot.scripts.lerobot_train \
  --policy.type act \
  --policy.device cuda \
  --dataset.repo_id "$REPO_IDS" \
  --dataset.root "$ROOTS" \
  --dataset.video_backend pyav \
  --tolerance_s 0.01 \
  --output_dir "$OUTPUT_DIR" \
  --job_name "hw3_fair_abc_10k_cosine_b256_c10_s1000" \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --policy.chunk_size "$CHUNK_SIZE" \
  --policy.n_action_steps "$CHUNK_SIZE" \
  --policy.push_to_hub false \
  --policy.repo_id "hw3_fair_abc_10k_cosine_b256_c10_s1000_act" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --persistent_workers true \
  --save_freq "$SAVE_FREQ" \
  --log_freq "$LOG_FREQ" \
  --wandb.enable false \
  --use_policy_training_preset false \
  --optimizer.type adamw \
  --optimizer.lr "$LR" \
  --optimizer.weight_decay "$WEIGHT_DECAY" \
  --optimizer.grad_clip_norm "$GRAD_CLIP_NORM" \
  --optimizer.betas "[0.9,0.999]" \
  --optimizer.eps 1e-8 \
  --scheduler.type cosine_decay_with_warmup \
  --scheduler.num_warmup_steps "$WARMUP_STEPS" \
  --scheduler.num_decay_steps "$DECAY_STEPS" \
  --scheduler.peak_lr "$LR" \
  --scheduler.decay_lr "$DECAY_LR"
