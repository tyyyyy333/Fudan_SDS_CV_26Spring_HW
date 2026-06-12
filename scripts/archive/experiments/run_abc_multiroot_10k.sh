#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-task2_abc_multiroot_b512_c10_w8_10k_v4}"
DATA_ROOT="${DATA_ROOT:-data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D}"
if [[ -n "${LEROBOT_TRAIN:-}" ]]; then
  read -r -a LEROBOT_TRAIN_CMD <<<"$LEROBOT_TRAIN"
else
  LEROBOT_TRAIN_CMD=(conda run -n hw3t2 lerobot-train)
fi
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"

REPO_IDS="[local/calvin_task_ABC_D_lerobot_0_4,local/calvin_task_ABC_D_lerobot_1_4,local/calvin_task_ABC_D_lerobot_2_4]"
ROOTS="[${DATA_ROOT}/calvin_task_ABC_D_lerobot_0_4,${DATA_ROOT}/calvin_task_ABC_D_lerobot_1_4,${DATA_ROOT}/calvin_task_ABC_D_lerobot_2_4]"

mkdir -p "logs" "outputs/task2/runs/${RUN_ID}"

exec "${LEROBOT_TRAIN_CMD[@]}" \
  --policy.type act \
  --policy.device cuda \
  --dataset.repo_id "${REPO_IDS}" \
  --dataset.root "${ROOTS}" \
  --dataset.video_backend pyav \
  --tolerance_s 0.01 \
  --output_dir "outputs/task2/runs/${RUN_ID}/abc_to_d_train" \
  --job_name hw3_abc_multiroot_b512_10k \
  --batch_size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --policy.optimizer_lr 0.0001 \
  --policy.chunk_size 10 \
  --policy.n_action_steps 10 \
  --policy.push_to_hub false \
  --policy.repo_id hw3_abc_multiroot_b512_10k_act \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor 2 \
  --persistent_workers true \
  --save_freq "${STEPS}" \
  --log_freq 20 \
  --wandb.enable false
