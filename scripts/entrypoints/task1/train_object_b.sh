#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "${CONDA_ENV:-hw3t1}"
PYTHON="$CONDA_ENV_PYTHON"
ENV_BIN="$CONDA_ENV_BIN"
RUN_ID="${RUN_ID:-object_b}"
EXP_DIR="${EXP_DIR:-$ROOT/outputs/task1/$RUN_ID}"
TRIAL_NAME="${TRIAL_NAME:-train}"
TRIAL_DIR="$EXP_DIR/$TRIAL_NAME"
PROMPT="${PROMPT:-a delicious hamburger}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
MAX_STEPS="${MAX_STEPS:-10000}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-64}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-runwayml/stable-diffusion-v1-5}"
RESUME="${RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-$TRIAL_DIR/ckpts/last.ckpt}"
SEED="${SEED:-0}"
USE_PERP_NEG="${USE_PERP_NEG:-false}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-100.0}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-hw3-task1}"
WANDB_NAME="${WANDB_NAME:-$RUN_ID}"
WANDB_MODE="${WANDB_MODE:-offline}"

# OmegaConf parses a bare `key=` override as None. The prompt processor
# requires a string, so preserve an intentionally empty negative prompt.
if [[ -z "$NEGATIVE_PROMPT" ]]; then
  NEGATIVE_PROMPT_OVERRIDE='system.prompt_processor.negative_prompt=""'
else
  NEGATIVE_PROMPT_OVERRIDE="system.prompt_processor.negative_prompt=$NEGATIVE_PROMPT"
fi

if [[ -e "$EXP_DIR" ]]; then
  if [[ "$RESUME" != "1" || ! -f "$RESUME_FROM" ]]; then
    echo "Refusing to overwrite existing output: $EXP_DIR" >&2
    echo "Set RESUME=1 with a valid RESUME_FROM checkpoint to continue it." >&2
    exit 2
  fi
fi

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=("resume=$RESUME_FROM")
fi

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export WANDB_MODE
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="$ENV_BIN:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTHONPATH="$ROOT/external/tiny-cuda-nn/bindings/torch/build/lib.linux-x86_64-cpython-310${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/external/threestudio"
exec "$PYTHON" launch.py \
  --config configs/dreamfusion-sd.yaml \
  --train \
  "system.prompt_processor.prompt=$PROMPT" \
  "$NEGATIVE_PROMPT_OVERRIDE" \
  "system.prompt_processor.pretrained_model_name_or_path=$DIFFUSION_MODEL" \
  "system.prompt_processor.use_perp_neg=$USE_PERP_NEG" \
  "system.guidance.pretrained_model_name_or_path=$DIFFUSION_MODEL" \
  system.guidance_type=stable-diffusion-guidance \
  "system.guidance.guidance_scale=$GUIDANCE_SCALE" \
  system.guidance.weighting_strategy=sds \
  system.loss.lambda_sds=1.0 \
  "system.loggers.wandb.enable=$WANDB_ENABLE" \
  "system.loggers.wandb.project=$WANDB_PROJECT" \
  "system.loggers.wandb.name=$WANDB_NAME" \
  data.width="$TRAIN_RESOLUTION" \
  data.height="$TRAIN_RESOLUTION" \
  trainer.max_steps="$MAX_STEPS" \
  "seed=$SEED" \
  "exp_root_dir=$ROOT/outputs/task1" \
  "name=$RUN_ID" \
  "tag=$TRIAL_NAME" \
  use_timestamp=false \
  "${EXTRA_ARGS[@]}"
