#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "${CONDA_ENV:-hw3t1}"
PYTHON="$CONDA_ENV_PYTHON"
RUN_ID="${RUN_ID:-object_c_v5}"
INPUT="${INPUT:-$ROOT/external/Magic123/data/hw3/object_c/rgba.png}"
WORKSPACE="${WORKSPACE:-$ROOT/outputs/task1/experiments/object_c/$RUN_ID}"
TEXT="${TEXT:-A high-resolution DSLR image of a white game controller}"
NEGATIVE="${NEGATIVE:-two heads, duplicate head, double face, mirrored front, extra face, extra object, duplicated geometry, floating parts, siamese, deformed, malformed}"
ITERS="${ITERS:-8000}"
LAMBDA_SD="${LAMBDA_SD:-1.0}"
LAMBDA_ZERO123="${LAMBDA_ZERO123:-80}"
GUIDANCE_SD="${GUIDANCE_SD:-50}"
GUIDANCE_ZERO123="${GUIDANCE_ZERO123:-10}"
LAMBDA_DEPTH="${LAMBDA_DEPTH:-0.1}"

if [[ -e "$WORKSPACE" ]]; then
  echo "Refusing to overwrite existing output: $WORKSPACE" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
cd "$ROOT/external/Magic123"
exec "$PYTHON" main.py \
  -O \
  --text "$TEXT" \
  --negative "$NEGATIVE" \
  --sd_version 1.5 \
  --image "$INPUT" \
  --workspace "$WORKSPACE" \
  --optim adam \
  --iters "$ITERS" \
  --guidance SD zero123 \
  --lambda_guidance "$LAMBDA_SD" "$LAMBDA_ZERO123" \
  --guidance_scale "$GUIDANCE_SD" "$GUIDANCE_ZERO123" \
  --dir_texts_neg \
  --latent_iter_ratio 0 \
  --normal_iter_ratio 0.2 \
  --lambda_depth "$LAMBDA_DEPTH" \
  --t_range 0.2 0.6 \
  --bg_radius -1 \
  --save_mesh
