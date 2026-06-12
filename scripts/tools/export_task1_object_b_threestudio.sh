#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "${CONDA_ENV:-hw3t1}"
PYTHON="$CONDA_ENV_PYTHON"
ENV_BIN="$CONDA_ENV_BIN"
RUN_ID="${RUN_ID:-object_b}"
TRIAL_NAME="${TRIAL_NAME:-train}"
WORKSPACE="${WORKSPACE:-$ROOT/outputs/task1/$RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE}"
TRIAL_DIR="$WORKSPACE/$TRIAL_NAME"
CONFIG="$TRIAL_DIR/configs/parsed.yaml"
CKPT="${CKPT:-$TRIAL_DIR/ckpts/last.ckpt}"
ISOSURFACE_THRESHOLD="${ISOSURFACE_THRESHOLD:-auto}"
ISOSURFACE_REMOVE_OUTLIERS="${ISOSURFACE_REMOVE_OUTLIERS:-false}"
KEEP_EXPORT_CACHE="${KEEP_EXPORT_CACHE:-false}"

if [[ ! -f "$CONFIG" || ! -f "$CKPT" ]]; then
  echo "Missing threestudio config or checkpoint under: $TRIAL_DIR" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="$ENV_BIN:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTHONPATH="$ROOT/external/tiny-cuda-nn/bindings/torch/build/lib.linux-x86_64-cpython-310${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/external/threestudio"
"$PYTHON" launch.py \
  --config "$CONFIG" \
  --export \
  "exp_root_dir=$(dirname "$WORKSPACE")" \
  "name=$(basename "$WORKSPACE")" \
  "tag=$TRIAL_NAME" \
  "resume=$CKPT" \
  system.exporter_type=mesh-exporter \
  system.exporter.fmt=obj-mtl \
  system.exporter.save_uv=true \
  system.exporter.save_texture=true \
  system.exporter.texture_size=1024 \
  system.exporter.context_type=cuda \
  system.loggers.wandb.enable=false \
  "system.geometry.isosurface_threshold=$ISOSURFACE_THRESHOLD" \
  "system.geometry.isosurface_remove_outliers=$ISOSURFACE_REMOVE_OUTLIERS"

CONFIGURED_TRIAL_DIR="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle).get("trial_dir", ""))
PY
)"
EXPORT_DIR="$(
  for save_dir in "$TRIAL_DIR/save" "$CONFIGURED_TRIAL_DIR/save"; do
    [[ -d "$save_dir" ]] || continue
    find "$save_dir" -maxdepth 1 -type d -name 'it*-export'
  done | sort -V | tail -1
)"
if [[ -z "$EXPORT_DIR" ]]; then
  echo "No exported asset directory found for: $TRIAL_DIR" >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR"
cp "$EXPORT_DIR/model.obj" "$OUTPUT_DIR/textured.obj"
cp "$EXPORT_DIR/model.mtl" "$OUTPUT_DIR/model.mtl"
cp "$EXPORT_DIR/texture_kd.jpg" "$OUTPUT_DIR/texture_kd.jpg"
if [[ "$KEEP_EXPORT_CACHE" != "true" ]]; then
  rm -rf "$EXPORT_DIR"
fi
echo "Exported Object B to: $OUTPUT_DIR"
