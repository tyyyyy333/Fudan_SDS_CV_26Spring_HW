#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "${TASK1_ENV:-hw3t1}"
TASK1_PY="$CONDA_ENV_PYTHON"
TASK1_BIN="$CONDA_ENV_BIN"
resolve_conda_env "${TASK2_ENV:-hw3t2}"
TASK2_PY="$CONDA_ENV_PYTHON"
GS_PYTHONPATH="$ROOT/external/2d-gaussian-splatting/submodules/simple-knn:$ROOT/external/2d-gaussian-splatting/submodules/diff-surfel-rasterization"
TCNN_PYTHONPATH="$ROOT/external/tiny-cuda-nn/bindings/torch/build/lib.linux-x86_64-cpython-310"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="$TASK1_BIN:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

echo "== Storage =="
df -h "$ROOT"

echo
echo "== Task 1: hw3t1 =="
PYTHONPATH="$GS_PYTHONPATH" "$TASK1_PY" - <<'PY'
import torch
import cv2
import diffusers
import huggingface_hub
import accelerate
import imageio_ffmpeg
import pytorch_lightning
import simple_knn._C
import diff_surfel_rasterization
from nerfacc.grid import ray_aabb_intersect

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda, torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("accelerate:", accelerate.__version__)
print("pytorch_lightning:", pytorch_lightning.__version__)
print("OpenCV:", cv2.__version__)
print("ffmpeg:", imageio_ffmpeg.get_ffmpeg_exe())
if torch.cuda.is_available():
    rays_o = torch.zeros((1, 3), device="cuda")
    rays_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    aabbs = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], device="cuda")
    ray_aabb_intersect(rays_o, rays_d, aabbs)
    print("nerfacc CUDA backend: OK")
PY

(
  cd "$ROOT/external/threestudio"
  PYTHONPATH="$TCNN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" "$TASK1_PY" - <<'PY'
import tinycudann
import threestudio
from threestudio.models.guidance.stable_diffusion_guidance import StableDiffusionGuidance

print("threestudio Stable Diffusion guidance: OK")
PY
)

(
  cd "$ROOT/external/Magic123"
  "$TASK1_PY" - <<'PY'
from guidance.sd_utils import StableDiffusion

print("Magic123 Stable Diffusion guidance: OK")
PY
)

echo
echo "== Task 2: hw3t2 + local LeRobot =="
PYTHONPATH="$ROOT/external/lerobot/src" "$TASK2_PY" - <<'PY'
import torch
import transformers
import huggingface_hub
import lerobot

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda, torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("lerobot:", lerobot.__file__)
PY

if [[ -x "$TASK1_BIN/tectonic" ]]; then
  "$TASK1_BIN/tectonic" --version
else
  tectonic --version
fi

echo
echo "Environment checks passed."
