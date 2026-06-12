#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHONPATH="$ROOT/src"
FINAL_ROOT="${FINAL_ROOT:-outputs/task1/final}"
QUALITY_ROOT="$FINAL_ROOT/quality"
SCENE_ROOT="$FINAL_ROOT/scene"
mkdir -p "$QUALITY_ROOT" "$SCENE_ROOT"

log() { echo "[auto_finish $(date +%H:%M:%S)] $*"; }

# --- Verify the accepted Object A model and its standalone views ---
A_MODEL="${A_MODEL:-outputs/task1/final/objects/object_a/model}"
A_SOURCE="${A_SOURCE:-data/task1/object_a/current}"
A_ITERATION="${A_ITERATION:-30000}"
DEFAULT_A_PLY="$A_MODEL/point_cloud/iteration_${A_ITERATION}/point_cloud.ply"
SUPPORTED_A_PLY="$A_MODEL/point_cloud/iteration_${A_ITERATION}/point_cloud_supported.ply"
if [[ -f "$SUPPORTED_A_PLY" ]]; then
    DEFAULT_A_PLY="$SUPPORTED_A_PLY"
    DEFAULT_A_SUPPORT_VIEWS=0
    DEFAULT_A_SUPPORT_RATIO=0
else
    DEFAULT_A_SUPPORT_VIEWS=5
    DEFAULT_A_SUPPORT_RATIO=0.70
fi
A_PLY="${A_PLY:-$DEFAULT_A_PLY}"
A_SUPPORT_VIEWS="${A_SUPPORT_VIEWS:-$DEFAULT_A_SUPPORT_VIEWS}"
A_SUPPORT_RATIO="${A_SUPPORT_RATIO:-$DEFAULT_A_SUPPORT_RATIO}"
A_VIEWS="${A_VIEWS:-outputs/task1/final/quality/object_a_views}"

if [ -f "$A_PLY" ]; then
    log "Object A model found"
else
    log "Missing accepted Object A model: $A_PLY"
    exit 2
fi

# Check NaN
log "Checking A PLY for NaN..."
NAN_COUNT=$(PYTHONPATH="$PYTHONPATH" conda run -n hw3t1 python -c "
from hw3cv.conversion import read_gaussian_ply
from pathlib import Path
import numpy as np
p = read_gaussian_ply(Path('$A_PLY'))
xyz = p['xyz']
n = np.isnan(xyz).any(axis=1).sum()
print(n)
" 2>/dev/null || echo "999")

log "NaN count: $NAN_COUNT"
if [ "$NAN_COUNT" -gt 0 ]; then
    log "WARNING: $NAN_COUNT NaN Gaussians found!"
fi

# Render A camera views
if [ ! -d "$A_VIEWS" ] || ! find "$A_VIEWS" -maxdepth 1 -name 'render_*.png' -print -quit | grep -q .; then
    log "Rendering Object A camera views..."
    rm -rf "$A_VIEWS"
    conda run -n hw3t1 python scripts/render_2dgs_camera_views.py \
        --source "$A_SOURCE" \
        --model "$A_MODEL" --iteration "$A_ITERATION" \
        --output "$A_VIEWS" \
        --num-views 16 --stride 5 --resolution 1 --white-background \
        2>&1 | tail -3
fi

# --- Verify/export Object B ---
B_WORKSPACE="${B_WORKSPACE:-outputs/task1/final/objects/object_b/training}"
B_MODEL="${B_MODEL:-outputs/task1/final/objects/object_b/model}"
B_OBJ="$B_MODEL/textured.obj"
B_CKPT="$B_WORKSPACE/train/ckpts/last.ckpt"

if [ -f "$B_CKPT" ] && { [ ! -f "$B_OBJ" ] || [ "$B_CKPT" -nt "$B_OBJ" ]; }; then
    log "Exporting Object B from $(basename "$B_WORKSPACE")"
    WORKSPACE="$B_WORKSPACE" OUTPUT_DIR="$B_MODEL" \
        bash scripts/tools/export_task1_object_b_threestudio.sh
elif [ -f "$B_OBJ" ]; then
    log "Object B canonical OBJ is newer than its checkpoint"
else
    log "Missing Object B OBJ and checkpoint: $B_OBJ / $B_CKPT"
    exit 2
fi

log "Looking for B mesh..."
B_MESH=$(find "$B_MODEL" -name "*.obj" -o -name "*.ply" 2>/dev/null | head -5)
log "B mesh candidates: $B_MESH"

# Render B turntable
log "Rendering Object B turntable..."
rm -rf "$QUALITY_ROOT/object_b_turntable"
if [ -f "$B_OBJ" ]; then
    conda run -n hw3t1 python scripts/render_obj_turntable.py \
        --obj "$B_OBJ" --outdir "$QUALITY_ROOT/object_b_turntable" \
        --frames 72 --samples 180000 --size 768 --up-axis z 2>&1 | tail -3
fi

# --- Final scene ---
log "Rendering final scene..."
FRAME_DIR="$SCENE_ROOT/frames"
rm -rf "$FRAME_DIR"

# Use Object C mesh (converted to gaussians by the renderer)
C_MESH="outputs/task1/final/objects/object_c/model/textured.obj"

conda run -n hw3t1 python scripts/render_task1_official_background_video.py \
    --camera-mode path --frames 360 --resolution 1 \
    --object-a "$A_PLY" \
    --object-b "$B_OBJ" --object-c "$C_MESH" \
    --object-a-source "$A_SOURCE" \
    --object-a-model "$A_MODEL" --object-a-iteration "$A_ITERATION" \
    --object-a-render-mode gaussian \
    --object-a-footprint-mode original \
    --no-object-a-dc-only \
    --object-a-max-scale-ratio 0.12 \
    --object-a-max-anisotropy 25 \
    --object-a-max-center-radius 1.6 \
    --object-a-support-min-views 0 \
    --object-a-support-min-ratio 0 \
    --object-a-center-percentile 100 \
    --object-points 143000 --mesh-points 180000 \
    --object-scene-scale 0.072 \
    --object-a-scale-multiplier 0.95 \
    --object-b-scale-multiplier 1.45 \
    --object-c-scale-multiplier 1.30 \
    --layout-mode ring \
    --layout-forward 0.0 --layout-up -0.12 \
    --object-a-radius 0.34 --object-a-angle -0.8 --object-a-height 0.20 \
    --object-b-radius 0.34 --object-b-angle 0.8 --object-b-height 0.25 \
    --object-c-radius 0.34 --object-c-angle 1.8 --object-c-height 0.25 \
    --object-a-yaw -1.20 --object-a-pitch 1.5707963268 \
    --object-b-yaw 0.25 --object-b-pitch 0.0 \
    --object-c-yaw -0.30 --object-c-pitch 0.0 \
    --frame-dir "$FRAME_DIR" \
    --output "$SCENE_ROOT/fused_scene_360.mp4" \
    --contact "$SCENE_ROOT/fused_scene_360_contact.jpg" \
    2>&1 | grep -E "Composite|Wrote|Error"

rm -rf "$FRAME_DIR"
log "ALL DONE!"
ls -lh "$SCENE_ROOT/fused_scene_360.mp4" "$SCENE_ROOT/fused_scene_360_contact.jpg"
