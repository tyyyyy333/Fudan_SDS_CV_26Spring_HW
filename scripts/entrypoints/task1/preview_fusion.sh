#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

A_MODEL="${A_MODEL:-outputs/task1/final/objects/object_a/model}"
A_SOURCE="${A_SOURCE:-data/task1/object_a/current}"
A_PLY="${A_PLY:-$A_MODEL/point_cloud/iteration_30000/point_cloud_supported.ply}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/task1/experiments/fusion/object_a_supported_faithful_preview}"

if [[ ! -f "$A_PLY" ]]; then
    echo "Missing foreground-supported Object A PLY: $A_PLY" >&2
    echo "Run scripts/filter_object_a_gaussians.py without --dry-run first." >&2
    exit 2
fi

rm -rf "${OUTPUT_ROOT}_frames"

conda run -n hw3t1 python scripts/render_task1_official_background_video.py \
    --camera-mode path --frames 12 --resolution 1 \
    --object-a "$A_PLY" \
    --object-b outputs/task1/final/objects/object_b/model/textured.obj \
    --object-c outputs/task1/final/objects/object_c/model/textured.obj \
    --object-a-source "$A_SOURCE" \
    --object-a-model "$A_MODEL" --object-a-iteration 30000 \
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
    --frame-dir "${OUTPUT_ROOT}_frames" \
    --output "${OUTPUT_ROOT}.mp4" \
    --contact "${OUTPUT_ROOT}_contact.jpg"

echo "Preview contact: ${OUTPUT_ROOT}_contact.jpg"
echo "Preview video:   ${OUTPUT_ROOT}.mp4"
