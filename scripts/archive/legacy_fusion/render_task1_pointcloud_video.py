#!/usr/bin/env python3
"""Render Task 1 fused Gaussian PLY to an MP4 point-cloud preview."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from hw3cv.conversion import read_gaussian_ply
from video_utils import encode_video_from_frames


C0 = 0.28209479177387814


def _look_at(camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0).astype(np.float32)


def _project(points: np.ndarray, camera: np.ndarray, target: np.ndarray,
             width: int, height: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = _look_at(camera, target)
    cam = (points - camera) @ rot.T
    z = cam[:, 2]
    valid = z > 1e-3
    focal = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    x = (cam[:, 0] * focal / np.maximum(z, 1e-3) + width * 0.5).astype(np.int32)
    y = (height * 0.5 - cam[:, 1] * focal / np.maximum(z, 1e-3)).astype(np.int32)
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid], z[valid], valid


def render_video(manifest_path: Path, fused_ply: Path, output_path: Path,
                 frames: int, width: int, height: int, max_points: int, seed: int,
                 point_radius: int, label: bool) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = read_gaussian_ply(fused_ply)
    xyz = params["xyz"].astype(np.float32)
    rgb = np.clip(params["f_dc"] * C0 + 0.5, 0.0, 1.0)

    rng = np.random.RandomState(seed)
    if len(xyz) > max_points:
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    lo, hi = np.percentile(xyz, [2, 98], axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    radius = float(np.linalg.norm(hi - lo) * 0.55)
    radius = max(radius, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_frames: list[np.ndarray] = []

    target = center + np.array([0.0, 0.0, radius * 0.05], dtype=np.float32)
    bg = np.array([18, 20, 24], dtype=np.uint8)
    colors = (np.clip(rgb, 0, 1)[:, ::-1] * 255).astype(np.uint8)

    for frame_idx in range(frames):
        angle = 2 * math.pi * frame_idx / frames
        camera = center + np.array(
            [math.cos(angle) * radius * 1.85, math.sin(angle) * radius * 1.85, radius * 0.7],
            dtype=np.float32,
        )
        x, y, z, valid = _project(xyz, camera, target, width, height, 50.0)
        valid_colors = colors[valid]
        order = np.argsort(z)[::-1]

        image = np.broadcast_to(bg, (height, width, 3)).copy()
        offsets = [(0, 0)]
        for r in range(1, max(1, point_radius) + 1):
            offsets.extend([(r, 0), (-r, 0), (0, r), (0, -r)])
            if r > 1:
                offsets.extend([(r, r), (r, -r), (-r, r), (-r, -r)])

        for dx, dy in offsets:
            xx = np.clip(x[order] + dx, 0, width - 1)
            yy = np.clip(y[order] + dy, 0, height - 1)
            image[yy, xx] = valid_colors[order]

        if label:
            cv2.putText(
                image,
                f"Task 1 fused scene | {frame_idx + 1:02d}/{frames:02d}",
                (24, height - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        video_frames.append(image)

    encode_video_from_frames(video_frames, output_path, fps=30)
    print(f"Rendered {frames} frames from {fused_ply} -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/task1/scene_manifest.json"))
    parser.add_argument("--fused-ply", type=Path, default=Path("outputs/task1/fused_scene.ply"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output or Path(manifest["render"]["output"])
    frames = args.frames or int(manifest["render"].get("frames", 60))
    render_video(
        args.manifest,
        args.fused_ply,
        output,
        frames,
        args.width,
        args.height,
        args.max_points,
        args.seed,
        args.point_radius,
        args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
