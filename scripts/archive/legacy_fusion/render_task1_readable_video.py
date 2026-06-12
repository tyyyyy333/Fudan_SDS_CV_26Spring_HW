#!/usr/bin/env python3
"""Render a readable Task 1 fused-scene preview video.

This renderer is for report-quality visualization when headless Blender is not
available. It keeps the three objects and the reconstructed background in one
scene, but normalizes each asset before placement so A/B/C remain visible.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from hw3cv.conversion import read_gaussian_ply, read_obj
from video_utils import encode_video_from_frames


C0 = 0.28209479177387814


def normalize_points(xyz: np.ndarray, target_radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    lo, hi = np.percentile(xyz, [2, 98], axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    radius = float(np.linalg.norm(hi - lo) * 0.5)
    scale = target_radius / max(radius, 1e-6)
    return (xyz - center) * scale, center, scale


def load_gaussian_points(path: Path, target_radius: float, max_points: int, seed: int):
    params = read_gaussian_ply(path)
    xyz, _, scale = normalize_points(params["xyz"].astype(np.float32), target_radius)
    rgb = np.clip(params["f_dc"].astype(np.float32) * C0 + 0.5, 0.0, 1.0)
    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb, scale


def sample_mesh_points(path: Path, target_radius: float, n_points: int, seed: int):
    vertices, faces, colors = read_obj(path)
    vertices = vertices.astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(faces) == 0:
        xyz = vertices
        rgb = colors if colors is not None else np.full_like(vertices, 0.65)
    else:
        tri = vertices[faces]
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
        probs = areas / max(float(areas.sum()), 1e-8)
        tri_idx = rng.choice(len(faces), size=n_points, p=probs)
        r1 = rng.random(n_points)
        r2 = rng.random(n_points)
        flip = r1 + r2 > 1
        r1[flip] = 1 - r1[flip]
        r2[flip] = 1 - r2[flip]
        tri_s = tri[tri_idx]
        xyz = tri_s[:, 0] + r1[:, None] * (tri_s[:, 1] - tri_s[:, 0]) + r2[:, None] * (tri_s[:, 2] - tri_s[:, 0])
        if colors is not None and len(colors) == len(vertices):
            ctri = colors[faces][tri_idx]
            rgb = ctri[:, 0] + r1[:, None] * (ctri[:, 1] - ctri[:, 0]) + r2[:, None] * (ctri[:, 2] - ctri[:, 0])
        else:
            rgb = np.full((n_points, 3), 0.70, dtype=np.float32)
    xyz, _, _ = normalize_points(xyz.astype(np.float32), target_radius)
    return xyz, np.clip(rgb.astype(np.float32), 0.0, 1.0)


def look_at(camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0).astype(np.float32)


def project(points: np.ndarray, camera: np.ndarray, target: np.ndarray, width: int, height: int, fov_deg: float):
    rot = look_at(camera, target)
    cam = (points - camera) @ rot.T
    z = cam[:, 2]
    valid = z > 1e-3
    focal = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    x = (cam[:, 0] * focal / np.maximum(z, 1e-3) + width * 0.5).astype(np.int32)
    y = (height * 0.5 - cam[:, 1] * focal / np.maximum(z, 1e-3)).astype(np.int32)
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid], z[valid], valid


def draw_points(image: np.ndarray, xyz: np.ndarray, rgb: np.ndarray, camera: np.ndarray, target: np.ndarray,
                radius: int, fov: float):
    height, width = image.shape[:2]
    x, y, z, valid = project(xyz, camera, target, width, height, fov)
    colors = (np.clip(rgb[valid], 0, 1)[:, ::-1] * 255).astype(np.uint8)
    order = np.argsort(z)[::-1]
    offsets = [(0, 0)]
    for r in range(1, radius + 1):
        offsets.extend([(r, 0), (-r, 0), (0, r), (0, -r), (r, r), (r, -r), (-r, r), (-r, -r)])
    for dx, dy in offsets:
        xx = np.clip(x[order] + dx, 0, width - 1)
        yy = np.clip(y[order] + dy, 0, height - 1)
        image[yy, xx] = colors[order]


def make_contact(video_path: Path, output: Path, frames: list[int]):
    cap = cv2.VideoCapture(str(video_path))
    tiles = []
    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            tiles.append(frame)
    cap.release()
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    small = [cv2.resize(t, (w // 2, h // 2), interpolation=cv2.INTER_AREA) for t in tiles]
    rows = []
    for i in range(0, len(small), 3):
        row = small[i:i + 3]
        while len(row) < 3:
            row.append(np.full_like(small[0], 245))
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output), grid)


def render(args):
    bg_xyz, bg_rgb, _ = load_gaussian_points(args.background, 1.85, args.bg_points, args.seed)
    a_xyz, a_rgb, _ = load_gaussian_points(args.object_a, 0.42, args.object_points, args.seed + 1)
    b_xyz, b_rgb = sample_mesh_points(args.object_b, 0.38, args.mesh_points, args.seed + 2)
    c_xyz, c_rgb = sample_mesh_points(args.object_c, 0.38, args.mesh_points, args.seed + 3)

    # A shallow "tabletop" layout: background behind, A/B/C in front.
    bg_xyz = bg_xyz + np.array([0.0, 0.42, -0.28], dtype=np.float32)
    a_xyz = a_xyz + np.array([-0.82, -0.78, 0.08], dtype=np.float32)
    b_xyz = b_xyz + np.array([0.0, -0.82, 0.06], dtype=np.float32)
    c_xyz = c_xyz + np.array([0.82, -0.78, 0.06], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    video_frames: list[np.ndarray] = []

    target = np.array([0.0, -0.20, 0.02], dtype=np.float32)
    for frame_idx in range(args.frames):
        angle = 2 * math.pi * frame_idx / args.frames
        camera = np.array([math.cos(angle) * 3.35, math.sin(angle) * 2.25 - 0.55, 1.20], dtype=np.float32)
        image = np.full((args.height, args.width, 3), (238, 238, 232), dtype=np.uint8)

        # Draw larger, slightly muted background first, then foreground assets.
        draw_points(image, bg_xyz, bg_rgb * 0.96 + 0.04, camera, target, args.bg_radius, args.fov)
        draw_points(image, a_xyz, a_rgb, camera, target, args.object_radius, args.fov)
        draw_points(image, b_xyz, b_rgb, camera, target, args.object_radius, args.fov)
        draw_points(image, c_xyz, c_rgb, camera, target, args.object_radius, args.fov)
        video_frames.append(image)

    encode_video_from_frames(video_frames, args.output, fps=30)
    make_contact(args.output, args.contact, [0, args.frames // 6, args.frames // 3, args.frames // 2, args.frames * 2 // 3, args.frames * 5 // 6])
    print(f"Wrote {args.output}")
    print(f"Wrote {args.contact}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=Path, default=Path("outputs/task1/background_2dgs_trained_7k/point_cloud/iteration_7000/point_cloud.ply"))
    parser.add_argument("--object-a", type=Path, default=Path("outputs/task1/object_a_2dgs_full/point_cloud/iteration_30000/point_cloud.ply"))
    parser.add_argument("--object-b", type=Path, default=Path("outputs/task1/object_b/textured.obj"))
    parser.add_argument("--object-c", type=Path, default=Path("outputs/task1/object_c/textured.obj"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task1/fused_scene_readable.mp4"))
    parser.add_argument("--contact", type=Path, default=Path("outputs/task1/fused_scene_readable_contact.jpg"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--bg-points", type=int, default=180_000)
    parser.add_argument("--object-points", type=int, default=80_000)
    parser.add_argument("--mesh-points", type=int, default=80_000)
    parser.add_argument("--bg-radius", type=int, default=2)
    parser.add_argument("--object-radius", type=int, default=3)
    parser.add_argument("--fov", type=float, default=46.0)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--seed", type=int, default=20260603)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
