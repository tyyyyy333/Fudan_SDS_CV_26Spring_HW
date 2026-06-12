#!/usr/bin/env python3
"""Render a true 3D Task 1 walkthrough from PLY/OBJ assets.

Unlike the presentation composite fallback, this script never uses 2D object
sprites. It samples/loads the reconstructed 3D assets, normalizes each asset to
a controlled display scale, places A/B/C in front of the reconstructed
background, and renders a software point-splat orbit video.
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


def robust_normalize(xyz: np.ndarray, radius: float, crop: float = 2.4):
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    shifted = xyz.astype(np.float32) - center
    scale = radius / max(float(np.linalg.norm(hi - lo) * 0.5), 1e-6)
    shifted *= scale
    keep = np.linalg.norm(shifted, axis=1) < crop * radius
    return shifted[keep], keep


def load_gaussian(path: Path, radius: float, max_points: int, seed: int, projection_image: Path | None = None):
    params = read_gaussian_ply(path)
    xyz, keep = robust_normalize(params["xyz"], radius)
    rgb = np.clip(params["f_dc"] * C0 + 0.5, 0.0, 1.0)[keep]
    if projection_image is not None and projection_image.exists():
        projected, valid = project_image_colors(params["xyz"], projection_image)
        projected = projected[keep]
        valid = valid[keep]
        rgb[valid] = 0.90 * projected[valid] + 0.10 * rgb[valid]
    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz.astype(np.float32), rgb.astype(np.float32)


def sample_mesh(
    path: Path,
    radius: float,
    n_points: int,
    seed: int,
    fallback_color,
    projection_image: Path | None = None,
):
    vertices, faces, colors = read_obj(path)
    vertices = vertices.astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(faces) == 0:
        xyz = vertices
        rgb = colors if colors is not None else np.tile(np.array(fallback_color, dtype=np.float32), (len(xyz), 1))
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
            rgb = np.tile(np.array(fallback_color, dtype=np.float32), (n_points, 1))
    if projection_image is not None and projection_image.exists():
        projected, mask = project_image_colors(xyz, projection_image)
        rgb[mask] = 0.85 * projected[mask] + 0.15 * rgb[mask]

    xyz, keep = robust_normalize(xyz, radius)
    return xyz.astype(np.float32), np.clip(rgb.astype(np.float32), 0, 1)[keep]


def project_image_colors(xyz: np.ndarray, image_path: Path):
    from PIL import Image

    rgba = Image.open(image_path).convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    image = arr[:, :, :3]
    alpha = arr[:, :, 3]
    if alpha.max() > 0.99 and alpha.min() < 0.99:
        mask = alpha > 0.05
    else:
        mask = image.mean(axis=2) > 0.035
    if not np.any(mask):
        return np.zeros((len(xyz), 3), dtype=np.float32), np.zeros(len(xyz), dtype=bool)

    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    px0, px1 = np.percentile(xyz[:, 0], [2, 98])
    pz0, pz1 = np.percentile(xyz[:, 2], [2, 98])
    u = np.clip((xyz[:, 0] - px0) / max(float(px1 - px0), 1e-6), 0.0, 1.0)
    v = np.clip((xyz[:, 2] - pz0) / max(float(pz1 - pz0), 1e-6), 0.0, 1.0)
    ix = np.round(x0 + u * (x1 - x0)).astype(np.int32)
    iy = np.round(y1 - v * (y1 - y0)).astype(np.int32)
    colors = image[iy, ix].astype(np.float32)
    valid = mask[iy, ix]
    return colors, valid


def transform(xyz: np.ndarray, loc, rot_z: float = 0.0):
    c, s = math.cos(rot_z), math.sin(rot_z)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return xyz @ R.T + np.array(loc, dtype=np.float32)


def look_at(camera: np.ndarray, target: np.ndarray):
    forward = target - camera
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0).astype(np.float32)


def project(xyz, camera, target, width, height, fov):
    rot = look_at(camera, target)
    cam = (xyz - camera) @ rot.T
    z = cam[:, 2]
    valid = z > 1e-3
    focal = 0.5 * width / math.tan(math.radians(fov) * 0.5)
    x = (cam[:, 0] * focal / np.maximum(z, 1e-3) + width * 0.5).astype(np.int32)
    y = (height * 0.5 - cam[:, 1] * focal / np.maximum(z, 1e-3)).astype(np.int32)
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid], z[valid], valid


def splat(image, xyz, rgb, camera, target, fov, radius):
    h, w = image.shape[:2]
    x, y, z, valid = project(xyz, camera, target, w, h, fov)
    colors = (np.clip(rgb[valid], 0, 1)[:, ::-1] * 255).astype(np.uint8)
    # Draw far points first so nearer reconstructed assets remain visible.
    order = np.argsort(z)[::-1]
    offsets = [(0, 0)]
    if radius >= 1:
        offsets += [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if radius >= 2:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
    for dx, dy in offsets:
        xx = np.clip(x[order] + dx, 0, w - 1)
        yy = np.clip(y[order] + dy, 0, h - 1)
        image[yy, xx] = colors[order]


def choose_object_a(default_path: Path, masked_path: Path):
    if masked_path.exists():
        return masked_path
    return default_path


def make_contact(video_path: Path, contact_path: Path, frames: list[int]):
    cap = cv2.VideoCapture(str(video_path))
    tiles = []
    for i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            tiles.append(cv2.resize(frame, (426, 240), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(tiles) >= 6:
        grid = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:6], axis=1)], axis=0)
        cv2.imwrite(str(contact_path), grid)


def render(args):
    object_a_path = choose_object_a(args.object_a, args.object_a_masked)
    print(f"Using Object A: {object_a_path}")
    bg_xyz, bg_rgb = load_gaussian(args.background, radius=2.08, max_points=args.bg_points, seed=args.seed)
    a_xyz, a_rgb = load_gaussian(
        object_a_path,
        radius=0.42,
        max_points=args.object_points,
        seed=args.seed + 1,
        projection_image=args.object_a_image,
    )
    b_xyz, b_rgb = sample_mesh(args.object_b, radius=0.38, n_points=args.mesh_points, seed=args.seed + 2, fallback_color=(0.1, 0.25, 0.95))
    c_xyz, c_rgb = sample_mesh(
        args.object_c,
        radius=0.40,
        n_points=args.mesh_points,
        seed=args.seed + 3,
        fallback_color=(0.78, 0.78, 0.82),
        projection_image=args.object_c_image,
    )

    bg_xyz = transform(bg_xyz, (0.0, 0.62, -0.46), 0.18)
    a_xyz = transform(a_xyz, (-0.86, -0.82, 0.08), -0.10)
    b_xyz = transform(b_xyz, (0.0, -0.84, 0.08), 0.25)
    c_xyz = transform(c_xyz, (0.88, -0.82, 0.08), -0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    video_frames: list[np.ndarray] = []

    target = np.array([0.0, -0.25, 0.0], dtype=np.float32)
    for frame_idx in range(args.frames):
        t = 2 * math.pi * frame_idx / args.frames
        camera = np.array([math.cos(t) * 3.35, math.sin(t) * 2.25 - 0.60, 1.25], dtype=np.float32)
        image = np.full((args.height, args.width, 3), (238, 238, 234), dtype=np.uint8)
        splat(image, bg_xyz, np.clip(bg_rgb * 1.12 + 0.04, 0, 1), camera, target, args.fov, args.bg_radius)
        splat(image, a_xyz, np.clip(a_rgb * 1.10 + 0.03, 0, 1), camera, target, args.fov, args.object_radius)
        splat(image, b_xyz, np.clip(b_rgb * 1.06 + 0.02, 0, 1), camera, target, args.fov, args.object_radius)
        splat(image, c_xyz, np.clip(c_rgb * 1.65 + 0.08, 0, 1), camera, target, args.fov, args.object_radius)
        video_frames.append(image)
    encode_video_from_frames(video_frames, args.output, fps=30)
    make_contact(args.output, args.contact, [0, args.frames // 6, args.frames // 3, args.frames // 2, args.frames * 2 // 3, args.frames * 5 // 6])
    print(f"Wrote {args.output}")
    print(f"Wrote {args.contact}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=Path, default=root / "outputs/task1/background_2dgs_trained_7k/point_cloud/iteration_7000/point_cloud.ply")
    parser.add_argument("--object-a", type=Path, default=root / "outputs/task1/object_a_2dgs_full/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--object-a-masked", type=Path, default=root / "outputs/task1/object_a_2dgs_masked_7k/point_cloud/iteration_7000/point_cloud.ply")
    parser.add_argument("--object-a-image", type=Path, default=root / "data/task1/object_a_masked_undistorted/images/frame_000030.jpg")
    parser.add_argument("--object-b", type=Path, default=root / "outputs/task1/object_b/textured.obj")
    parser.add_argument("--object-c", type=Path, default=root / "outputs/task1/object_c/textured.obj")
    parser.add_argument("--object-c-image", type=Path, default=root / "data/task1/object_c.png")
    parser.add_argument("--output", type=Path, default=root / "outputs/task1/fused_scene_true3d.mp4")
    parser.add_argument("--contact", type=Path, default=root / "outputs/task1/fused_scene_true3d_contact.jpg")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--bg-points", type=int, default=220_000)
    parser.add_argument("--object-points", type=int, default=140_000)
    parser.add_argument("--mesh-points", type=int, default=140_000)
    parser.add_argument("--bg-radius", type=int, default=2)
    parser.add_argument("--object-radius", type=int, default=2)
    parser.add_argument("--fov", type=float, default=44.0)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--seed", type=int, default=20260603)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
