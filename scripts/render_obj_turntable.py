#!/usr/bin/env python3
"""Render an OBJ mesh into a server-friendly turntable video and contact sheet."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

from video_utils import encode_video_from_pattern

def load_surface_points(obj: Path, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(obj, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a single mesh in {obj}, got {type(mesh).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"OBJ has no renderable geometry: {obj}")

    rng = np.random.default_rng(seed)
    points, face_index = trimesh.sample.sample_surface(mesh, samples, seed=rng)

    visual = mesh.visual
    if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None and len(visual.vertex_colors) == len(mesh.vertices):
        vertex_colors = np.asarray(visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    else:
        vertex_colors = np.asarray(visual.to_color().vertex_colors[:, :3], dtype=np.float32) / 255.0
    face_colors = vertex_colors[np.asarray(mesh.faces)[face_index]].mean(axis=1)

    return points.astype(np.float32), np.clip(face_colors.astype(np.float32), 0.0, 1.0)


def normalize(points: np.ndarray) -> np.ndarray:
    centered = points - np.median(points, axis=0, keepdims=True)
    radius = np.percentile(np.linalg.norm(centered, axis=1), 98)
    return centered / max(float(radius), 1e-6)


def orient_up_axis(points: np.ndarray, up_axis: str) -> np.ndarray:
    if up_axis == "y":
        return points
    if up_axis == "z":
        # The software turntable rotates around Y. Map a Z-up asset to Y-up
        # while preserving a right-handed frame.
        return points[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0], dtype=np.float32)
    raise ValueError(f"Unsupported up axis: {up_axis}")


def rotate(points: np.ndarray, yaw_deg: float, elevation_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    elev = math.radians(elevation_deg)
    ry = np.array(
        [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]],
        dtype=np.float32,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(elev), -math.sin(elev)], [0.0, math.sin(elev), math.cos(elev)]],
        dtype=np.float32,
    )
    return points @ ry.T @ rx.T


def render_frame(
    points: np.ndarray,
    colors: np.ndarray,
    yaw: float,
    elevation: float,
    size: int,
    point_radius: int,
    background: tuple[int, int, int],
) -> np.ndarray:
    pts = rotate(points, yaw, elevation)
    scale = size * 0.38
    px = np.round(pts[:, 0] * scale + size / 2).astype(np.int32)
    py = np.round(-pts[:, 1] * scale + size / 2).astype(np.int32)
    depth = pts[:, 2]

    image = np.full((size, size, 3), background, dtype=np.uint8)
    order = np.argsort(depth)
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    for idx in order[valid[order]]:
        color = tuple(int(v) for v in rgb[idx])
        if point_radius <= 1:
            image[py[idx], px[idx]] = color
        else:
            cv2.circle(image, (int(px[idx]), int(py[idx])), point_radius, color, -1, lineType=cv2.LINE_AA)

    return image


def contact_sheet(frames: list[np.ndarray], output: Path, cols: int = 4) -> None:
    thumbs = [Image.fromarray(frame).resize((320, 320), Image.Resampling.LANCZOS) for frame in frames]
    rows = math.ceil(len(thumbs) / cols)
    canvas = Image.new("RGB", (cols * 320, rows * 348), "white")
    draw = ImageDraw.Draw(canvas)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 320
        y = (i // cols) * 348
        draw.text((x + 8, y + 7), f"view {i:02d}", fill=(20, 20, 20))
        canvas.paste(thumb, (x, y + 28))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, required=True, help="Input OBJ path.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for frames, contact sheet, and video.")
    parser.add_argument("--frames", type=int, default=72, help="Number of turntable frames.")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
    parser.add_argument("--size", type=int, default=768, help="Square render resolution.")
    parser.add_argument("--samples", type=int, default=180_000, help="Surface samples used for software rendering.")
    parser.add_argument("--elevation", type=float, default=15.0, help="Camera elevation in degrees.")
    parser.add_argument("--start-yaw", type=float, default=0.0, help="Starting yaw angle in degrees.")
    parser.add_argument("--point-radius", type=int, default=2, help="Rasterized point radius in pixels.")
    parser.add_argument("--up-axis", choices=["y", "z"], default="y", help="Vertical axis used by the OBJ.")
    parser.add_argument("--seed", type=int, default=20260608, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.outdir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("*.jpg"):
        old.unlink()

    points, colors = load_surface_points(args.obj, args.samples, args.seed)
    points = orient_up_axis(points, args.up_axis)
    points = normalize(points)

    contact_frames: list[np.ndarray] = []
    contact_indices = set(np.linspace(0, args.frames - 1, min(8, args.frames), dtype=int).tolist())
    for i in range(args.frames):
        yaw = args.start_yaw + 360.0 * i / args.frames
        frame = render_frame(points, colors, yaw, args.elevation, args.size, args.point_radius, (255, 255, 255))
        if i in contact_indices:
            contact_frames.append(frame)
        cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    contact = args.outdir / "contact.jpg"
    video = args.outdir / "turntable.mp4"
    contact_sheet(contact_frames, contact)
    encode_video_from_pattern(frame_dir / "%05d.jpg", video, args.fps, args.frames)
    print(f"Wrote frames: {frame_dir}")
    print(f"Wrote contact: {contact}")
    print(f"Wrote video: {video}")


if __name__ == "__main__":
    main()
