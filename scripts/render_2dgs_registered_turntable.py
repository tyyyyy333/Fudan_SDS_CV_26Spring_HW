#!/usr/bin/env python3
"""Render a stable Object A turntable along its registered camera sequence."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = ROOT / "external" / "2d-gaussian-splatting"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(GS_ROOT))
sys.path.insert(0, str(GS_ROOT / "submodules" / "simple-knn"))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from video_utils import encode_video_from_pattern  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--max-scale-ratio", type=float, default=0.025)
    parser.add_argument("--min-scale-ratio", type=float, default=0.0015)
    parser.add_argument("--center-percentile", type=float, default=99.0)
    parser.add_argument("--min-opacity", type=float, default=0.02)
    parser.add_argument(
        "--camera-frames",
        type=int,
        nargs="+",
        help="Render these source frame numbers in the supplied order.",
    )
    parser.add_argument(
        "--exclude-range",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
        help="Exclude an inclusive source-frame range before sampling.",
    )
    return parser.parse_args()


def frame_number(camera) -> int:
    name = getattr(camera, "image_name", "")
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def stabilize_gaussians(
    gaussians: GaussianModel,
    center_percentile: float,
    min_scale_ratio: float,
    max_scale_ratio: float,
    min_opacity: float,
) -> tuple[int, int, float]:
    xyz = gaussians._xyz.detach()
    center = xyz.median(dim=0).values
    distance = torch.linalg.norm(xyz - center, dim=1)
    radius = float(torch.quantile(distance, center_percentile / 100.0).item())
    opacity = gaussians.get_opacity[:, 0]
    keep = torch.isfinite(xyz).all(dim=1) & (distance <= radius) & (opacity >= min_opacity)
    before = int(len(xyz))

    gaussians._xyz = nn.Parameter(gaussians._xyz.detach()[keep].requires_grad_(False))
    gaussians._features_dc = nn.Parameter(gaussians._features_dc.detach()[keep].requires_grad_(False))
    gaussians._features_rest = nn.Parameter(gaussians._features_rest.detach()[keep].requires_grad_(False))
    gaussians._opacity = nn.Parameter(gaussians._opacity.detach()[keep].requires_grad_(False))
    gaussians._rotation = nn.Parameter(gaussians._rotation.detach()[keep].requires_grad_(False))

    linear_scale = torch.exp(gaussians._scaling.detach()[keep])
    lower = max(radius * min_scale_ratio, 1e-6)
    upper = max(radius * max_scale_ratio, lower)
    linear_scale = linear_scale.clamp(min=lower, max=upper)
    gaussians._scaling = nn.Parameter(torch.log(linear_scale).requires_grad_(False))
    gaussians.max_radii2D = torch.zeros(len(gaussians._xyz), device=xyz.device)
    return before, int(keep.sum().item()), radius


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    return image.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()


def contact_sheet(frame_dir: Path, output: Path, source_frames: list[int]) -> None:
    frames = len(source_frames)
    indices = np.linspace(0, frames - 1, min(16, frames), dtype=int)
    tile_w, tile_h, label_h, cols = 480, 270, 28, 4
    canvas = Image.new("RGB", (cols * tile_w, math.ceil(len(indices) / cols) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for rank, index in enumerate(indices):
        image = Image.open(frame_dir / f"{index:05d}.jpg").convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = rank % cols * tile_w
        y = rank // cols * (tile_h + label_h)
        draw.text((x + 8, y + 6), f"source frame {source_frames[index]:04d}", fill="black")
        canvas.paste(image, (x + (tile_w - image.width) // 2, y + label_h))
    canvas.save(output, quality=95)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir()

    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    namespace = parser.parse_args(
        [
            "--source_path",
            str(args.source.resolve()),
            "--model_path",
            str(args.model.resolve()),
            "--resolution",
            str(args.resolution),
            "--white_background",
        ]
    )
    dataset = model.extract(namespace)
    pipe = pipeline.extract(namespace)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = sorted(scene.getTrainCameras() + scene.getTestCameras(), key=frame_number)
    if args.exclude_range:
        cameras = [
            camera
            for camera in cameras
            if not any(start <= frame_number(camera) <= end for start, end in args.exclude_range)
        ]
    if args.camera_frames:
        cameras_by_frame = {frame_number(camera): camera for camera in cameras}
        missing = [number for number in args.camera_frames if number not in cameras_by_frame]
        if missing:
            raise ValueError(f"Requested source frames are not registered: {missing}")
        selected_cameras = [cameras_by_frame[number] for number in args.camera_frames]
    else:
        selected = np.linspace(0, len(cameras) - 1, args.frames, dtype=int)
        selected_cameras = [cameras[index] for index in selected]
    selected_source_frames = [frame_number(camera) for camera in selected_cameras]
    before, after, radius = stabilize_gaussians(
        gaussians,
        args.center_percentile,
        args.min_scale_ratio,
        args.max_scale_ratio,
        args.min_opacity,
    )
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    for output_index, camera in enumerate(selected_cameras):
        with torch.no_grad():
            image = tensor_to_rgb(render(camera, gaussians, pipe, background)["render"])
        cv2.imwrite(
            str(frame_dir / f"{output_index:05d}.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 96],
        )

    rendered_frames = len(selected_cameras)
    contact_sheet(frame_dir, args.output / "contact_16views.jpg", selected_source_frames)
    video_name = f"turntable_{rendered_frames}.mp4"
    encode_video_from_pattern(frame_dir / "%05d.jpg", args.output / video_name, args.fps, rendered_frames)
    keyframes = args.output / "keyframes"
    shutil.rmtree(keyframes, ignore_errors=True)
    keyframes.mkdir()
    for index in np.linspace(0, rendered_frames - 1, min(24, rendered_frames), dtype=int):
        shutil.copy2(frame_dir / f"{index:05d}.jpg", keyframes / f"{index:05d}.jpg")
    shutil.rmtree(frame_dir)
    (args.output / "render_summary.txt").write_text(
        f"registered_cameras={len(cameras)}\n"
        f"rendered_source_frames={','.join(map(str, selected_source_frames))}\n"
        f"gaussians_before={before}\n"
        f"gaussians_after={after}\n"
        f"robust_radius={radius:.8f}\n"
        f"min_scale_ratio={args.min_scale_ratio}\n"
        f"max_scale_ratio={args.max_scale_ratio}\n"
        f"min_opacity={args.min_opacity}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
