#!/usr/bin/env python3
"""Render a trained 2DGS model on a dense synthetic orbit for coverage inspection."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = ROOT / "external" / "2d-gaussian-splatting"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(GS_ROOT))
sys.path.insert(0, str(GS_ROOT / "submodules" / "simple-knn"))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.render_utils import focus_point_fn, generate_path, transform_poses_pca  # noqa: E402
from video_utils import encode_video_from_pattern  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--contact-views", type=int, default=16)
    parser.add_argument("--keep-keyframes", type=int, default=24)
    parser.add_argument("--white-background", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def camera_poses(cameras: list) -> np.ndarray:
    c2ws = np.array(
        [
            np.linalg.inv(np.asarray(cam.world_view_transform.T.detach().cpu().numpy()))
            for cam in cameras
        ]
    )
    return c2ws[:, :3, :] @ np.diag([1.0, -1.0, -1.0, 1.0])


def camera_coverage(cameras: list) -> tuple[np.ndarray, np.ndarray, float, float]:
    poses, _ = transform_poses_pca(camera_poses(cameras))
    center = focus_point_fn(poses)
    offsets = poses[:, :3, 3] - center[None, :]
    azimuth = np.mod(np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])), 360.0)
    horizontal = np.linalg.norm(offsets[:, :2], axis=1)
    elevation = np.degrees(np.arctan2(offsets[:, 2], horizontal))
    ordered = np.sort(azimuth)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
    largest_gap = float(gaps.max())
    coverage = 360.0 - largest_gap
    return azimuth, elevation, coverage, largest_gap


def save_coverage(
    cameras: list,
    output: Path,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    coverage: float,
    largest_gap: float,
) -> None:
    with (output / "source_camera_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "azimuth_deg", "elevation_deg"])
        for camera, az, el in sorted(
            zip(cameras, azimuth, elevation), key=lambda item: item[1]
        ):
            writer.writerow([camera.image_name, f"{az:.4f}", f"{el:.4f}"])

    fig = plt.figure(figsize=(6.4, 5.5), dpi=180)
    axis = fig.add_subplot(111, projection="polar")
    axis.scatter(np.radians(azimuth), np.ones_like(azimuth), c=elevation, cmap="coolwarm", s=34)
    axis.set_yticks([])
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_title(
        f"Registered source cameras: {len(cameras)}\n"
        f"azimuth coverage {coverage:.1f} deg; largest unseen gap {largest_gap:.1f} deg",
        pad=20,
    )
    colorbar = fig.colorbar(axis.collections[0], ax=axis, pad=0.10, shrink=0.8)
    colorbar.set_label("elevation (deg)")
    fig.tight_layout()
    fig.savefig(output / "source_camera_coverage.png", bbox_inches="tight")
    plt.close(fig)

    (output / "coverage_summary.txt").write_text(
        "\n".join(
            [
                f"registered_cameras={len(cameras)}",
                f"azimuth_coverage_deg={coverage:.4f}",
                f"largest_unseen_gap_deg={largest_gap:.4f}",
                f"elevation_min_deg={elevation.min():.4f}",
                f"elevation_max_deg={elevation.max():.4f}",
                "note=The rendered orbit is synthetic. It tests novel-view behavior but does not add training coverage.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def save_contact(frame_dir: Path, output: Path, frames: int, views: int) -> None:
    indices = np.linspace(0, frames - 1, min(views, frames), dtype=int)
    tile_width, tile_height, label_height = 320, 180, 26
    cols = 4
    rows = math.ceil(len(indices) / cols)
    canvas = Image.new("RGB", (cols * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for rank, index in enumerate(indices):
        image = Image.open(frame_dir / f"{index:05d}.jpg").convert("RGB")
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (rank % cols) * tile_width
        y = (rank // cols) * (tile_height + label_height)
        phase = 360.0 * index / frames
        draw.text((x + 7, y + 5), f"orbit {phase:5.1f} deg / frame {index:03d}", fill=(20, 20, 20))
        canvas.paste(image, (x + (tile_width - image.width) // 2, y + label_height))
    canvas.save(output, quality=95)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True)

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
        ]
        + (["--white_background"] if args.white_background else [])
    )
    dataset = model.extract(namespace)
    pipe = pipeline.extract(namespace)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
    if len(cameras) < 3:
        raise RuntimeError("At least three registered cameras are required")

    azimuth, elevation, coverage, largest_gap = camera_coverage(cameras)
    save_coverage(cameras, args.output, azimuth, elevation, coverage, largest_gap)

    orbit = generate_path(cameras, n_frames=args.frames)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    for index, camera in enumerate(tqdm(orbit, desc="render A orbit")):
        with torch.no_grad():
            rgb = tensor_to_rgb(render(camera, gaussians, pipe, background)["render"])
        cv2.imwrite(
            str(frame_dir / f"{index:05d}.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        )

    save_contact(frame_dir, args.output / "contact_16views.jpg", args.frames, args.contact_views)
    encode_video_from_pattern(
        frame_dir / "%05d.jpg",
        args.output / "turntable_120.mp4",
        args.fps,
        args.frames,
    )

    keyframe_dir = args.output / "keyframes"
    shutil.rmtree(keyframe_dir, ignore_errors=True)
    keyframe_dir.mkdir()
    for index in np.linspace(0, args.frames - 1, min(args.keep_keyframes, args.frames), dtype=int):
        shutil.copy2(frame_dir / f"{index:05d}.jpg", keyframe_dir / f"{index:05d}.jpg")
    shutil.rmtree(frame_dir)

    print(f"Registered-camera azimuth coverage: {coverage:.1f} deg")
    print(f"Largest unseen azimuth gap: {largest_gap:.1f} deg")
    print(f"Wrote video: {args.output / 'turntable_120.mp4'}")
    print(f"Wrote contact: {args.output / 'contact_16views.jpg'}")
    print(f"Wrote keyframes: {keyframe_dir}")


if __name__ == "__main__":
    main()
