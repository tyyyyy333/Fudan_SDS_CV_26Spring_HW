#!/usr/bin/env python3
"""Prune 2DGS floaters using multi-view foreground masks.

The Object A reconstruction was trained from masked RGB frames on a white
canvas. The cup is present, but white-canvas boundary floaters survive in the
Gaussian checkpoint. This script projects each Gaussian center into the training
cameras and keeps only Gaussians that repeatedly land on non-white foreground
pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = ROOT / "external/2d-gaussian-splatting"
sys.path.insert(0, str(GS_ROOT))
sys.path.insert(0, str(GS_ROOT / "submodules/simple-knn"))

from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


def build_dataset(source: Path, model: Path, resolution: int, white_background: bool):
    return SimpleNamespace(
        sh_degree=3,
        source_path=str(source.resolve()),
        model_path=str(model.resolve()),
        images="images",
        resolution=resolution,
        white_background=white_background,
        data_device="cuda",
        eval=True,
        render_items=["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"],
    )


def foreground_mask(image: torch.Tensor, white_threshold: float, dark_threshold: float):
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    mean = arr.mean(axis=2)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    return (mean < white_threshold) & ((mean > dark_threshold) | (chroma > 0.02))


def project_xyz(xyz: np.ndarray, camera):
    h, w = int(camera.image_height), int(camera.image_width)
    full = camera.full_proj_transform.detach().cpu().numpy()
    pts = np.concatenate([xyz.astype(np.float32), np.ones((len(xyz), 1), dtype=np.float32)], axis=1)
    clip = pts @ full
    valid = np.abs(clip[:, 3]) > 1e-6
    ndc = np.zeros((len(xyz), 3), dtype=np.float32)
    ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
    x = ((ndc[:, 0] + 1.0) * 0.5 * w).astype(np.int32)
    y = ((1.0 - ndc[:, 1]) * 0.5 * h).astype(np.int32)
    valid &= (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    valid &= (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return x, y, valid


def subset_param(param: torch.Tensor, keep: torch.Tensor):
    return nn.Parameter(param.detach()[keep].clone().requires_grad_(True))


def prune(args):
    dataset = build_dataset(args.source, args.model, args.resolution, args.white_background)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    xyz = gaussians.get_xyz.detach().cpu().numpy()

    visible = np.zeros(len(xyz), dtype=np.int32)
    foreground = np.zeros(len(xyz), dtype=np.int32)
    for cam in tqdm(cameras, desc="project masks"):
        mask = foreground_mask(cam.original_image, args.white_threshold, args.dark_threshold)
        x, y, valid = project_xyz(xyz, cam)
        valid_idx = np.where(valid)[0]
        visible[valid_idx] += 1
        fg = mask[y[valid_idx], x[valid_idx]]
        foreground[valid_idx[fg]] += 1

    ratio = foreground / np.maximum(visible, 1)
    keep_np = (visible >= args.min_visible) & (
        (foreground >= args.min_foreground) | (ratio >= args.min_ratio)
    )

    if args.min_opacity > 0:
        opacity = 1.0 / (1.0 + np.exp(-gaussians._opacity.detach().cpu().numpy().reshape(-1)))
        keep_np &= opacity >= args.min_opacity

    if args.robust_crop > 0:
        lo, hi = np.percentile(xyz[keep_np], [args.crop_low, args.crop_high], axis=0)
        center = (lo + hi) * 0.5
        radius = np.linalg.norm(hi - lo) * 0.5
        dist = np.linalg.norm(xyz - center[None, :], axis=1)
        keep_np &= dist < args.robust_crop * radius

    for axis, low, high in ((0, args.x_low, args.x_high), (1, args.y_low, args.y_high), (2, args.z_low, args.z_high)):
        if low is not None:
            keep_np &= xyz[:, axis] >= low
        if high is not None:
            keep_np &= xyz[:, axis] <= high

    keep = torch.from_numpy(keep_np).to(device=gaussians.get_xyz.device)
    before = int(len(keep_np))
    after = int(keep_np.sum())
    if after < args.min_keep:
        raise RuntimeError(f"Pruning was too aggressive: kept {after}/{before}")

    gaussians._xyz = subset_param(gaussians._xyz, keep)
    gaussians._features_dc = subset_param(gaussians._features_dc, keep)
    gaussians._features_rest = subset_param(gaussians._features_rest, keep)
    gaussians._opacity = subset_param(gaussians._opacity, keep)
    gaussians._scaling = subset_param(gaussians._scaling, keep)
    gaussians._rotation = subset_param(gaussians._rotation, keep)
    gaussians.max_radii2D = torch.zeros((after,), device="cuda")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(args.output))
    print(f"kept {after}/{before} gaussians")
    print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/task1/object_a_masked_undistorted")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/task1/object_a_2dgs_masked_7k")
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task1/object_a_2dgs_masked_7k_pruned/point_cloud/iteration_7000/point_cloud.ply")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--white-background", action="store_true", default=True)
    parser.add_argument("--white-threshold", type=float, default=0.965)
    parser.add_argument("--dark-threshold", type=float, default=0.04)
    parser.add_argument("--min-visible", type=int, default=2)
    parser.add_argument("--min-foreground", type=int, default=1)
    parser.add_argument("--min-ratio", type=float, default=0.18)
    parser.add_argument("--robust-crop", type=float, default=1.65)
    parser.add_argument("--crop-low", type=float, default=3.0)
    parser.add_argument("--crop-high", type=float, default=97.0)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--x-low", type=float, default=None)
    parser.add_argument("--x-high", type=float, default=None)
    parser.add_argument("--y-low", type=float, default=None)
    parser.add_argument("--y-high", type=float, default=None)
    parser.add_argument("--z-low", type=float, default=None)
    parser.add_argument("--z-high", type=float, default=None)
    parser.add_argument("--min-keep", type=int, default=8000)
    prune(parser.parse_args())


if __name__ == "__main__":
    main()
