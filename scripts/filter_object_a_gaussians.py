#!/usr/bin/env python3
"""Remove white-background floaters and retain Object A's main 3D component."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3cv.conversion import read_gaussian_ply, write_gaussian_ply  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--min-support-views", type=int, default=5)
    parser.add_argument("--min-support-ratio", type=float, default=0.70)
    parser.add_argument("--min-opacity", type=float, default=0.02)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument(
        "--radius-factor",
        type=float,
        default=8.0,
        help="Graph radius as a multiple of the median nearest-neighbor distance.",
    )
    parser.add_argument(
        "--report-factors",
        type=float,
        nargs="*",
        default=[4.0, 6.0, 8.0, 10.0, 12.0],
        help="Print component sizes for these factors before writing the selected result.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute support/component statistics without writing a PLY.",
    )
    parser.add_argument(
        "--keep-all-supported",
        action="store_true",
        help="Keep every foreground-supported Gaussian instead of one graph component.",
    )
    return parser.parse_args()


def foreground_mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    mean = image.mean(axis=2)
    chroma = image.max(axis=2) - image.min(axis=2)
    return (mean < 0.965) | (chroma > 0.035)


def support_counts(
    xyz: np.ndarray,
    cameras_path: Path,
    image_dir: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    hits = np.zeros(len(xyz), dtype=np.uint16)
    visible = np.zeros(len(xyz), dtype=np.uint16)
    used_cameras = 0

    for index, camera in enumerate(cameras, 1):
        image_path = image_dir / f"{camera['img_name']}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{camera['img_name']}.png"
        if not image_path.exists():
            continue

        mask = foreground_mask(image_path)
        height, width = mask.shape
        rotation = np.asarray(camera["rotation"], dtype=np.float32)
        position = np.asarray(camera["position"], dtype=np.float32)
        camera_xyz = (xyz - position) @ rotation
        depth = camera_xyz[:, 2]
        projectable = (depth > 0.0) & np.isfinite(camera_xyz).all(axis=1)
        indices = np.flatnonzero(projectable)
        if not len(indices):
            continue

        u = np.rint(
            float(camera["fx"]) * camera_xyz[indices, 0] / depth[indices]
            + float(camera["width"]) * 0.5
        ).astype(np.int32)
        v = np.rint(
            float(camera["fy"]) * camera_xyz[indices, 1] / depth[indices]
            + float(camera["height"]) * 0.5
        ).astype(np.int32)
        # cameras.json dimensions should match the images, but scaling keeps
        # this robust if quality images were resized after training.
        u = np.rint(u * width / int(camera["width"])).astype(np.int32)
        v = np.rint(v * height / int(camera["height"])).astype(np.int32)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_indices = indices[inside]
        visible[valid_indices] += 1
        hits[valid_indices] += mask[v[inside], u[inside]]
        used_cameras += 1
        if index % 50 == 0 or index == len(cameras):
            print(f"projected_cameras={index}/{len(cameras)}", flush=True)

    return hits, visible, used_cameras


def load_or_compute_support(
    xyz: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    if args.cache and args.cache.exists():
        cached = np.load(args.cache)
        if int(cached["point_count"]) == len(xyz):
            print(f"Using support cache: {args.cache}")
            return cached["hits"], cached["visible"], int(cached["used_cameras"])
        print("Ignoring support cache because the input point count changed")

    hits, visible, used_cameras = support_counts(xyz, args.cameras, args.images)
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache,
            hits=hits,
            visible=visible,
            used_cameras=np.asarray(used_cameras),
            point_count=np.asarray(len(xyz)),
        )
        print(f"Wrote support cache: {args.cache}")
    return hits, visible, used_cameras


def component_labels(
    xyz: np.ndarray,
    knn: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(xyz)
    neighbors = min(max(knn, 2), count)
    distances, indices = cKDTree(xyz).query(xyz, k=neighbors, workers=-1)
    rows = np.repeat(np.arange(count, dtype=np.int32), neighbors - 1)
    cols = indices[:, 1:].reshape(-1).astype(np.int32)
    edge_keep = distances[:, 1:].reshape(-1) <= radius
    rows = rows[edge_keep]
    cols = cols[edge_keep]
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(count, count),
    )
    _, labels = connected_components(graph.maximum(graph.T), directed=False)
    return labels, np.bincount(labels)


def subset_params(params: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    result = {
        key: value[keep]
        for key, value in params.items()
        if key in {"xyz", "f_dc", "opacity", "scales", "rotations", "f_rest"}
    }
    return result


def write_filtered(params: dict[str, np.ndarray], path: Path) -> None:
    extra_names = None
    extra_data = None
    if "f_rest" in params:
        coefficient_count = params["f_rest"].shape[1]
        extra_names = [f"f_rest_{index}" for index in range(coefficient_count * 3)]
        extra_data = params["f_rest"].transpose(0, 2, 1).reshape(len(params["xyz"]), -1)
    write_gaussian_ply(params, path, extra_names, extra_data)


def main() -> None:
    args = parse_args()
    params = read_gaussian_ply(args.input)
    xyz = params["xyz"].astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-params["opacity"].astype(np.float32)))
    finite = np.isfinite(xyz).all(axis=1)

    hits, visible, used_cameras = load_or_compute_support(xyz, args)
    support_ratio = hits.astype(np.float32) / np.maximum(visible, 1)
    foreground = (
        finite
        & (opacity >= args.min_opacity)
        & (hits >= args.min_support_views)
        & (support_ratio >= args.min_support_ratio)
    )
    foreground_indices = np.flatnonzero(foreground)
    foreground_xyz = xyz[foreground]
    if len(foreground_xyz) < 2:
        raise RuntimeError("Foreground filtering retained fewer than two Gaussians")

    nearest = cKDTree(foreground_xyz).query(foreground_xyz, k=2, workers=-1)[0][:, 1]
    median_nearest = float(np.median(nearest[nearest > 0]))
    print(
        f"input={len(xyz)} cameras={used_cameras} "
        f"foreground_supported={len(foreground_xyz)} "
        f"median_nearest_distance={median_nearest:.8f}"
    )

    factors = sorted(set(args.report_factors + [args.radius_factor]))
    chosen_labels = None
    chosen_sizes = None
    for factor in factors:
        radius = median_nearest * factor
        labels, sizes = component_labels(foreground_xyz, args.knn, radius)
        top = np.sort(sizes)[-8:][::-1]
        print(
            f"factor={factor:g} radius={radius:.8f} components={len(sizes)} "
            f"top_sizes={','.join(map(str, top))}"
        )
        if factor == args.radius_factor:
            chosen_labels, chosen_sizes = labels, sizes

    assert chosen_labels is not None and chosen_sizes is not None
    if args.keep_all_supported:
        keep_indices = foreground_indices
        selected_label = "all-supported"
    else:
        largest_label = int(np.argmax(chosen_sizes))
        component_local = chosen_labels == largest_label
        keep_indices = foreground_indices[component_local]
        selected_label = "largest-component"
    keep = np.zeros(len(xyz), dtype=bool)
    keep[keep_indices] = True
    kept_xyz = xyz[keep]
    lo, hi = np.percentile(kept_xyz, [2, 98], axis=0)
    print(
        f"selected_factor={args.radius_factor:g} selection={selected_label} selected={keep.sum()} "
        f"bbox_p02={lo.tolist()} bbox_p98={hi.tolist()}"
    )

    if args.dry_run:
        print("Dry run: no PLY written")
        return

    filtered = subset_params(params, keep)
    write_filtered(filtered, args.output)
    print(f"Wrote filtered Gaussian PLY: {args.output}")


if __name__ == "__main__":
    main()
