#!/usr/bin/env python3
"""Render Task 1 with an official 2DGS background and 3D A/B/C assets.

This script keeps the background on the assignment path: it is rendered through
the 2D Gaussian Splatting rasterizer, not through the lightweight point-cloud
preview renderer used for earlier diagnostics. Objects A/B/C are transformed
once into the same world coordinate system and then projected through the same
camera matrices for every frame, so their appearance stays geometrically
consistent across viewpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external/2d-gaussian-splatting"
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(EXT / "submodules/simple-knn"))
sys.path.insert(0, str(ROOT / "src"))

from gaussian_renderer import render as render_gaussian  # noqa: E402
from video_utils import encode_video_from_pattern  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.render_utils import focus_point_fn, generate_path  # noqa: E402

from hw3cv.conversion import _normals_to_quaternions, read_gaussian_ply, read_obj  # noqa: E402


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = np.array(
    [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
     -1.0925484305920792, 0.5462742152960396],
    dtype=np.float64,
)
C3 = np.array(
    [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
     0.3731763325901154, -0.4570457994644658, 1.445305721320277,
     -0.5900435899266435],
    dtype=np.float64,
)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def robust_normalize(xyz: np.ndarray, radius: float, crop: float = 2.4):
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    shifted = xyz.astype(np.float32) - center
    scale = radius / max(float(np.linalg.norm(hi - lo) * 0.5), 1e-6)
    shifted *= scale
    keep = np.linalg.norm(shifted, axis=1) < crop * radius
    return shifted[keep], keep


def robust_normalize_with_scale(xyz: np.ndarray, radius: float, crop: float = 2.4):
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    shifted = xyz.astype(np.float32) - center
    scale = radius / max(float(np.linalg.norm(hi - lo) * 0.5), 1e-6)
    shifted *= scale
    keep = np.linalg.norm(shifted, axis=1) < crop * radius
    return shifted[keep], keep, float(scale)


def project_image_colors(xyz: np.ndarray, image_path: Path):
    rgba = Image.open(image_path).convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    image = arr[:, :, :3]
    alpha = arr[:, :, 3]
    if alpha.max() > 0.99 and alpha.min() < 0.99:
        mask = alpha > 0.05
    else:
        mean = image.mean(axis=2)
        chroma = image.max(axis=2) - image.min(axis=2)
        # Most masked object references in this project are RGB images on a
        # white canvas. Treat near-white canvas pixels as invalid projection
        # colors so the 3D asset does not turn into a pale blob.
        mask = (mean < 0.965) & ((mean > 0.04) | (chroma > 0.02))
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


def camera_mask_support(
    xyz: np.ndarray,
    cameras_json: Path,
    image_dir: Path,
    min_views: int,
    min_ratio: float,
):
    import json

    cameras = json.loads(cameras_json.read_text())
    hits = np.zeros(len(xyz), dtype=np.int16)
    visible = np.zeros(len(xyz), dtype=np.int16)
    for camera in cameras:
        image_path = image_dir / f"{camera['img_name']}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{camera['img_name']}.png"
        if not image_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        mean = image.mean(axis=2)
        chroma = image.max(axis=2) - image.min(axis=2)
        foreground = (mean < 0.965) | (chroma > 0.035)

        rotation = np.asarray(camera["rotation"], dtype=np.float32)
        position = np.asarray(camera["position"], dtype=np.float32)
        camera_xyz = (xyz - position) @ rotation
        depth = camera_xyz[:, 2]
        projectable = (depth > 0.0) & np.isfinite(camera_xyz).all(axis=1)
        u = np.full(len(xyz), -1, dtype=np.int32)
        v = np.full(len(xyz), -1, dtype=np.int32)
        idx = np.flatnonzero(projectable)
        u[idx] = np.rint(
            float(camera["fx"]) * camera_xyz[idx, 0] / depth[idx]
            + float(camera["width"]) * 0.5
        ).astype(np.int32)
        v[idx] = np.rint(
            float(camera["fy"]) * camera_xyz[idx, 1] / depth[idx]
            + float(camera["height"]) * 0.5
        ).astype(np.int32)
        valid = (
            projectable
            & (u >= 0)
            & (u < int(camera["width"]))
            & (v >= 0)
            & (v < int(camera["height"]))
        )
        visible += valid
        valid_indices = np.flatnonzero(valid)
        hits[valid_indices] += foreground[v[valid_indices], u[valid_indices]]

    support_ratio = hits / np.maximum(visible, 1)
    keep = (hits >= min_views) & (support_ratio >= min_ratio)
    print(
        f"Kept camera-mask-supported Gaussian centers: {keep.sum()}/{len(keep)} "
        f"(min_views={min_views}, min_ratio={min_ratio})"
    )
    return keep


def load_gaussian_points(path: Path, radius: float, max_points: int, seed: int, projection_image: Path | None):
    params = read_gaussian_ply(path)
    xyz_raw = params["xyz"].astype(np.float32)
    xyz, keep = robust_normalize(xyz_raw, radius)
    rgb = np.clip(params["f_dc"] * C0 + 0.5, 0.0, 1.0)[keep]
    if projection_image is not None and projection_image.exists():
        projected, valid = project_image_colors(xyz_raw, projection_image)
        projected = projected[keep]
        valid = valid[keep]
        rgb[valid] = 0.92 * projected[valid] + 0.08 * rgb[valid]
    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz.astype(np.float32), np.clip(rgb.astype(np.float32), 0, 1)


def sample_mesh_points(path: Path, radius: float, n_points: int, seed: int, fallback_color, projection_image: Path | None):
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
        flip = r1 + r2 > 1.0
        r1[flip] = 1.0 - r1[flip]
        r2[flip] = 1.0 - r2[flip]
        tri_s = tri[tri_idx]
        xyz = tri_s[:, 0] + r1[:, None] * (tri_s[:, 1] - tri_s[:, 0]) + r2[:, None] * (tri_s[:, 2] - tri_s[:, 0])
        if colors is not None and len(colors) == len(vertices):
            ctri = colors[faces][tri_idx]
            rgb = ctri[:, 0] + r1[:, None] * (ctri[:, 1] - ctri[:, 0]) + r2[:, None] * (ctri[:, 2] - ctri[:, 0])
        else:
            rgb = np.tile(np.array(fallback_color, dtype=np.float32), (n_points, 1))

    if projection_image is not None and projection_image.exists():
        projected, mask = project_image_colors(xyz, projection_image)
        rgb[mask] = 0.88 * projected[mask] + 0.12 * rgb[mask]

    xyz, keep = robust_normalize(xyz, radius)
    return xyz.astype(np.float32), np.clip(rgb.astype(np.float32), 0, 1)[keep]


def sample_mesh_gaussians(
    path: Path,
    radius: float,
    n_points: int,
    seed: int,
    fallback_color,
    projection_image: Path | None,
    footprint_scale: float,
    opacity_probability: float,
):
    """Sample a textured mesh into surface-aligned 2D Gaussian surfels.

    UV/albedo or vertex RGB supplies diffuse color. The surfel normal comes
    from the sampled triangle, and its tangent radius is derived from the
    normalized surface area per sample. PBR channels such as roughness,
    metallic, transmission, and environment-dependent specular response are
    intentionally not synthesized by this radiance-field renderer.
    """
    vertices, faces, colors = read_obj(path)
    vertices = vertices.astype(np.float32)
    if len(faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")

    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_areas = np.linalg.norm(cross, axis=1)
    valid_faces = double_areas > 1e-12
    if not np.any(valid_faces):
        raise ValueError(f"Mesh has only degenerate faces: {path}")
    probs = np.where(valid_faces, double_areas, 0.0).astype(np.float64)
    probs /= probs.sum()

    rng = np.random.default_rng(seed)
    tri_idx = rng.choice(len(faces), size=n_points, p=probs)
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    flip = r1 + r2 > 1.0
    r1[flip] = 1.0 - r1[flip]
    r2[flip] = 1.0 - r2[flip]
    tri_s = tri[tri_idx]
    xyz_raw = (
        tri_s[:, 0]
        + r1[:, None] * (tri_s[:, 1] - tri_s[:, 0])
        + r2[:, None] * (tri_s[:, 2] - tri_s[:, 0])
    )

    normals = cross[tri_idx]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    if colors is not None and len(colors) == len(vertices):
        ctri = colors[faces][tri_idx]
        rgb = (
            ctri[:, 0]
            + r1[:, None] * (ctri[:, 1] - ctri[:, 0])
            + r2[:, None] * (ctri[:, 2] - ctri[:, 0])
        )
    else:
        rgb = np.tile(np.asarray(fallback_color, dtype=np.float32), (n_points, 1))

    if projection_image is not None and projection_image.exists():
        projected, mask = project_image_colors(xyz_raw, projection_image)
        rgb[mask] = 0.88 * projected[mask] + 0.12 * rgb[mask]

    xyz, keep, normalization_scale = robust_normalize_with_scale(xyz_raw, radius)
    rgb = np.clip(rgb[keep], 0.0, 1.0)
    normals = normals[keep]
    n = len(xyz)
    f_dc = ((rgb - 0.5) / C0).astype(np.float32)
    f_rest = np.zeros((n, 15, 3), dtype=np.float32)
    opacity_probability = float(np.clip(opacity_probability, 1e-4, 1.0 - 1e-4))
    opacity = np.full(
        (n, 1),
        math.log(opacity_probability / (1.0 - opacity_probability)),
        dtype=np.float32,
    )
    normalized_surface_area = float(double_areas.sum() * 0.5 * normalization_scale**2)
    tangent_radius = footprint_scale * math.sqrt(
        normalized_surface_area / max(math.pi * n, 1.0)
    )
    tangent_radius = float(np.clip(tangent_radius, radius * 0.0015, radius * 0.035))
    scale = np.full((n, 2), math.log(max(tangent_radius, 1e-6)), dtype=np.float32)
    rotation = _normals_to_quaternions(normals).astype(np.float32)
    print(
        f"Mesh surfels {path.name}: n={n}, normalized_area={normalized_surface_area:.6f}, "
        f"tangent_radius={tangent_radius:.6f}, opacity={opacity_probability:.3f}"
    )
    return {
        "xyz": xyz,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacity": opacity,
        "scales": scale,
        "rotations": rotation,
    }


def canonical_basis(right: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    return np.stack([right, forward, up], axis=1).astype(np.float32)


def sh_basis_degree3(directions: np.ndarray) -> np.ndarray:
    """Evaluate the real SH basis used by the bundled 2DGS renderer."""
    d = np.asarray(directions, dtype=np.float64)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    return np.stack(
        [
            np.full_like(x, C0),
            -C1 * y,
            C1 * z,
            -C1 * x,
            C2[0] * xy,
            C2[1] * yz,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * xz,
            C2[4] * (xx - yy),
            C3[0] * y * (3.0 * xx - yy),
            C3[1] * xy * z,
            C3[2] * y * (4.0 * zz - xx - yy),
            C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            C3[4] * x * (4.0 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3.0 * yy),
        ],
        axis=1,
    )


def rotate_sh_features(
    f_dc: np.ndarray,
    f_rest: np.ndarray,
    local_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate degree-3 SH coefficients with the Gaussian asset."""
    if f_rest.shape[1] < 15 or np.allclose(local_to_world, np.eye(3), atol=1e-7):
        return f_dc, f_rest

    # A fixed, well-conditioned spherical sample set is sufficient because
    # degree-3 SH has only 16 basis functions.
    n = 128
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    world_dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    local_dirs = world_dirs @ np.asarray(local_to_world, dtype=np.float64)
    transform = np.linalg.lstsq(
        sh_basis_degree3(world_dirs),
        sh_basis_degree3(local_dirs),
        rcond=None,
    )[0]

    coeff = np.concatenate([f_dc[:, None, :], f_rest[:, :15, :]], axis=1)
    rotated = np.einsum("ij,njc->nic", transform, coeff, optimize=True)
    result_rest = f_rest.copy()
    result_rest[:, :15, :] = rotated[:, 1:, :]
    return rotated[:, 0, :].astype(np.float32), result_rest.astype(np.float32)


def load_object_calibration(path: Path | None):
    if path is None:
        return None
    calibration = json.loads(path.read_text())
    return {
        "center": np.asarray(calibration["center"], dtype=np.float32),
        "radius": float(calibration["radius"]),
        "basis": np.asarray(calibration["basis"], dtype=np.float32),
    }


def load_gaussian_asset(
    path: Path,
    radius: float,
    max_points: int,
    seed: int,
    source_basis: np.ndarray | None = None,
    min_opacity: float = 0.0,
    max_scale_ratio: float = 0.0,
    max_anisotropy: float = 0.0,
    max_center_radius: float = 0.0,
    max_neighbor_distance: float = 0.0,
    neighbor_count: int = 8,
    projection_image: Path | None = None,
    largest_component_voxel: float = 0.0,
    source_slab_halfwidth: float = 0.0,
    support_cameras_json: Path | None = None,
    support_image_dir: Path | None = None,
    support_min_views: int = 0,
    support_min_ratio: float = 0.0,
    center_percentile: float = 100.0,
    source_center: np.ndarray | None = None,
    source_radius: float = 0.0,
):
    params = read_gaussian_ply(path)
    xyz_raw = params["xyz"].astype(np.float32)
    rotations = params["rotations"].astype(np.float32)
    f_dc_raw = params["f_dc"].astype(np.float32)
    if "f_rest" in params:
        f_rest_raw = params["f_rest"].astype(np.float32)
    else:
        f_rest_raw = np.zeros((len(xyz_raw), 15, 3), dtype=np.float32)
    opacity_raw = params["opacity"].reshape(-1, 1).astype(np.float32)
    scales_raw = params["scales"][:, :2].astype(np.float32)

    if projection_image is not None and projection_image.exists():
        projected_rgb, projected_valid = project_image_colors(xyz_raw, projection_image)
        if np.any(projected_valid):
            fallback_rgb = np.median(projected_rgb[projected_valid], axis=0)
            projected_rgb[~projected_valid] = fallback_rgb
            f_dc_raw = (projected_rgb - 0.5) / C0

    if (
        support_min_views > 0
        and support_min_ratio > 0.0
        and support_cameras_json is not None
        and support_cameras_json.exists()
        and support_image_dir is not None
        and support_image_dir.exists()
    ):
        support_keep = camera_mask_support(
            xyz_raw,
            support_cameras_json,
            support_image_dir,
            support_min_views,
            support_min_ratio,
        )
        xyz_raw = xyz_raw[support_keep]
        rotations = rotations[support_keep]
        f_dc_raw = f_dc_raw[support_keep]
        f_rest_raw = f_rest_raw[support_keep]
        opacity_raw = opacity_raw[support_keep]
        scales_raw = scales_raw[support_keep]

    if center_percentile < 100.0 and len(xyz_raw):
        center = np.median(xyz_raw, axis=0)
        distance = np.linalg.norm(xyz_raw - center[None, :], axis=1)
        radius_limit = float(np.percentile(distance, center_percentile))
        radial_keep = distance <= radius_limit
        print(
            f"Kept robust-center Gaussian centers: {radial_keep.sum()}/{len(radial_keep)} "
            f"(percentile={center_percentile}, radius={radius_limit:.6f})"
        )
        xyz_raw = xyz_raw[radial_keep]
        rotations = rotations[radial_keep]
        f_dc_raw = f_dc_raw[radial_keep]
        f_rest_raw = f_rest_raw[radial_keep]
        opacity_raw = opacity_raw[radial_keep]
        scales_raw = scales_raw[radial_keep]

    if min_opacity > 0.0:
        opacity_probability = 1.0 / (1.0 + np.exp(-opacity_raw[:, 0]))
        visible = opacity_probability >= min_opacity
        xyz_raw = xyz_raw[visible]
        rotations = rotations[visible]
        f_dc_raw = f_dc_raw[visible]
        f_rest_raw = f_rest_raw[visible]
        opacity_raw = opacity_raw[visible]
        scales_raw = scales_raw[visible]

    # Reject invalid and extreme-color records before coordinate normalization.
    # Scale filtering must happen after normalization because the source PLY and
    # the destination scene use different units.
    max_brightness = np.abs(f_dc_raw).max(axis=1)
    rogue_keep = (
        np.isfinite(xyz_raw).all(axis=1)
        & np.isfinite(scales_raw).all(axis=1)
        & np.isfinite(rotations).all(axis=1)
        & (max_brightness < 5.0)
    )
    xyz_raw = xyz_raw[rogue_keep]
    rotations = rotations[rogue_keep]
    f_dc_raw = f_dc_raw[rogue_keep]
    f_rest_raw = f_rest_raw[rogue_keep]
    opacity_raw = opacity_raw[rogue_keep]
    scales_raw = scales_raw[rogue_keep]

    if source_basis is not None:
        if source_center is None:
            lo, hi = np.percentile(xyz_raw, [5, 95], axis=0)
            center = ((lo + hi) * 0.5).astype(np.float32)
        else:
            center = source_center.astype(np.float32)
        xyz_raw = (xyz_raw - center) @ source_basis
        local_rot = quaternions_to_matrices(rotations)
        rotations = matrices_to_quaternions(source_basis.T[None, :, :] @ local_rot)
        f_dc_raw, f_rest_raw = rotate_sh_features(
            f_dc_raw, f_rest_raw, source_basis.T,
        )
        if source_slab_halfwidth > 0.0:
            local_lo, local_hi = np.percentile(xyz_raw, [2, 98], axis=0)
            local_radius = max(float(np.linalg.norm(local_hi - local_lo) * 0.5), 1e-6)
            depth = xyz_raw[:, 1]
            hist, edges = np.histogram(depth, bins=128, range=(local_lo[1], local_hi[1]))
            peak = int(np.argmax(hist))
            depth_center = float((edges[peak] + edges[peak + 1]) * 0.5)
            slab_keep = np.abs(depth - depth_center) <= source_slab_halfwidth * local_radius
            print(
                f"Kept source-camera depth slab: {slab_keep.sum()}/{len(slab_keep)} "
                f"(center={depth_center:.4f}, halfwidth={source_slab_halfwidth})"
            )
            xyz_raw = xyz_raw[slab_keep]
            rotations = rotations[slab_keep]
            f_dc_raw = f_dc_raw[slab_keep]
            f_rest_raw = f_rest_raw[slab_keep]
            opacity_raw = opacity_raw[slab_keep]
            scales_raw = scales_raw[slab_keep]

    if source_radius > 0.0:
        norm_scale = radius / source_radius
        xyz = xyz_raw * norm_scale
        keep = np.ones(len(xyz_raw), dtype=bool)
    else:
        xyz, keep, norm_scale = robust_normalize_with_scale(xyz_raw, radius)
    f_dc = f_dc_raw[keep]
    f_rest = f_rest_raw[keep]
    opacity = opacity_raw[keep]
    scales = scales_raw[keep] + math.log(max(norm_scale, 1e-8))
    rotations = rotations[keep]

    if largest_component_voxel > 0.0 and len(xyz):
        try:
            from scipy.ndimage import label

            normalized_xyz = xyz / max(radius, 1e-8)
            voxel = np.floor(
                (normalized_xyz - normalized_xyz.min(axis=0)) / largest_component_voxel
            ).astype(np.int32)
            occupancy = np.zeros(tuple(voxel.max(axis=0) + 1), dtype=bool)
            occupancy[tuple(voxel.T)] = True
            labels, _ = label(occupancy, np.ones((3, 3, 3), dtype=bool))
            point_labels = labels[tuple(voxel.T)]
            counts = np.bincount(point_labels)
            counts[0] = 0
            largest_label = int(np.argmax(counts))
            component_keep = point_labels == largest_label
            print(
                f"Kept largest Gaussian center component: {component_keep.sum()}/{len(component_keep)} "
                f"(voxel={largest_component_voxel})"
            )
            xyz = xyz[component_keep]
            f_dc = f_dc[component_keep]
            f_rest = f_rest[component_keep]
            opacity = opacity[component_keep]
            scales = scales[component_keep]
            rotations = rotations[component_keep]
        except ImportError:
            print("Warning: scipy unavailable; skipping largest-component filtering")

    center_keep = np.ones(len(xyz), dtype=bool)
    if max_center_radius > 0.0:
        center_keep &= np.linalg.norm(xyz, axis=1) <= radius * max_center_radius
    if max_neighbor_distance > 0.0 and len(xyz) > neighbor_count:
        try:
            from scipy.spatial import cKDTree

            normalized_xyz = xyz / max(radius, 1e-8)
            distances, _ = cKDTree(normalized_xyz).query(
                normalized_xyz,
                k=neighbor_count + 1,
                workers=-1,
            )
            center_keep &= distances[:, -1] <= max_neighbor_distance
        except ImportError:
            print("Warning: scipy unavailable; skipping Gaussian center-density filtering")
    removed_centers = int((~center_keep).sum())
    if removed_centers:
        print(
            f"Filtered {removed_centers}/{len(center_keep)} sparse Gaussian centers from {path.name} "
            f"(max_center_radius={max_center_radius}, "
            f"max_neighbor_distance={max_neighbor_distance}, neighbor_count={neighbor_count})"
        )
        xyz = xyz[center_keep]
        f_dc = f_dc[center_keep]
        f_rest = f_rest[center_keep]
        opacity = opacity[center_keep]
        scales = scales[center_keep]
        rotations = rotations[center_keep]

    if max_scale_ratio > 0.0 or max_anisotropy > 0.0:
        linear_scales = np.exp(np.clip(scales, -80.0, 80.0))
        largest_axis = linear_scales.max(axis=1)
        smallest_axis = np.maximum(linear_scales.min(axis=1), 1e-12)
        shape_keep = np.ones(len(xyz), dtype=bool)
        if max_scale_ratio > 0.0:
            shape_keep &= largest_axis <= radius * max_scale_ratio
        if max_anisotropy > 0.0:
            shape_keep &= largest_axis / smallest_axis <= max_anisotropy
        removed = int((~shape_keep).sum())
        if removed:
            print(
                f"Filtered {removed}/{len(shape_keep)} rogue Gaussians from {path.name} "
                f"(max_scale_ratio={max_scale_ratio}, max_anisotropy={max_anisotropy})"
            )
        xyz = xyz[shape_keep]
        f_dc = f_dc[shape_keep]
        f_rest = f_rest[shape_keep]
        opacity = opacity[shape_keep]
        scales = scales[shape_keep]
        rotations = rotations[shape_keep]

    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, f_dc, f_rest, opacity, scales, rotations = (
            xyz[idx],
            f_dc[idx],
            f_rest[idx],
            opacity[idx],
            scales[idx],
            rotations[idx],
        )
    return {
        "xyz": xyz.astype(np.float32),
        "f_dc": f_dc.astype(np.float32),
        "f_rest": f_rest.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "scales": scales.astype(np.float32),
        "rotations": rotations.astype(np.float32),
    }


def transform_points(
    xyz: np.ndarray,
    loc: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    yaw: float,
    pitch: float = 0.0,
    roll: float = 0.0,
):
    basis = local_basis(right, forward, up, yaw, pitch, roll)
    return xyz @ basis.T + loc.astype(np.float32)


def local_basis(
    right: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
):
    """Build a world basis from local yaw(up), pitch(right), and roll(forward)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float32)
    scene_basis = np.stack([right, forward, up], axis=1).astype(np.float32)
    return scene_basis @ rz @ rx @ ry


def quaternions_to_matrices(q: np.ndarray):
    q = q.astype(np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3), dtype=np.float32)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrices_to_quaternions(m: np.ndarray):
    q = np.empty((len(m), 4), dtype=np.float32)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    pos = tr > 0
    s = np.sqrt(np.maximum(tr[pos] + 1.0, 1e-8)) * 2
    q[pos, 0] = 0.25 * s
    q[pos, 1] = (m[pos, 2, 1] - m[pos, 1, 2]) / s
    q[pos, 2] = (m[pos, 0, 2] - m[pos, 2, 0]) / s
    q[pos, 3] = (m[pos, 1, 0] - m[pos, 0, 1]) / s

    rem = ~pos
    if np.any(rem):
        mr = m[rem]
        qr = q[rem]
        cond0 = (mr[:, 0, 0] > mr[:, 1, 1]) & (mr[:, 0, 0] > mr[:, 2, 2])
        cond1 = ~cond0 & (mr[:, 1, 1] > mr[:, 2, 2])
        cond2 = ~(cond0 | cond1)
        for cond, axis in ((cond0, 0), (cond1, 1), (cond2, 2)):
            if not np.any(cond):
                continue
            mm = mr[cond]
            if axis == 0:
                s = np.sqrt(np.maximum(1.0 + mm[:, 0, 0] - mm[:, 1, 1] - mm[:, 2, 2], 1e-8)) * 2
                qr[cond, 0] = (mm[:, 2, 1] - mm[:, 1, 2]) / s
                qr[cond, 1] = 0.25 * s
                qr[cond, 2] = (mm[:, 0, 1] + mm[:, 1, 0]) / s
                qr[cond, 3] = (mm[:, 0, 2] + mm[:, 2, 0]) / s
            elif axis == 1:
                s = np.sqrt(np.maximum(1.0 + mm[:, 1, 1] - mm[:, 0, 0] - mm[:, 2, 2], 1e-8)) * 2
                qr[cond, 0] = (mm[:, 0, 2] - mm[:, 2, 0]) / s
                qr[cond, 1] = (mm[:, 0, 1] + mm[:, 1, 0]) / s
                qr[cond, 2] = 0.25 * s
                qr[cond, 3] = (mm[:, 1, 2] + mm[:, 2, 1]) / s
            else:
                s = np.sqrt(np.maximum(1.0 + mm[:, 2, 2] - mm[:, 0, 0] - mm[:, 1, 1], 1e-8)) * 2
                qr[cond, 0] = (mm[:, 1, 0] - mm[:, 0, 1]) / s
                qr[cond, 1] = (mm[:, 0, 2] + mm[:, 2, 0]) / s
                qr[cond, 2] = (mm[:, 1, 2] + mm[:, 2, 1]) / s
                qr[cond, 3] = 0.25 * s
        q[rem] = qr
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    return q


def transform_gaussian_asset(
    asset: dict[str, np.ndarray],
    loc: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
):
    transformed = dict(asset)
    basis = local_basis(right, forward, up, yaw, pitch, roll)
    transformed["xyz"] = asset["xyz"] @ basis.T + loc.astype(np.float32)
    local_rot = quaternions_to_matrices(asset["rotations"])
    world_rot = basis[None, :, :] @ local_rot
    transformed["rotations"] = matrices_to_quaternions(world_rot)
    transformed["f_dc"], transformed["f_rest"] = rotate_sh_features(
        asset["f_dc"], asset["f_rest"], basis,
    )
    return transformed


def ring_position(
    center: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    scene_radius: float,
    radius: float,
    angle: float,
    height: float,
) -> np.ndarray:
    """Place an asset around the scene focus; angle 0 is camera-facing/front."""
    radial = math.sin(angle) * right - math.cos(angle) * forward
    return center + radial * (radius * scene_radius) + up * (height * scene_radius)


def use_dc_color_only(asset: dict[str, np.ndarray]):
    """Disable view-dependent SH terms after a world-space asset rotation."""
    result = dict(asset)
    result["f_rest"] = np.zeros_like(asset["f_rest"])
    return result


def use_isotropic_footprints(
    asset: dict[str, np.ndarray],
    radius: float,
    scale_ratio: float,
    opacity: float,
):
    """Use stable small disks while preserving trained centers/colors/opacities."""
    result = dict(asset)
    scale = math.log(max(radius * scale_ratio, 1e-6))
    result["scales"] = np.full_like(asset["scales"], scale)
    result["rotations"] = np.zeros_like(asset["rotations"])
    result["rotations"][:, 0] = 1.0
    if opacity > 0.0:
        probability = min(max(opacity, 1e-4), 1.0 - 1e-4)
        result["opacity"] = np.full_like(
            asset["opacity"],
            math.log(probability / (1.0 - probability)),
        )
    return result


def clamp_gaussian_footprints(
    asset: dict[str, np.ndarray],
    radius: float,
    min_scale_ratio: float,
    max_scale_ratio: float,
    min_opacity: float,
):
    """Retain trained surfel orientation while bounding unstable screen footprints."""
    result = dict(asset)
    linear = np.exp(np.clip(asset["scales"], -80.0, 80.0))
    lower = max(radius * min_scale_ratio, 1e-5)
    upper = max(radius * max_scale_ratio, lower)
    result["scales"] = np.log(np.clip(linear, lower, upper)).astype(np.float32)
    if min_opacity > 0.0:
        probability = 1.0 / (1.0 + np.exp(-asset["opacity"]))
        probability = np.clip(probability, min_opacity, 1.0 - 1e-5)
        result["opacity"] = np.log(probability / (1.0 - probability)).astype(np.float32)
    return result


def append_assets_to_gaussians(gaussians, assets: list[dict[str, np.ndarray]]):
    if not assets:
        return
    device = gaussians.get_xyz.device
    xyz = torch.cat([gaussians._xyz.detach()] + [torch.tensor(a["xyz"], dtype=torch.float32, device=device) for a in assets], dim=0)
    f_dc = torch.cat(
        [gaussians._features_dc.detach()]
        + [torch.tensor(a["f_dc"][:, None, :], dtype=torch.float32, device=device) for a in assets],
        dim=0,
    )
    f_rest = torch.cat(
        [gaussians._features_rest.detach()]
        + [torch.tensor(a["f_rest"], dtype=torch.float32, device=device) for a in assets],
        dim=0,
    )
    opacity = torch.cat(
        [gaussians._opacity.detach()]
        + [torch.tensor(a["opacity"], dtype=torch.float32, device=device) for a in assets],
        dim=0,
    )
    scales = torch.cat(
        [gaussians._scaling.detach()]
        + [torch.tensor(a["scales"], dtype=torch.float32, device=device) for a in assets],
        dim=0,
    )
    rotations = torch.cat(
        [gaussians._rotation.detach()]
        + [torch.tensor(a["rotations"], dtype=torch.float32, device=device) for a in assets],
        dim=0,
    )
    gaussians._xyz = nn.Parameter(xyz.requires_grad_(False))
    gaussians._features_dc = nn.Parameter(f_dc.requires_grad_(False))
    gaussians._features_rest = nn.Parameter(f_rest.requires_grad_(False))
    gaussians._opacity = nn.Parameter(opacity.requires_grad_(False))
    gaussians._scaling = nn.Parameter(scales.requires_grad_(False))
    gaussians._rotation = nn.Parameter(rotations.requires_grad_(False))
    gaussians.max_radii2D = torch.zeros((xyz.shape[0],), dtype=torch.float32, device=device)


def camera_poses(cameras):
    return np.array([np.linalg.inv(np.asarray(cam.world_view_transform.T.detach().cpu().numpy())) for cam in cameras])


def scene_anchor(cameras, tabletop: bool = False):
    c2ws = camera_poses(cameras)
    pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    focus = focus_point_fn(pose)
    cam_pos = c2ws[:, :3, 3]
    mean_cam = cam_pos.mean(axis=0)
    forward = normalize(focus - mean_cam)
    up = normalize(pose[:, :3, 1].mean(axis=0))
    right = normalize(np.cross(forward, up))
    up = normalize(np.cross(right, forward))
    if tabletop:
        # This sequence is captured from above the table. The mean viewing
        # direction is therefore the downward table normal, while the averaged
        # camera-up vector lies in the table plane. Rebuild a conventional
        # right/forward/up frame so local +Z assets stand on the table.
        table_up = -forward
        table_forward = up
        right = normalize(np.cross(table_forward, table_up))
        forward = normalize(np.cross(table_up, right))
        up = table_up
    radius = float(np.percentile(np.linalg.norm(cam_pos - focus[None, :], axis=1), 60))
    return focus.astype(np.float32), right.astype(np.float32), forward.astype(np.float32), up.astype(np.float32), max(radius, 1e-3)


def model_anchor(source: Path, model: Path, iteration: int, resolution: int, white_background: bool):
    dataset = SimpleNamespace(
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
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    cameras = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)
    focus, right, forward, up, radius = scene_anchor(cameras)
    del scene, gaussians
    torch.cuda.empty_cache()
    return focus, right, forward, up, radius


def project_points(xyz: np.ndarray, camera):
    h, w = int(camera.image_height), int(camera.image_width)
    full = camera.full_proj_transform.detach().cpu().numpy()
    world_view = camera.world_view_transform.detach().cpu().numpy()
    pts = np.concatenate([xyz.astype(np.float32), np.ones((len(xyz), 1), dtype=np.float32)], axis=1)
    clip = pts @ full
    view = pts @ world_view
    valid = np.abs(clip[:, 3]) > 1e-6
    ndc = np.zeros((len(xyz), 3), dtype=np.float32)
    ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
    x = ((ndc[:, 0] + 1.0) * 0.5 * w).astype(np.int32)
    y = ((1.0 - ndc[:, 1]) * 0.5 * h).astype(np.int32)
    valid &= (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    valid &= (x >= 0) & (x < w) & (y >= 0) & (y < h)
    depth = view[:, 2].astype(np.float32)
    return x[valid], y[valid], depth[valid], valid


def splat_points(frame: np.ndarray, xyz: np.ndarray, rgb: np.ndarray, camera, radius: int):
    h, w = frame.shape[:2]
    x, y, depth, valid = project_points(xyz, camera)
    colors = (np.clip(rgb[valid], 0, 1) * 255.0).astype(np.uint8)
    if len(x) == 0:
        return
    order = np.argsort(depth)[::-1]
    offsets = [(0, 0)]
    if radius >= 1:
        offsets += [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if radius >= 2:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
    if radius >= 3:
        offsets += [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

    for dx, dy in offsets:
        xx = np.clip(x[order] + dx, 0, w - 1)
        yy = np.clip(y[order] + dy, 0, h - 1)
        frame[yy, xx] = colors[order]


def render_background_frame(camera, gaussians, pipe, background):
    with torch.no_grad():
        result = render_gaussian(camera, gaussians, pipe, background)["render"]
    rgb = result.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (rgb * 255.0).astype(np.uint8)


def ensure_even_frame(frame: np.ndarray):
    h, w = frame.shape[:2]
    return frame[: h - (h % 2), : w - (w % 2)]


def make_contact(frame_dir: Path, contact_path: Path, frames: list[int]):
    tiles = []
    for frame_idx in frames:
        p = frame_dir / f"{frame_idx:05d}.jpg"
        if p.exists():
            if cv2 is not None:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is not None:
                    tiles.append(cv2.resize(im, (426, 240), interpolation=cv2.INTER_AREA))
            else:
                im = np.asarray(Image.open(p).convert("RGB").resize((426, 240), Image.Resampling.LANCZOS))
                tiles.append(im[:, :, ::-1])
    if len(tiles) >= 6:
        grid = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:6], axis=1)], axis=0)
        if cv2 is not None:
            cv2.imwrite(str(contact_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            Image.fromarray(grid[:, :, ::-1]).save(contact_path, quality=95)


def build_cameras(scene, mode: str, frames: int):
    train = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)
    if mode == "train":
        idx = np.linspace(0, len(train) - 1, frames).round().astype(int)
        return [train[i] for i in idx]
    return generate_path(train, n_frames=frames)


def render(args):
    dataset = SimpleNamespace(
        sh_degree=3,
        source_path=str(args.background_source.resolve()),
        model_path=str(args.background_model.resolve()),
        images="images",
        resolution=args.resolution,
        white_background=False,
        data_device="cuda",
        eval=True,
        render_items=["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"],
    )
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = build_cameras(scene, args.camera_mode, args.frames)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    focus, right, forward, up, scene_radius = scene_anchor(cameras, tabletop=True)
    object_radius = args.object_scene_scale * scene_radius
    base = focus + forward * (args.layout_forward * scene_radius) + up * (args.layout_up * scene_radius) + right * (args.layout_right * scene_radius)
    lateral = args.object_spacing * scene_radius
    if args.layout_mode == "ring":
        a_pos = ring_position(
            base, right, forward, up, scene_radius,
            args.object_a_radius, args.object_a_angle, args.object_a_height,
        )
        b_pos = ring_position(
            base, right, forward, up, scene_radius,
            args.object_b_radius, args.object_b_angle, args.object_b_height,
        )
        c_pos = ring_position(
            base, right, forward, up, scene_radius,
            args.object_c_radius, args.object_c_angle, args.object_c_height,
        )
    else:
        a_pos = base - right * (lateral + args.object_a_lateral_offset * scene_radius)
        b_pos = base
        c_pos = base + right * lateral

    if args.gaussian_composite:
        a_basis = None
        a_center = None
        a_source_radius = 0.0
        calibration = load_object_calibration(args.object_a_calibration)
        if calibration is not None:
            a_basis = calibration["basis"]
            a_center = calibration["center"]
            a_source_radius = calibration["radius"]
        elif args.object_a_canonicalize:
            _, a_right, a_forward, a_up, _ = model_anchor(
                args.object_a_source, args.object_a_model,
                args.object_a_iteration, args.resolution, True,
            )
            a_basis = canonical_basis(a_right, a_forward, a_up)
        a_asset = load_gaussian_asset(
            args.object_a, object_radius * 1.04 * args.object_a_scale_multiplier,
            args.object_points, args.seed + 1, a_basis, args.object_a_min_opacity,
            args.object_a_max_scale_ratio, args.object_a_max_anisotropy,
            args.object_a_max_center_radius, args.object_a_max_neighbor_distance,
            args.object_a_neighbor_count, args.object_a_image,
            args.object_a_largest_component_voxel,
            args.object_a_source_slab_halfwidth,
            args.object_a_model / "cameras.json",
            args.object_a_source / "images",
            args.object_a_support_min_views,
            args.object_a_support_min_ratio,
            args.object_a_center_percentile,
            a_center,
            a_source_radius,
        )
        if args.object_a_dc_only:
            a_asset = use_dc_color_only(a_asset)
        if args.object_a_footprint_mode == "isotropic":
            a_asset = use_isotropic_footprints(
                a_asset,
                object_radius * 1.04 * args.object_a_scale_multiplier,
                args.object_a_footprint_scale_ratio,
                args.object_a_footprint_opacity,
            )
        elif args.object_a_footprint_mode == "clamped":
            a_asset = clamp_gaussian_footprints(
                a_asset,
                object_radius * 1.04 * args.object_a_scale_multiplier,
                args.object_a_footprint_min_scale_ratio,
                args.object_a_footprint_scale_ratio,
                args.object_a_footprint_opacity,
            )
        if args.object_a_render_mode == "points":
            a_overlay = transform_gaussian_asset(
                a_asset,
                a_pos, right, forward, up,
                args.object_a_yaw, args.object_a_pitch, args.object_a_roll,
            )
            a_overlay_xyz = a_overlay["xyz"]
            a_overlay_rgb = np.clip(a_overlay["f_dc"] * C0 + 0.5, 0.0, 1.0)
        if args.object_b_gaussian is not None and args.object_b_gaussian.exists():
            b_asset = load_gaussian_asset(
                args.object_b_gaussian,
                object_radius * 0.96 * args.object_b_scale_multiplier,
                args.mesh_points,
                args.seed + 2,
            )
        else:
            b_asset = sample_mesh_gaussians(
                args.object_b,
                object_radius * 0.96 * args.object_b_scale_multiplier,
                args.mesh_points,
                args.seed + 2,
                (0.1, 0.25, 0.95),
                None,
                args.mesh_surfel_scale,
                args.mesh_surfel_opacity,
            )
        if args.object_c_gaussian is not None and args.object_c_gaussian.exists():
            c_asset = load_gaussian_asset(
                args.object_c_gaussian,
                object_radius * args.object_c_scale_multiplier,
                args.mesh_points,
                args.seed + 3,
            )
        else:
            c_asset = sample_mesh_gaussians(
                args.object_c,
                object_radius * args.object_c_scale_multiplier,
                args.mesh_points,
                args.seed + 3,
                (0.78, 0.78, 0.82),
                args.object_c_image if args.object_c_reproject_image else None,
                args.mesh_surfel_scale,
                args.mesh_surfel_opacity,
            )
        assets = []
        if args.object_a_render_mode == "gaussian":
            assets.append(transform_gaussian_asset(
                a_asset, a_pos, right, forward, up,
                args.object_a_yaw, args.object_a_pitch, args.object_a_roll,
            ))
        assets.extend([
            transform_gaussian_asset(
                b_asset, b_pos, right, forward, up,
                args.object_b_yaw, args.object_b_pitch, args.object_b_roll,
            ),
            transform_gaussian_asset(
                c_asset, c_pos, right, forward, up,
                args.object_c_yaw, args.object_c_pitch, args.object_c_roll,
            ),
        ])
        append_assets_to_gaussians(gaussians, assets)
        print(f"Composite Gaussian count: {gaussians.get_xyz.shape[0]}")
    else:
        a_xyz, a_rgb = load_gaussian_points(args.object_a, object_radius * 1.04, args.object_points, args.seed + 1, args.object_a_image)
        if args.object_b_gaussian is not None and args.object_b_gaussian.exists():
            b_xyz, b_rgb = load_gaussian_points(
                args.object_b_gaussian,
                object_radius * 0.96 * args.object_b_scale_multiplier,
                args.mesh_points,
                args.seed + 2,
                None,
            )
        else:
            b_xyz, b_rgb = sample_mesh_points(
                args.object_b,
                object_radius * 0.96 * args.object_b_scale_multiplier,
                args.mesh_points,
                args.seed + 2,
                (0.1, 0.25, 0.95),
                None,
            )
        if args.object_c_gaussian is not None and args.object_c_gaussian.exists():
            c_xyz, c_rgb = load_gaussian_points(
                args.object_c_gaussian,
                object_radius * args.object_c_scale_multiplier,
                args.mesh_points,
                args.seed + 3,
                None,
            )
        else:
            c_xyz, c_rgb = sample_mesh_points(
                args.object_c,
                object_radius * args.object_c_scale_multiplier,
                args.mesh_points,
                args.seed + 3,
                (0.78, 0.78, 0.82),
                args.object_c_image,
            )

        a_world = transform_points(
            a_xyz, a_pos, right, forward, up,
            args.object_a_yaw, args.object_a_pitch, args.object_a_roll,
        )
        b_world = transform_points(
            b_xyz, b_pos, right, forward, up,
            args.object_b_yaw, args.object_b_pitch, args.object_b_roll,
        )
        c_world = transform_points(
            c_xyz, c_pos, right, forward, up,
            args.object_c_yaw, args.object_c_pitch, args.object_c_roll,
        )
        a_rgb = np.clip(a_rgb * 1.08 + 0.02, 0, 1)
        b_rgb = np.clip(b_rgb * 1.05 + 0.02, 0, 1)
        c_rgb = np.clip(c_rgb * 1.55 + 0.08, 0, 1)

    args.frame_dir.mkdir(parents=True, exist_ok=True)
    for old in args.frame_dir.glob("*.jpg"):
        old.unlink()

    for i, cam in enumerate(tqdm(cameras, desc="render official-bg frames")):
        frame = render_background_frame(cam, gaussians, pipe, background)
        if args.gaussian_composite and args.object_a_render_mode == "points":
            splat_points(frame, a_overlay_xyz, a_overlay_rgb, cam, args.object_a_point_radius)
        elif not args.gaussian_composite:
            splat_points(frame, a_world, a_rgb, cam, args.object_radius)
            splat_points(frame, b_world, b_rgb, cam, args.object_radius)
            splat_points(frame, c_world, c_rgb, cam, args.object_radius)
        frame = ensure_even_frame(frame)
        frame_path = args.frame_dir / f"{i:05d}.jpg"
        if cv2 is not None:
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        else:
            Image.fromarray(frame).save(frame_path, quality=96)

    make_contact(args.frame_dir, args.contact, [0, args.frames // 6, args.frames // 3, args.frames // 2, args.frames * 2 // 3, args.frames * 5 // 6])
    encode_video_from_pattern(args.frame_dir / "%05d.jpg", args.output, args.fps, args.frames)
    print(f"Wrote frames: {args.frame_dir}")
    print(f"Wrote contact: {args.contact}")
    print(f"Wrote video: {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-source", type=Path, default=ROOT / "data/background/kitchen")
    parser.add_argument("--background-model", type=Path, default=ROOT / "outputs/task1/final/environment/kitchen_2dgs")
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--camera-mode", choices=["train", "path"], default="path")
    parser.add_argument("--object-a", type=Path, default=ROOT / "outputs/task1/final/objects/object_a/model/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--object-a-source", type=Path, default=ROOT / "data/task1/object_a/current")
    parser.add_argument("--object-a-model", type=Path, default=ROOT / "outputs/task1/final/objects/object_a/model")
    parser.add_argument(
        "--object-a-calibration",
        type=Path,
        default=ROOT / "outputs/task1/final/objects/object_a/fusion_calibration.json",
        help="Fixed center, source radius, and source-to-canonical basis for A.",
    )
    parser.add_argument("--object-a-iteration", type=int, default=30000)
    parser.add_argument(
        "--object-a-image",
        type=Path,
        default=None,
        help="Optional diagnostic color projection. By default, retain trained 2DGS colors.",
    )
    parser.add_argument("--object-b", type=Path, default=ROOT / "outputs/task1/final/objects/object_b/model/textured.obj")
    parser.add_argument("--object-c", type=Path, default=ROOT / "outputs/task1/final/objects/object_c/model/textured.obj")
    parser.add_argument("--object-b-gaussian", type=Path, default=None)
    parser.add_argument("--object-c-gaussian", type=Path, default=None)
    parser.add_argument("--object-c-image", type=Path, default=ROOT / "data/task1/object_c.png")
    parser.add_argument("--frame-dir", type=Path, default=ROOT / "outputs/task1/final/scene/frames")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task1/final/scene/fused_scene_360.mp4")
    parser.add_argument("--contact", type=Path, default=ROOT / "outputs/task1/final/scene/fused_scene_360_contact.jpg")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--object-points", type=int, default=143_000)
    parser.add_argument("--mesh-points", type=int, default=180_000)
    parser.add_argument("--object-radius", type=int, default=2)
    parser.add_argument("--object-scene-scale", type=float, default=0.072)
    parser.add_argument("--object-spacing", type=float, default=0.18)
    parser.add_argument("--layout-mode", choices=["ring", "line"], default="ring")
    parser.add_argument("--layout-forward", type=float, default=0.0)
    parser.add_argument("--layout-up", type=float, default=-0.12)
    parser.add_argument("--layout-right", type=float, default=0.0)
    parser.add_argument("--object-a-scale-multiplier", type=float, default=0.95)
    parser.add_argument("--object-b-scale-multiplier", type=float, default=1.45)
    parser.add_argument("--object-c-scale-multiplier", type=float, default=1.30)
    # Polar position: radius (0=center, ~0.3=outer), angle (0=front, π/2=right, π=back, -π/2=left), height (up offset)
    parser.add_argument("--object-a-radius", type=float, default=0.34)
    parser.add_argument("--object-a-angle", type=float, default=-0.8)
    parser.add_argument("--object-a-height", type=float, default=0.20)
    parser.add_argument("--object-b-radius", type=float, default=0.34)
    parser.add_argument("--object-b-angle", type=float, default=0.8)
    parser.add_argument("--object-b-height", type=float, default=0.25)
    parser.add_argument("--object-c-radius", type=float, default=0.34)
    parser.add_argument("--object-c-angle", type=float, default=1.8)
    parser.add_argument("--object-c-height", type=float, default=0.25)
    parser.add_argument("--object-a-dc-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--object-a-min-opacity", type=float, default=0.0)
    parser.add_argument(
        "--object-a-render-mode",
        choices=["points", "gaussian"],
        default="gaussian",
        help="Render A as transformed Gaussians (formal path) or diagnostic points.",
    )
    parser.add_argument("--object-a-point-radius", type=int, default=6)
    parser.add_argument(
        "--mesh-surfel-scale",
        type=float,
        default=1.45,
        help="Coverage multiplier for the area-derived B/C surfel tangent radius.",
    )
    parser.add_argument(
        "--mesh-surfel-opacity",
        type=float,
        default=0.92,
        help="Uniform opacity probability for surface-aligned B/C surfels.",
    )
    parser.add_argument(
        "--object-c-reproject-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic only: project the single input image over C; disabled for final UV/albedo rendering.",
    )
    parser.add_argument(
        "--object-a-footprint-mode",
        choices=["original", "isotropic", "clamped"],
        default="original",
        help="Retain trained 2DGS footprints; stabilized modes are diagnostic only.",
    )
    parser.add_argument(
        "--object-a-footprint-scale-ratio",
        type=float,
        default=0.03,
        help="Isotropic A footprint radius relative to the normalized object radius.",
    )
    parser.add_argument(
        "--object-a-footprint-min-scale-ratio",
        type=float,
        default=0.003,
        help="Minimum trained surfel axis retained by clamped footprint mode.",
    )
    parser.add_argument(
        "--object-a-footprint-opacity",
        type=float,
        default=0.24,
        help="Uniform opacity used by stabilized A footprints; <=0 preserves trained opacity.",
    )
    parser.add_argument(
        "--object-a-max-scale-ratio",
        type=float,
        default=0.0,
        help="Drop A splats whose normalized long axis exceeds this fraction of A's radius.",
    )
    parser.add_argument(
        "--object-a-max-anisotropy",
        type=float,
        default=0.0,
        help="Drop A splats with an extreme long-axis/short-axis ratio.",
    )
    parser.add_argument(
        "--object-a-max-center-radius",
        type=float,
        default=0.0,
        help="Drop A centers outside this normalized radius around the robust object center.",
    )
    parser.add_argument(
        "--object-a-max-neighbor-distance",
        type=float,
        default=0.0,
        help="Drop A centers whose kth nearest neighbor is farther than this normalized distance.",
    )
    parser.add_argument(
        "--object-a-neighbor-count",
        type=int,
        default=8,
        help="Neighbor rank used by A's spatial-density filter.",
    )
    parser.add_argument(
        "--object-a-largest-component-voxel",
        type=float,
        default=0.0,
        help="Keep only A's largest 3D center component at this normalized voxel size; <=0 disables.",
    )
    parser.add_argument(
        "--object-a-source-slab-halfwidth",
        type=float,
        default=0.0,
        help="When canonicalizing A, keep the dominant source-camera depth slab at this radius fraction.",
    )
    parser.add_argument("--object-a-support-min-views", type=int, default=5)
    parser.add_argument("--object-a-support-min-ratio", type=float, default=0.70)
    parser.add_argument(
        "--object-a-center-percentile",
        type=float,
        default=100.0,
        help="Before normalization, retain this radial percentile around A's median center.",
    )
    # Object rotation: yaw(around up), pitch(around right, 3.14=flip), roll(around forward)
    parser.add_argument("--object-a-yaw", type=float, default=-1.20)
    parser.add_argument("--object-a-pitch", type=float, default=1.5707963268)
    parser.add_argument("--object-a-roll", type=float, default=0.0)
    parser.add_argument("--object-b-yaw", type=float, default=0.25)
    parser.add_argument("--object-b-pitch", type=float, default=0.0)
    parser.add_argument("--object-b-roll", type=float, default=0.0)
    parser.add_argument("--object-c-yaw", type=float, default=-0.30)
    parser.add_argument("--object-c-pitch", type=float, default=0.0)
    parser.add_argument("--object-c-roll", type=float, default=0.0)
    parser.add_argument("--gaussian-composite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object-a-lateral-offset", type=float, default=0.0)
    parser.add_argument("--object-a-canonicalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260604)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
