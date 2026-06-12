"""Scene fusion: merge 2DGS + AIGC assets into a unified Gaussian representation.

The core challenge (explicitly called out in the assignment) is that 2DGS
produces explicit Gaussian surfels, while threestudio/Magic123 produce
meshes or implicit fields.  This module implements both fusion strategies:

  Strategy 1 — "code-level merging" (代码级拼接):
    Convert AIGC meshes to Gaussian surfels → merge into a single PLY that
    can be fed back into the 2DGS renderer or imported into Blender.

  Strategy 2 — Blender-based (recommended for final video):
    Export each asset in its native format and let Blender handle the
    multi-representation rendering (PLY point clouds + OBJ meshes).

The fused output is always a unified Gaussian PLY file so downstream
tools (2DGS viewer, Blender, our render script) can consume it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .conversion import (
    merge_gaussian_plys,
    mesh_to_gaussian_ply,
    read_gaussian_ply,
    read_obj,
    read_simple_ply,
    write_gaussian_ply,
    write_simple_ply,
)


# ======================================================================
# Asset descriptors
# ======================================================================

class SceneAsset:
    """Describes one asset in the scene with its transform."""

    def __init__(self, name: str, source_path: Path, kind: str,
                 location: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float] = (0, 0, 0),  # Euler XYZ, radians
                 scale: float = 1.0,
                 normalize_radius: Optional[float] = None,
                 normalize_crop: float = 2.4):
        self.name = name
        self.source_path = Path(source_path)
        self.kind = kind          # "2dgs" | "mesh" | "ply"
        self.location = np.array(location, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = scale
        self.normalize_radius = normalize_radius
        self.normalize_crop = normalize_crop

    def load_as_gaussian(self, mesh_to_gs_samples: int = 100_000) -> Dict[str, np.ndarray]:
        """Load this asset and return Gaussian parameter arrays in world space."""
        if self.kind == "2dgs" or (self.kind == "ply" and self._is_gs_ply()):
            params = read_gaussian_ply(self.source_path)
        elif self.kind == "mesh" or self.source_path.suffix == ".obj":
            verts, faces, colors = read_obj(self.source_path)
            params = mesh_to_gaussian_ply(verts, faces, vertex_colors=colors,
                                          n_samples=mesh_to_gs_samples)
        elif self.kind == "ply":
            # Plain PLY: sample as Gaussian with default params
            xyz, rgb = read_simple_ply(self.source_path)
            N = len(xyz)
            f_dc = np.zeros((N, 3), dtype=np.float32)
            if rgb is not None:
                f_dc = ((rgb - 0.5) / 0.28209479177387814).astype(np.float32)
            params = {
                "xyz": xyz,
                "f_dc": f_dc,
                "opacity": np.full(N, _inv_sigmoid_np(0.5), dtype=np.float32),
                "scales": np.full((N, 3), np.log(0.005), dtype=np.float32),
                "rotations": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1)),
            }
        else:
            raise ValueError(f"Unknown asset kind: {self.kind}")

        if self.normalize_radius is not None:
            params = _robust_normalize_gaussian_params(
                params, self.normalize_radius, self.normalize_crop
            )

        # Apply transform: scale, then rotate, then translate
        return _transform_gaussian_params(params, self.scale, self.rotation, self.location)

    def _is_gs_ply(self) -> bool:
        """Check if a PLY file has the 2DGS property set."""
        try:
            params = read_gaussian_ply(self.source_path)
            return "f_dc" in params and "rotations" in params
        except Exception:
            return False


# ======================================================================
# Fused scene
# ======================================================================

class FusedScene:
    """Holds a collection of SceneAssets and merges them.

    Usage:
        scene = FusedScene()
        scene.add_background("outputs/background_2dgs/.../point_cloud.ply")
        scene.add_object("object_a", "outputs/object_a_2dgs/.../point_cloud.ply",
                         location=(0, 0, 0), scale=1.0)
        scene.add_object("object_b", "outputs/object_b/textured.obj",
                         location=(0.6, 0, 0), scale=0.35, kind="mesh")
        scene.add_object("object_c", "outputs/object_c/textured.obj",
                         location=(-0.6, 0, 0), scale=0.35, kind="mesh")
        scene.export("outputs/fused_scene.ply")
    """

    def __init__(self):
        self.background: Optional[SceneAsset] = None
        self.objects: List[SceneAsset] = []

    def add_background(self, path: Path, kind: str = "2dgs",
                       normalize_radius=None, normalize_crop=2.4,
                       location=(0, 0, 0), rotation=(0, 0, 0),
                       scale=1.0) -> None:
        self.background = SceneAsset(
            "background", path, kind, location, rotation, scale,
            normalize_radius=normalize_radius,
            normalize_crop=normalize_crop,
        )

    def add_object(self, name: str, path: Path, *,
                   location=(0, 0, 0), rotation=(0, 0, 0),
                   scale=1.0, kind="2dgs", normalize_radius=None,
                   normalize_crop=2.4) -> None:
        self.objects.append(SceneAsset(
            name, path, kind, location, rotation, scale,
            normalize_radius=normalize_radius,
            normalize_crop=normalize_crop,
        ))

    def export(self, output_path: Path, mesh_to_gs_samples: int = 100_000) -> Path:
        """Merge all assets into a single Gaussian PLY."""
        parts: List[Dict[str, np.ndarray]] = []

        if self.background is not None:
            parts.append(self.background.load_as_gaussian(mesh_to_gs_samples))

        for obj in self.objects:
            parts.append(obj.load_as_gaussian(mesh_to_gs_samples))

        if not parts:
            raise RuntimeError("No assets in scene")

        merged = merge_gaussian_plys(parts)
        return write_gaussian_ply(merged, output_path)

    def export_manifest(self, output_path: Path,
                        camera_path: str = "orbit", frames: int = 180,
                        video_output: Optional[Path] = None) -> Path:
        """Write a JSON manifest for the Blender render script.

        The manifest describes each asset's source path, type, and transform
        so the Blender script can load and render them.
        """
        manifest: dict = {
            "background": self._asset_dict(self.background) if self.background else None,
            "objects": [self._asset_dict(o) for o in self.objects],
            "render": {
                "camera_path": camera_path,
                "frames": frames,
                "output": str(video_output or output_path.parent / "fused_scene.mp4"),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        return output_path

    @staticmethod
    def _asset_dict(asset: SceneAsset) -> dict:
        return {
            "name": asset.name,
            "type": asset.kind,
            "path": str(asset.source_path),
            "location": asset.location.tolist(),
            "rotation": asset.rotation.tolist(),
            "scale": asset.scale,
            "normalize_radius": asset.normalize_radius,
            "normalize_crop": asset.normalize_crop,
        }


# ======================================================================
# Convenience: build FusedScene from a config-like dict
# ======================================================================

def build_fused_scene(background_path: Optional[Path],
                      object_specs: List[dict]) -> FusedScene:
    """Build a FusedScene from a list of object specs.

    Each spec is a dict with keys:
      name, path, kind (default "2dgs"), location (default origin),
      rotation (default zero), scale (default 1.0)

    Example:
        scene = build_fused_scene(
            background_path=Path("outputs/bg.ply"),
            object_specs=[
                {"name": "obj_a", "path": Path("outputs/a.ply"), "kind": "2dgs"},
                {"name": "obj_b", "path": Path("outputs/b.obj"), "kind": "mesh",
                 "location": (0.6, 0, 0), "scale": 0.35},
            ],
        )
    """
    scene = FusedScene()
    if background_path is not None and background_path.exists():
        scene.add_background(background_path)
    for spec in object_specs:
        path = spec["path"]
        if not path.exists():
            continue
        scene.add_object(
            name=spec["name"],
            path=path,
            location=spec.get("location", (0, 0, 0)),
            rotation=spec.get("rotation", (0, 0, 0)),
            scale=spec.get("scale", 1.0),
            kind=spec.get("kind", "2dgs"),
            normalize_radius=spec.get("normalize_radius"),
            normalize_crop=spec.get("normalize_crop", 2.4),
        )
    return scene


# ======================================================================
# Internal helpers
# ======================================================================

def _inv_sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-8, 1 - 1e-8) / (1 - np.clip(x, 1e-8, 1 - 1e-8)))


def _robust_normalize_gaussian_params(
    params: Dict[str, np.ndarray],
    radius: float,
    crop: float,
) -> Dict[str, np.ndarray]:
    """Center and scale one asset into a stable display radius."""
    xyz = params["xyz"].astype(np.float32)
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = radius / max(float(np.linalg.norm(hi - lo) * 0.5), 1e-6)
    normalized_xyz = (xyz - center) * scale
    keep = np.linalg.norm(normalized_xyz, axis=1) < crop * radius

    result: Dict[str, np.ndarray] = {}
    n = len(xyz)
    for key, value in params.items():
        if isinstance(value, np.ndarray) and len(value) == n:
            result[key] = value[keep]
        else:
            result[key] = value
    result["xyz"] = normalized_xyz[keep].astype(np.float32)
    result["scales"] = (params["scales"][keep] + np.log(scale)).astype(np.float32)
    return result


def _euler_to_rotmat(euler: np.ndarray) -> np.ndarray:
    """Euler XYZ (radians) → 3×3 rotation matrix."""
    cx, cy, cz = np.cos(euler)
    sx, sy, sz = np.sin(euler)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _transform_gaussian_params(params: Dict[str, np.ndarray],
                               scale: float, rotation: np.ndarray,
                               translation: np.ndarray) -> Dict[str, np.ndarray]:
    """Apply similarity transform to a set of Gaussian parameters in-place-like."""
    R = _euler_to_rotmat(rotation)
    xyz = (params["xyz"] * scale) @ R.T + translation
    new_scales = params["scales"] + np.log(scale)  # log-scale: multiply by scale

    # Rotate quaternions
    q_rot = _euler_to_quat(rotation)
    old_quats = params["rotations"]
    new_quats = np.zeros_like(old_quats)
    for i in range(len(old_quats)):
        new_quats[i] = _quat_mul(q_rot, old_quats[i])

    result = dict(params)
    result["xyz"] = xyz.astype(np.float32)
    result["scales"] = new_scales.astype(np.float32)
    result["rotations"] = new_quats.astype(np.float32)
    return result


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    cx, cy, cz = np.cos(euler * 0.5)
    sx, sy, sz = np.sin(euler * 0.5)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])
