"""Task 1 pipeline: COLMAP → 2DGS / threestudio / Magic123 → scene fusion.

This module orchestrates external tools — it does NOT reimplement the ML
algorithms.  Its job is to:
  1. Generate proper configuration files for each tool.
  2. Chain the multi-step pipeline (e.g., COLMAP → 2DGS training).
  3. Convert output formats so different tools' outputs can be merged.
  4. Produce the fused scene (PLY + Blender manifest) and the final video.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import HW3Config, Task1ObjectConfig
from .conversion import (
    mesh_to_gaussian_ply,
    read_gaussian_ply,
    read_obj,
    read_simple_ply,
    write_gaussian_ply,
)
from .data import build_scene_manifest as _build_manifest_dict
from .data import check_task1_assets, expected_task1_assets, validate_capture_directory
from .scene_fusion import FusedScene, build_fused_scene


# ======================================================================
# COLMAP pipeline
# ======================================================================

def build_colmap_command(image_dir: Path, output_dir: Path,
                         quality: str = "medium") -> List[str]:
    """Build the COLMAP automatic reconstruction command.

    Uses COLMAP's automatic reconstruction pipeline:
      colmap automatic_reconstructor --image_path ... --workspace_path ...

    Args:
        image_dir: directory of source images (or extracted video frames).
        output_dir: workspace directory for COLMAP output.
        quality: "low", "medium", or "high" (passed to COLMAP).
    """
    return [
        "colmap", "automatic_reconstructor",
        "--image_path", str(image_dir),
        "--workspace_path", str(output_dir),
        "--quality", quality,
        "--single_camera", "1",
        "--dense", "0",  # sparse only; 2DGS only needs sparse
    ]


def extract_frames_command(video_path: Path, output_dir: Path,
                           fps: int = 2) -> List[str]:
    """Build an ffmpeg command to extract frames from a video.

    Args:
        video_path: input video file.
        output_dir: directory for extracted frames (named frame_%06d.jpg).
        fps: frames per second to extract.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_dir / "frame_%06d.jpg"),
    ]


# ======================================================================
# 2DGS training
# ======================================================================

def build_train_2dgs_command(config: HW3Config, target: str) -> List[str]:
    """Build the 2DGS training command for a given target.

    Uses the official 2d-gaussian-splatting train.py script.

    Args:
        config: HW3Config.
        target: "object_a" or "background".

    Returns:
        Shell command as a list of strings.
    """
    if target == "background":
        source = config.task1.background.source
        output = config.task1.background.output
    elif target == "object_a":
        obj_a = config.task1.objects["object_a"]
        if obj_a.source is None:
            raise ValueError("object_a requires a source capture directory")
        source = obj_a.source
        output = obj_a.output
    else:
        raise ValueError("target must be object_a or background")
    return ["python", "train.py", "-s", str(source), "-m", str(output), "--eval"]


# ======================================================================
# threestudio (text-to-3D, object B)
# ======================================================================

def build_threestudio_command(config: HW3Config) -> List[str]:
    """Build the threestudio text-to-3D command for object_b."""
    obj = config.task1.objects["object_b"]
    if not obj.prompt:
        raise ValueError("object_b requires a text prompt in config")
    return [
        "python", "launch.py",
        "--config", "configs/dreamfusion-sd.yaml",
        "--train",
        f"system.prompt_processor.prompt={obj.prompt}",
        "system.prompt_processor.pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        "system.guidance.pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        f"exp_root_dir={obj.output.parent}",
        f"name={obj.output.name}",
        "tag=run",
    ]


def threestudio_export_mesh_command(trial_dir: Path, output_obj: Path) -> List[str]:
    """Export the trained threestudio model to a textured OBJ mesh."""
    trial_dir = Path(trial_dir)
    return [
        "python", "launch.py",
        "--config", str(trial_dir / "configs" / "parsed.yaml"),
        "--export",
        "resume=" + str(trial_dir / "ckpts" / "last.ckpt"),
        "system.exporter_type=mesh-exporter",
        "system.exporter.fmt=obj",
        "system.exporter.save_uv=false",
        "system.exporter.save_texture=true",
        "system.exporter.context_type=cuda",
    ]


# ======================================================================
# Magic123 (image-to-3D, object C)
# ======================================================================

def build_magic123_command(config: HW3Config) -> List[str]:
    """Build the Magic123 single-image-to-3D command for object_c."""
    obj = config.task1.objects["object_c"]
    if obj.image is None:
        raise ValueError("object_c requires an input image path configured")
    return [
        "python", "main.py",
        "-O",
        "--image", str(obj.image),
        "--workspace", str(obj.output),
        "--text", "A high-resolution DSLR image of a white game controller",
        "--guidance", "SD", "zero123",
        "--lambda_guidance", "1.0", "80",
        "--save_mesh",
    ]


# ======================================================================
# AIGC mesh → Gaussian conversion (post-processing)
# ======================================================================

def convert_mesh_to_gs_ply(mesh_path: Path, output_path: Path,
                           n_samples: int = 200_000,
                           base_scale: float = 0.005) -> Path:
    """Convert a textured mesh (OBJ from threestudio/Magic123) to a Gaussian PLY.

    This implements the "code-level merging" strategy described in the
    assignment: sampling the AIGC mesh surface, creating 2D Gaussian surfels
    aligned with the local geometry, and writing a 2DGS-compatible PLY file
    that can be merged with the 2DGS background.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    verts, faces, colors = read_obj(mesh_path)
    params = mesh_to_gaussian_ply(verts, faces, vertex_colors=colors,
                                  n_samples=n_samples, base_scale=base_scale)
    return write_gaussian_ply(params, output_path)


def convert_simple_ply_to_gs_ply(ply_path: Path, output_path: Path,
                                 base_opacity: float = 0.5,
                                 base_scale: float = 0.006) -> Path:
    """Convert a COLMAP/simple xyz+rgb PLY into a Gaussian-compatible PLY."""
    xyz, rgb = read_simple_ply(ply_path)
    n_points = len(xyz)
    colors = rgb if rgb is not None else np.full((n_points, 3), 0.5, dtype=np.float32)
    f_dc = ((colors - 0.5) / 0.28209479177387814).astype(np.float32)
    opacity = np.full(n_points, np.log(base_opacity / (1 - base_opacity)), dtype=np.float32)
    scales = np.full((n_points, 3), np.log(base_scale), dtype=np.float32)
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_points, 1))
    return write_gaussian_ply(
        {
            "xyz": xyz.astype(np.float32),
            "f_dc": f_dc,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
        },
        output_path,
    )


# ======================================================================
# Scene fusion orchestration
# ======================================================================

def make_scene_manifest(config: HW3Config, manifest_path: Path | None = None) -> Path:
    """Generate the Blender scene manifest JSON.

    Requires all task1 assets to exist.  Run after training/generation is complete.
    """
    assets = expected_task1_assets(config.outputs.task1)
    missing = [name for name, path in assets.items() if not path.exists()]
    if missing:
        detail = ", ".join(f"{name}: {assets[name]}" for name in missing)
        raise FileNotFoundError(f"Missing task1 assets: {detail}")

    output_path = manifest_path or config.task1.scene.manifest
    manifest = _build_manifest_dict(assets, config.task1.scene.render_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return output_path


def fuse_scene_from_manifest(config: HW3Config,
                             manifest_path: Optional[Path] = None,
                             fused_ply_output: Optional[Path] = None) -> Path:
    """Load the scene manifest, merge all assets into a unified Gaussian PLY.

    This is the "code-level merging" approach: AIGC meshes are converted to
    Gaussian surfels on-the-fly and merged with the 2DGS point clouds.
    """
    mf_path = manifest_path or config.task1.scene.manifest
    if not mf_path.exists():
        raise FileNotFoundError(f"Manifest not found: {mf_path}. Run make-scene-manifest first.")

    manifest = json.loads(mf_path.read_text(encoding="utf-8"))
    bg = manifest.get("background")
    objects = manifest.get("objects", [])

    specs = []
    for obj in objects:
        path = Path(obj["path"])
        if not path.exists():
            print(f"  [SKIP] {obj['name']}: {path} not found")
            continue
        specs.append({
            "name": obj["name"],
            "path": path,
            "kind": obj.get("type", "2dgs"),
            "location": tuple(obj.get("location", [0, 0, 0])),
            "rotation": tuple(obj.get("rotation", [0, 0, 0])),
            "scale": obj.get("scale", 1.0),
            "normalize_radius": obj.get("normalize_radius"),
            "normalize_crop": obj.get("normalize_crop", 2.4),
        })

    scene = FusedScene()
    if bg and bg.get("path"):
        bg_path = Path(bg["path"])
        if bg_path.exists():
            scene.add_background(
                bg_path,
                kind=bg.get("type", "2dgs"),
                location=tuple(bg.get("location", [0, 0, 0])),
                rotation=tuple(bg.get("rotation", [0, 0, 0])),
                scale=bg.get("scale", 1.0),
                normalize_radius=bg.get("normalize_radius"),
                normalize_crop=bg.get("normalize_crop", 2.4),
            )

    for spec in specs:
        scene.add_object(
            name=spec["name"],
            path=spec["path"],
            kind=spec.get("kind", "2dgs"),
            location=spec.get("location", (0, 0, 0)),
            rotation=spec.get("rotation", (0, 0, 0)),
            scale=spec.get("scale", 1.0),
            normalize_radius=spec.get("normalize_radius"),
            normalize_crop=spec.get("normalize_crop", 2.4),
        )

    output = fused_ply_output or config.outputs.task1 / "fused_scene.ply"
    scene.export(output)
    print(f"Fused scene → {output}")
    return output


def build_render_scene_command(config: HW3Config) -> List[str]:
    """Build the Blender rendering command for the scene manifest."""
    local_blender = config.project_root / "blender-4.2.0-linux-x64" / "blender"
    blender_exe = str(local_blender) if local_blender.exists() else (shutil.which("blender") or "blender")
    return [
        blender_exe, "--background", "--python",
        str(config.project_root / "scripts" / "render_scene.py"),
        "--", "--manifest", str(config.task1.scene.manifest),
    ]


# ======================================================================
# Data validation (re-exports from data.py for CLI convenience)
# ======================================================================

def expected_assets(config: HW3Config) -> Dict[str, Path]:
    return expected_task1_assets(config.outputs.task1)


def check_assets(config: HW3Config) -> Dict[str, bool]:
    status, _ = check_task1_assets(config.outputs.task1)
    return status


def validate_capture(config: HW3Config, target: str) -> Dict[str, object]:
    if target == "object_a":
        src = config.task1.objects["object_a"].source
    elif target == "background":
        src = config.task1.background.source
    else:
        raise ValueError(f"target must be object_a or background, got {target}")
    if src is None:
        return {"valid": False, "image_count": 0, "error": "No source directory configured"}
    return validate_capture_directory(src)


def _require_prompt(obj: Task1ObjectConfig, target: str) -> None:
    if not obj.prompt:
        raise ValueError(f"{target} requires a text prompt")
