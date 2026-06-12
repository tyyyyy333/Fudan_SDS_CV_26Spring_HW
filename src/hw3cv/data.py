"""Data validation and preparation utilities for HW3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CALVIN
# ---------------------------------------------------------------------------

CALVIN_ENVS = ["A", "B", "C", "D"]


def collect_calvin_episodes(calvin_root: Path, environments: List[str]) -> List[Dict[str, str]]:
    """Collect episode paths from CALVIN environment directories."""
    episodes: List[Dict[str, str]] = []
    for env in environments:
        env_dir = calvin_root / env
        if not env_dir.exists():
            continue
        for path in sorted(env_dir.rglob("*.npz")):
            episodes.append({"environment": env, "path": str(path)})
    return episodes


def validate_calvin_directory(calvin_root: Path) -> Dict[str, List[str]]:
    """Check which CALVIN environment directories exist and have episodes.

    Returns a dict mapping env name → list of episode filenames (empty if missing).
    """
    result: Dict[str, List[str]] = {}
    for env in CALVIN_ENVS:
        env_dir = calvin_root / env
        if not env_dir.is_dir():
            result[env] = []
            continue
        episodes = sorted(p.name for p in env_dir.rglob("*.npz"))
        result[env] = episodes
    return result


def get_calvin_stats(calvin_root: Path) -> Dict[str, object]:
    """Return episode counts and total .npz file sizes per environment."""
    stats: Dict[str, object] = {}
    for env in CALVIN_ENVS:
        env_dir = calvin_root / env
        episodes = sorted(env_dir.rglob("*.npz")) if env_dir.is_dir() else []
        total_bytes = sum(p.stat().st_size for p in episodes)
        stats[env] = {
            "episode_count": len(episodes),
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        }
    return stats


# ---------------------------------------------------------------------------
# Capture / Task1
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def validate_capture_directory(path: Path) -> Dict[str, object]:
    """Check that a capture directory contains usable images for COLMAP/2DGS."""
    if not path.is_dir():
        return {"valid": False, "image_count": 0, "error": f"Directory not found: {path}"}
    images = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {
        "valid": len(images) >= 1,
        "image_count": len(images),
        "formats": sorted(set(p.suffix.lower() for p in images)),
    }


def expected_task1_assets(outputs_task1: Path) -> Dict[str, Path]:
    """Return expected paths for all task1 output assets."""
    return {
        "object_a": outputs_task1 / "final" / "objects" / "object_a" / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        "object_b": outputs_task1 / "final" / "objects" / "object_b" / "model" / "textured.obj",
        "object_c": outputs_task1 / "final" / "objects" / "object_c" / "model" / "textured.obj",
        "background": outputs_task1 / "final" / "environment" / "kitchen_2dgs" / "point_cloud" / "iteration_7000" / "point_cloud.ply",
    }


def check_task1_assets(outputs_task1: Path) -> Tuple[Dict[str, bool], List[str]]:
    """Check which task1 assets exist. Returns (status_map, missing_list)."""
    assets = expected_task1_assets(outputs_task1)
    status = {name: path.exists() for name, path in assets.items()}
    missing = [name for name, ok in status.items() if not ok]
    return status, missing


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_scene_manifest(
    assets: Dict[str, Path],
    output_path: Path,
    *,
    camera_path: str = "orbit",
    frames: int = 360,
    object_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    object_scales: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Build a scene manifest dict without writing it to disk."""
    positions = object_positions or {
        "object_a": (0.0, 0.0, 0.0),
        "object_b": (0.6, 0.0, 0.0),
        "object_c": (-0.6, 0.0, 0.0),
    }
    scales = object_scales or {
        "object_a": 1.0,
        "object_b": 0.35,
        "object_c": 0.35,
    }

    objects = []
    for name in ("object_a", "object_b", "object_c"):
        kind = "2dgs" if name == "object_a" else "mesh"
        objects.append({
            "name": name,
            "type": kind,
            "path": str(assets[name]),
            "location": list(positions[name]),
            "rotation": [0.0, 0.0, 0.0],
            "scale": scales[name],
        })

    return {
        "background": {"type": "2dgs", "path": str(assets["background"])},
        "objects": objects,
        "render": {
            "camera_path": camera_path,
            "frames": frames,
            "output": str(output_path),
        },
    }


def write_scene_manifest(manifest: Dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
