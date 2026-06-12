#!/usr/bin/env python3
"""Blender scene rendering script for HW3 Task 1.

Loads a scene manifest and renders a fused multi-view video from 2DGS point
clouds and AIGC mesh assets.  Designed to run both inside Blender (actual
rendering) and as a plain Python script (validation + instructions).

Usage:
    blender --background --python scripts/render_scene.py -- --manifest scene.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Outside-Blender helpers
# ---------------------------------------------------------------------------

def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("background", "objects", "render"):
        if key not in manifest:
            raise KeyError(f"Manifest missing required key: {key}")
    return manifest


def _check_assets(manifest: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    bg_path = Path(manifest["background"]["path"])
    if not bg_path.exists():
        missing.append(f"background: {bg_path}")
    for obj in manifest["objects"]:
        p = Path(obj["path"])
        if not p.exists():
            missing.append(f"{obj['name']}: {p}")
    return missing


# ---------------------------------------------------------------------------
# Blender helpers (only called when bpy is available)
# ---------------------------------------------------------------------------

def _clear_default_scene() -> None:
    import bpy
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)


def _setup_world() -> None:
    import bpy
    world = bpy.data.worlds.new("HW3_World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Strength"].default_value = 0.8


def _setup_lighting() -> None:
    import bpy
    # Key light
    bpy.ops.object.light_add(type="SUN", location=(8, 5, 12))
    bpy.context.object.data.energy = 3.5
    bpy.context.object.data.angle = 0.1
    bpy.context.object.name = "Key_Light"
    # Fill
    bpy.ops.object.light_add(type="AREA", location=(-4, -2, 3))
    bpy.context.object.data.energy = 80.0
    bpy.context.object.data.size = 4.0
    bpy.context.object.name = "Fill_Light"


def _import_ply_pointcloud(ply_path: Path, name: str, location: List[float], scale: float) -> Any:
    """Import a 2DGS PLY file and set up a Geometry-Nodes point cloud render."""
    import bpy

    try:
        bpy.ops.wm.ply_import(filepath=str(ply_path))
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=str(ply_path))
    obj = bpy.context.selected_objects[0]
    obj.name = name
    obj.location = location
    obj.scale = (scale, scale, scale)

    # Geometry Nodes: render each Gaussian centre as a small disc
    gn_mod = obj.modifiers.new(name="PointCloudGN", type="NODES")
    node_group = bpy.data.node_groups.new("PointCloudNodes", "GeometryNodeTree")
    gn_mod.node_group = node_group

    if hasattr(node_group, "interface"):
        node_group.interface.new_socket(
            name="Geometry",
            in_out="INPUT",
            socket_type="NodeSocketGeometry",
        )
        node_group.interface.new_socket(
            name="Geometry",
            in_out="OUTPUT",
            socket_type="NodeSocketGeometry",
        )

    nodes = node_group.nodes
    links = node_group.links
    nodes.clear()

    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")

    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    set_radius = nodes.new("GeometryNodeSetPointRadius")
    radius_input = nodes.new("ShaderNodeValue")
    radius_input.outputs[0].default_value = 0.008

    links.new(group_in.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], set_radius.inputs["Points"])
    links.new(radius_input.outputs["Value"], set_radius.inputs["Radius"])
    links.new(set_radius.outputs["Points"], group_out.inputs["Geometry"])

    # Material with vertex-colour passthrough
    mat = bpy.data.materials.new(name=f"{name}_PointMat")
    mat.use_nodes = True
    mat_nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links
    mat_nodes.clear()

    attr_node = mat_nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "Col"
    output_node = mat_nodes.new("ShaderNodeOutputMaterial")
    bsdf = mat_nodes.new("ShaderNodeBsdfPrincipled")
    mat_links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
    mat_links.new(attr_node.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.6
    mat_links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    obj.data.materials.append(mat)
    return obj


def _import_obj_mesh(obj_path: Path, name: str, location: List[float], scale: float) -> Any:
    import bpy
    try:
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    obj = bpy.context.selected_objects[0]
    obj.name = name
    obj.location = location
    obj.scale = (scale, scale, scale)
    return obj


def _create_orbit_camera(frames: int, distance: float = 6.0, height: float = 2.0) -> None:
    import bpy

    bpy.ops.object.camera_add(location=(distance, 0, height))
    cam = bpy.context.object
    cam.name = "Orbit_Camera"
    bpy.context.scene.camera = cam

    # Empty target at origin for track-to constraint
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0.5))
    target = bpy.context.object
    target.name = "Camera_Target"

    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    cam.rotation_euler = (math.radians(75), 0, 0)
    cam.keyframe_insert(data_path="rotation_euler", frame=1)

    # Animate orbit
    for frame in range(1, frames + 1):
        angle = 2 * math.pi * (frame - 1) / frames
        cam.location = (
            distance * math.cos(angle),
            distance * math.sin(angle),
            height,
        )
        cam.keyframe_insert(data_path="location", frame=frame)

    cam.data.lens = 35


def _setup_render(output_path: Path, frames: int) -> None:
    import bpy
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 32
    scene.render.filepath = str(output_path.with_suffix(""))
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = frames


# ---------------------------------------------------------------------------
# Main render entrypoint
# ---------------------------------------------------------------------------

def render_scene_blender(manifest: Dict[str, Any]) -> None:
    import bpy

    _clear_default_scene()
    _setup_world()
    _setup_lighting()

    # Background (2DGS point cloud)
    bg = manifest["background"]
    bg_path = Path(bg["path"])
    if bg_path.exists():
        _import_ply_pointcloud(bg_path, "Background", [0, 0, 0], 1.0)
    else:
        print(f"[WARN] background asset not found: {bg_path}")

    # Foreground objects
    for spec in manifest["objects"]:
        path = Path(spec["path"])
        loc = spec.get("location", [0, 0, 0])
        scl = spec.get("scale", 1.0)

        if not path.exists():
            print(f"[WARN] skipping missing asset {spec['name']}: {path}")
            continue

        kind = spec.get("type", "mesh")
        if kind == "2dgs" or path.suffix == ".ply":
            _import_ply_pointcloud(path, spec["name"], loc, scl)
        else:
            _import_obj_mesh(path, spec["name"], loc, scl)

    render_cfg = manifest["render"]
    frames = render_cfg.get("frames", 180)
    _create_orbit_camera(frames)
    _setup_render(Path(render_cfg["output"]), frames)

    print(f"Starting render: {frames} frames → {render_cfg['output']}")
    bpy.ops.render.render(animation=True)
    print("Render complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="HW3 Task 1 scene renderer")
    parser.add_argument("--manifest", type=Path)
    cli_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(cli_args)
    if args.manifest is None:
        args.manifest = Path("outputs/task1/scene_manifest.json")

    manifest = validate_manifest(args.manifest)
    missing = _check_assets(manifest)
    if missing:
        print("Missing assets (render will skip them):")
        for m in missing:
            print(f"  - {m}")

    try:
        import bpy  # noqa: F401
    except ImportError:
        output = Path(manifest["render"]["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        placeholder = output.with_suffix(".placeholder.txt")
        placeholder.write_text(
            "Scene manifest validated. Run inside Blender for final video:\n"
            "  blender --background --python scripts/render_scene.py -- --manifest <path>\n",
            encoding="utf-8",
        )
        print(f"Outside Blender — wrote {placeholder}")
        return 0

    render_scene_blender(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
