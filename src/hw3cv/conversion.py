"""Format conversion utilities between 2DGS, AIGC mesh, and scene-fusion formats.

Three main conversion paths:

  1. AIGC mesh (OBJ from threestudio/Magic123) → Gaussian PLY
     Sample surface points, estimate normals, create flat 2D Gaussian surfels
     aligned to the local surface.  This enables "代码级拼接" — merging
     AIGC assets directly into the 2DGS representation.

  2. 2DGS PLY ↔ Gaussian parameter arrays
     Read/write the custom PLY format used by 2d-gaussian-splatting that
     carries xyz, f_dc (SH DC), opacity, scale, rotation quaternion, and
     optional higher-order SH coefficients.

  3. General point-cloud PLY ↔ numpy arrays
     Lightweight read/write for standard xyz+rgb point clouds used by
     COLMAP and Blender.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# 1.  AIGC mesh → Gaussian PLY  (the key technical contribution)
# ======================================================================

def _sample_mesh_surface_with_barycentrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample mesh triangles and keep the barycentric coordinates."""
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Cannot sample an empty mesh")

    rng = np.random.RandomState(seed)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        raise ValueError("Cannot sample a mesh with only degenerate faces")
    probs = np.zeros_like(areas, dtype=np.float64)
    probs[valid] = areas[valid].astype(np.float64)
    probs /= probs.sum()

    tri_idx = rng.choice(len(faces), size=n_samples, p=probs)
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]

    p0 = v0[tri_idx]
    p1 = v1[tri_idx]
    p2 = v2[tri_idx]
    points = p0 + r1[:, None] * (p1 - p0) + r2[:, None] * (p2 - p0)

    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8)
    normals = face_normals[tri_idx]
    return (
        points.astype(np.float32),
        normals.astype(np.float32),
        tri_idx.astype(np.int64),
        r1.astype(np.float32),
        r2.astype(np.float32),
    )


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, n_samples: int,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly sample points and normals from a triangle mesh surface.

    Args:
        vertices: (V, 3) float32
        faces:    (F, 3) int32, 0-indexed
        n_samples: number of surface samples

    Returns:
        points:  (n_samples, 3)
        normals: (n_samples, 3)  — face normals (smooth if vertex normals available)
    """
    points, normals, _, _, _ = _sample_mesh_surface_with_barycentrics(
        vertices, faces, n_samples, seed=seed
    )
    return points, normals


def mesh_to_gaussian_ply(vertices: np.ndarray, faces: np.ndarray,
                         vertex_colors: Optional[np.ndarray] = None,
                         n_samples: int = 100_000,
                         base_opacity: float = 0.8,
                         base_scale: float = 0.005,
                         seed: int = 42) -> Dict[str, np.ndarray]:
    """Convert a textured mesh to a 2DGS-compatible Gaussian parameter set.

    Each surface sample becomes one 2D Gaussian surfel:
      - centre = sample point
      - normal = surfel normal (disk orientation)
      - scale  = small in normal direction, larger tangentially → flat disk
      - colour = sampled from mesh texture / vertex colours
      - opacity = uniform base value

    Returns a dict of numpy arrays suitable for write_gaussian_ply().
    """
    points, normals, tri_idx, r1, r2 = _sample_mesh_surface_with_barycentrics(
        vertices, faces, n_samples, seed=seed
    )
    N = len(points)

    # Colours: use vertex colours averaged per face, or default grey
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        v0_c = vertex_colors[faces[:, 0]]
        v1_c = vertex_colors[faces[:, 1]]
        v2_c = vertex_colors[faces[:, 2]]
        colors = (1 - r1 - r2)[:, None] * v0_c[tri_idx] + \
                 r1[:, None] * v1_c[tri_idx] + \
                 r2[:, None] * v2_c[tri_idx]
    else:
        colors = np.full((N, 3), 0.5, dtype=np.float32)

    # Convert from [0,1] RGB to SH DC (the 0th-band coefficient)
    f_dc = (colors - 0.5) / 0.28209479177387814  # C0 = 1/(2*sqrt(pi))

    # Opacity: inverse-sigmoid so that sigmoid(opacity_raw) ≈ base_opacity
    opacity_raw = np.full(N, _inv_sigmoid(base_opacity), dtype=np.float32)

    # Scale: flat disk — tiny along normal, larger tangentially
    log_scale_normal = np.full(N, np.log(base_scale * 0.1), dtype=np.float32)   # along normal (thin)
    log_scale_tangent1 = np.full(N, np.log(base_scale), dtype=np.float32)       # tangent 1
    log_scale_tangent2 = np.full(N, np.log(base_scale), dtype=np.float32)       # tangent 2

    # Rotation: quaternion that rotates world-z to the surface normal
    quats = _normals_to_quaternions(normals)

    return {
        "xyz": points,
        "f_dc": f_dc.astype(np.float32),
        "opacity": opacity_raw,
        "scale_names": ["scale_0", "scale_1", "scale_2"],
        "scales": np.stack([log_scale_tangent1, log_scale_tangent2, log_scale_normal], axis=-1),
        "rot_names": ["rot_0", "rot_1", "rot_2", "rot_3"],
        "rotations": quats.astype(np.float32),
    }


# ======================================================================
# 2.  Read / write 2DGS PLY format
# ======================================================================

# Fields written by 2d-gaussian-splatting's PointCloud.write_ply
GS_PROPERTY_ORDER = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]
# Optional: f_rest_0 ... f_rest_44  (for SH degree 3: (3+1)^2 - 1 = 15 bands * 3 channels = 45 values)


def read_gaussian_ply(path: Path) -> Dict[str, np.ndarray]:
    """Read a 2DGS/3DGS PLY file and return per-property numpy arrays.

    Returns keys: xyz (N,3), f_dc (N,3), opacity (N,), scales (N,3),
    rotations (N,4), and optionally f_rest (N, M, 3) if higher-order SH present.
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        vertex_count = 0
        properties: List[str] = []
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])

        fmt = "<" + "f" * len(properties)
        stride = struct.calcsize(fmt)
        body = f.read(vertex_count * stride)

    data = np.frombuffer(body, dtype=np.float32).reshape(vertex_count, len(properties))

    result: Dict[str, np.ndarray] = {}

    # Positions
    xi = _indices(properties, ["x", "y", "z"])
    result["xyz"] = data[:, xi]

    # SH DC
    dci = _indices(properties, ["f_dc_0", "f_dc_1", "f_dc_2"])
    result["f_dc"] = data[:, dci]

    # Opacity (raw logit)
    if "opacity" in properties:
        oi = properties.index("opacity")
        result["opacity"] = data[:, oi]

    # Scales (log-scale)
    scale_names = [p for p in properties if p.startswith("scale_")]
    if len(scale_names) >= 3:
        si = _indices(properties, ["scale_0", "scale_1", "scale_2"])
        result["scales"] = data[:, si]
    elif len(scale_names) == 2:
        si = _indices(properties, ["scale_0", "scale_1"])
        thin_scale = np.full((vertex_count, 1), np.log(0.001), dtype=np.float32)
        result["scales"] = np.concatenate([data[:, si], thin_scale], axis=1)
    else:
        raise ValueError(f"Unsupported Gaussian PLY scale fields in {path}: {scale_names}")

    # Rotations (quaternion)
    ri = _indices(properties, ["rot_0", "rot_1", "rot_2", "rot_3"])
    result["rotations"] = data[:, ri]

    # Higher-order SH
    rest_names = [p for p in properties if p.startswith("f_rest_")]
    if rest_names:
        rest_names = sorted(rest_names, key=lambda name: int(name.split("_")[-1]))
        rest_idx = [properties.index(n) for n in rest_names]
        f_rest = data[:, rest_idx]
        n_rest = len(rest_names)
        # Official 2DGS stores SH residuals as channel-major
        # (RGB, coefficients) and loads them as (coefficients, RGB).
        result["f_rest"] = f_rest.reshape(vertex_count, 3, n_rest // 3).transpose(0, 2, 1)

    result["property_names"] = np.array(properties)
    return result


def write_gaussian_ply(params: Dict[str, np.ndarray], path: Path,
                       extra_properties: Optional[List[str]] = None,
                       extra_data: Optional[np.ndarray] = None) -> Path:
    """Write a 2DGS-compatible PLY file from a parameter dict.

    Args:
        params: dict with at minimum 'xyz' (N,3), 'f_dc' (N,3),
                'opacity' (N,), 'scales' (N,3), 'rotations' (N,4).
        path: output PLY path.
        extra_properties: additional property names (e.g. ['f_rest_0', ...]).
        extra_data: (N, K) array for the extra properties.
    """
    N = len(params["xyz"])
    props = list(GS_PROPERTY_ORDER)
    columns = [
        params["xyz"][:, 0], params["xyz"][:, 1], params["xyz"][:, 2],
        params["f_dc"][:, 0], params["f_dc"][:, 1], params["f_dc"][:, 2],
        params["opacity"],
        params["scales"][:, 0], params["scales"][:, 1], params["scales"][:, 2],
        params["rotations"][:, 0], params["rotations"][:, 1],
        params["rotations"][:, 2], params["rotations"][:, 3],
    ]
    if extra_properties and extra_data is not None:
        props.extend(extra_properties)
        for k in range(extra_data.shape[1]):
            columns.append(extra_data[:, k])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for prop in props:
            f.write(f"property float {prop}\n".encode())
        f.write(b"end_header\n")
        for i in range(N):
            for col in columns:
                f.write(struct.pack("<f", float(col[i])))
    return path


def merge_gaussian_plys(ply_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate multiple Gaussian parameter dicts into one.

    If f_rest is present in some but not all PLYs, missing ones get zeros.
    """
    xyz = np.concatenate([p["xyz"] for p in ply_list], axis=0)
    f_dc = np.concatenate([p["f_dc"] for p in ply_list], axis=0)
    opacity = np.concatenate([p["opacity"] for p in ply_list], axis=0)
    scales = np.concatenate([p["scales"] for p in ply_list], axis=0)
    rotations = np.concatenate([p["rotations"] for p in ply_list], axis=0)

    result = {"xyz": xyz, "f_dc": f_dc, "opacity": opacity,
              "scales": scales, "rotations": rotations}

    # Handle f_rest
    has_rest = [p for p in ply_list if "f_rest" in p and p["f_rest"] is not None]
    if has_rest:
        n_rest = has_rest[0]["f_rest"].shape[1]
        rest_parts = []
        for p in ply_list:
            if "f_rest" in p and p["f_rest"] is not None:
                rest_parts.append(p["f_rest"])
            else:
                rest_parts.append(np.zeros((len(p["xyz"]), n_rest, 3), dtype=np.float32))
        result["f_rest"] = np.concatenate(rest_parts, axis=0)

    return result


# ======================================================================
# 3.  Simple xyz+rgb PLY I/O (for Colmap, Blender interchange)
# ======================================================================

def read_simple_ply(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a simple PLY with x y z [nx ny nz] [red green blue].

    Returns (xyz, rgb_or_None).
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break

        vertex_count = 0
        prop_specs: List[Tuple[str, str]] = []  # (name, type)
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                prop_specs.append((parts[-1], parts[1]))  # (name, type)

        # Read raw bytes
        fmt_parts = []
        for _, ptype in prop_specs:
            if ptype == "float":
                fmt_parts.append("f")
            elif ptype in ("uchar", "uint8"):
                fmt_parts.append("B")
            else:
                fmt_parts.append("f")  # fallback
        fmt = "<" + "".join(fmt_parts)
        stride = struct.calcsize(fmt)
        body = f.read(vertex_count * stride)

    # Parse
    names = [n for n, _ in prop_specs]
    records = []
    for i in range(vertex_count):
        values = list(struct.unpack_from(fmt, body, i * stride))
        records.append(values)

    data = np.array(records, dtype=np.float32)

    xi = _indices(names, ["x", "y", "z"])
    xyz = data[:, xi]

    rgb = None
    rgb_names = ["red", "green", "blue"]
    if all(n in names for n in rgb_names):
        ri = _indices(names, ["red", "green", "blue"])
        rgb = data[:, ri] / 255.0

    return xyz, rgb


def write_simple_ply(xyz: np.ndarray, path: Path,
                     rgb: Optional[np.ndarray] = None) -> Path:
    """Write a simple PLY with xyz positions and optional RGB colours."""
    N = len(xyz)
    has_rgb = rgb is not None and len(rgb) == N

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        if has_rgb:
            f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        for i in range(N):
            f.write(struct.pack("<3f", *xyz[i]))
            if has_rgb:
                col = (np.clip(rgb[i], 0, 1) * 255).astype(np.uint8)
                f.write(struct.pack("<3B", *col))
    return path


# ======================================================================
# 4.  OBJ mesh I/O
# ======================================================================

def read_obj(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read OBJ geometry and return (vertices, triangular_faces, colors).

    The reader supports both common colour encodings produced by the Task 1
    tools: per-vertex RGB values on ``v`` lines and MTL/UV texture maps.
    Textured meshes are expanded by (vertex, uv) corner so every returned
    vertex has a single RGB value. This is intentionally simple but keeps
    mesh-to-Gaussian conversion from dropping Magic123 textures.
    """
    verts: List[List[float]] = []
    vertex_colors: List[List[float]] = []
    uvs: List[List[float]] = []
    raw_faces: List[List[Tuple[int, Optional[int]]]] = []
    mtllibs: List[str] = []

    with open(path) as f:
        for line in f:
            if line.startswith("mtllib "):
                mtllibs.extend(line.split()[1:])
            elif line.startswith("v "):
                parts = line.split()
                verts.append([float(x) for x in parts[1:4]])
                if len(parts) >= 7:
                    vertex_colors.append([float(x) for x in parts[4:7]])
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                corners: List[Tuple[int, Optional[int]]] = []
                for token in line.split()[1:]:
                    chunks = token.split("/")
                    vi = _obj_index(chunks[0], len(verts))
                    ti = None
                    if len(chunks) > 1 and chunks[1]:
                        ti = _obj_index(chunks[1], len(uvs))
                    corners.append((vi, ti))
                raw_faces.extend(_triangulate_face(corners))

    v_arr = np.array(verts, dtype=np.float32)
    if not raw_faces:
        return v_arr, np.empty((0, 3), dtype=np.int32), None

    texture = _load_obj_diffuse_texture(path, mtllibs)
    if texture is not None and uvs and all(ti is not None for face in raw_faces for _, ti in face):
        expanded_verts: List[np.ndarray] = []
        expanded_colors: List[np.ndarray] = []
        faces: List[List[int]] = []
        corner_to_index: Dict[Tuple[int, int], int] = {}
        uv_arr = np.array(uvs, dtype=np.float32)

        for face in raw_faces:
            tri: List[int] = []
            for vi, ti_opt in face:
                ti = int(ti_opt)  # all texture indices were checked above
                key = (vi, ti)
                if key not in corner_to_index:
                    corner_to_index[key] = len(expanded_verts)
                    expanded_verts.append(v_arr[vi])
                    expanded_colors.append(_sample_texture(texture, uv_arr[ti]))
                tri.append(corner_to_index[key])
            faces.append(tri)
        return (
            np.array(expanded_verts, dtype=np.float32),
            np.array(faces, dtype=np.int32),
            np.array(expanded_colors, dtype=np.float32),
        )

    faces_arr = np.array([[vi for vi, _ in face] for face in raw_faces], dtype=np.int32)
    colors_arr = None
    if vertex_colors and len(vertex_colors) == len(verts):
        colors_arr = np.array(vertex_colors, dtype=np.float32)
    return v_arr, faces_arr, colors_arr


def _obj_index(value: str, count: int) -> int:
    idx = int(value)
    return idx - 1 if idx > 0 else count + idx


def _triangulate_face(corners: List[Tuple[int, Optional[int]]]) -> List[List[Tuple[int, Optional[int]]]]:
    if len(corners) < 3:
        return []
    if len(corners) == 3:
        return [corners]
    return [[corners[0], corners[i], corners[i + 1]] for i in range(1, len(corners) - 1)]


def _load_obj_diffuse_texture(obj_path: Path, mtllibs: List[str]) -> Optional[np.ndarray]:
    for name in mtllibs:
        mtl_path = obj_path.parent / name
        if not mtl_path.exists():
            continue
        for line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("map_Kd "):
                continue
            tex_name = line.split(None, 1)[1].strip().strip('"')
            tex_path = obj_path.parent / tex_name
            if not tex_path.exists():
                continue
            from PIL import Image

            img = Image.open(tex_path).convert("RGB")
            return np.asarray(img, dtype=np.float32) / 255.0
    return None


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = texture.shape[:2]
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    x = int(round(u * (w - 1)))
    y = int(round((1.0 - v) * (h - 1)))
    return texture[y, x]


# ======================================================================
# Internal helpers
# ======================================================================

def _inv_sigmoid(x: float) -> float:
    import math
    return math.log(x / (1 - x))


def _indices(property_list: List[str], names: List[str]) -> List[int]:
    return [property_list.index(n) for n in names]


def _normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    """Find a quaternion rotation that maps world Z=[0,0,1] to each normal.

    Returns (N, 4) quaternions (w, x, y, z).
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    N = len(normals)
    quats = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        n = normals[i]
        n = n / (np.linalg.norm(n) + 1e-8)
        # Axis-angle: cross(z, n) is rotation axis, dot(z, n) is cos(angle)
        v = np.cross(z_axis, n)
        c = np.dot(z_axis, n)
        # Half-angle quaternion
        if c < -0.9999:  # antiparallel: 180° around arbitrary perpendicular axis
            quats[i] = [0.0, 1.0, 0.0, 0.0]
        else:
            s = np.sqrt((1 + c) * 2)
            quats[i] = [s * 0.5, v[0] / s, v[1] / s, v[2] / s]

    return quats
