"""Tests for conversion utilities."""

import numpy as np

from hw3cv.conversion import (
    mesh_to_gaussian_ply,
    merge_gaussian_plys,
    read_gaussian_ply,
    read_obj,
    read_simple_ply,
    sample_mesh_surface,
    write_gaussian_ply,
    write_simple_ply,
)


class TestMeshToGaussian:
    def test_sample_mesh_surface(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        pts, normals = sample_mesh_surface(verts, faces, n_samples=50, seed=0)
        assert pts.shape == (50, 3)
        assert normals.shape == (50, 3)
        # Normal should be [0,0,1] or [0,0,-1]
        assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1e-4)

    def test_mesh_to_gaussian_ply_output_shapes(self):
        verts = np.random.randn(100, 3).astype(np.float32) * 0.5
        faces = np.random.randint(0, 100, (80, 3)).astype(np.int32)
        params = mesh_to_gaussian_ply(verts, faces, n_samples=500, seed=0)
        assert params["xyz"].shape == (500, 3)
        assert params["f_dc"].shape == (500, 3)
        assert params["opacity"].shape == (500,)
        assert params["scales"].shape == (500, 3)
        assert params["rotations"].shape == (500, 4)


class TestGaussianPlyIO:
    def test_write_read_roundtrip(self, tmp_path):
        N = 20
        params = {
            "xyz": np.random.randn(N, 3).astype(np.float32) * 0.1,
            "f_dc": np.random.randn(N, 3).astype(np.float32) * 0.1,
            "opacity": np.random.randn(N).astype(np.float32),
            "scales": np.random.randn(N, 3).astype(np.float32),
            "rotations": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1)),
        }
        out = tmp_path / "test.ply"
        write_gaussian_ply(params, out)
        loaded = read_gaussian_ply(out)
        np.testing.assert_allclose(params["xyz"], loaded["xyz"], atol=1e-5)

    def test_merge_preserves_count(self):
        N1, N2 = 10, 15
        p1 = {
            "xyz": np.zeros((N1, 3), dtype=np.float32),
            "f_dc": np.zeros((N1, 3), dtype=np.float32),
            "opacity": np.zeros(N1, dtype=np.float32),
            "scales": np.zeros((N1, 3), dtype=np.float32),
            "rotations": np.zeros((N1, 4), dtype=np.float32),
        }
        p2 = {
            "xyz": np.ones((N2, 3), dtype=np.float32),
            "f_dc": np.ones((N2, 3), dtype=np.float32),
            "opacity": np.ones(N2, dtype=np.float32),
            "scales": np.ones((N2, 3), dtype=np.float32),
            "rotations": np.ones((N2, 4), dtype=np.float32),
        }
        merged = merge_gaussian_plys([p1, p2])
        assert len(merged["xyz"]) == N1 + N2


class TestSimplePlyIO:
    def test_write_read_roundtrip(self, tmp_path):
        xyz = np.random.randn(50, 3).astype(np.float32)
        rgb = np.random.rand(50, 3).astype(np.float32)
        out = tmp_path / "test.ply"
        write_simple_ply(xyz, out, rgb=rgb)
        loaded_xyz, loaded_rgb = read_simple_ply(out)
        assert len(loaded_xyz) == 50
        assert loaded_rgb is not None


class TestObjIO:
    def test_read_obj(self, tmp_path):
        obj = tmp_path / "test.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        verts, faces, colors = read_obj(obj)
        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)
