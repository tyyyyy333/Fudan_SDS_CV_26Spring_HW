"""Tests for scene fusion module."""

import numpy as np

from hw3cv.conversion import write_gaussian_ply
from hw3cv.scene_fusion import FusedScene, SceneAsset, build_fused_scene


class TestSceneAsset:
    def test_load_2dgs_as_gaussian(self, tmp_path):
        N = 10
        params = {
            "xyz": np.random.randn(N, 3).astype(np.float32) * 0.1,
            "f_dc": np.random.randn(N, 3).astype(np.float32) * 0.1,
            "opacity": np.random.randn(N).astype(np.float32),
            "scales": np.random.randn(N, 3).astype(np.float32),
            "rotations": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1)),
        }
        ply = tmp_path / "test.ply"
        write_gaussian_ply(params, ply)

        asset = SceneAsset("test", ply, "2dgs", location=(1, 0, 0), scale=2.0)
        gs = asset.load_as_gaussian()
        # Position should be scaled and translated
        np.testing.assert_allclose(gs["xyz"], params["xyz"] * 2.0 + [1, 0, 0], atol=1e-5)


class TestFusedScene:
    def test_export_merges_background_and_objects(self, tmp_path):
        N = 5
        p1 = {
            "xyz": np.zeros((N, 3), dtype=np.float32),
            "f_dc": np.zeros((N, 3), dtype=np.float32),
            "opacity": np.zeros(N, dtype=np.float32),
            "scales": np.zeros((N, 3), dtype=np.float32),
            "rotations": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1)),
        }
        bg_ply = tmp_path / "bg.ply"
        write_gaussian_ply(p1, bg_ply)

        p2 = {
            "xyz": np.ones((N, 3), dtype=np.float32),
            "f_dc": np.ones((N, 3), dtype=np.float32),
            "opacity": np.ones(N, dtype=np.float32),
            "scales": np.ones((N, 3), dtype=np.float32),
            "rotations": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1)),
        }
        obj_ply = tmp_path / "obj.ply"
        write_gaussian_ply(p2, obj_ply)

        scene = FusedScene()
        scene.add_background(bg_ply)
        scene.add_object("test", obj_ply, location=(0, 0, 0), scale=1.0)

        out = tmp_path / "fused.ply"
        scene.export(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_manifest(self, tmp_path):
        N = 3
        p = {
            "xyz": np.zeros((N, 3), dtype=np.float32),
            "f_dc": np.zeros((N, 3), dtype=np.float32),
            "opacity": np.zeros(N, dtype=np.float32),
            "scales": np.zeros((N, 3), dtype=np.float32),
            "rotations": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1)),
        }
        ply = tmp_path / "test.ply"
        write_gaussian_ply(p, ply)

        scene = FusedScene()
        scene.add_object("obj", ply, location=(0.5, 0, 0), scale=2.0)
        mf = tmp_path / "manifest.json"
        scene.export_manifest(mf)
        assert mf.exists()


class TestBuildFusedScene:
    def test_from_specs(self, tmp_path):
        N = 5
        p = {
            "xyz": np.zeros((N, 3), dtype=np.float32),
            "f_dc": np.zeros((N, 3), dtype=np.float32),
            "opacity": np.zeros(N, dtype=np.float32),
            "scales": np.zeros((N, 3), dtype=np.float32),
            "rotations": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1)),
        }
        ply = tmp_path / "test.ply"
        write_gaussian_ply(p, ply)

        scene = build_fused_scene(
            background_path=None,
            object_specs=[{"name": "obj", "path": ply, "kind": "2dgs", "scale": 0.5}],
        )
        assert len(scene.objects) == 1
        assert scene.objects[0].scale == 0.5
