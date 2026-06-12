"""Tests for hw3cv.data module."""

from pathlib import Path

from hw3cv.data import (
    build_scene_manifest,
    check_task1_assets,
    collect_calvin_episodes,
    expected_task1_assets,
    get_calvin_stats,
    validate_calvin_directory,
    validate_capture_directory,
)


class TestCalvin:
    def test_collect_episodes_empty_when_dir_missing(self, tmp_path):
        episodes = collect_calvin_episodes(tmp_path / "no_such_dir", ["A", "B"])
        assert episodes == []

    def test_collect_episodes_finds_npz_files(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "A" / "ep_001.npz").write_text("")
        (tmp_path / "B").mkdir()
        (tmp_path / "B" / "ep_002.npz").write_text("")

        episodes = collect_calvin_episodes(tmp_path, ["A", "B"])

        assert len(episodes) == 2
        assert episodes[0]["environment"] == "A"
        assert episodes[1]["environment"] == "B"

    def test_validate_calvin_directory_reports_all_envs(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "A" / "a.npz").write_text("")

        result = validate_calvin_directory(tmp_path)

        assert set(result.keys()) == {"A", "B", "C", "D"}
        assert len(result["A"]) == 1
        assert result["B"] == []

    def test_get_calvin_stats_computes_sizes(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "A" / "a.npz").write_bytes(b"x" * 1024)

        stats = get_calvin_stats(tmp_path)

        assert stats["A"]["episode_count"] == 1
        assert stats["A"]["total_size_bytes"] == 1024
        assert stats["B"]["episode_count"] == 0


class TestCapture:
    def test_missing_directory(self):
        result = validate_capture_directory(Path("/no/such/dir"))
        assert result["valid"] is False
        assert result["image_count"] == 0

    def test_counts_images(self, tmp_path):
        (tmp_path / "img_01.jpg").write_text("")
        (tmp_path / "img_02.png").write_text("")
        (tmp_path / "notes.txt").write_text("")

        result = validate_capture_directory(tmp_path)

        assert result["valid"] is True
        assert result["image_count"] == 2

    def test_empty_directory_not_valid(self, tmp_path):
        result = validate_capture_directory(tmp_path)

        assert result["valid"] is False
        assert result["image_count"] == 0


class TestAssets:
    def test_check_assets_returns_status_for_all_keys(self):
        status, missing = check_task1_assets(Path("/no/such/outputs"))

        assert set(status.keys()) == {"object_a", "object_b", "object_c", "background"}
        assert len(missing) == 4
        assert all(v is False for v in status.values())

    def test_expected_assets_paths_match_convention(self):
        assets = expected_task1_assets(Path("out/task1"))
        assert str(assets["object_a"]).endswith(
            "final/objects/object_a/model/point_cloud/iteration_30000/point_cloud.ply"
        )
        assert str(assets["background"]).endswith(
            "final/environment/kitchen_2dgs/point_cloud/iteration_7000/point_cloud.ply"
        )
        assert str(assets["object_b"]).endswith(
            "final/objects/object_b/model/textured.obj"
        )
        assert str(assets["object_c"]).endswith(
            "final/objects/object_c/model/textured.obj"
        )


class TestManifest:
    def test_build_scene_manifest_structure(self, tmp_path):
        assets = {k: tmp_path / f"{k}.dummy" for k in ["object_a", "object_b", "object_c", "background"]}
        for p in assets.values():
            p.write_text("")

        manifest = build_scene_manifest(assets, Path("out/video.mp4"))

        assert manifest["background"]["type"] == "2dgs"
        assert len(manifest["objects"]) == 3
        assert manifest["objects"][0]["name"] == "object_a"
        assert manifest["objects"][0]["type"] == "2dgs"
        assert manifest["objects"][1]["type"] == "mesh"
        assert manifest["render"]["frames"] == 360
        assert manifest["render"]["output"] == "out/video.mp4"
