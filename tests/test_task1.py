import json
from dataclasses import replace
from pathlib import Path

import pytest

from hw3cv.config import OutputsConfig, load_config
from hw3cv.data import expected_task1_assets
from hw3cv.task1 import build_magic123_command, build_threestudio_command, build_train_2dgs_command, make_scene_manifest


def test_build_train_2dgs_background_command_uses_configured_paths():
    config = load_config()

    command = build_train_2dgs_command(config, "background")

    assert command[:4] == ["python", "train.py", "-s", str(Path.cwd() / "data" / "background" / "kitchen")]
    assert "-m" in command
    assert str(Path.cwd() / "outputs" / "task1" / "final" / "environment" / "kitchen_2dgs") in command


def test_build_threestudio_command_selects_expected_framework():
    config = load_config()

    command = build_threestudio_command(config)

    assert command[:3] == ["python", "launch.py", "--config"]
    assert "system.prompt_processor.prompt=a delicious hamburger" in command


def test_build_magic123_command_uses_configured_image():
    config = load_config()

    command = build_magic123_command(config)

    assert command[:2] == ["python", "main.py"]
    assert str(Path.cwd() / "data" / "task1" / "object_c" / "rgba.png") in command


def test_make_scene_manifest_requires_assets(tmp_path):
    base = load_config()
    config = replace(
        base,
        outputs=OutputsConfig(
            task1=tmp_path / "missing-task1",
            task2=tmp_path / "missing-task2",
        ),
    )

    with pytest.raises(FileNotFoundError, match="object_a"):
        make_scene_manifest(config, tmp_path / "scene.json")


def test_make_scene_manifest_writes_valid_json_when_assets_exist(tmp_path):
    base = load_config()
    output_root = tmp_path / "outputs" / "task1"
    config = replace(
        base,
        outputs=OutputsConfig(task1=output_root, task2=tmp_path / "outputs" / "task2"),
        task1=replace(
            base.task1,
            scene=replace(
                base.task1.scene,
                render_output=output_root / "final" / "scene" / "fused_scene_360.mp4",
            ),
        ),
    )
    assets = expected_task1_assets(output_root)
    for asset in assets.values():
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text("placeholder", encoding="utf-8")

    manifest_path = tmp_path / "scene.json"
    make_scene_manifest(config, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["background"]["path"].endswith("kitchen_2dgs/point_cloud/iteration_7000/point_cloud.ply")
    assert [item["name"] for item in manifest["objects"]] == ["object_a", "object_b", "object_c"]
    assert manifest["render"]["output"].endswith("outputs/task1/final/scene/fused_scene_360.mp4")
    assert manifest["render"]["frames"] == 360
