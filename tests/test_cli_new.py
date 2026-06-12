"""Tests for CLI subcommands."""

import json
import subprocess
import sys


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hw3cv.cli", *args],
        check=check,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": "src"},
    )


class TestTask1CheckAssets:
    def test_reports_all_asset_names(self):
        result = _run("task1", "check-assets", check=False)
        for name in ["object_a", "object_b", "object_c", "background"]:
            assert name in result.stdout
        assert result.returncode in {0, 1}


class TestTask1CheckCapture:
    def test_reports_invalid_for_missing_dir(self):
        result = _run("task1", "check-capture", "--target", "object_a", check=False)
        data = json.loads(result.stdout)
        assert data["valid"] is False


class TestTask2CheckCalvin:
    def test_reports_all_envs(self):
        result = _run("task2", "check-calvin")
        for env in ["A", "B", "C", "D"]:
            assert env in result.stdout


class TestTask2CalvinStats:
    def test_outputs_json(self):
        result = _run("task2", "calvin-stats")
        stats = json.loads(result.stdout)
        assert set(stats.keys()) == {"A", "B", "C", "D"}


class TestTask2Compare:
    def test_prints_table(self):
        result = _run("task2", "compare")
        assert "single_b" in result.stdout
        assert "abc_to_d" in result.stdout


class TestTask1RenderScene:
    def test_prints_or_runs_render_command(self):
        result = _run("task1", "render-scene", "--dry-run")
        assert "blender" in result.stdout.lower()


class TestTask1Magic123:
    def test_magic123_dry_run(self):
        result = _run("task1", "magic123", "--dry-run")
        assert "python main.py" in result.stdout


class TestTask1Threestudio:
    def test_threestudio_dry_run(self):
        result = _run("task1", "threestudio", "--dry-run")
        assert "python launch.py" in result.stdout


class TestTask2WandbConfig:
    def test_outputs_json(self):
        result = _run("task2", "wandb-config", "--exp", "single_b")
        data = json.loads(result.stdout)
        assert data["project"] == "cv-hw3"
