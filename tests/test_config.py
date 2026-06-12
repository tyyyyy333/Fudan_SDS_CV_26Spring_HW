from pathlib import Path

from hw3cv.config import load_config


def test_load_config_resolves_project_relative_paths():
    config = load_config()

    assert config.project_root == Path.cwd()
    assert config.external_dir == Path.cwd() / "external"
    assert config.outputs.task1 == Path.cwd() / "outputs" / "task1"
    assert config.task1.objects["object_a"].source == Path.cwd() / "data" / "task1" / "object_a" / "current"
    assert config.task2.environments["D"] == Path.cwd() / "data" / "calvin" / "D"


def test_load_config_accepts_custom_json(tmp_path):
    config_path = tmp_path / "custom.json"
    config_path.write_text(
        """
        {
          "external_dir": "vendor",
          "outputs": {"task1": "runs/one", "task2": "runs/two"},
          "task1": {
            "objects": {
              "object_a": {"source": "captures/a", "output": "assets/a"},
              "object_b": {"prompt": "a red toy robot", "output": "assets/b"},
              "object_c": {"image": "images/c.png", "output": "assets/c"}
            },
            "background": {"source": "scenes/garden", "output": "assets/bg"},
            "scene": {"manifest": "scene/manifest.json", "render_output": "scene/out.mp4"}
          },
          "task2": {
            "calvin_root": "calvin",
            "environments": {"A": "calvin/A", "B": "calvin/B", "C": "calvin/C", "D": "calvin/D"},
            "experiments": {
              "single_b": {"train_envs": ["B"], "eval_env": "D", "output": "act/single_b"},
              "abc_to_d": {"train_envs": ["A", "B", "C"], "eval_env": "D", "output": "act/abc_to_d"}
            },
            "act": {"batch_size": 8, "learning_rate": 0.0001, "epochs": 2, "chunk_size": 16}
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_root == tmp_path
    assert config.external_dir == tmp_path / "vendor"
    assert config.task1.objects["object_b"].prompt == "a red toy robot"
    assert config.task2.experiments["abc_to_d"].train_envs == ["A", "B", "C"]
