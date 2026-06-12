import json

from hw3cv.config import load_config
from hw3cv.task2 import build_eval_act_command, build_train_act_command, index_calvin_experiment


def _write_episode(path, name):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text("episode", encoding="utf-8")


def test_index_calvin_single_b_uses_b_for_train_and_d_for_eval(tmp_path):
    config = load_config()
    _write_episode(tmp_path / "B", "episode_000001.npz")
    _write_episode(tmp_path / "D", "episode_000101.npz")

    index = index_calvin_experiment(config, "single_b", tmp_path)

    assert index["experiment"] == "single_b"
    assert [item["environment"] for item in index["train_episodes"]] == ["B"]
    assert [item["environment"] for item in index["eval_episodes"]] == ["D"]


def test_index_calvin_abc_to_d_collects_all_training_envs(tmp_path):
    config = load_config()
    for env in ["A", "B", "C", "D"]:
        _write_episode(tmp_path / env, f"episode_{env}.npz")

    index = index_calvin_experiment(config, "abc_to_d", tmp_path)

    assert [item["environment"] for item in index["train_episodes"]] == ["A", "B", "C"]
    assert [item["environment"] for item in index["eval_episodes"]] == ["D"]


def test_build_act_commands_include_index_and_metrics_paths(tmp_path):
    config = load_config()
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"experiment": "abc_to_d"}), encoding="utf-8")

    train_command = build_train_act_command(config, "abc_to_d", index_path)
    eval_command = build_eval_act_command(config, "abc_to_d", index_path)

    assert train_command[:5] == ["conda", "run", "-n", "hw3t2", "lerobot-train"]
    assert ["--policy.type", "act"] == train_command[5:7]
    assert "--dataset.root" in train_command
    assert ["--batch_size", "512"] == train_command[
        train_command.index("--batch_size") : train_command.index("--batch_size") + 2
    ]
    assert eval_command[:3] == ["python", "-m", "lerobot.scripts.eval"]
    assert any(part.endswith("abc_to_d_train/eval_metrics.json") for part in eval_command)
