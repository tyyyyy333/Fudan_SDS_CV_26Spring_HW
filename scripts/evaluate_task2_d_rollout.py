#!/usr/bin/env python
"""CALVIN-D simulator rollout entry for LeRobot ACT checkpoints.

This is the closed-loop zero-shot evaluation entry required by Task 2. It needs
the official CALVIN simulator package (`calvin_env`) and a CALVIN dataset root
with `validation/` assets. The local workspace currently only has the
LeRobot-format HF shards, which are enough for offline teacher-forced loss but
not enough for simulator rollout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "external" / "lerobot" / "src"
CALVIN_ROOT = ROOT / "external" / "calvin"
CALVIN_MODELS = CALVIN_ROOT / "calvin_models"
CALVIN_ENV = CALVIN_ROOT / "calvin_env"

for path in [LEROBOT_SRC, CALVIN_ROOT, CALVIN_MODELS, CALVIN_ENV]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/task2/runs/task2_fair_abc_10k_cosine_b256_c10_s1000"
    / "abc_to_d_train/checkpoints/010000/pretrained_model"
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dependency_status(dataset_path: Path, checkpoint: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add("checkpoint", checkpoint.exists(), _rel(checkpoint))
    add("calvin_env_source", any(CALVIN_ENV.iterdir()) if CALVIN_ENV.exists() else False, _rel(CALVIN_ENV))
    add("dataset_path", dataset_path.exists(), _rel(dataset_path))
    add("dataset_validation", (dataset_path / "validation").exists(), _rel(dataset_path / "validation"))

    for module_name in [
        "hydra",
        "omegaconf",
        "calvin_env.envs.play_table_env",
        "calvin_agent.evaluation.evaluate_policy",
        "lerobot.policies",
    ]:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            add(f"import:{module_name}", False, f"{type(exc).__name__}: {exc}")
        else:
            add(f"import:{module_name}", True, "ok")

    missing = [item for item in checks if not item["ok"]]
    return {
        "status": "ready" if not missing else "missing_requirements",
        "checks": checks,
        "missing": missing,
        "note": (
            "Closed-loop rollout requires official CALVIN simulator assets. "
            "LeRobot HF shards cannot replace dataset_path/validation."
        ),
    }


def _to_chw_float(image: Any) -> torch.Tensor:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dims, got shape {arr.shape}")
    if arr.shape[0] in (1, 3):
        tensor = torch.as_tensor(arr)
    else:
        tensor = torch.as_tensor(arr).permute(2, 0, 1)
    tensor = tensor.float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tuple(tensor.shape[-2:]) != (256, 256):
        tensor = F.interpolate(tensor.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]
    return tensor.clamp(0.0, 1.0)


def _get_rgb(obs: dict[str, Any], key: str) -> Any:
    rgb_obs = obs.get("rgb_obs", {})
    if key in rgb_obs:
        return rgb_obs[key]
    if key in obs:
        return obs[key]
    raise KeyError(f"CALVIN observation does not contain rgb key {key!r}")


class LeRobotACTCalvinModel:
    """Adapter from CALVIN evaluation API to a saved LeRobot ACT policy."""

    def __init__(self, checkpoint: Path, device: str):
        from lerobot.policies import get_policy_class
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

        policy_cls = get_policy_class("act")
        self.policy = policy_cls.from_pretrained(checkpoint)
        self.policy.to(device)
        self.policy.eval()
        self.device = device

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            device_processor={"device": device},
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            device_processor={"device": "cpu"},
        )

    def reset(self) -> None:
        self.policy.reset()

    @torch.inference_mode()
    def step(self, obs: dict[str, Any], goal: str) -> np.ndarray:  # noqa: ARG002
        state = torch.as_tensor(obs["robot_obs"], dtype=torch.float32)
        batch = {
            "observation.state": state,
            "observation.images.image": _to_chw_float(_get_rgb(obs, "rgb_static")),
            "observation.images.wrist_image": _to_chw_float(_get_rgb(obs, "rgb_gripper")),
        }
        batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.squeeze(0).cpu().numpy()


def summarize_rollout_results(results: list[int]) -> dict[str, Any]:
    counts = {str(i): int(sum(1 for result in results if result == i)) for i in range(6)}
    success_len = {}
    for i in range(1, 6):
        success_len[f"success_len_{i}"] = float(sum(result >= i for result in results) / len(results))
    return {
        "num_sequences": len(results),
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "counts": counts,
        **success_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=ROOT / "data/calvin/task_ABC_D")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/task2/d_rollout")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = dependency_status(args.dataset_path, args.checkpoint)
    if args.check_only or status["status"] != "ready":
        out = {
            **status,
            "dataset_path": _rel(args.dataset_path),
            "checkpoint": _rel(args.checkpoint),
            "metric_type": "calvin_d_simulator_rollout_readiness",
        }
        (args.output_dir / "readiness.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2))
        if status["status"] != "ready" and not args.check_only:
            raise SystemExit(2)
        return

    from calvin_agent.evaluation.evaluate_policy import evaluate_policy, make_env

    model = LeRobotACTCalvinModel(args.checkpoint, args.device)
    env = make_env(args.dataset_path)
    results = evaluate_policy(model, env, epoch="lerobot_act", eval_log_dir=args.output_dir, debug=args.debug)
    summary = {
        "metric_type": "calvin_d_simulator_rollout",
        "dataset_path": _rel(args.dataset_path),
        "checkpoint": _rel(args.checkpoint),
        "results": summarize_rollout_results(results),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
