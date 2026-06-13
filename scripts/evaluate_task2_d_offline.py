#!/usr/bin/env python
"""Offline D-shard evaluation for HW3 Task 2 ACT checkpoints.

This computes teacher-forced ACT losses on the held-out D LeRobot shard. It is
not a CALVIN simulator rollout success metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "external" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import ACTION, POLICY_PREPROCESSOR_DEFAULT_NAME  # noqa: E402


RUNS = {
    "b_only_fair_10k": ROOT
    / "outputs/task2/runs/task2_fair_b_10k_cosine_b256_c10_s1000"
    / "single_b_train/checkpoints/010000/pretrained_model",
    "abc_fair_10k": ROOT
    / "outputs/task2/runs/task2_fair_abc_10k_cosine_b256_c10_s1000"
    / "abc_to_d_train/checkpoints/010000/pretrained_model",
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def evaluate_one(
    name: str,
    checkpoint: Path,
    dataset: LeRobotDataset,
    *,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    device: str,
) -> dict:
    policy_cls = get_policy_class("act")
    policy = policy_cls.from_pretrained(checkpoint)
    policy.to(device)
    # ACT only returns VAE KL terms in training mode. We keep no_grad below, so
    # this is teacher-forced loss evaluation rather than optimization.
    policy.train()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        device_processor={"device": device},
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    total_examples = 0
    total_loss = 0.0
    total_l1 = 0.0
    total_kld = 0.0
    batches = 0

    with torch.inference_mode():
        for batch in dataloader:
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            loss, loss_dict = policy(batch)
            n = int(batch[ACTION].shape[0])
            total_examples += n
            total_loss += float(loss.item()) * n
            total_l1 += float(loss_dict["l1_loss"]) * n
            total_kld += float(loss_dict.get("kld_loss", 0.0)) * n
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break

    return {
        "name": name,
        "checkpoint": display_path(checkpoint),
        "eval_dataset": display_path(Path(dataset.root)),
        "batches": batches,
        "examples": total_examples,
        "loss": total_loss / total_examples,
        "l1_loss": total_l1 / total_examples,
        "kld_loss": total_kld / total_examples,
        "kl_weight": float(policy.config.kl_weight),
        "metric_type": "offline_teacher_forced_d_shard_loss",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d-root",
        type=Path,
        default=ROOT
        / "data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D/calvin_task_ABC_D_lerobot_3_4",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/task2/eval_d_offline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--runs",
        default=",".join(RUNS),
        help=f"Comma-separated run names to evaluate. Available: {','.join(RUNS)}",
    )
    args = parser.parse_args()

    selected_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in RUNS]
    if unknown:
        raise SystemExit(f"Unknown run names: {unknown}. Available: {list(RUNS)}")

    first_cfg = PreTrainedConfig.from_pretrained(next(iter(RUNS.values())))
    d_meta = LeRobotDatasetMetadata(
        "local/calvin_task_ABC_D_lerobot_3_4",
        root=args.d_root,
    )
    delta_timestamps = resolve_delta_timestamps(first_cfg, d_meta)
    dataset = LeRobotDataset(
        "local/calvin_task_ABC_D_lerobot_3_4",
        root=args.d_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
        tolerance_s=0.01,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        name: evaluate_one(
            name,
            RUNS[name],
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            device=args.device,
        )
        for name in selected_names
    }
    summary = {
        "note": "Offline held-out D teacher-forced ACT loss. This is not simulator rollout success rate.",
        "d_root": display_path(args.d_root),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "results": results,
    }

    json_path = args.output_dir / "metrics.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Task 2 Offline D Evaluation",
        "",
        "This is a teacher-forced ACT loss evaluation on the held-out D shard, not simulator rollout success.",
        "",
        "| Experiment | Examples | Total loss | L1 loss | KLD loss | Checkpoint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, result in results.items():
        md.append(
            f"| {name} | {result['examples']} | {result['loss']:.4f} | "
            f"{result['l1_loss']:.4f} | {result['kld_loss']:.4f} | `{result['checkpoint']}` |"
        )
    (args.output_dir / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
