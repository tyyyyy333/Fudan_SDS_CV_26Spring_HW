#!/usr/bin/env python3
"""Export retained evaluation metrics to an offline W&B report run.

This is deliberately labeled as checkpoint re-evaluation. It does not rewrite
historical ACT training logs or imply that validation ran during training.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOGS = {
    "B-only": ROOT / "logs/task2_fair_b_10k_cosine.log",
    "A+B+C": ROOT / "logs/task2_fair_abc_10k_cosine.log",
}


def training_points(path: Path):
    pattern = re.compile(
        r"step:\S+ .*?loss:([0-9.]+).*?grdn:([0-9.]+).*?lr:([0-9.eE+-]+)"
    )
    record_index = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            record_index += 1
            yield (
                record_index * 20,
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(ROOT / "outputs/task2/wandb"))
    full_metrics = json.loads(
        (ROOT / "outputs/task2/zero_shot_d_action_error_full/metrics.json").read_text()
    )
    robust = json.loads(
        (ROOT / "outputs/task2/action_chunking_robustness/metrics.json").read_text()
    )

    for model, log_path in TRAIN_LOGS.items():
        run = wandb.init(
            project="hw3-task2",
            group="fair-act-10k",
            name=f"{model.lower().replace('+', '').replace('-', '_')}-10k",
            reinit=True,
            config={
                "model": model,
                "training_steps": 10000,
                "batch_size": 256,
                "learning_rate": 1e-4,
                "scheduler": "500-step warmup + cosine to 1e-5",
                "seed": 1000,
                "chunk_size": 10,
            },
        )
        for step, loss, grad_norm, lr in training_points(log_path):
            run.log(
                {
                    "train/loss": loss,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": lr,
                },
                step=step,
            )
        run.finish()

    full = full_metrics["results"]
    run = wandb.init(
        project="hw3-task2",
        group="fair-act-10k",
        name="fair-10k-d-evaluation",
        reinit=True,
        config={
            "metric_origin": "post-training checkpoint re-evaluation",
            "held_out_environment": "D",
            "training_seed": 1000,
            "chunk_size": 10,
            "samples_full_d": 92274,
        },
    )
    table = wandb.Table(columns=["model", "training_step", "D_action_l1", "D_total_loss"])
    table.add_data(
        "B-only",
        10000,
        full["b_only_fair_10k"]["l1_loss"],
        full["b_only_fair_10k"]["loss"],
    )
    table.add_data(
        "A+B+C",
        10000,
        full["abc_fair_10k"]["l1_loss"],
        full["abc_fair_10k"]["loss"],
    )
    run.log({"full_D_zero_shot": table})

    clean = robust["results"]
    horizon = wandb.Table(columns=["model", "chunk_position", "action_l1"])
    for model in ["B-only", "A+B+C"]:
        for index, value in enumerate(clean[model]["D"]["clean"]["horizon_l1"], start=1):
            horizon.add_data(model, index, value)
    run.log({"clean_D_chunk_horizon": horizon})
    run.finish()


if __name__ == "__main__":
    main()
