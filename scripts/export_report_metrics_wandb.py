#!/usr/bin/env python3
"""Export retained evaluation metrics to an offline W&B report run.

This is deliberately labeled as checkpoint re-evaluation. It does not rewrite
historical ACT training logs or imply that validation ran during training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(ROOT / "outputs/task2/wandb"))
    eval_metrics = json.loads(
        (
            ROOT
            / "outputs/task2/experiments/eval_d_offline_scheduler/metrics.json"
        ).read_text()
    )
    full_metrics = json.loads(
        (ROOT / "outputs/task2/zero_shot_d_action_error_full/metrics.json").read_text()
    )
    robust = json.loads(
        (ROOT / "outputs/task2/action_chunking_robustness/metrics.json").read_text()
    )

    run = wandb.init(
        project="hw3-task2",
        name="act-checkpoint-reevaluation",
        config={
            "metric_origin": "post-training checkpoint re-evaluation",
            "held_out_environment": "D",
            "training_seed": 1000,
            "chunk_size": 10,
            "samples_full_d": 92274,
        },
    )
    scheduler = eval_metrics["results"]
    for step, key in [(10000, "abc_cosine_10k"), (20000, "abc_cosine_20k"), (30000, "abc_cosine_30k")]:
        values = scheduler[key]
        run.log(
            {
                "checkpoint_step": step,
                "validation_D/total_loss": values["loss"],
                "validation_D/action_l1": values["l1_loss"],
                "validation_D/kld": values["kld_loss"],
            },
            step=step,
        )

    full = full_metrics["results"]
    table = wandb.Table(columns=["model", "training_step", "D_action_l1", "D_total_loss"])
    table.add_data("B-only", 5000, full["single_b"]["l1_loss"], full["single_b"]["loss"])
    table.add_data(
        "A+B+C",
        30000,
        full["abc_cosine_30k"]["l1_loss"],
        full["abc_cosine_30k"]["loss"],
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
