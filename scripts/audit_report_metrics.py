#!/usr/bin/env python3
"""Audit the numeric claims used by the HW3 report against final artifacts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            if line.startswith(b"element vertex "):
                return int(line.split()[2])


def parse_log_timestamp(line: str) -> datetime | None:
    match = re.search(r"\[(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]", line)
    if match:
        return datetime.strptime(match.group(1), "%y/%m/%d %H:%M:%S")
    match = re.search(r"\[(\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]", line)
    return (
        datetime.strptime(f"2026/{match.group(1)}", "%Y/%m/%d %H:%M:%S")
        if match
        else None
    )


def log_span_seconds(path: Path) -> int:
    timestamps = [
        timestamp
        for timestamp in (
            parse_log_timestamp(line)
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        if timestamp is not None
    ]
    if not timestamps:
        raise ValueError(f"No timestamps found in {path}")
    return int((timestamps[-1] - timestamps[0]).total_seconds())


def training_minutes(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"training takes ([0-9.]+) minutes", text)
    if not matches:
        raise ValueError(f"No training duration found in {path}")
    return float(matches[-1])


def matched_timestamp(path: Path, pattern: str, *, first: bool) -> datetime:
    matches = [
        timestamp
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if pattern in line
        for timestamp in [parse_log_timestamp(line)]
        if timestamp is not None
    ]
    if not matches:
        raise ValueError(f"No timestamp matching {pattern!r} in {path}")
    return matches[0] if first else matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "report_cvpr" / "metric_audit.json",
    )
    args = parser.parse_args()

    task2_full = json.loads(
        (ROOT / "outputs/task2/zero_shot_d_action_error_full/metrics.json").read_text()
    )["results"]
    chunk_data = json.loads(
        (ROOT / "outputs/task2/action_chunking_robustness/metrics.json").read_text()
    )
    chunk = chunk_data["results"]
    checkpoints = json.loads(
        (ROOT / "outputs/task2/experiments/eval_d_offline_scheduler/metrics.json").read_text()
    )["results"]

    b_full = task2_full["single_b"]
    abc_full = task2_full["abc_cosine_30k"]
    b_clean = chunk["B-only"]["D"]["clean"]
    abc_clean = chunk["A+B+C"]["D"]["clean"]

    a_model = ROOT / (
        "outputs/task1/final/objects/object_a/model/point_cloud/"
        "iteration_30000/point_cloud.ply"
    )
    a_supported = a_model.with_name("point_cloud_supported.ply")
    support_cache = np.load(a_model.with_name("support_cache.npz"))
    camera_count = len(
        json.loads(
            (
                ROOT / "outputs/task1/final/objects/object_a/model/cameras.json"
            ).read_text()
        )
    )

    a_log = ROOT / "logs/object_a_curated469_clean30k.log"
    b_log = ROOT / "logs/object_b_hamburger_official_v1.log"
    c_coarse = ROOT / (
        "outputs/task1/final/objects/object_c/training/coarse/log_object_c_v5.txt"
    )
    c_fine = ROOT / (
        "outputs/task1/final/objects/object_c/training/fine/log_object_c_v5_fine.txt"
    )

    a_text = a_log.read_text(encoding="utf-8", errors="replace")
    a_test = re.search(
        r"Evaluating test: L1 ([0-9.]+) PSNR ([0-9.]+)", a_text
    )
    a_train = re.search(
        r"Evaluating train: L1 ([0-9.]+) PSNR ([0-9.]+)", a_text
    )
    b_text = b_log.read_text(encoding="utf-8", errors="replace")
    b_elapsed = re.findall(r"10000it \[(\d+):(\d+),", b_text)
    a_elapsed = re.findall(r"30000/30000 \[(\d+):(\d+)<", a_text)
    if not a_test or not a_train or not b_elapsed:
        raise ValueError("Could not parse one or more final training logs")
    b_minutes, b_seconds = map(int, b_elapsed[-1])
    a_minutes, a_seconds = map(int, a_elapsed[-1])
    c_pipeline_start = matched_timestamp(c_coarse, "Start Training", first=True)
    c_pipeline_end = matched_timestamp(c_fine, "Finished saving mesh", first=False)

    audit = {
        "task1": {
            "object_a": {
                "registered_cameras": camera_count,
                "gaussians_raw": ply_vertex_count(a_model),
                "gaussians_supported": ply_vertex_count(a_supported),
                "support_cache_cameras": int(support_cache["used_cameras"]),
                "test_l1": float(a_test.group(1)),
                "test_psnr_db": float(a_test.group(2)),
                "train_l1": float(a_train.group(1)),
                "train_psnr_db": float(a_train.group(2)),
                "optimization_loop_seconds": a_minutes * 60 + a_seconds,
                "end_to_end_log_span_seconds": log_span_seconds(a_log),
            },
            "object_b": {
                "steps": 10000,
                "training_seconds": b_minutes * 60 + b_seconds,
            },
            "object_c": {
                "coarse_training_minutes": training_minutes(c_coarse),
                "fine_training_minutes": training_minutes(c_fine),
                "combined_training_minutes": training_minutes(c_coarse)
                + training_minutes(c_fine),
                "coarse_log_span_seconds": log_span_seconds(c_coarse),
                "fine_log_span_seconds": log_span_seconds(c_fine),
                "pipeline_training_start_to_final_export_seconds": int(
                    (c_pipeline_end - c_pipeline_start).total_seconds()
                ),
            },
        },
        "task2": {
            "full_d": {
                "examples": b_full["examples"],
                "b_only_l1": b_full["l1_loss"],
                "abc_l1": abc_full["l1_loss"],
                "l1_relative_improvement_percent": (
                    (b_full["l1_loss"] - abc_full["l1_loss"])
                    / b_full["l1_loss"]
                    * 100
                ),
                "total_loss_relative_improvement_percent": (
                    (b_full["loss"] - abc_full["loss"]) / b_full["loss"] * 100
                ),
            },
            "chunk_subset_d": {
                "examples": b_clean["examples"],
                "b_only_l1": b_clean["l1"],
                "abc_l1": abc_clean["l1"],
                "b_only_arm_cosine": b_clean["arm_cosine_mean"],
                "abc_arm_cosine": abc_clean["arm_cosine_mean"],
                "b_only_tail_head_ratio": b_clean["tail_head_ratio"],
                "abc_tail_head_ratio": abc_clean["tail_head_ratio"],
                "b_only_variation_ratio": b_clean["variation_ratio_pred_to_gt"],
                "abc_variation_ratio": abc_clean["variation_ratio_pred_to_gt"],
            },
            "paired_clean_d": chunk_data["derived"]["clean_D_model_difference"],
            "visual_shift_b_to_d": chunk_data["visual_shift_B_to_D"],
            "controlled_perturbations": chunk_data["derived"]["perturbation"],
            "abc_checkpoint_l1": {
                "10k": checkpoints["abc_cosine_10k"]["l1_loss"],
                "20k": checkpoints["abc_cosine_20k"]["l1_loss"],
                "30k": checkpoints["abc_cosine_30k"]["l1_loss"],
            },
        },
        "interpretation_limits": [
            "B-only and A+B+C use different training steps and LR schedules.",
            "Task2 metrics are offline teacher-forced action errors, not rollout success.",
            "Object B/C have no multi-view ground truth; no PSNR or Chamfer is claimed.",
        ],
    }

    assert audit["task1"]["object_a"]["registered_cameras"] == 469
    assert audit["task1"]["object_a"]["gaussians_raw"] == 66136
    assert audit["task1"]["object_a"]["gaussians_supported"] == 42800
    assert audit["task2"]["full_d"]["examples"] == 92274
    assert abs(
        audit["task2"]["full_d"]["l1_relative_improvement_percent"] - 17.601031
    ) < 1e-5
    assert (
        audit["task2"]["paired_clean_d"]["ci95_low"] > 0
    ), "Paired clean-D confidence interval unexpectedly crosses zero"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
