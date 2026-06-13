#!/usr/bin/env python3
"""Verify that the final B-only and ABC ACT checkpoints differ only by data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B = ROOT / (
    "outputs/task2/runs/task2_fair_b_10k_cosine_b256_c10_s1000/"
    "single_b_train/checkpoints/010000/pretrained_model"
)
DEFAULT_ABC = ROOT / (
    "outputs/task2/runs/task2_fair_abc_10k_cosine_b256_c10_s1000/"
    "abc_to_d_train/checkpoints/010000/pretrained_model"
)


def load_config(checkpoint: Path) -> dict:
    path = checkpoint / "train_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def comparable_view(config: dict) -> dict:
    policy = config["policy"]
    return {
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "seed": config["seed"],
        "cudnn_deterministic": config["cudnn_deterministic"],
        "num_workers": config["num_workers"],
        "prefetch_factor": config["prefetch_factor"],
        "persistent_workers": config["persistent_workers"],
        "tolerance_s": config["tolerance_s"],
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "policy": {
            key: policy[key]
            for key in (
                "chunk_size",
                "n_action_steps",
                "vision_backbone",
                "pretrained_backbone_weights",
                "dim_model",
                "n_heads",
                "dim_feedforward",
                "n_encoder_layers",
                "n_decoder_layers",
                "use_vae",
                "latent_dim",
                "n_vae_encoder_layers",
                "dropout",
                "kl_weight",
                "normalization_mapping",
                "use_amp",
                "temporal_ensemble_coeff",
            )
        },
        "image_transforms": config["dataset"]["image_transforms"],
        "video_backend": config["dataset"]["video_backend"],
        "return_uint8": config["dataset"]["return_uint8"],
        "use_imagenet_stats": config["dataset"]["use_imagenet_stats"],
        "sample_weighting": config["sample_weighting"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-checkpoint", type=Path, default=DEFAULT_B)
    parser.add_argument("--abc-checkpoint", type=Path, default=DEFAULT_ABC)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/task2/fair_comparison/config_audit.json",
    )
    args = parser.parse_args()

    b_config = load_config(args.b_checkpoint)
    abc_config = load_config(args.abc_checkpoint)
    b_view = comparable_view(b_config)
    abc_view = comparable_view(abc_config)
    differences = {
        key: {"b_only": b_view[key], "abc": abc_view[key]}
        for key in b_view
        if b_view[key] != abc_view[key]
    }
    report = {
        "fair": not differences,
        "controlled_fields": b_view,
        "allowed_difference": {
            "b_only_dataset": b_config["dataset"]["repo_id"],
            "abc_dataset": abc_config["dataset"]["repo_id"],
        },
        "unexpected_differences": differences,
        "b_checkpoint": str(args.b_checkpoint),
        "abc_checkpoint": str(args.abc_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if differences:
        raise SystemExit("Task2 checkpoints are not a fair controlled comparison")


if __name__ == "__main__":
    main()
