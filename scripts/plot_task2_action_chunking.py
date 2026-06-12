#!/usr/bin/env python3
"""Create report figures from Task 2 action-chunking robustness metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_NAMES = ["B-only", "A+B+C"]
CAMERA_KEYS = ["observation.images.image", "observation.images.wrist_image"]
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
CONDITION_LABELS = {
    "clean": "Clean D",
    "dark": "Dark (x0.55)",
    "bright": "Bright (x1.35)",
    "low_contrast": "Low contrast",
    "warm_color": "Warm color",
    "blur": "Blur (9x9)",
    "noise": "Gaussian noise",
    "camera_shift": "Camera shift",
    "center_occlusion": "Center occlusion",
    "static_drop": "Static camera drop",
    "wrist_drop": "Wrist camera drop",
}


def plot_results(payload: dict, output_dir: Path) -> None:
    results = payload["results"]
    derived = payload["derived"]
    visual = payload["visual_stats"]
    colors = {"B-only": "#8C8C8C", "A+B+C": "#276B9A"}
    positions = np.arange(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=220)
    for model_name in MODEL_NAMES:
        axes[0].plot(
            positions,
            results[model_name]["D"]["clean"]["horizon_l1"],
            marker="o",
            color=colors[model_name],
            label=model_name,
        )
        axes[1].plot(
            positions,
            results[model_name]["D"]["clean"]["arm_cosine_by_horizon"],
            marker="o",
            color=colors[model_name],
            label=model_name,
        )
    axes[0].set_title("Action error across chunk positions")
    axes[0].set_xlabel("Chunk position k")
    axes[0].set_ylabel("Normalized Action L1")
    axes[1].set_title("6-DoF direction agreement")
    axes[1].set_xlabel("Chunk position k")
    axes[1].set_ylabel("Cosine similarity")
    for ax in axes:
        ax.set_xticks(positions)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "chunk_horizon_analysis.png", bbox_inches="tight")
    plt.close(fig)

    perturbations = [condition for condition in CONDITION_LABELS if condition != "clean"]
    x = np.arange(len(perturbations))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.0), dpi=220, sharex=True)
    for index, model_name in enumerate(MODEL_NAMES):
        absolute_values = [
            results[model_name]["D"][condition]["l1"]
            for condition in perturbations
        ]
        relative_values = [
            derived["perturbation"][model_name][condition]["relative_degradation_pct"]
            for condition in perturbations
        ]
        offset = x + (index - 0.5) * width
        axes[0].bar(offset, absolute_values, width, label=model_name, color=colors[model_name])
        axes[1].bar(offset, relative_values, width, label=model_name, color=colors[model_name])
    axes[0].set_ylabel("Normalized Action L1")
    axes[0].set_title("Absolute error under controlled visual shifts")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Increase from clean D (%)")
    axes[1].set_title("Relative degradation from each model's clean-D baseline")
    axes[1].set_xticks(x, [CONDITION_LABELS[c] for c in perturbations], rotation=28, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "visual_perturbation_degradation.png", bbox_inches="tight")
    plt.close(fig)

    heatmap_conditions = ["clean", "dark", "low_contrast", "blur", "noise", "camera_shift", "center_occlusion"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220, sharey=True)
    clean_reference = {
        model_name: np.asarray(results[model_name]["D"]["clean"]["horizon_l1"]) for model_name in MODEL_NAMES
    }
    for ax, model_name in zip(axes, MODEL_NAMES):
        matrix = []
        for condition in heatmap_conditions:
            current = np.asarray(results[model_name]["D"][condition]["horizon_l1"])
            matrix.append((current / clean_reference[model_name] - 1.0) * 100.0)
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=-5, vmax=40)
        ax.set_title(model_name)
        ax.set_xticks(np.arange(10), np.arange(1, 11))
        ax.set_xlabel("Chunk position k")
        ax.set_yticks(np.arange(len(heatmap_conditions)), [CONDITION_LABELS[c] for c in heatmap_conditions])
    axes[0].set_ylabel("Visual condition")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="L1 increase from clean D (%)", shrink=0.86)
    fig.suptitle("Where visual shift enters the 10-step action chunk", y=0.99)
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.12, top=0.86, wspace=0.08)
    fig.savefig(output_dir / "horizon_shift_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 3.9), dpi=220)
    x = np.arange(len(ACTION_LABELS))
    for index, model_name in enumerate(MODEL_NAMES):
        values = [
            results[model_name]["D"]["clean"]["action_dimension_l1"][label] for label in ACTION_LABELS
        ]
        ax.bar(x + (index - 0.5) * width, values, width, label=model_name, color=colors[model_name])
    ax.set_xticks(x, ACTION_LABELS)
    ax.set_ylabel("Normalized L1")
    ax.set_title("Clean D error by action dimension")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "action_dimension_error.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=220)
    x = np.arange(2)
    for index, model_name in enumerate(MODEL_NAMES):
        values = [results[model_name][env]["clean"]["l1"] for env in ["B", "D"]]
        axes[0].bar(x + (index - 0.5) * width, values, width, label=model_name, color=colors[model_name])
        ratios = [results[model_name][env]["clean"]["tail_head_ratio"] for env in ["B", "D"]]
        axes[1].bar(x + (index - 0.5) * width, ratios, width, label=model_name, color=colors[model_name])
    axes[0].set_title("In-domain B versus unseen D")
    axes[0].set_ylabel("Normalized Action L1")
    axes[1].set_title("Within-chunk tail/head amplification")
    axes[1].set_ylabel("Mean L1(k=8..10) / L1(k=1..3)")
    for ax in axes:
        ax.set_xticks(x, ["B", "D"])
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_environment_gap.png", bbox_inches="tight")
    plt.close(fig)

    camera_conditions = ["clean", "static_drop", "wrist_drop"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
    x = np.arange(len(camera_conditions))
    for index, model_name in enumerate(MODEL_NAMES):
        values = [results[model_name]["D"][condition]["l1"] for condition in camera_conditions]
        ax.bar(x + (index - 0.5) * width, values, width, label=model_name, color=colors[model_name])
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in camera_conditions])
    ax.set_ylabel("Normalized Action L1")
    ax.set_title("Dual-camera ablation on unseen D")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "camera_ablation.png", bbox_inches="tight")
    plt.close(fig)

    condition_groups = {
        "Photometric": ["dark", "bright", "low_contrast", "warm_color"],
        "Corruption": ["blur", "noise"],
        "Spatial": ["camera_shift", "center_occlusion"],
        "Camera ablation": ["static_drop", "wrist_drop"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=220)
    x = np.arange(len(condition_groups))
    for index, model_name in enumerate(MODEL_NAMES):
        group_l1 = [
            np.mean([results[model_name]["D"][condition]["l1"] for condition in conditions])
            for conditions in condition_groups.values()
        ]
        group_drift = [
            np.mean(
                [
                    results[model_name]["D"][condition]["prediction_drift_l1"]
                    for condition in conditions
                ]
            )
            for conditions in condition_groups.values()
        ]
        offset = x + (index - 0.5) * width
        axes[0].bar(offset, group_l1, width, label=model_name, color=colors[model_name])
        axes[1].bar(offset, group_drift, width, label=model_name, color=colors[model_name])
    axes[0].set_title("Mean action error by shift family")
    axes[0].set_ylabel("Normalized Action L1")
    axes[1].set_title("Prediction sensitivity to the same frames")
    axes[1].set_ylabel("Prediction drift L1")
    for ax in axes:
        ax.set_xticks(x, list(condition_groups))
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=220)
    bin_centers = (np.arange(32) + 0.5) / 32
    channel_colors = ["#C94A4A", "#3D8B5F", "#3D65A5"]
    for ax, key in zip(axes, CAMERA_KEYS):
        for env_name, linestyle in [("B", "--"), ("D", "-")]:
            hist = np.asarray(visual[env_name][key]["histogram"])
            for channel in range(3):
                ax.plot(
                    bin_centers,
                    hist[channel],
                    color=channel_colors[channel],
                    linestyle=linestyle,
                    alpha=0.9,
                    label=f"{env_name}-{['R', 'G', 'B'][channel]}",
                )
        ax.set_title("Static camera" if key.endswith(".image") else "Wrist camera")
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Low-level appearance shift from environment B to D", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "visual_distribution_histograms.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/task2/action_chunking_robustness/metrics.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text())
    plot_results(payload, args.metrics.parent)
    print(args.metrics.parent)


if __name__ == "__main__":
    main()
