#!/usr/bin/env python
"""Analyze ACT action chunking under CALVIN B->D visual distribution shift.

The analysis is offline and teacher-forced. It measures:
1. clean in-domain B versus unseen D action error,
2. error at each of the 10 positions in an action chunk,
3. temporal action-difference error and action-dimension error,
4. controlled visual perturbations and camera ablations on D,
5. low-level RGB histogram shift between B and D.

It does not fabricate simulator rollout success rates.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "external" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies import get_policy_class  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import ACTION, POLICY_PREPROCESSOR_DEFAULT_NAME  # noqa: E402


DATA_ROOT = ROOT / "data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D"
CHECKPOINTS = {
    "B-only": ROOT
    / "outputs/task2/runs/task2_single_b_actual40g_b512_c10_w8_5k"
    / "single_b_train/checkpoints/005000/pretrained_model",
    "A+B+C": ROOT
    / "outputs/task2/runs/task2_abc_scheduler_b512_c10_w8_30k"
    / "abc_to_d_train/checkpoints/030000/pretrained_model",
}
ENVIRONMENTS = {
    "B": (1, "local/calvin_task_ABC_D_lerobot_1_4"),
    "D": (3, "local/calvin_task_ABC_D_lerobot_3_4"),
}
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
PERTURBATION_CONDITIONS = list(CONDITION_LABELS)


def load_dataset(env_name: str, config: PreTrainedConfig) -> LeRobotDataset:
    shard, repo_id = ENVIRONMENTS[env_name]
    root = DATA_ROOT / f"calvin_task_ABC_D_lerobot_{shard}_4"
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    return LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=resolve_delta_timestamps(config, meta),
        video_backend="pyav",
        return_uint8=True,
        tolerance_s=0.01,
    )


def make_loader(dataset: LeRobotDataset, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers else None,
        persistent_workers=num_workers > 0,
    )


def prepare_raw_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out = dict(batch)
    for key in CAMERA_KEYS:
        image = out[key]
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        out[key] = image
    return out


def shifted_image(image: torch.Tensor, dx: int = 16, dy: int = -8) -> torch.Tensor:
    shifted = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    if dx > 0:
        shifted[..., :dx] = 0.5
    elif dx < 0:
        shifted[..., dx:] = 0.5
    if dy > 0:
        shifted[..., :dy, :] = 0.5
    elif dy < 0:
        shifted[..., dy:, :] = 0.5
    return shifted


def perturb_batch(batch: dict[str, Any], condition: str, batch_index: int) -> dict[str, Any]:
    out = dict(batch)
    for camera_index, key in enumerate(CAMERA_KEYS):
        image = batch[key].clone()
        if condition == "clean":
            pass
        elif condition == "dark":
            image.mul_(0.55)
        elif condition == "bright":
            image.mul_(1.35).clamp_(0.0, 1.0)
        elif condition == "low_contrast":
            image = ((image - 0.5) * 0.5 + 0.5).clamp_(0.0, 1.0)
        elif condition == "warm_color":
            scale = image.new_tensor([1.20, 1.00, 0.72]).view(1, 3, 1, 1)
            image = (image * scale).clamp_(0.0, 1.0)
        elif condition == "blur":
            image = F.avg_pool2d(image, kernel_size=9, stride=1, padding=4)
        elif condition == "noise":
            generator = torch.Generator(device=image.device)
            generator.manual_seed(20260609 + batch_index * 17 + camera_index)
            noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
            image = (image + noise * 0.08).clamp_(0.0, 1.0)
        elif condition == "camera_shift":
            image = shifted_image(image)
        elif condition == "center_occlusion":
            height, width = image.shape[-2:]
            y0, y1 = int(height * 0.34), int(height * 0.66)
            x0, x1 = int(width * 0.34), int(width * 0.66)
            image[..., y0:y1, x0:x1] = 0.5
        elif condition == "static_drop":
            if key.endswith(".image"):
                image.fill_(0.5)
        elif condition == "wrist_drop":
            if key.endswith(".wrist_image"):
                image.fill_(0.5)
        else:
            raise ValueError(f"Unknown condition: {condition}")
        out[key] = image
    return out


@dataclass
class MetricAccumulator:
    chunk_size: int = 10
    action_dim: int = 7
    abs_horizon_sum: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    abs_horizon_count: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    abs_dim_sum: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    abs_dim_count: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    abs_horizon_dim_sum: np.ndarray = field(default_factory=lambda: np.zeros((10, 7), dtype=np.float64))
    abs_horizon_dim_count: np.ndarray = field(default_factory=lambda: np.zeros((10, 7), dtype=np.float64))
    cosine_horizon_sum: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    cosine_horizon_count: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    drift_horizon_sum: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    drift_horizon_count: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    delta_error_sum: float = 0.0
    delta_error_count: float = 0.0
    pred_variation_sum: float = 0.0
    gt_variation_sum: float = 0.0
    variation_count: float = 0.0
    gripper_correct: float = 0.0
    gripper_count: float = 0.0
    examples: int = 0
    batch_l1: list[float] = field(default_factory=list)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        is_pad: torch.Tensor,
        clean_prediction: torch.Tensor | None = None,
    ) -> None:
        prediction = prediction.detach().float().cpu()
        target = target.detach().float().cpu()
        is_pad = is_pad.detach().bool().cpu()
        valid = ~is_pad
        abs_error = (prediction - target).abs()
        valid_3d = valid.unsqueeze(-1)

        horizon_sum = (abs_error * valid_3d).sum(dim=(0, 2)).numpy()
        horizon_count = (valid.sum(dim=0) * abs_error.shape[-1]).numpy()
        self.abs_horizon_sum += horizon_sum
        self.abs_horizon_count += horizon_count
        self.abs_dim_sum += (abs_error * valid_3d).sum(dim=(0, 1)).numpy()
        self.abs_dim_count += valid.sum().item()
        self.abs_horizon_dim_sum += (abs_error * valid_3d).sum(dim=0).numpy()
        self.abs_horizon_dim_count += valid.unsqueeze(-1).expand_as(abs_error).sum(dim=0).numpy()

        arm_pred = prediction[..., :6]
        arm_target = target[..., :6]
        cosine = F.cosine_similarity(arm_pred, arm_target, dim=-1, eps=1e-8)
        self.cosine_horizon_sum += (cosine * valid).sum(dim=0).numpy()
        self.cosine_horizon_count += valid.sum(dim=0).numpy()

        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        gt_delta = target[:, 1:] - target[:, :-1]
        adjacent_3d = adjacent_valid.unsqueeze(-1)
        self.delta_error_sum += float(((pred_delta - gt_delta).abs() * adjacent_3d).sum())
        self.delta_error_count += float(adjacent_valid.sum() * prediction.shape[-1])
        self.pred_variation_sum += float((pred_delta.abs() * adjacent_3d).sum())
        self.gt_variation_sum += float((gt_delta.abs() * adjacent_3d).sum())
        self.variation_count += float(adjacent_valid.sum() * prediction.shape[-1])

        gripper_pred = prediction[..., 6] >= 0
        gripper_target = target[..., 6] >= 0
        self.gripper_correct += float(((gripper_pred == gripper_target) & valid).sum())
        self.gripper_count += float(valid.sum())

        if clean_prediction is not None:
            clean_prediction = clean_prediction.detach().float().cpu()
            drift = (prediction - clean_prediction).abs()
            self.drift_horizon_sum += (drift * valid_3d).sum(dim=(0, 2)).numpy()
            self.drift_horizon_count += horizon_count

        batch_denominator = valid.sum().item() * prediction.shape[-1]
        self.batch_l1.append(float((abs_error * valid_3d).sum()) / max(batch_denominator, 1))
        self.examples += prediction.shape[0]

    def finalize(self) -> dict[str, Any]:
        horizon = np.divide(
            self.abs_horizon_sum,
            self.abs_horizon_count,
            out=np.zeros_like(self.abs_horizon_sum),
            where=self.abs_horizon_count > 0,
        )
        dimension = np.divide(
            self.abs_dim_sum,
            self.abs_dim_count,
            out=np.zeros_like(self.abs_dim_sum),
            where=self.abs_dim_count > 0,
        )
        horizon_dim = np.divide(
            self.abs_horizon_dim_sum,
            self.abs_horizon_dim_count,
            out=np.zeros_like(self.abs_horizon_dim_sum),
            where=self.abs_horizon_dim_count > 0,
        )
        cosine = np.divide(
            self.cosine_horizon_sum,
            self.cosine_horizon_count,
            out=np.zeros_like(self.cosine_horizon_sum),
            where=self.cosine_horizon_count > 0,
        )
        drift = np.divide(
            self.drift_horizon_sum,
            self.drift_horizon_count,
            out=np.zeros_like(self.drift_horizon_sum),
            where=self.drift_horizon_count > 0,
        )
        x = np.arange(1, self.chunk_size + 1)
        slope = float(np.polyfit(x, horizon, 1)[0])
        overall = float(self.abs_horizon_sum.sum() / self.abs_horizon_count.sum())
        head = float(horizon[:3].mean())
        tail = float(horizon[-3:].mean())
        return {
            "examples": self.examples,
            "l1": overall,
            "first_step_l1": float(horizon[0]),
            "last_step_l1": float(horizon[-1]),
            "head_l1_k1_3": head,
            "tail_l1_k8_10": tail,
            "tail_head_ratio": tail / head if head else math.nan,
            "endpoint_amplification": float(horizon[-1] / horizon[0]) if horizon[0] else math.nan,
            "horizon_slope": slope,
            "horizon_l1": horizon.tolist(),
            "action_dimension_l1": {label: float(value) for label, value in zip(ACTION_LABELS, dimension)},
            "horizon_dimension_l1": horizon_dim.tolist(),
            "arm_cosine_by_horizon": cosine.tolist(),
            "arm_cosine_mean": float(np.mean(cosine)),
            "delta_action_l1": self.delta_error_sum / max(self.delta_error_count, 1),
            "pred_action_variation": self.pred_variation_sum / max(self.variation_count, 1),
            "gt_action_variation": self.gt_variation_sum / max(self.variation_count, 1),
            "variation_ratio_pred_to_gt": self.pred_variation_sum / max(self.gt_variation_sum, 1e-12),
            "gripper_sign_accuracy": self.gripper_correct / max(self.gripper_count, 1),
            "prediction_drift_l1": float(self.drift_horizon_sum.sum() / max(self.drift_horizon_count.sum(), 1)),
            "prediction_drift_by_horizon": drift.tolist(),
            "batch_l1": self.batch_l1,
        }


@dataclass
class VisualStats:
    bins: int = 32
    count: dict[str, int] = field(default_factory=lambda: {key: 0 for key in CAMERA_KEYS})
    sum_rgb: dict[str, np.ndarray] = field(
        default_factory=lambda: {key: np.zeros(3, dtype=np.float64) for key in CAMERA_KEYS}
    )
    sum_sq_rgb: dict[str, np.ndarray] = field(
        default_factory=lambda: {key: np.zeros(3, dtype=np.float64) for key in CAMERA_KEYS}
    )
    hist: dict[str, np.ndarray] = field(
        default_factory=lambda: {key: np.zeros((3, 32), dtype=np.float64) for key in CAMERA_KEYS}
    )

    def update(self, batch: dict[str, Any]) -> None:
        for key in CAMERA_KEYS:
            image = batch[key].detach().float().cpu()
            image = F.avg_pool2d(image, kernel_size=8, stride=8)
            pixels = image.permute(1, 0, 2, 3).reshape(3, -1)
            self.count[key] += pixels.shape[1]
            self.sum_rgb[key] += pixels.sum(dim=1).numpy()
            self.sum_sq_rgb[key] += (pixels**2).sum(dim=1).numpy()
            for channel in range(3):
                self.hist[key][channel] += np.histogram(
                    pixels[channel].numpy(), bins=self.bins, range=(0.0, 1.0)
                )[0]

    def finalize(self) -> dict[str, Any]:
        result = {}
        for key in CAMERA_KEYS:
            count = max(self.count[key], 1)
            mean = self.sum_rgb[key] / count
            variance = np.maximum(self.sum_sq_rgb[key] / count - mean**2, 0)
            hist = self.hist[key]
            hist = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
            result[key] = {
                "mean_rgb": mean.tolist(),
                "std_rgb": np.sqrt(variance).tolist(),
                "histogram": hist.tolist(),
            }
        return result


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    midpoint = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (midpoint + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (midpoint + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def compare_visual_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in CAMERA_KEYS:
        b = stats["B"][key]
        d = stats["D"][key]
        b_hist = np.asarray(b["histogram"])
        d_hist = np.asarray(d["histogram"])
        per_channel = [js_divergence(b_hist[i], d_hist[i]) for i in range(3)]
        b_mean = np.asarray(b["mean_rgb"])
        d_mean = np.asarray(d["mean_rgb"])
        result[key] = {
            "histogram_js_per_channel": per_channel,
            "histogram_js_mean": float(np.mean(per_channel)),
            "mean_rgb_l2": float(np.linalg.norm(b_mean - d_mean)),
            "mean_luminance_B": float(b_mean.mean()),
            "mean_luminance_D": float(d_mean.mean()),
        }
    return result


def save_perturbation_examples(batch: dict[str, Any], output: Path) -> None:
    conditions = PERTURBATION_CONDITIONS
    tile_size = 180
    rows = len(CAMERA_KEYS)
    canvas = Image.new("RGB", (len(conditions) * tile_size, rows * (tile_size + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, condition in enumerate(conditions):
        shifted = perturb_batch(batch, condition, 0)
        draw.text((col * tile_size + 4, 4), CONDITION_LABELS[condition], fill="black")
        for row, key in enumerate(CAMERA_KEYS):
            image = shifted[key][0].permute(1, 2, 0).mul(255).byte().numpy()
            pil = Image.fromarray(image).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
            canvas.paste(pil, (col * tile_size, row * (tile_size + 28) + 28))
    canvas.save(output, quality=92)


def bootstrap_paired_difference(a: list[float], b: list[float], seed: int = 20260609) -> dict[str, float]:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n = min(len(a_arr), len(b_arr))
    differences = a_arr[:n] - b_arr[:n]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(5000, n))
    samples = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "paired_effect_size_dz": float(differences.mean() / max(differences.std(ddof=1), 1e-12)),
    }


def evaluate_model(
    model_name: str,
    checkpoint: Path,
    loaders: dict[str, torch.utils.data.DataLoader],
    *,
    device: str,
    max_batches: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_cls = get_policy_class("act")
    policy = policy_cls.from_pretrained(checkpoint)
    policy.to(device).eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        device_processor={"device": device},
    )

    results: dict[str, Any] = {}
    visual_stats: dict[str, Any] = {}
    for env_name, loader in loaders.items():
        conditions = PERTURBATION_CONDITIONS if env_name == "D" else ["clean"]
        accumulators = {condition: MetricAccumulator() for condition in conditions}
        stats = VisualStats()
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            raw_batch = prepare_raw_batch(batch)
            if model_name == "B-only":
                stats.update(raw_batch)
                if env_name == "D" and batch_index == 0:
                    save_perturbation_examples(raw_batch, output_dir / "visual_shift_examples.jpg")

            clean_prediction = None
            for condition in conditions:
                condition_batch = perturb_batch(raw_batch, condition, batch_index)
                processed = preprocessor(condition_batch)
                prediction = policy.predict_action_chunk(processed)
                if condition == "clean":
                    clean_prediction = prediction.detach()
                accumulators[condition].update(
                    prediction,
                    processed[ACTION],
                    processed["action_is_pad"],
                    None if condition == "clean" else clean_prediction,
                )

        results[env_name] = {condition: accumulator.finalize() for condition, accumulator in accumulators.items()}
        if model_name == "B-only":
            visual_stats[env_name] = stats.finalize()

    del policy
    torch.cuda.empty_cache()
    return results, visual_stats


def add_derived_metrics(results: dict[str, Any]) -> dict[str, Any]:
    derived: dict[str, Any] = {"cross_environment": {}, "perturbation": {}}
    for model_name, model_results in results.items():
        clean_b = model_results["B"]["clean"]
        clean_d = model_results["D"]["clean"]
        derived["cross_environment"][model_name] = {
            "B_l1": clean_b["l1"],
            "D_l1": clean_d["l1"],
            "absolute_gap": clean_d["l1"] - clean_b["l1"],
            "relative_gap_pct": (clean_d["l1"] / clean_b["l1"] - 1.0) * 100.0,
            "B_tail_head_ratio": clean_b["tail_head_ratio"],
            "D_tail_head_ratio": clean_d["tail_head_ratio"],
            "tail_head_gap": clean_d["tail_head_ratio"] - clean_b["tail_head_ratio"],
        }
        clean_l1 = clean_d["l1"]
        derived["perturbation"][model_name] = {}
        for condition, metrics in model_results["D"].items():
            derived["perturbation"][model_name][condition] = {
                "l1": metrics["l1"],
                "relative_degradation_pct": (metrics["l1"] / clean_l1 - 1.0) * 100.0,
                "prediction_drift_l1": metrics["prediction_drift_l1"],
                "tail_head_ratio": metrics["tail_head_ratio"],
                "endpoint_amplification": metrics["endpoint_amplification"],
                "delta_action_l1": metrics["delta_action_l1"],
            }

    derived["clean_D_model_difference"] = bootstrap_paired_difference(
        results["B-only"]["D"]["clean"]["batch_l1"],
        results["A+B+C"]["D"]["clean"]["batch_l1"],
    )
    return derived


def strip_batch_values(data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(data)
    for model_results in data.values():
        for env_results in model_results.values():
            for metrics in env_results.values():
                metrics.pop("batch_l1", None)
    return data


def save_csv(results: dict[str, Any], derived: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "environment",
                "condition",
                "examples",
                "l1",
                "first_step_l1",
                "last_step_l1",
                "tail_head_ratio",
                "horizon_slope",
                "delta_action_l1",
                "variation_ratio_pred_to_gt",
                "arm_cosine_mean",
                "gripper_sign_accuracy",
                "prediction_drift_l1",
            ]
        )
        for model_name, model_results in results.items():
            for env_name, env_results in model_results.items():
                for condition, metrics in env_results.items():
                    writer.writerow(
                        [
                            model_name,
                            env_name,
                            condition,
                            metrics["examples"],
                            metrics["l1"],
                            metrics["first_step_l1"],
                            metrics["last_step_l1"],
                            metrics["tail_head_ratio"],
                            metrics["horizon_slope"],
                            metrics["delta_action_l1"],
                            metrics["variation_ratio_pred_to_gt"],
                            metrics["arm_cosine_mean"],
                            metrics["gripper_sign_accuracy"],
                            metrics["prediction_drift_l1"],
                        ]
                    )

    with (output_dir / "horizon_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "environment", "condition", "chunk_position", "l1", "prediction_drift_l1"])
        for model_name, model_results in results.items():
            for env_name, env_results in model_results.items():
                for condition, metrics in env_results.items():
                    for index, (l1_value, drift) in enumerate(
                        zip(metrics["horizon_l1"], metrics["prediction_drift_by_horizon"]), start=1
                    ):
                        writer.writerow([model_name, env_name, condition, index, l1_value, drift])


def plot_results(results: dict[str, Any], derived: dict[str, Any], visual: dict[str, Any], output_dir: Path) -> None:
    colors = {"B-only": "#8C8C8C", "A+B+C": "#276B9A"}
    positions = np.arange(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=220)
    for model_name in CHECKPOINTS:
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

    clean_conditions = [condition for condition in PERTURBATION_CONDITIONS if condition != "clean"]
    x = np.arange(len(clean_conditions))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.2, 4.4), dpi=220)
    for index, model_name in enumerate(CHECKPOINTS):
        values = [
            derived["perturbation"][model_name][condition]["relative_degradation_pct"]
            for condition in clean_conditions
        ]
        ax.bar(x + (index - 0.5) * width, values, width, label=model_name, color=colors[model_name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in clean_conditions], rotation=28, ha="right")
    ax.set_ylabel("Action L1 degradation from clean D (%)")
    ax.set_title("Robustness to controlled visual distribution shifts")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "visual_perturbation_degradation.png", bbox_inches="tight")
    plt.close(fig)

    heatmap_conditions = ["clean", "dark", "low_contrast", "blur", "noise", "camera_shift", "center_occlusion"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220, sharey=True)
    clean_reference = {
        model_name: np.asarray(results[model_name]["D"]["clean"]["horizon_l1"]) for model_name in CHECKPOINTS
    }
    for ax, model_name in zip(axes, CHECKPOINTS):
        matrix = []
        for condition in heatmap_conditions:
            current = np.asarray(results[model_name]["D"][condition]["horizon_l1"])
            matrix.append((current / clean_reference[model_name] - 1.0) * 100.0)
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=-5, vmax=80)
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
    for index, model_name in enumerate(CHECKPOINTS):
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
    for index, model_name in enumerate(CHECKPOINTS):
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
    for index, model_name in enumerate(CHECKPOINTS):
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


def write_markdown(
    results: dict[str, Any],
    derived: dict[str, Any],
    visual_shift: dict[str, Any],
    output_dir: Path,
    sample_count: int,
) -> None:
    lines = [
        "# Task 2 Action Chunking Robustness",
        "",
        f"Controlled analysis on the first {sample_count:,} ordered samples from B and D.",
        "The main full-D aggregate remains in `outputs/task2/zero_shot_d_action_error_full/metrics.json`.",
        "",
        "## Clean cross-environment metrics",
        "",
        "| Model | B L1 | D L1 | D gap | D first step | D last step | D tail/head | Delta-action L1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in CHECKPOINTS:
        cross = derived["cross_environment"][model_name]
        d = results[model_name]["D"]["clean"]
        lines.append(
            f"| {model_name} | {cross['B_l1']:.4f} | {cross['D_l1']:.4f} | "
            f"{cross['relative_gap_pct']:+.2f}% | {d['first_step_l1']:.4f} | "
            f"{d['last_step_l1']:.4f} | {d['tail_head_ratio']:.3f} | {d['delta_action_l1']:.4f} |"
        )
    paired = derived["clean_D_model_difference"]
    lines += [
        "",
        "Paired clean-D batch difference (B-only minus A+B+C): "
        f"{paired['mean_difference']:.4f}, 95% bootstrap CI "
        f"[{paired['ci95_low']:.4f}, {paired['ci95_high']:.4f}], "
        f"paired effect size dz={paired['paired_effect_size_dz']:.2f}.",
        "",
        "## Visual perturbation degradation",
        "",
        "| Condition | B-only L1 | B-only change | A+B+C L1 | A+B+C change |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in PERTURBATION_CONDITIONS:
        b = derived["perturbation"]["B-only"][condition]
        abc = derived["perturbation"]["A+B+C"][condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {b['l1']:.4f} | {b['relative_degradation_pct']:+.2f}% | "
            f"{abc['l1']:.4f} | {abc['relative_degradation_pct']:+.2f}% |"
        )
    lines += [
        "",
        "## Low-level B to D visual shift",
        "",
        "| Camera | RGB histogram JS | Mean RGB L2 | Mean luminance B | Mean luminance D |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in CAMERA_KEYS:
        value = visual_shift[key]
        camera = "static" if key.endswith(".image") else "wrist"
        lines.append(
            f"| {camera} | {value['histogram_js_mean']:.5f} | {value['mean_rgb_l2']:.5f} | "
            f"{value['mean_luminance_B']:.4f} | {value['mean_luminance_D']:.4f} |"
        )
    lines += [
        "",
        "## Generated figures",
        "",
        "- `chunk_horizon_analysis.png`",
        "- `visual_perturbation_degradation.png`",
        "- `horizon_shift_heatmap.png`",
        "- `action_dimension_error.png`",
        "- `cross_environment_gap.png`",
        "- `camera_ablation.png`",
        "- `robustness_summary.png`",
        "- `visual_distribution_histograms.png`",
        "- `visual_shift_examples.jpg`",
        "",
        "These are offline action-prediction metrics, not simulator rollout success rates.",
    ]
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/task2/action_chunking_robustness"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    get_policy_class("act")
    config = PreTrainedConfig.from_pretrained(next(iter(CHECKPOINTS.values())))
    datasets = {env_name: load_dataset(env_name, config) for env_name in ENVIRONMENTS}
    loaders = {
        env_name: make_loader(dataset, args.batch_size, args.num_workers)
        for env_name, dataset in datasets.items()
    }

    all_results: dict[str, Any] = {}
    visual_stats: dict[str, Any] = {}
    for model_name, checkpoint in CHECKPOINTS.items():
        print(f"Evaluating {model_name}: {checkpoint}")
        model_results, model_visual = evaluate_model(
            model_name,
            checkpoint,
            loaders,
            device=args.device,
            max_batches=args.max_batches,
            output_dir=args.output_dir,
        )
        all_results[model_name] = model_results
        visual_stats.update(model_visual)
        partial = {
            "config": vars(args) | {"output_dir": str(args.output_dir)},
            "results": strip_batch_values(all_results),
        }
        (args.output_dir / "partial_metrics.json").write_text(
            json.dumps(partial, indent=2) + "\n", encoding="utf-8"
        )

    visual_shift = compare_visual_stats(visual_stats)
    derived = add_derived_metrics(all_results)
    serializable_results = strip_batch_values(all_results)
    summary = {
        "protocol": {
            "metric_type": "offline_teacher_forced_action_chunking_robustness",
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "samples_per_environment": min(len(datasets["B"]), args.batch_size * args.max_batches),
            "chunk_size": 10,
            "n_action_steps": 10,
            "temporal_ensemble": False,
            "environments": {
                name: str(Path(dataset.root).relative_to(ROOT)) for name, dataset in datasets.items()
            },
            "checkpoints": {name: str(path.relative_to(ROOT)) for name, path in CHECKPOINTS.items()},
            "note": "D is unseen during training. Metrics are offline and are not rollout success rates.",
        },
        "results": serializable_results,
        "derived": derived,
        "visual_stats": visual_stats,
        "visual_shift_B_to_D": visual_shift,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_csv(serializable_results, derived, args.output_dir)
    write_markdown(
        serializable_results,
        derived,
        visual_shift,
        args.output_dir,
        summary["protocol"]["samples_per_environment"],
    )
    print(json.dumps({"output_dir": str(args.output_dir), "derived": derived}, indent=2))


if __name__ == "__main__":
    main()
