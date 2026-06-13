#!/usr/bin/env python3
"""Build report figures directly from current experiment artifacts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report_cvpr" / "figures"


def contain(path: Path, size: tuple[int, int], background: str = "white") -> Image.Image:
    if not path.exists():
        canvas = Image.new("RGB", size, (238, 238, 238))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), f"missing\n{path.name}", fill=(90, 90, 90))
        return canvas
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def crop_contact_cell(path: Path, columns: int, rows: int, index: int) -> Image.Image:
    """Extract one full-resolution cell from a regular Magic123 contact sheet."""
    image = Image.open(path).convert("RGB")
    column = index % columns
    row = index // columns
    x0 = round(column * image.width / columns)
    x1 = round((column + 1) * image.width / columns)
    y0 = round(row * image.height / rows)
    y1 = round((row + 1) * image.height / rows)
    return image.crop((x0 + 2, y0 + 2, x1 - 2, y1 - 2))


def trim_white(image: Image.Image, threshold: int = 246, padding: int = 20) -> Image.Image:
    """Remove near-white margins while preserving a small visual border."""
    rgb = np.asarray(image.convert("RGB"))
    foreground = np.any(rgb < threshold, axis=2)
    if not foreground.any():
        return image.convert("RGB")
    ys, xs = np.nonzero(foreground)
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, image.width)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, image.height)
    return image.convert("RGB").crop((x0, y0, x1, y1))


def image_grid_from_images(
    images: list[Image.Image],
    output: Path,
    cols: int,
    cell: tuple[int, int],
    labels: list[str],
) -> None:
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell[0], rows * cell[1]), "white")
    for index, source in enumerate(images):
        image = ImageOps.contain(source.convert("RGB"), (cell[0], cell[1] - 32), Image.Resampling.LANCZOS)
        x = (index % cols) * cell[0]
        y = (index // cols) * cell[1]
        canvas.paste(image, (x + (cell[0] - image.width) // 2, y + 32))
    fig, ax = plt.subplots(figsize=(cols * 3.3, rows * 2.7), dpi=220)
    ax.imshow(canvas)
    ax.axis("off")
    for index, label in enumerate(labels):
        x = (index % cols) * cell[0] + 10
        y = (index // cols) * cell[1] + 19
        ax.text(x, y, label, fontsize=8, va="center", color="#111111")
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def image_grid(
    paths: list[Path],
    output: Path,
    cols: int,
    cell: tuple[int, int],
    labels: list[str] | None = None,
) -> None:
    if not paths:
        paths = [Path("__missing__")]
        labels = labels or ["missing inputs"]
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell[0], rows * cell[1]), "white")
    for index, path in enumerate(paths):
        image = contain(path, (cell[0], cell[1] - 28))
        x = (index % cols) * cell[0]
        y = (index // cols) * cell[1] + 28
        canvas.paste(image, (x, y))
    fig, ax = plt.subplots(figsize=(cols * 2.5, rows * 1.8), dpi=220)
    ax.imshow(canvas)
    ax.axis("off")
    if labels:
        for index, label in enumerate(labels):
            x = (index % cols) * cell[0] + 8
            y = (index // cols) * cell[1] + 17
            ax.text(x, y, label, fontsize=7, va="center", color="#111111")
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_object_a_views() -> None:
    view_dir = ROOT / "outputs/task1/final/quality/object_a_views"
    paths = [view_dir / f"gt_{i:02d}.png" for i in range(1,10,3)]
    paths += [view_dir / f"render_{i:02d}.png" for i in range(1,10,3)]
    image_grid(
        paths,
        OUT / "task1_a_views.png",
        cols=3,
        cell=(430, 315),
        labels=["Input view 1", "Input view 2", "Input view 3", "2DGS render 1", "2DGS render 2", "2DGS render 3"],
    )


def load_scalar(event_path: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    return (
        np.asarray([event.step for event in events], dtype=np.float64),
        np.asarray([event.value for event in events], dtype=np.float64),
    )


def make_task1_a_training_curve() -> None:
    event_path = next(
        (ROOT / "outputs/task1/final/objects/object_a/model").glob("events.out.tfevents*")
    )
    steps, total = load_scalar(event_path, "train_loss_patches/total_loss")
    test_steps, test_psnr = load_scalar(event_path, "test/loss_viewpoint - psnr")
    train_steps, train_psnr = load_scalar(event_path, "train/loss_viewpoint - psnr")
    fig, left = plt.subplots(figsize=(7.4, 3.5), dpi=220)
    left.plot(
        steps,
        moving_average(total, 151),
        color="#1F5A94",
        linewidth=1.2,
        label="Smoothed total loss",
    )
    left.set_xlabel("2DGS optimization step")
    left.set_ylabel("Training loss", color="#1F5A94")
    left.tick_params(axis="y", labelcolor="#1F5A94")
    left.grid(alpha=0.22)
    right = left.twinx()
    right.scatter(test_steps, test_psnr, color="#B24C32", marker="o", s=25, label="Test PSNR")
    right.scatter(train_steps, train_psnr, color="#4B8B3B", marker="s", s=25, label="Train PSNR")
    right.set_ylabel("PSNR (dB)")
    handles = left.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = left.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    left.legend(handles, labels, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "task1_a_training_curve.png", bbox_inches="tight")
    plt.close(fig)


def make_task1_c_training_curve() -> None:
    paths = [
        next((ROOT / "outputs/task1/final/objects/object_c/training/coarse").glob("**/events.out.tfevents*")),
        next((ROOT / "outputs/task1/final/objects/object_c/training/fine").glob("**/events.out.tfevents*")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=220)
    for ax, path, title in zip(axes, paths, ["Coarse NeRF", "Fine DMTet"]):
        steps, sd = load_scalar(path, "train/loss_sds")
        _, zero = load_scalar(path, "train/loss_zero123")
        ax.plot(steps, moving_average(sd, 51), color="#1F5A94", linewidth=1.1, label="SD loss")
        ax.plot(steps, moving_average(zero, 51), color="#B24C32", linewidth=1.1, label="Zero123 loss")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Optimization step")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Logged weighted guidance loss")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "task1_c_training_curve.png", bbox_inches="tight")
    plt.close(fig)


def make_task1_quality_overview() -> None:
    environment_dir = ROOT / "outputs/task1/final/quality/environment_kitchen"
    environment_contact = environment_dir / "contact.jpg"
    image_grid(
        [environment_dir / f"render_{index:02d}.png" for index in range(8)],
        environment_contact,
        cols=4,
        cell=(360, 260),
        labels=[f"view {index}" for index in range(8)],
    )
    paths = [
        ROOT / "outputs/task1/final/quality/object_a_turntable_dense/keyframes/00000.jpg",
        ROOT / "outputs/task1/final/quality/object_b_turntable/frames/00000.jpg",
        ROOT / "outputs/task1/final/quality/object_c_turntable/frames/00060.jpg",
        ROOT / "outputs/task1/final/quality/environment_kitchen/render_00.png",
    ]
    image_grid(
        paths,
        OUT / "task1_independent_quality.png",
        cols=2,
        cell=(660, 480),
        labels=["A: registered-camera render", "B: textured-mesh turntable",
                "C: v5 concave rear view", "Kitchen: standalone 2DGS render"],
    )


def read_obj(path: Path, max_points: int = 180_000) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            values = [float(value) for value in line.split()[1:]]
            vertices.append(values[:3])
            if len(values) >= 6:
                colors.append(values[3:6])
    xyz = np.asarray(vertices, dtype=np.float32)
    if colors and len(colors) == len(vertices):
        rgb = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    else:
        height = xyz[:, 1]
        height = (height - height.min()) / max(float(np.ptp(height)), 1e-6)
        rgb = plt.cm.viridis(height)[:, :3]
    if len(xyz) > max_points:
        rng = np.random.default_rng(23300200022)
        keep = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb


def project(xyz: np.ndarray, azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = xyz - np.median(xyz, axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(centered, axis=1), 97)
    centered /= max(float(scale), 1e-6)
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    rx = np.array(
        [[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]],
        dtype=np.float32,
    )
    rotated = centered @ rz.T @ rx.T
    return rotated[:, 0], rotated[:, 2], rotated[:, 1]


def render_obj(path: Path, output: Path, azimuth: float, elevation: float) -> None:
    xyz, rgb = read_obj(path)
    px, py, depth = project(xyz, azimuth, elevation)
    order = np.argsort(depth)
    fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=220)
    ax.scatter(px[order], py[order], c=rgb[order], s=0.35, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_generated_assets() -> None:
    b_projection = (
        ROOT
        / "outputs/task1/final/objects/object_b/training/train/save/it10000-0.png"
    )
    c_projection = ROOT / "outputs/task1/final/quality/object_c_turntable/frames/00045.jpg"
    images = [
        Image.open(b_projection),
        Image.open(ROOT / "data/object_C.jpg"),
        trim_white(Image.open(c_projection)),
    ]
    image_grid_from_images(
        images,
        OUT / "task1_generated_assets.png",
        cols=3,
        cell=(500, 390),
        labels=["B: threestudio/SDS mesh", "C: provided image", "C: Magic123 mesh"],
    )


def make_task1_b_training_curve() -> None:
    path = (
        ROOT
        / "outputs/task1/final/objects/object_b/training/train/csv_logs/version_0/metrics.csv"
    )
    data = pd.read_csv(path)
    data = data[data["train/loss_sds"].notna()].copy()
    steps = data["step"].to_numpy(dtype=np.float64)
    loss = data["train/loss_sds"].to_numpy(dtype=np.float64)
    grad = data["train/grad_norm"].to_numpy(dtype=np.float64)
    keep = np.isfinite(steps) & np.isfinite(loss)
    steps, loss, grad = steps[keep], loss[keep], grad[keep]
    order = np.argsort(steps)
    steps, loss, grad = steps[order], loss[order], grad[order]

    window = min(101, max(5, len(loss) // 40))
    fig, left = plt.subplots(figsize=(7.4, 3.5), dpi=220)
    left.plot(steps, moving_average(loss, window), color="#1F5A94", linewidth=1.3)
    left.set_xlabel("SDS optimization step")
    left.set_ylabel("SDS loss (moving average)", color="#1F5A94")
    left.tick_params(axis="y", labelcolor="#1F5A94")
    left.grid(alpha=0.22)
    right = left.twinx()
    right.plot(
        steps,
        moving_average(np.clip(grad, 0, np.nanpercentile(grad, 98)), window),
        color="#B24C32",
        linewidth=1.0,
        alpha=0.8,
    )
    right.set_ylabel("Gradient norm (98% clipped)", color="#B24C32")
    right.tick_params(axis="y", labelcolor="#B24C32")
    fig.tight_layout()
    fig.savefig(OUT / "task1_b_training_curve.png", bbox_inches="tight")
    plt.close(fig)


def make_task1_b_progress() -> None:
    save_dir = ROOT / "outputs/task1/final/objects/object_b/training/train/save"
    candidates = [800, 2800, 5000, 7600, 10000]
    paths = [save_dir / f"it{step}-0.png" for step in candidates]
    image_grid(
        paths,
        OUT / "task1_b_progress.png",
        cols=5,
        cell=(360, 250),
        labels=[f"{step} steps" for step in candidates],
    )


def make_task1_c_versions() -> None:
    evidence = ROOT / "outputs/task1/experiments/failure_evidence/object_c_v6"
    images = [
        Image.open(ROOT / "outputs/task1/final/quality/object_c_turntable/frames/00000.jpg"),
        Image.open(ROOT / "outputs/task1/final/quality/object_c_turntable/frames/00060.jpg"),
        crop_contact_cell(evidence / "object_c_v6_fine_ep0050_lambertian.jpg", 8, 13, 0),
        crop_contact_cell(evidence / "object_c_v6_fine_ep0050_lambertian.jpg", 8, 13, 50),
    ]
    image_grid_from_images(
        images,
        OUT / "task1_c_versions.png",
        cols=2,
        cell=(620, 500),
        labels=["v5 fine: front (final)", "v5 fine: rear", "v6 fine: front", "v6 fine: rear (rejected)"],
    )


def make_task1_fusion_evolution() -> None:
    evidence = ROOT / "outputs/task1/experiments/failure_evidence/fusion"
    paths = [
        evidence / "pre_refactor_20260611_1715_contact.jpg",
        evidence / "surface_aligned_preview_contact.jpg",
        ROOT / "outputs/task1/final/scene/fused_scene_360_contact.jpg",
    ]
    image_grid(
        paths,
        OUT / "task1_fusion_evolution.png",
        cols=3,
        cell=(480, 320),
        labels=["Legacy insertion", "Surface-aligned preview", "Final 360-frame fusion"],
    )


def make_scene_contact() -> None:
    source = ROOT / "outputs/task1/final/scene/fused_scene_360_contact.jpg"
    if not source.exists():
        source = ROOT / "outputs/task1/experiments/failure_evidence/fusion/surface_aligned_preview_contact.jpg"
    contain(source, (1280, 720)).save(OUT / "task1_scene_contact.jpg", quality=96)


def parse_training_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern = re.compile(r"step:\S+ .*?loss:([0-9.]+).*?lr:([0-9.eE+-]+)")
    steps: list[float] = []
    losses: list[float] = []
    lrs: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        steps.append(float((len(steps) + 1) * 20))
        losses.append(float(match.group(1)))
        lrs.append(float(match.group(2)))
    return np.asarray(steps), np.asarray(losses), np.asarray(lrs)


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.pad(smoothed, (window - 1, 0), mode="edge")


def make_task2_training_curve() -> None:
    logs = {
        "B-only": ROOT / "logs/task2_fair_b_10k_cosine.log",
        "A+B+C": ROOT / "logs/task2_fair_abc_10k_cosine.log",
    }
    missing = [path.name for path in logs.values() if not path.exists()]
    if missing:
        fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=220)
        ax.text(0.5, 0.5, f"missing training log\n{', '.join(missing)}", ha="center", va="center")
        ax.axis("off")
        fig.savefig(OUT / "task2_training_curve.png", bbox_inches="tight")
        plt.close(fig)
        return
    colors = {"B-only": "#8C8C8C", "A+B+C": "#276B9A"}
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=220)
    for label, path in logs.items():
        steps, losses, _ = parse_training_log(path)
        ax.plot(
            steps,
            moving_average(losses, window=15),
            color=colors[label],
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Smoothed ACT training loss")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "task2_training_curve.png", bbox_inches="tight")
    plt.close(fig)


def make_task2_evaluation() -> None:
    metrics_path = ROOT / "outputs/task2/zero_shot_d_action_error_full/metrics.json"
    if not metrics_path.exists():
        metrics_path = ROOT / "outputs/task2/experiments/eval_d_offline_scheduler/metrics.json"
    metrics = json.loads(metrics_path.read_text())["results"]
    keys = ["b_only_fair_10k", "abc_fair_10k"]
    labels_map = {
        "b_only_fair_10k": "B only\n10k",
        "abc_fair_10k": "A+B+C\n10k",
    }
    labels = [labels_map[key] for key in keys]
    values = [metrics[key]["loss"] for key in keys]
    l1_values = [metrics[key]["l1_loss"] for key in keys]
    colors_map = {
        "b_only_fair_10k": "#8C8C8C",
        "abc_fair_10k": "#276B9A",
    }
    colors = [colors_map[key] for key in keys]

    fig, ax = plt.subplots(figsize=(6.2, 3.7), dpi=220)
    positions = np.arange(len(keys))
    ax.bar(positions, values, width=0.66, color=colors, label="Total ACT loss")
    ax.plot(positions, l1_values, color="#222222", marker="o", linewidth=1.2, label="Action L1")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Held-out D teacher-forced loss")
    ax.set_ylim(max(0.0, min(values + l1_values) * 0.85), max(values + l1_values) * 1.18)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    for x, value in zip(positions, values):
        ax.text(x, value + 0.004, f"{value:.4f}", ha="center", fontsize=8)
    ax.text(
        0.99,
        0.97,
        "92,274 held-out D examples\nNot simulator rollout success",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(OUT / "task2_offline_eval.png", bbox_inches="tight")
    plt.close(fig)


def copy_task2_robustness_figures() -> None:
    source_dir = ROOT / "outputs" / "task2" / "action_chunking_robustness"
    files = {
        "chunk_horizon_analysis.png": "task2_chunk_horizon.png",
        "visual_perturbation_degradation.png": "task2_visual_shifts.png",
        "horizon_shift_heatmap.png": "task2_horizon_shift_heatmap.png",
        "action_dimension_error.png": "task2_action_dimensions.png",
        "robustness_summary.png": "task2_robustness_summary.png",
        "camera_ablation.png": "task2_camera_ablation.png",
        "visual_distribution_histograms.png": "task2_visual_histograms.png",
    }
    for source_name, destination_name in files.items():
        source = source_dir / source_name
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Run scripts/plot_task2_action_chunking.py first."
            )
        shutil.copy2(source, OUT / destination_name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_object_a_views()
    make_task1_a_training_curve()
    make_generated_assets()
    make_task1_b_training_curve()
    make_task1_b_progress()
    make_task1_c_versions()
    make_task1_c_training_curve()
    make_task1_quality_overview()
    make_task1_fusion_evolution()
    make_scene_contact()
    make_task2_training_curve()
    make_task2_evaluation()
    copy_task2_robustness_figures()


if __name__ == "__main__":
    main()
