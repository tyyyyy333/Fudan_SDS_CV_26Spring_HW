import ast
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "report_assets"
ASSET_DIR.mkdir(exist_ok=True)

BG = "#f6f3ee"
INK = "#1f2a33"
GRID = "#cfc6ba"
BLUE = "#355c7d"
SAND = "#cfa56a"
GREEN = "#4c7a62"
RED = "#9a4f5a"
SLATE = "#97a9b9"
WHITE = "#fffdf9"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(path, rect=None):
    plt.tight_layout(rect=rect)
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def annotate_bars(ax, bars, fmt="{:.2f}", yoff=0.5):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + yoff,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "fc": WHITE, "ec": "none", "alpha": 0.9},
        )


def style_axis(ax):
    ax.grid(axis="y", color=GRID, alpha=0.5, linewidth=0.9)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=11)


def task1_model_comparison():
    runs = {
        "Baseline": load_json(ROOT / "outputs/task1/baseline_pretrained_resnet34/summary.json"),
        "Scratch": load_json(ROOT / "outputs/task1/scratch_resnet34_fair/summary.json"),
        "SE": load_json(ROOT / "outputs/task1/se_resnet34/summary.json"),
        "CBAM": load_json(ROOT / "outputs/task1/cbam_resnet34/summary.json"),
        "Swin-T": load_json(ROOT / "outputs/task1/swin_tiny/summary.json"),
        "Tuned": load_json(ROOT / "outputs/task1/tuned_best/summary.json"),
    }
    labels = list(runs.keys())
    val = [100.0 * runs[k]["best_val_acc"] for k in labels]
    test = [100.0 * runs[k]["test_acc"] for k in labels]
    x = np.arange(len(labels))
    width = 0.36

    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars1 = ax.bar(x - width / 2, val, width, label="Best Val Acc", color=SLATE)
    bars2 = ax.bar(x + width / 2, test, width, label="Test Acc", color=BLUE)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Task 1  Structure Comparison")
    ax.set_xticks(x, labels)
    ax.set_ylim(50, 100)
    style_axis(ax)
    annotate_bars(ax, bars1, "{:.1f}", 0.5)
    annotate_bars(ax, bars2, "{:.1f}", 0.5)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save_fig(ASSET_DIR / "task1_model_comparison.png")


def task1_history():
    base_hist = load_json(ROOT / "outputs/task1/baseline_pretrained_resnet34/history.json")
    swin_hist = load_json(ROOT / "outputs/task1/swin_tiny/history.json")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=False)

    base_epoch = [r["epoch"] for r in base_hist]
    swin_epoch = [r["epoch"] for r in swin_hist]
    axes[0].plot(base_epoch, [r["train_loss"] for r in base_hist], color=SLATE, linewidth=2.6, label="Baseline Train")
    axes[0].plot(base_epoch, [r["val_loss"] for r in base_hist], color=BLUE, linewidth=2.6, label="Baseline Val")
    axes[0].plot(swin_epoch, [r["train_loss"] for r in swin_hist], color=SAND, linewidth=2.6, label="Swin Train")
    axes[0].plot(swin_epoch, [r["val_loss"] for r in swin_hist], color=RED, linewidth=2.6, label="Swin Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    style_axis(axes[0])

    axes[1].plot(base_epoch, [100.0 * r["train_acc"] for r in base_hist], color=SLATE, linewidth=2.6, label="Baseline Train")
    axes[1].plot(base_epoch, [100.0 * r["val_acc"] for r in base_hist], color=BLUE, linewidth=2.6, label="Baseline Val")
    axes[1].plot(swin_epoch, [100.0 * r["train_acc"] for r in swin_hist], color=SAND, linewidth=2.6, label="Swin Train")
    axes[1].plot(swin_epoch, [100.0 * r["val_acc"] for r in swin_hist], color=RED, linewidth=2.6, label="Swin Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    style_axis(axes[1])

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Task 1  Training Curves", y=1.02, fontsize=14, fontweight="bold")
    save_fig(ASSET_DIR / "task1_training_curves.png")


def task1_tune_trials():
    trials = load_json(ROOT / "outputs/task1/tuning/trials_summary.json")
    complete = [trial for trial in trials if "COMPLETE" in trial["state"]]
    labels = [f"T{trial['number']}" for trial in complete]
    val = [100.0 * trial["best_val_acc"] for trial in complete]
    head_lr = [trial["params"]["head_lr"] for trial in complete]
    epochs = [trial["params"]["epochs"] for trial in complete]
    freeze = [trial["params"]["freeze_backbone_epochs"] for trial in complete]

    freeze_markers = {0: "o", 1: "s", 3: "^"}
    epoch_colors = {30: SLATE, 50: BLUE, 70: SAND, 100: RED}

    apply_style()
    fig, ax1 = plt.subplots(figsize=(11.2, 5.2))
    bar_colors = [epoch_colors.get(ep, GREEN) for ep in epochs]
    bars = ax1.bar(labels, val, color=bar_colors, width=0.62, edgecolor=INK, linewidth=0.8)
    ax1.set_ylabel("Best Val Acc (%)")
    ax1.set_ylim(92, 96)
    style_axis(ax1)
    fig.suptitle("Task 1  Tuning Trials", y=0.75, fontsize=14, fontweight="bold")
    annotate_bars(ax1, bars, "{:.1f}", 0.08)

    ax2 = ax1.twinx()
    ax2.plot(labels, head_lr, color=RED, marker="o", markersize=6, linewidth=2.5, alpha=0.5, label="Head LR")
    ax2.set_ylabel("Head LR")
    ax2.set_yscale("log")

    for bar, epoch, frz, lr in zip(bars, epochs, freeze, head_lr):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            92.15,
            f"ep {epoch}\nfreeze {frz}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax1.scatter(
            bar.get_x() + bar.get_width() / 2,
            min(95.85, bar.get_height() - 0.08),
            marker=freeze_markers.get(frz, "o"),
            s=58,
            color=WHITE,
            edgecolor=INK,
            linewidth=1.2,
            zorder=5,
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    epoch_handles = [Patch(facecolor=color, edgecolor=INK, label=f"epoch {ep}") for ep, color in epoch_colors.items()]
    freeze_handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor=WHITE, markeredgecolor=INK, markersize=8, label=f"freeze {frz}")
        for frz, marker in freeze_markers.items()
    ]
    lr_handle = [Line2D([0], [0], color=RED, linewidth=2.5, marker="o", label="Head LR")]
    ax1.legend(
        handles=epoch_handles + freeze_handles + lr_handle,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
    )

    save_fig(ASSET_DIR / "task1_tune_trials.png", rect=[0, 0, 1, 0.86])


def task2_detection_metrics():
    summary = load_json(ROOT / "outputs/task2/high_score/summary.json")
    results = summary["results"]
    marker = "results_dict: "
    start = results.index(marker) + len(marker)
    end = results.index("}\nsave_dir:") + 1
    metrics = ast.literal_eval(results[start:end])

    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    values = [
        metrics["metrics/precision(B)"],
        metrics["metrics/recall(B)"],
        metrics["metrics/mAP50(B)"],
        metrics["metrics/mAP50-95(B)"],
    ]

    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, values, color=[BLUE, GREEN, SAND, RED], width=0.6)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Score")
    ax.set_title("Task 2  Detector Metrics")
    style_axis(ax)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label="Precision"),
            Patch(facecolor=GREEN, label="Recall"),
            Patch(facecolor=SAND, label="mAP50"),
            Patch(facecolor=RED, label="mAP50-95"),
        ],
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    annotate_bars(ax, bars, "{:.3f}", 0.015)
    save_fig(ASSET_DIR / "task2_detector_metrics.png")


def task2_training_curves():
    rows = []
    path = ROOT / "runs/detect/outputs/task2/high_score_train/visdrone_yolov8m/results.csv"
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key.strip(): float(value) for key, value in row.items()})

    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"] for row in rows]
    val_loss = [row["val/box_loss"] + row["val/cls_loss"] + row["val/dfl_loss"] for row in rows]
    map50 = [row["metrics/mAP50(B)"] for row in rows]
    map5095 = [row["metrics/mAP50-95(B)"] for row in rows]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(epochs, train_loss, color=BLUE, linewidth=2.8, label="Train total loss")
    axes[0].plot(epochs, val_loss, color=RED, linewidth=2.8, label="Val total loss")
    axes[0].fill_between(epochs, train_loss, color=BLUE, alpha=0.08)
    axes[0].fill_between(epochs, val_loss, color=RED, alpha=0.08)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    style_axis(axes[0])
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(epochs, map50, color=GREEN, linewidth=2.8, label="Val mAP50")
    axes[1].plot(epochs, map5095, color=SAND, linewidth=2.8, label="Val mAP50-95")
    axes[1].fill_between(epochs, map50, color=GREEN, alpha=0.08)
    axes[1].fill_between(epochs, map5095, color=SAND, alpha=0.08)
    axes[1].set_title("Validation mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    style_axis(axes[1])
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Task 2  Training Curves", y=1.02, fontsize=14, fontweight="bold")
    save_fig(ASSET_DIR / "task2_training_curves.png")


def task2_tracking_summary():
    summary = load_json(ROOT / "outputs/task2/high_score_track/summary.json")
    occ = load_json(ROOT / "outputs/task2/high_score_track/occlusion_analysis.json")

    labels = ["Line Count", "Switch", "Lost", "Kept"]
    values = [summary["line_count"], occ["switch_count"], occ["lost_count"], occ["kept_count"]]

    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, values, color=[BLUE, RED, SAND, GREEN], width=0.6)
    ax.set_ylabel("Count")
    ax.set_title("Task 2  Tracking Summary")
    style_axis(ax)
    annotate_bars(ax, bars, "{:.0f}", 3)
    save_fig(ASSET_DIR / "task2_tracking_summary.png")


def task3_loss_comparison():
    runs = {
        "CE": load_json(ROOT / "outputs/task3/ce_only/summary.json"),
        "Dice": load_json(ROOT / "outputs/task3/dice_only/summary.json"),
        "CE + Dice": load_json(ROOT / "outputs/task3/ce_dice/summary.json"),
    }
    labels = list(runs.keys())
    test = [runs[k]["test_miou"] for k in labels]
    pixel = [runs[k]["test_pixel_acc"] for k in labels]
    border = [runs[k]["test_classwise_iou"][2] for k in labels]
    x = np.arange(len(labels))
    width = 0.24

    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    bars1 = ax.bar(x - width, test, width, label="Test mIoU", color=BLUE)
    bars2 = ax.bar(x, pixel, width, label="Pixel Acc", color=GREEN)
    bars3 = ax.bar(x + width, border, width, label="Border IoU", color=SAND)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Task 3  Loss Comparison")
    style_axis(ax)
    ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.15, 1))
    annotate_bars(ax, bars1, "{:.3f}", 0.01)
    annotate_bars(ax, bars2, "{:.3f}", 0.01)
    annotate_bars(ax, bars3, "{:.3f}", 0.01)
    save_fig(ASSET_DIR / "task3_loss_comparison.png")


def task3_training_curves():
    hist = load_json(ROOT / "outputs/task3/ce_dice/history.json")
    epochs = [row["epoch"] for row in hist]
    train_loss = [row["train_loss"] for row in hist]
    val_loss = [row["val_loss"] for row in hist]
    train_miou = [row["train_miou"] for row in hist]
    val_miou = [row["val_miou"] for row in hist]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(epochs, train_loss, color=BLUE, linewidth=2.8, label="Train loss")
    axes[0].plot(epochs, val_loss, color=RED, linewidth=2.8, label="Val loss")
    axes[0].fill_between(epochs, train_loss, color=BLUE, alpha=0.08)
    axes[0].fill_between(epochs, val_loss, color=RED, alpha=0.08)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    style_axis(axes[0])
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(epochs, train_miou, color=GREEN, linewidth=2.8, label="Train mIoU")
    axes[1].plot(epochs, val_miou, color=SAND, linewidth=2.8, label="Val mIoU")
    axes[1].fill_between(epochs, train_miou, color=GREEN, alpha=0.08)
    axes[1].fill_between(epochs, val_miou, color=SAND, alpha=0.08)
    axes[1].set_title("mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    style_axis(axes[1])
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Task 3  Training Curves", y=1.02, fontsize=14, fontweight="bold")
    save_fig(ASSET_DIR / "task3_training_curves.png")


def resize_and_crop(img, size):
    return img.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def labeled_panel(images, labels, out_path, title=None, cell_size=(420, 280), margin=22, label_h=34, title_h=0):
    n = len(images)
    canvas = Image.new("RGB", (margin * (n + 1) + cell_size[0] * n, margin * 2 + title_h + label_h + cell_size[1]), BG)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((margin, margin), title, fill=INK)
    top = margin + title_h
    for idx, (img_path, label) in enumerate(zip(images, labels)):
        x = margin + idx * (cell_size[0] + margin)
        draw.text((x, top), label, fill=INK)
        img = resize_and_crop(Image.open(img_path), cell_size)
        canvas.paste(img, (x, top + label_h))
        draw.rounded_rectangle(
            (x - 2, top + label_h - 2, x + cell_size[0] + 2, top + label_h + cell_size[1] + 2),
            radius=10,
            outline=GRID,
            width=3,
        )
    canvas.save(out_path)


def task2_visual_panel():
    images = [
        ROOT / "outputs/task2/high_score_track/crossing_frames/frame_000209_track_248_negative_to_positive.jpg",
        ROOT / "outputs/task2/high_score_track/occlusion_frames/frame_000078.jpg",
    ]
    labels = ["Crossing event", "ID transition window"]
    labeled_panel(images, labels, ASSET_DIR / "task2_visual_cases.png", "Task 2  Qualitative Results", title_h=16)


def task3_visual_panel():
    base = ROOT / "outputs/task3/ce_dice/prediction_exports"
    images = [
        base / "test_00000_input.png",
        base / "test_00000_gt.png",
        base / "test_00000_pred.png",
        base / "test_00000_overlay.png",
    ]
    labels = ["Input", "Ground Truth", "Prediction", "Overlay"]
    labeled_panel(images, labels, ASSET_DIR / "task3_visual_cases.png", "Task 3  Qualitative Results", title_h=16)


def main():
    task1_model_comparison()
    task1_history()
    task1_tune_trials()
    task2_detection_metrics()
    task2_training_curves()
    task2_tracking_summary()
    task2_visual_panel()
    task3_loss_comparison()
    task3_training_curves()
    task3_visual_panel()
    print("report assets generated")


if __name__ == "__main__":
    main()
