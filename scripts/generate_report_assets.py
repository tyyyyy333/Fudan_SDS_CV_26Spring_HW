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


def save_fig(path):
    plt.tight_layout()
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


def draw_box(draw, xy, text, fill, outline=INK, radius=14):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=3)
    x0, y0, x1, y1 = xy
    draw.multiline_text((x0 + 16, y0 + 14), text, fill=INK, spacing=6)


def draw_arrow(draw, start, end, fill=INK, width=5):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    dx = ex - sx
    dy = ey - sy
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        p1 = (ex - 14 * sign, ey - 9)
        p2 = (ex - 14 * sign, ey + 9)
    else:
        sign = 1 if dy >= 0 else -1
        p1 = (ex - 9, ey - 14 * sign)
        p2 = (ex + 9, ey - 14 * sign)
    draw.polygon([end, p1, p2], fill=fill)


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


def task1_model_structure():
    canvas = Image.new("RGB", (1600, 520), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), "Task 1  Backbone Variants", fill=INK)

    boxes = [
        ((40, 90, 500, 430), "Baseline / SE / CBAM\n\nInput 256x256\nStem Conv7x7\nResNet34 stages\n[3, 4, 6, 3]\nGlobal AvgPool\nDropout + FC37\n\nSE  add channel squeeze-excitation\nCBAM  add channel + spatial attention", "#dbe7f2"),
        ((570, 90, 1030, 430), "Swin-Tiny\n\nPatch4 embedding\nWindow MSA blocks\nHierarchical stages\nPatch merging\nGlobal pooling\nLinear classifier 37\n\nPretrained on ImageNet", "#efe3d2"),
        ((1100, 90, 1560, 430), "Fine-tuning policy\n\nBackbone LR 5e-5\nHead LR 5e-4\nAdamW\nWarmup + Cosine\nMixup + CutMix\nTTA horizontal flip\nOptuna tunes LR / freeze / epochs", "#dfeadf"),
    ]
    for xy, text, fill in boxes:
        draw_box(draw, xy, text, fill)
    draw_arrow(draw, (500, 260), (570, 260))
    draw_arrow(draw, (1030, 260), (1100, 260))
    canvas.save(ASSET_DIR / "task1_model_structure.png")


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
    ax1.set_title("Task 1  Tuning Trials")
    annotate_bars(ax1, bars, "{:.1f}", 0.08)

    ax2 = ax1.twinx()
    ax2.plot(labels, head_lr, color=RED, marker="o", markersize=6, linewidth=2.5, label="Head LR")
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
    ax1.legend(handles=epoch_handles + freeze_handles + lr_handle, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))

    save_fig(ASSET_DIR / "task1_tune_trials.png")


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


def task2_model_structure():
    canvas = Image.new("RGB", (1700, 500), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), "Task 2  Detection and Tracking Pipeline", fill=INK)
    boxes = [
        ((40, 130, 300, 370), "VisDrone train/val\n+\ndemo.mp4", "#dbe7f2"),
        ((370, 130, 660, 370), "YOLOv8m detector\n\n896 input\nSGD\n120 epochs\nPredict boxes + classes", "#efe3d2"),
        ((730, 130, 1020, 370), "BoT-SORT tracker\n\nKalman prediction\nIoU matching\nReID appearance\nTrack IDs", "#dfeadf"),
        ((1090, 130, 1380, 370), "Scene analysis\n\nCrossing events\nID switch / lost\nFrame records\nTransition windows", "#f2dede"),
        ((1450, 130, 1660, 370), "Outputs\n\ntracked.mp4\ncrossing_frames\nocclusion_frames\nsummary.json", "#e3e0f3"),
    ]
    for xy, text, fill in boxes:
        draw_box(draw, xy, text, fill)
    for x0, x1 in [(300, 370), (660, 730), (1020, 1090), (1380, 1450)]:
        draw_arrow(draw, (x0, 250), (x1, 250))
    canvas.save(ASSET_DIR / "task2_model_structure.png")


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
    ax.legend(frameon=False, ncol=3, loc="upper left")
    annotate_bars(ax, bars1, "{:.3f}", 0.01)
    annotate_bars(ax, bars2, "{:.3f}", 0.01)
    annotate_bars(ax, bars3, "{:.3f}", 0.01)
    save_fig(ASSET_DIR / "task3_loss_comparison.png")


def task3_model_structure():
    canvas = Image.new("RGB", (1760, 540), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), "Task 3  U-Net Architecture", fill=INK)
    boxes = [
        ((40, 140, 240, 380), "Input\n3 x 256 x 256", "#dbe7f2"),
        ((300, 90, 540, 430), "Encoder\n\nDoubleConv 64\nDown 128\nDown 256\nDown 512", "#efe3d2"),
        ((620, 150, 840, 370), "Bottleneck\n\nDown 1024\nDropout 0.2", "#f2dede"),
        ((920, 90, 1160, 430), "Decoder\n\nUp 512\nUp 256\nUp 128\nUp 64", "#dfeadf"),
        ((1240, 140, 1460, 380), "Output\n1x1 Conv\n3 classes", "#e3e0f3"),
        ((1510, 110, 1720, 410), "Skip connections\n\nConcat encoder\nfeatures at each scale\nfor boundary recovery", "#f5eac8"),
    ]
    for xy, text, fill in boxes:
        draw_box(draw, xy, text, fill)
    for x0, x1 in [(240, 300), (540, 620), (840, 920), (1160, 1240)]:
        draw_arrow(draw, (x0, 260), (x1, 260))
    draw_arrow(draw, (540, 160), (920, 160))
    draw_arrow(draw, (540, 230), (920, 230))
    draw_arrow(draw, (540, 300), (920, 300))
    draw_arrow(draw, (540, 370), (920, 370))
    draw_arrow(draw, (1460, 260), (1510, 260))
    canvas.save(ASSET_DIR / "task3_model_structure.png")


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
    task1_model_structure()
    task1_model_comparison()
    task1_history()
    task1_tune_trials()
    task2_model_structure()
    task2_detection_metrics()
    task2_training_curves()
    task2_tracking_summary()
    task2_visual_panel()
    task3_model_structure()
    task3_loss_comparison()
    task3_training_curves()
    task3_visual_panel()
    print("report assets generated")


if __name__ == "__main__":
    main()
