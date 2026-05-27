import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / ".vendor"
if VENDOR_ROOT.exists():
    sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CLASS_NAMES = ("pet", "background", "border")
NUM_CLASSES = 3


def _load_project_runtime():
    from hw2_cv.cli import load_run_config, print_json
    from hw2_cv.runner import autocast_context
    from hw2_cv.task3.data import CLASS_NAMES as data_class_names
    from hw2_cv.task3.data import NUM_CLASSES as data_num_classes
    from hw2_cv.task3.data import build_dataloaders
    from hw2_cv.task3.metrics import SegmentationMetric
    from hw2_cv.task3.models import build_model
    from hw2_cv.task3.visualize import _denormalize, colorize_mask, overlay_mask
    from hw2_cv.utils import ensure_dir, save_json

    return {
        "load_run_config": load_run_config,
        "print_json": print_json,
        "autocast_context": autocast_context,
        "class_names": data_class_names,
        "num_classes": data_num_classes,
        "build_dataloaders": build_dataloaders,
        "SegmentationMetric": SegmentationMetric,
        "build_model": build_model,
        "_denormalize": _denormalize,
        "colorize_mask": colorize_mask,
        "overlay_mask": overlay_mask,
        "ensure_dir": ensure_dir,
        "save_json": save_json,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate gradient-guided anchor refinement for Task 3 segmentation."
    )
    parser.add_argument("--config", type=str, default="configs/task3_unet.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/task3/ce_dice/best.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/task3_anchor_refine")
    parser.add_argument("--conf-threshold", type=float, default=0.90)
    parser.add_argument("--grad-threshold", type=float, default=0.12)
    parser.add_argument("--boundary-grad-threshold", type=float, default=0.20)
    parser.add_argument("--uncertain-threshold", type=float, default=0.75)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--diffusion-strength", type=float, default=0.55)
    parser.add_argument("--fusion-weight", type=float, default=0.65)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--max-export", type=int, default=12)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def _to_gray01(images, mean, std):
    mean_tensor = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _gradient_magnitude(images, mean, std):
    gray = _to_gray01(images, mean=mean, std=std)
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=images.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=images.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    flat = grad.flatten(1)
    max_per_image = flat.amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return grad / max_per_image


def refine_probabilities(
    probs,
    images,
    mean,
    std,
    conf_threshold,
    grad_threshold,
    boundary_grad_threshold,
    uncertain_threshold,
    iterations,
    diffusion_strength,
    fusion_weight,
    kernel_size,
):
    confidence = probs.max(dim=1, keepdim=True).values
    grad = _gradient_magnitude(images, mean=mean, std=std)

    anchors = (confidence >= conf_threshold) & (grad <= grad_threshold)
    refine_region = (confidence <= uncertain_threshold) | (grad >= boundary_grad_threshold)

    refined = probs.clone()
    padding = kernel_size // 2
    for _ in range(iterations):
        local_mean = F.avg_pool2d(refined, kernel_size=kernel_size, stride=1, padding=padding)
        proposal = (1.0 - diffusion_strength) * refined + diffusion_strength * local_mean
        refined = torch.where(anchors, probs, proposal)
        refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)

    fused = fusion_weight * probs + (1.0 - fusion_weight) * refined
    fused = torch.where(refine_region, fused, probs)
    return fused / fused.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _metric_dict(metric):
    return {
        "miou": metric.mean_iou(),
        "pixel_acc": metric.pixel_accuracy(),
        "classwise_iou": {
            name: value for name, value in zip(CLASS_NAMES, metric.classwise_iou())
        },
    }


def _export_case(output_dir, image_id, image, target_mask, baseline_mask, refined_mask, mean, std):
    ensure_dir(output_dir)
    image_np = _denormalize(image, mean=mean, std=std)
    target_np = colorize_mask(target_mask)
    baseline_np = colorize_mask(baseline_mask)
    refined_np = colorize_mask(refined_mask)
    baseline_overlay = overlay_mask(image_np, baseline_np)
    refined_overlay = overlay_mask(image_np, refined_np)

    panels = [image_np, target_np, baseline_overlay, refined_overlay]
    labels = ["input", "gt", "baseline", "refined"]
    top = np.full((28, image_np.shape[1] * 4, 3), 255, dtype=np.uint8)
    for idx, label in enumerate(labels):
        cv2.putText(
            top,
            label,
            (idx * image_np.shape[1] + 8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 45, 70),
            1,
            cv2.LINE_AA,
        )
    grid = np.concatenate(panels, axis=1)
    canvas = np.concatenate([top, grid], axis=0)
    path = output_dir / f"{image_id}_anchor_refine_compare.png"
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return str(path)


@torch.no_grad()
def evaluate_anchor_refinement(model, loader, config, args, device):
    mean = tuple(config["data"]["normalize"]["mean"])
    std = tuple(config["data"]["normalize"]["std"])
    baseline_metric = SegmentationMetric(num_classes=NUM_CLASSES)
    refined_metric = SegmentationMetric(num_classes=NUM_CLASSES)
    exported = []
    export_dir = ensure_dir(Path(args.output_dir) / "prediction_exports")

    model.eval()
    progress = tqdm(loader, desc="anchor-refine-test", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast_context(device, args.amp):
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            refined_probs = refine_probabilities(
                probs=probs,
                images=images,
                mean=mean,
                std=std,
                conf_threshold=args.conf_threshold,
                grad_threshold=args.grad_threshold,
                boundary_grad_threshold=args.boundary_grad_threshold,
                uncertain_threshold=args.uncertain_threshold,
                iterations=args.iterations,
                diffusion_strength=args.diffusion_strength,
                fusion_weight=args.fusion_weight,
                kernel_size=args.kernel_size,
            )

        baseline_metric.update(probs, masks)
        refined_metric.update(refined_probs, masks)

        if len(exported) < args.max_export:
            baseline_masks = probs.argmax(dim=1).detach().cpu()
            refined_masks = refined_probs.argmax(dim=1).detach().cpu()
            images_cpu = images.detach().cpu()
            masks_cpu = masks.detach().cpu()
            for position, image_id in enumerate(batch["image_id"]):
                if len(exported) >= args.max_export:
                    break
                exported.append(
                    _export_case(
                        output_dir=export_dir,
                        image_id=image_id,
                        image=images_cpu[position],
                        target_mask=masks_cpu[position],
                        baseline_mask=baseline_masks[position],
                        refined_mask=refined_masks[position],
                        mean=mean,
                        std=std,
                    )
                )

    baseline = _metric_dict(baseline_metric)
    refined = _metric_dict(refined_metric)
    delta = {
        "miou": refined["miou"] - baseline["miou"],
        "pixel_acc": refined["pixel_acc"] - baseline["pixel_acc"],
        "classwise_iou": {
            name: refined["classwise_iou"][name] - baseline["classwise_iou"][name]
            for name in CLASS_NAMES
        },
    }
    return baseline, refined, delta, exported


def main():
    global CLASS_NAMES, NUM_CLASSES
    global SegmentationMetric, _denormalize, colorize_mask, overlay_mask
    global autocast_context, ensure_dir, save_json

    args = parse_args()
    runtime = _load_project_runtime()
    load_run_config = runtime["load_run_config"]
    print_json = runtime["print_json"]
    build_dataloaders = runtime["build_dataloaders"]
    build_model = runtime["build_model"]
    CLASS_NAMES = runtime["class_names"]
    NUM_CLASSES = runtime["num_classes"]
    SegmentationMetric = runtime["SegmentationMetric"]
    _denormalize = runtime["_denormalize"]
    colorize_mask = runtime["colorize_mask"]
    overlay_mask = runtime["overlay_mask"]
    autocast_context = runtime["autocast_context"]
    ensure_dir = runtime["ensure_dir"]
    save_json = runtime["save_json"]

    config = load_run_config(args.config)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data = build_dataloaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = ensure_dir(args.output_dir)
    baseline, refined, delta, exported = evaluate_anchor_refinement(
        model=model,
        loader=data["test_loader"],
        config=config,
        args=args,
        device=device,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "parameters": {
            "conf_threshold": args.conf_threshold,
            "grad_threshold": args.grad_threshold,
            "boundary_grad_threshold": args.boundary_grad_threshold,
            "uncertain_threshold": args.uncertain_threshold,
            "iterations": args.iterations,
            "diffusion_strength": args.diffusion_strength,
            "fusion_weight": args.fusion_weight,
            "kernel_size": args.kernel_size,
        },
        "baseline": baseline,
        "refined": refined,
        "delta": delta,
        "exported_prediction_paths": exported,
    }
    save_json(summary, output_dir / "summary.json")
    print_json(summary)


if __name__ == "__main__":
    main()
