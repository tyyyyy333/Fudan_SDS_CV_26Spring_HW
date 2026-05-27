import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / ".vendor"
if VENDOR_ROOT.exists():
    sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.cli import print_json
from hw2_cv.task1.data import NUM_CLASSES, build_dataloaders
from hw2_cv.task1.engine import evaluate
from hw2_cv.task1.models import build_model
from hw2_cv.utils import save_json


def parse_args():
    parser = argparse.ArgumentParser("Evaluate a Task 1 checkpoint with optional TTA.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tta-horizontal-flip", action="store_true")
    parser.add_argument(
        "--tta-mode",
        choices=["none", "hflip", "scale_hflip"],
        default="none",
        help="TTA policy. scale_hflip averages scales 0.875/1.0/1.125 with horizontal flips.",
    )
    parser.add_argument("--ttt", action="store_true", help="Run unlabeled entropy-minimization TTT.")
    parser.add_argument("--ttt-steps", type=int, default=1)
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--ttt-episodic", action="store_true", help="Reset the model before every batch.")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _center_crop_or_pad(images, size):
    _, _, height, width = images.shape
    if height < size or width < size:
        pad_h = max(size - height, 0)
        pad_w = max(size - width, 0)
        images = F.pad(
            images,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            mode="reflect",
        )
        _, _, height, width = images.shape
    top = (height - size) // 2
    left = (width - size) // 2
    return images[:, :, top : top + size, left : left + size]


def _scale_batch(images, scale):
    if scale == 1.0:
        return images
    size = images.shape[-1]
    scaled_size = max(int(round(size * scale)), 1)
    scaled = F.interpolate(
        images,
        size=(scaled_size, scaled_size),
        mode="bilinear",
        align_corners=False,
    )
    return _center_crop_or_pad(scaled, size)


def _tta_views(images, mode):
    if mode == "none":
        return [images]
    if mode == "hflip":
        return [images, torch.flip(images, dims=[3])]
    if mode == "scale_hflip":
        views = []
        for scale in (0.875, 1.0, 1.125):
            scaled = _scale_batch(images, scale)
            views.append(scaled)
            views.append(torch.flip(scaled, dims=[3]))
        return views
    raise ValueError(f"Unsupported TTA mode: {mode}")


def _predict_with_tta(model, images, mode):
    logits = None
    for view in _tta_views(images, mode):
        view_logits = model(view)
        logits = view_logits if logits is None else logits + view_logits
    return logits / len(_tta_views(images, mode))


def _entropy_loss(logits):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean()


def _configure_ttt_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapt_params = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()
            module.requires_grad_(True)
            if module.weight is not None:
                adapt_params.append(module.weight)
            if module.bias is not None:
                adapt_params.append(module.bias)
    return adapt_params


@torch.no_grad()
def _evaluate_logits(logits, targets, criterion):
    loss = criterion(logits, targets)
    probabilities = F.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    return loss, predictions


def evaluate_with_ttt(model, loader, criterion, device, tta_mode, steps, lr, episodic, initial_state):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    adapt_params = _configure_ttt_model(model)
    optimizer = torch.optim.Adam(adapt_params, lr=lr) if adapt_params else None
    progress = tqdm(loader, desc=f"test-ttt-{tta_mode}", leave=False)

    for batch in progress:
        if episodic:
            model.load_state_dict(initial_state)
            adapt_params = _configure_ttt_model(model)
            optimizer = torch.optim.Adam(adapt_params, lr=lr) if adapt_params else None

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if optimizer is not None:
            model.eval()
            _configure_ttt_model(model)
            for _ in range(max(steps, 0)):
                optimizer.zero_grad(set_to_none=True)
                logits = _predict_with_tta(model, images, tta_mode)
                loss = _entropy_loss(logits)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = _predict_with_tta(model, images, tta_mode)
            loss, predictions = _evaluate_logits(logits, targets, criterion)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == targets).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    dataloaders = build_dataloaders(config)
    model = build_model(config, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["train"].get("label_smoothing", 0.0))
    ).to(device)

    tta_mode = args.tta_mode
    if args.tta_horizontal_flip:
        tta_mode = "hflip"

    if args.ttt:
        initial_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        metrics = evaluate_with_ttt(
            model=model,
            loader=dataloaders["test_loader"],
            criterion=criterion,
            device=device,
            tta_mode=tta_mode,
            steps=args.ttt_steps,
            lr=args.ttt_lr,
            episodic=args.ttt_episodic,
            initial_state=initial_state,
        )
    elif tta_mode == "scale_hflip":
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress = tqdm(dataloaders["test_loader"], desc="test-scale-hflip", leave=False)
        with torch.no_grad():
            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                logits = _predict_with_tta(model, images, tta_mode)
                loss, predictions = _evaluate_logits(logits, targets, criterion)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (predictions == targets).sum().item()
                total_samples += batch_size
        metrics = {
            "loss": total_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
        }
    else:
        metrics = evaluate(
            model=model,
            loader=dataloaders["test_loader"],
            criterion=criterion,
            device=device,
            amp=bool(config["train"].get("amp", True)),
            stage="test-tta" if tta_mode == "hflip" else "test",
            collect_outputs=False,
            tta_horizontal_flip=tta_mode == "hflip",
        )
    summary = {
        "checkpoint": str(checkpoint_path),
        "tta_mode": tta_mode,
        "ttt": bool(args.ttt),
        "ttt_steps": int(args.ttt_steps) if args.ttt else 0,
        "ttt_lr": float(args.ttt_lr) if args.ttt else 0.0,
        "ttt_episodic": bool(args.ttt_episodic) if args.ttt else False,
        "test_acc": metrics["acc"],
        "test_loss": metrics["loss"],
    }
    if args.output:
        save_json(summary, Path(args.output))
    print_json(summary)


if __name__ == "__main__":
    main()
