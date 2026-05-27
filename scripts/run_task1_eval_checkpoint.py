import argparse
import sys
from pathlib import Path

import torch
from torch import nn

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
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


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

    metrics = evaluate(
        model=model,
        loader=dataloaders["test_loader"],
        criterion=criterion,
        device=device,
        amp=bool(config["train"].get("amp", True)),
        stage="test-tta" if args.tta_horizontal_flip else "test",
        collect_outputs=False,
        tta_horizontal_flip=args.tta_horizontal_flip,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "tta_horizontal_flip": bool(args.tta_horizontal_flip),
        "test_acc": metrics["acc"],
        "test_loss": metrics["loss"],
    }
    if args.output:
        save_json(summary, Path(args.output))
    print_json(summary)


if __name__ == "__main__":
    main()
