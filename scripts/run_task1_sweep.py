import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / ".vendor"
if VENDOR_ROOT.exists():
    sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.cli import load_run_config, parse_config_args, print_json, run_sweep
from hw2_cv.task1.train import run_training


DEFAULT_EXPERIMENTS = [
    {
        "tag": "baseline_pretrained_resnet34",
        "meta": {"group": "structure"},
        "model": {"family": "resnet", "variant": "resnet34", "pretrained": True},
    },
    {
        "tag": "scratch_resnet34_fair",
        "meta": {"group": "ablation"},
        "model": {"family": "resnet", "variant": "resnet34", "pretrained": False},
        "optimizer": {"head_lr": 0.0005, "backbone_lr": 0.0005},
        "train": {
            "epochs": 100,
            "freeze_backbone_epochs": 0,
            "label_smoothing": 0.0,
            "early_stopping": {"enabled": False},
        },
    },
    {
        "tag": "se_resnet34",
        "meta": {"group": "structure"},
        "model": {"family": "resnet", "variant": "se_resnet34", "pretrained": True},
    },
    {
        "tag": "cbam_resnet34",
        "meta": {"group": "structure"},
        "model": {"family": "resnet", "variant": "cbam_resnet34", "pretrained": True},
    },
    {
        "tag": "swin_tiny",
        "meta": {"group": "structure"},
        "model": {"family": "timm", "variant": "swin_tiny_patch4_window7_224", "pretrained": True},
        "optimizer": {"head_lr": 0.0005, "backbone_lr": 0.00005},
    },
    {
        "tag": "se_resnet34_epoch50",
        "meta": {"group": "hyperparam_epoch"},
        "model": {"family": "resnet", "variant": "se_resnet34", "pretrained": True},
        "train": {
            "epochs": 50,
            "freeze_backbone_epochs": 1,
            "early_stopping": {"enabled": False},
        },
    },
    {
        "tag": "se_resnet34_epoch100",
        "meta": {"group": "hyperparam_epoch"},
        "model": {"family": "resnet", "variant": "se_resnet34", "pretrained": True},
        "train": {
            "epochs": 100,
            "freeze_backbone_epochs": 1,
            "early_stopping": {"enabled": False},
        },
    },
    {
        "tag": "se_resnet34_lr_low",
        "meta": {"group": "hyperparam_lr"},
        "model": {"family": "resnet", "variant": "se_resnet34", "pretrained": True},
        "optimizer": {"head_lr": 0.0005, "backbone_lr": 0.00005},
        "train": {"early_stopping": {"enabled": False}},
    },
    {
        "tag": "se_resnet34_lr_high",
        "meta": {"group": "hyperparam_lr"},
        "model": {"family": "resnet", "variant": "se_resnet34", "pretrained": True},
        "optimizer": {"head_lr": 0.002, "backbone_lr": 0.0002},
        "train": {"early_stopping": {"enabled": False}},
    },
]


def parse_args():
    return parse_config_args("Run Task 1 sweep experiments.", "Base YAML config.")


def main():
    args = parse_args()
    base_config = load_run_config(args.config)
    all_summaries = run_sweep(base_config, DEFAULT_EXPERIMENTS, run_training)
    print_json(all_summaries)


if __name__ == "__main__":
    main()
