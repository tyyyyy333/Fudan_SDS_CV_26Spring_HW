#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / ".vendor"
if VENDOR_ROOT.exists():
    sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.pets import build_pet_source, detect_pet_source, split_trainval_indices
from hw2_cv.runner import autocast_context
from hw2_cv.task1.data import NUM_CLASSES as TASK1_NUM_CLASSES
from hw2_cv.task1.models import build_model as build_classifier
from hw2_cv.task3.data import NUM_CLASSES as TASK3_NUM_CLASSES
from hw2_cv.task3.losses import build_criterion as build_segmentation_criterion
from hw2_cv.task3.metrics import SegmentationMetric
from hw2_cv.task3.models import build_model as build_segmenter
from hw2_cv.utils import build_scheduler, ensure_dir, load_yaml, prepare_run, save_json


def parse_args():
    parser = argparse.ArgumentParser("Jointly train Task 3 segmentation-guided Task 1 classification.")
    parser.add_argument("--config", type=str, default="configs/task1_task3_joint.yaml")
    return parser.parse_args()


def build_joint_transform(image_size, augment, mean, std, augmentation_cfg):
    steps = [A.Resize(height=image_size, width=image_size)]
    if augment:
        hflip_prob = float(augmentation_cfg.get("horizontal_flip_prob", 0.5))
        if hflip_prob > 0:
            steps.append(A.HorizontalFlip(p=hflip_prob))
        affine_cfg = augmentation_cfg.get("shift_scale_rotate", {})
        if affine_cfg.get("enabled", False):
            steps.append(
                A.ShiftScaleRotate(
                    shift_limit=float(affine_cfg.get("shift_limit", 0.05)),
                    scale_limit=float(affine_cfg.get("scale_limit", 0.1)),
                    rotate_limit=int(affine_cfg.get("rotate_limit", 15)),
                    border_mode=0,
                    fill=float(affine_cfg.get("fill", 0)),
                    fill_mask=1,
                    p=float(affine_cfg.get("probability", 0.5)),
                )
            )
        color_cfg = augmentation_cfg.get("color_jitter", {})
        if color_cfg.get("enabled", False):
            steps.append(
                A.ColorJitter(
                    brightness=float(color_cfg.get("brightness", 0.2)),
                    contrast=float(color_cfg.get("contrast", 0.2)),
                    saturation=float(color_cfg.get("saturation", 0.2)),
                    hue=float(color_cfg.get("hue", 0.02)),
                    p=float(color_cfg.get("probability", 0.3)),
                )
            )
    steps.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(steps)


class JointPetDataset(Dataset):
    def __init__(self, source, indices, transform):
        self.source = source
        self.indices = indices if indices is not None else list(range(len(source)))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        image = self.source.get_image(base_index)
        mask = self.source.get_mask_array(base_index)
        transformed = self.transform(image=np.array(image), mask=mask)
        return {
            "image": transformed["image"],
            "target": int(self.source.get_label(base_index)),
            "mask": transformed["mask"].long().clamp_(0, TASK3_NUM_CLASSES - 1),
            "index": int(base_index),
            "image_id": self.source.get_image_id(base_index),
        }


def build_joint_dataloaders(config):
    data_cfg = config["data"]
    mean = tuple(data_cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225]))
    train_source = build_pet_source(
        root=data_cfg["root"],
        split="trainval",
        target="segmentation",
        download=data_cfg.get("download", False),
    )
    test_source = build_pet_source(
        root=data_cfg["root"],
        split="test",
        target="segmentation",
        download=data_cfg.get("download", False),
    )
    train_indices, val_indices = split_trainval_indices(
        root=data_cfg["root"],
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(config.get("seed", 42)),
        download=data_cfg.get("download", False),
    )
    train_transform = build_joint_transform(
        int(data_cfg["image_size"]),
        True,
        mean,
        std,
        data_cfg.get("augmentation", {}),
    )
    eval_transform = build_joint_transform(
        int(data_cfg["image_size"]),
        False,
        mean,
        std,
        data_cfg.get("augmentation", {}),
    )
    loader_kwargs = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(data_cfg["num_workers"]) > 0,
    }
    return {
        "train_loader": DataLoader(JointPetDataset(train_source, train_indices, train_transform), shuffle=True, **loader_kwargs),
        "val_loader": DataLoader(JointPetDataset(train_source, val_indices, eval_transform), shuffle=False, **loader_kwargs),
        "test_loader": DataLoader(JointPetDataset(test_source, None, eval_transform), shuffle=False, **loader_kwargs),
        "dataset_source": detect_pet_source(data_cfg["root"]),
    }


class JointSegGuidedClassifier(nn.Module):
    def __init__(self, segmenter, classifier, mean, std, background_scale):
        super().__init__()
        self.segmenter = segmenter
        self.classifier = classifier
        self.background_scale = float(background_scale)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def guided_image(self, images, seg_logits):
        probabilities = torch.softmax(seg_logits, dim=1)
        foreground = (probabilities[:, 0:1] + probabilities[:, 2:3]).clamp(0.0, 1.0)
        attention = self.background_scale + (1.0 - self.background_scale) * foreground
        rgb = images * self.std + self.mean
        guided_rgb = rgb * attention
        return (guided_rgb - self.mean) / self.std

    def forward(self, images):
        seg_logits = self.segmenter(images)
        cls_logits = self.classifier(self.guided_image(images, seg_logits))
        return cls_logits, seg_logits


def build_optimizer(model, config):
    opt_cfg = config["optimizer"]
    head_names = getattr(model.classifier, "_head_parameter_names", set())
    classifier_head = []
    classifier_backbone = []
    for name, parameter in model.classifier.named_parameters():
        if name in head_names:
            classifier_head.append(parameter)
        else:
            classifier_backbone.append(parameter)
    param_groups = [
        {"params": classifier_head, "lr": float(opt_cfg["classifier_head_lr"]), "weight_decay": float(opt_cfg.get("weight_decay", 0.0))},
        {"params": classifier_backbone, "lr": float(opt_cfg["classifier_backbone_lr"]), "weight_decay": float(opt_cfg.get("weight_decay", 0.0))},
        {"params": model.segmenter.parameters(), "lr": float(opt_cfg["segmentation_lr"]), "weight_decay": float(opt_cfg.get("weight_decay", 0.0))},
    ]
    return torch.optim.AdamW(param_groups)


def classification_loss(criterion, logits, targets):
    return criterion(logits, targets)


def align_seg_logits(seg_logits, masks):
    if seg_logits.shape[-2:] == masks.shape[-2:]:
        return seg_logits
    return F.interpolate(seg_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)


def train_one_epoch(model, loader, cls_criterion, seg_criterion, optimizer, device, scaler, config, epoch):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_correct = 0
    total_samples = 0
    seg_weight = float(config["segmentation"].get("loss_weight", 0.25))
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(config["train"].get("amp", True))):
            cls_logits, seg_logits = model(images)
            seg_logits = align_seg_logits(seg_logits, masks)
            cls_loss = classification_loss(cls_criterion, cls_logits, targets)
            seg_loss = seg_criterion(seg_logits, masks)
            loss = cls_loss + seg_weight * seg_loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_seg_loss += seg_loss.item() * batch_size
        total_correct += (cls_logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        if step % max(int(config["train"].get("log_interval", 20)), 1) == 0:
            progress.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")
    return {
        "loss": total_loss / max(total_samples, 1),
        "cls_loss": total_cls_loss / max(total_samples, 1),
        "seg_loss": total_seg_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(model, loader, cls_criterion, seg_criterion, device, config, stage):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_correct = 0
    total_samples = 0
    metric = SegmentationMetric(num_classes=TASK3_NUM_CLASSES)
    seg_weight = float(config["segmentation"].get("loss_weight", 0.25))
    tta_hflip = bool(config.get("evaluation", {}).get("tta_horizontal_flip", False))
    for batch in tqdm(loader, desc=stage, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast_context(device, bool(config["train"].get("amp", True))):
            cls_logits, seg_logits = model(images)
            if tta_hflip:
                flipped_cls_logits, flipped_seg_logits = model(torch.flip(images, dims=[3]))
                cls_logits = 0.5 * (cls_logits + flipped_cls_logits)
                seg_logits = 0.5 * (seg_logits + torch.flip(flipped_seg_logits, dims=[3]))
            seg_logits = align_seg_logits(seg_logits, masks)
            cls_loss = classification_loss(cls_criterion, cls_logits, targets)
            seg_loss = seg_criterion(seg_logits, masks)
            loss = cls_loss + seg_weight * seg_loss
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_seg_loss += seg_loss.item() * batch_size
        total_correct += (cls_logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        metric.update(seg_logits.detach().cpu(), masks.detach().cpu())
    return {
        "loss": total_loss / max(total_samples, 1),
        "cls_loss": total_cls_loss / max(total_samples, 1),
        "seg_loss": total_seg_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "miou": metric.mean_iou(),
        "pixel_acc": metric.pixel_accuracy(),
        "classwise_iou": metric.classwise_iou(),
    }


def load_segmenter(config, device):
    seg_cfg = load_yaml(config["segmentation"]["config"])
    segmenter = build_segmenter(seg_cfg)
    checkpoint_path = config["segmentation"].get("checkpoint")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        segmenter.load_state_dict(checkpoint["model_state_dict"])
    return segmenter.to(device), seg_cfg


def run_training(config):
    config, output_dir, device = prepare_run(config)
    data = build_joint_dataloaders(config)
    segmenter, seg_cfg = load_segmenter(config, device)
    classifier_cfg = {"model": config["classifier"], "data": config["data"]}
    classifier = build_classifier(classifier_cfg, num_classes=TASK1_NUM_CLASSES).to(device)
    mean = tuple(config["data"]["normalize"]["mean"])
    std = tuple(config["data"]["normalize"]["std"])
    model = JointSegGuidedClassifier(
        segmenter=segmenter,
        classifier=classifier,
        mean=mean,
        std=std,
        background_scale=float(config["guidance"].get("background_scale", 0.15)),
    ).to(device)
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"].get("label_smoothing", 0.0))).to(device)
    seg_criterion = build_segmentation_criterion(seg_cfg, num_classes=TASK3_NUM_CLASSES).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, float(config["optimizer"]["classifier_head_lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and bool(config["train"].get("amp", True))))

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    best_path = output_dir / "best.pt"
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, data["train_loader"], cls_criterion, seg_criterion, optimizer, device, scaler, config, epoch)
        val_metrics = evaluate(model, data["val_loader"], cls_criterion, seg_criterion, device, config, "val")
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_seg_loss": train_metrics["seg_loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_cls_loss": val_metrics["cls_loss"],
            "val_seg_loss": val_metrics["seg_loss"],
            "val_acc": val_metrics["acc"],
            "val_miou": val_metrics["miou"],
            "classifier_head_lr": float(optimizer.param_groups[0]["lr"]),
            "classifier_backbone_lr": float(optimizer.param_groups[1]["lr"]),
            "segmentation_lr": float(optimizer.param_groups[2]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_acc={train_metrics['acc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_miou={val_metrics['miou']:.4f}",
            flush=True,
        )
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, data["test_loader"], cls_criterion, seg_criterion, device, config, "test")
    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "profile": config["profile"],
        "dataset_source": data["dataset_source"],
        "segmentation_checkpoint": config["segmentation"].get("checkpoint"),
        "guidance": config["guidance"],
        "segmentation_loss_weight": float(config["segmentation"].get("loss_weight", 0.25)),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "test_miou": test_metrics["miou"],
        "test_pixel_acc": test_metrics["pixel_acc"],
        "test_classwise_iou": test_metrics["classwise_iou"],
        "checkpoint": str(best_path),
    }
    save_json(history, output_dir / "history.json")
    save_json(summary, output_dir / "summary.json")
    return summary


def main():
    args = parse_args()
    config = load_yaml(args.config)
    summary = run_training(config)
    print(summary)


if __name__ == "__main__":
    main()
