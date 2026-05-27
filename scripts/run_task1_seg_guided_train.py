import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / ".vendor"
if VENDOR_ROOT.exists():
    sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.cli import load_run_config, print_json
from hw2_cv.pets import build_pet_source, detect_pet_source, split_trainval_indices
from hw2_cv.runner import autocast_context, run_supervised_training
from hw2_cv.task1.data import NUM_CLASSES, build_transforms
from hw2_cv.task1.engine import evaluate, train_one_epoch
from hw2_cv.task1.models import build_model, build_optimizer, set_backbone_trainable
from hw2_cv.task1.train import _build_mixup_fn, _history_row
from hw2_cv.task3.models import build_model as build_segmentation_model
from hw2_cv.utils import build_scheduler, ensure_dir, load_yaml, prepare_run, save_json


def parse_args():
    parser = argparse.ArgumentParser("Train Task 1 with Task 3 segmentation-guided inputs.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-cache", action="store_true")
    return parser.parse_args()


def _safe_mask_name(split, image_id):
    return f"{split}_{image_id}.png".replace("/", "_")


def _mask_path(cache_dir, split, image_id):
    return Path(cache_dir) / _safe_mask_name(split, image_id)


def _normalize_image_tensor(image, mean, std):
    array = np.array(image.resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


@torch.no_grad()
def build_mask_cache(config, device):
    guide_cfg = config["data"]["segmentation_guidance"]
    cache_dir = ensure_dir(guide_cfg["cache_dir"])
    seg_cfg = load_yaml(config["segmentation"]["config"])
    seg_ckpt = torch.load(config["segmentation"]["checkpoint"], map_location="cpu", weights_only=False)
    seg_model = build_segmentation_model(seg_cfg).to(device)
    seg_model.load_state_dict(seg_ckpt["model_state_dict"])
    seg_model.eval()

    mean = tuple(seg_cfg["data"]["normalize"]["mean"])
    std = tuple(seg_cfg["data"]["normalize"]["std"])
    batch_size = int(config["segmentation"].get("batch_size", 64))

    for split in ("trainval", "test"):
        source = build_pet_source(
            root=config["data"]["root"],
            split=split,
            target="category",
            download=config["data"].get("download", False),
        )
        pending = []
        for index in range(len(source)):
            image_id = source.get_image_id(index)
            output_path = _mask_path(cache_dir, split, image_id)
            if not output_path.exists():
                pending.append((index, image_id, output_path))

        if not pending:
            continue

        progress = tqdm(range(0, len(pending), batch_size), desc=f"cache-mask-{split}", leave=False)
        for start in progress:
            rows = pending[start : start + batch_size]
            images = [
                _normalize_image_tensor(source.get_image(index), mean=mean, std=std)
                for index, _image_id, _path in rows
            ]
            batch = torch.stack(images).to(device, non_blocking=True)
            with autocast_context(device, bool(seg_cfg["train"].get("amp", True))):
                logits = seg_model(batch)
            masks = logits.float().argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            for mask, (_index, _image_id, output_path) in zip(masks, rows):
                Image.fromarray(mask, mode="L").save(output_path)


def _apply_guidance(image, mask, mode, background_scale):
    image = image.resize(mask.size, Image.BILINEAR).convert("RGB")
    image_np = np.array(image, dtype=np.float32)
    mask_np = np.array(mask, dtype=np.uint8)

    if mode == "pet_only_dim_background":
        keep = mask_np == 0
    elif mode == "pet_border_dim_background":
        keep = (mask_np == 0) | (mask_np == 2)
    else:
        raise ValueError(f"Unsupported segmentation guidance mode: {mode}")

    guided = image_np * background_scale
    guided[keep] = image_np[keep]
    return Image.fromarray(np.clip(guided, 0, 255).astype(np.uint8), mode="RGB")


class SegmentationGuidedClassificationDataset(Dataset):
    def __init__(self, root, split, transform, download, cache_dir, mode, background_scale, indices=None, base_dataset=None):
        self.base_dataset = base_dataset or build_pet_source(
            root=root,
            split=split,
            target="category",
            download=download,
        )
        self.split = split
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.background_scale = float(background_scale)
        self.indices = indices if indices is not None else list(range(len(self.base_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        image = self.base_dataset.get_image(base_index)
        target = self.base_dataset.get_label(base_index)
        image_id = self.base_dataset.get_image_id(base_index)
        mask = Image.open(_mask_path(self.cache_dir, self.split, image_id)).convert("L")
        image = _apply_guidance(
            image=image,
            mask=mask,
            mode=self.mode,
            background_scale=self.background_scale,
        )
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": int(target),
            "index": int(base_index),
            "image_id": image_id,
        }


def build_guided_dataloaders(config):
    data_cfg = config["data"]
    guide_cfg = data_cfg["segmentation_guidance"]
    train_transform, eval_transform = build_transforms(data_cfg)

    trainval_source = build_pet_source(
        root=data_cfg["root"],
        split="trainval",
        target="category",
        download=data_cfg.get("download", False),
    )
    test_source = build_pet_source(
        root=data_cfg["root"],
        split="test",
        target="category",
        download=data_cfg.get("download", False),
    )
    train_indices, val_indices = split_trainval_indices(
        root=data_cfg["root"],
        val_ratio=data_cfg["val_ratio"],
        seed=config.get("seed", 42),
        download=data_cfg.get("download", False),
    )

    common_kwargs = {
        "root": data_cfg["root"],
        "download": data_cfg.get("download", False),
        "cache_dir": guide_cfg["cache_dir"],
        "mode": guide_cfg.get("mode", "pet_border_dim_background"),
        "background_scale": guide_cfg.get("background_scale", 0.15),
    }
    train_dataset = SegmentationGuidedClassificationDataset(
        split="trainval",
        transform=train_transform,
        indices=train_indices,
        base_dataset=trainval_source,
        **common_kwargs,
    )
    val_dataset = SegmentationGuidedClassificationDataset(
        split="trainval",
        transform=eval_transform,
        indices=val_indices,
        base_dataset=trainval_source,
        **common_kwargs,
    )
    test_dataset = SegmentationGuidedClassificationDataset(
        split="test",
        transform=eval_transform,
        indices=None,
        base_dataset=test_source,
        **common_kwargs,
    )

    loader_kwargs = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(data_cfg["num_workers"]) > 0,
    }
    class_names = trainval_source.get_class_names() or [str(index) for index in range(NUM_CLASSES)]
    return {
        "train_loader": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val_loader": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test_loader": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        "dataset_source": detect_pet_source(data_cfg["root"]),
        "class_names": class_names,
    }


def run_guided_training(config, skip_cache=False):
    config, output_dir, device = prepare_run(config)
    if not skip_cache:
        build_mask_cache(config, device)

    dataloaders = build_guided_dataloaders(config)
    model = build_model(config, num_classes=NUM_CLASSES).to(device)

    freeze_epochs = int(config["train"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["train"].get("label_smoothing", 0.0))
    ).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, float(config["optimizer"]["head_lr"]))
    mixup_fn = _build_mixup_fn(config)

    def before_epoch(epoch, current_model):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_backbone_trainable(current_model, True)

    result = run_supervised_training(
        config=config,
        output_dir=output_dir,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        test_loader=dataloaders["test_loader"],
        train_one_epoch=train_one_epoch,
        evaluate=evaluate,
        score_name="acc",
        checkpoint_score_name="val_acc",
        history_row=_history_row,
        train_kwargs={"mixup_fn": mixup_fn},
        val_kwargs={"collect_outputs": False},
        test_kwargs={
            "collect_outputs": False,
            "tta_horizontal_flip": bool(config.get("evaluation", {}).get("tta_horizontal_flip", False)),
        },
        before_epoch=before_epoch,
        run_name=f"task1/{config['model']['variant']}/seg_guided",
        run_test=True,
    )

    test_metrics = result["test_metrics"]
    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "profile": config["profile"],
        "dataset_source": dataloaders["dataset_source"],
        "variant": config["model"]["variant"],
        "pretrained": bool(config["model"].get("pretrained", True)),
        "image_size": int(config["data"]["image_size"]),
        "epochs": int(config["train"]["epochs"]),
        "best_epoch": result["best_epoch"],
        "best_val_acc": result["best_score"],
        "test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "segmentation_checkpoint": config["segmentation"]["checkpoint"],
        "segmentation_guidance": config["data"]["segmentation_guidance"],
        "checkpoint": str(output_dir / "best.pt"),
    }
    save_json(result["history"], output_dir / "history.json")
    save_json(summary, output_dir / "summary.json")
    return summary


def main():
    args = parse_args()
    config = load_run_config(args.config)
    summary = run_guided_training(config, skip_cache=args.skip_cache)
    print_json(summary)


if __name__ == "__main__":
    main()
