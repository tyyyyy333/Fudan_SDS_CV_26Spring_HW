import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from hw2_cv.pets import build_pet_source, detect_pet_source, split_trainval_indices
from hw2_cv.utils import IMAGENET_MEAN, IMAGENET_STD


NUM_CLASSES = 3
CLASS_NAMES = ["pet", "background", "border"]
BACKGROUND_CLASS_ID = 1


class OxfordPetSegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        image_size,
        download,
        augment,
        normalize_mean,
        normalize_std,
        augmentation_cfg,
        indices=None,
        base_dataset=None,
    ):
        self.base_dataset = base_dataset
        if self.base_dataset is None:
            self.base_dataset = build_pet_source(
                root=root,
                split=split,
                target="segmentation",
                download=download,
            )
        self.indices = indices if indices is not None else list(range(len(self.base_dataset)))
        self.transform = build_transform(
            image_size=image_size,
            augment=augment,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            augmentation_cfg=augmentation_cfg,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        image = self.base_dataset.get_image(base_index)
        mask_np = self.base_dataset.get_mask_array(base_index)
        image_np = np.array(image)
        transformed = self.transform(image=image_np, mask=mask_np)
        image_id = self.base_dataset.get_image_id(base_index)
        return {
            "image": transformed["image"],
            "mask": transformed["mask"].long().clamp_(0, NUM_CLASSES - 1),
            "index": int(base_index),
            "image_id": image_id,
        }


def build_transform(image_size, augment, normalize_mean, normalize_std, augmentation_cfg):
    transforms_list = [A.Resize(height=image_size, width=image_size)]
    if augment:
        horizontal_flip_prob = float(augmentation_cfg.get("horizontal_flip_prob", 0.5))
        vertical_flip_prob = float(augmentation_cfg.get("vertical_flip_prob", 0.0))
        if horizontal_flip_prob > 0:
            transforms_list.append(A.HorizontalFlip(p=horizontal_flip_prob))
        if vertical_flip_prob > 0:
            transforms_list.append(A.VerticalFlip(p=vertical_flip_prob))

        affine_cfg = augmentation_cfg.get("shift_scale_rotate", {})
        if affine_cfg.get("enabled", True):
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=float(affine_cfg.get("shift_limit", 0.05)),
                    scale_limit=float(affine_cfg.get("scale_limit", 0.1)),
                    rotate_limit=int(affine_cfg.get("rotate_limit", 15)),
                    border_mode=0,
                    fill=float(affine_cfg.get("fill", 0)),
                    fill_mask=int(affine_cfg.get("fill_mask", BACKGROUND_CLASS_ID)),
                    p=float(affine_cfg.get("probability", 0.5)),
                )
            )

        color_cfg = augmentation_cfg.get("color_jitter", augmentation_cfg.get("color", {}))
        if color_cfg.get("enabled", True):
            transforms_list.append(
                A.ColorJitter(
                    brightness=float(color_cfg.get("brightness", 0.2)),
                    contrast=float(color_cfg.get("contrast", 0.2)),
                    saturation=float(color_cfg.get("saturation", 0.2)),
                    hue=float(color_cfg.get("hue", 0.02)),
                    p=float(color_cfg.get("probability", 0.3)),
                )
            )

    transforms_list.extend(
        [
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms_list)


def estimate_class_weights(dataset, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    if hasattr(dataset, "base_dataset") and hasattr(dataset, "indices"):
        for base_index in dataset.indices:
            mask_array = dataset.base_dataset.get_mask_array(base_index)
            mask_tensor = torch.from_numpy(mask_array).long().clamp_(0, num_classes - 1)
            bincount = torch.bincount(mask_tensor.view(-1), minlength=num_classes).to(torch.float64)
            counts += bincount
    else:
        for sample in dataset:
            mask = sample["mask"]
            bincount = torch.bincount(mask.view(-1), minlength=num_classes).to(torch.float64)
            counts += bincount

    counts = torch.clamp(counts, min=1.0)
    inverse = counts.sum() / counts
    normalized = inverse / inverse.mean()
    return normalized.tolist()


def build_dataloaders(config):
    data_cfg = config["data"]
    normalize_cfg = data_cfg.get("normalize", {})
    normalize_mean = tuple(normalize_cfg.get("mean", IMAGENET_MEAN))
    normalize_std = tuple(normalize_cfg.get("std", IMAGENET_STD))
    augmentation_cfg = data_cfg.get("augmentation", {})
    trainval_source = build_pet_source(
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
        val_ratio=data_cfg["val_ratio"],
        seed=config.get("seed", 42),
        download=data_cfg.get("download", False),
    )

    train_dataset = OxfordPetSegmentationDataset(
        root=data_cfg["root"],
        split="trainval",
        image_size=int(data_cfg["image_size"]),
        download=data_cfg.get("download", False),
        augment=True,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augmentation_cfg=augmentation_cfg,
        indices=train_indices,
        base_dataset=trainval_source,
    )
    val_dataset = OxfordPetSegmentationDataset(
        root=data_cfg["root"],
        split="trainval",
        image_size=int(data_cfg["image_size"]),
        download=data_cfg.get("download", False),
        augment=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augmentation_cfg=augmentation_cfg,
        indices=val_indices,
        base_dataset=trainval_source,
    )
    test_dataset = OxfordPetSegmentationDataset(
        root=data_cfg["root"],
        split="test",
        image_size=int(data_cfg["image_size"]),
        download=data_cfg.get("download", False),
        augment=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augmentation_cfg=augmentation_cfg,
        indices=None,
        base_dataset=test_source,
    )

    loader_kwargs = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(data_cfg["num_workers"]) > 0,
    }
    return {
        "train_loader": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val_loader": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test_loader": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "dataset_source": detect_pet_source(data_cfg["root"]),
        "class_names": CLASS_NAMES,
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
    }
