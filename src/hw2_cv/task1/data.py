from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hw2_cv.pets import (
    build_pet_source,
    detect_pet_source,
    split_trainval_indices,
)
from hw2_cv.utils import IMAGENET_MEAN, IMAGENET_STD


NUM_CLASSES = 37


class OxfordPetClassificationDataset(Dataset):
    def __init__(self, root, split, transform, download, indices=None, base_dataset=None):
        self.base_dataset = base_dataset
        if self.base_dataset is None:
            self.base_dataset = build_pet_source(
                root=root,
                split=split,
                target="category",
                download=download,
            )
        self.transform = transform
        self.indices = indices if indices is not None else list(range(len(self.base_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        image = self.base_dataset.get_image(base_index)
        target = self.base_dataset.get_label(base_index)
        if self.transform is not None:
            image = self.transform(image)
        image_id = self.base_dataset.get_image_id(base_index)
        return {
            "image": image,
            "target": int(target),
            "index": int(base_index),
            "image_id": image_id,
        }


def build_transforms(data_cfg):
    image_size = int(data_cfg["image_size"])
    augmentation_cfg = data_cfg.get("augmentation", {})
    normalize_cfg = data_cfg.get("normalize", {})
    evaluation_cfg = data_cfg.get("evaluation", {})
    mean = tuple(normalize_cfg.get("mean", IMAGENET_MEAN))
    std = tuple(normalize_cfg.get("std", IMAGENET_STD))

    train_steps = []
    if augmentation_cfg.get("random_resized_crop", True):
        train_steps.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(augmentation_cfg.get("crop_scale", (0.75, 1.0))),
                ratio=tuple(augmentation_cfg.get("crop_ratio", (0.75, 1.3333333333))),
            )
        )
    else:
        train_steps.append(transforms.Resize((image_size, image_size)))

    horizontal_flip_prob = float(augmentation_cfg.get("horizontal_flip_prob", 0.5))
    if horizontal_flip_prob > 0:
        train_steps.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))

    color_jitter_cfg = augmentation_cfg.get("color_jitter", {})
    if color_jitter_cfg:
        train_steps.append(
            transforms.ColorJitter(
                brightness=float(color_jitter_cfg.get("brightness", 0.2)),
                contrast=float(color_jitter_cfg.get("contrast", 0.2)),
                saturation=float(color_jitter_cfg.get("saturation", 0.2)),
                hue=float(color_jitter_cfg.get("hue", 0.0)),
            )
        )

    rand_augment_cfg = augmentation_cfg.get("randaugment", {})
    if rand_augment_cfg.get("enabled", False):
        train_steps.append(
            transforms.RandAugment(
                num_ops=int(rand_augment_cfg.get("num_ops", 2)),
                magnitude=int(rand_augment_cfg.get("magnitude", 7)),
            )
        )

    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    random_erasing_cfg = augmentation_cfg.get("random_erasing", {})
    if random_erasing_cfg.get("enabled", False):
        train_steps.append(
            transforms.RandomErasing(
                p=float(random_erasing_cfg.get("probability", 0.1)),
                scale=tuple(random_erasing_cfg.get("scale", (0.02, 0.1))),
                ratio=tuple(random_erasing_cfg.get("ratio", (0.3, 3.3))),
            )
        )

    eval_resize_ratio = float(evaluation_cfg.get("resize_ratio", 1.14))
    eval_resize_size = max(int(round(image_size * eval_resize_ratio)), image_size)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def build_dataloaders(config):
    data_cfg = config["data"]
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

    train_dataset = OxfordPetClassificationDataset(
        root=data_cfg["root"],
        split="trainval",
        transform=train_transform,
        download=data_cfg.get("download", False),
        indices=train_indices,
        base_dataset=trainval_source,
    )
    val_dataset = OxfordPetClassificationDataset(
        root=data_cfg["root"],
        split="trainval",
        transform=eval_transform,
        download=data_cfg.get("download", False),
        indices=val_indices,
        base_dataset=trainval_source,
    )
    test_dataset = OxfordPetClassificationDataset(
        root=data_cfg["root"],
        split="test",
        transform=eval_transform,
        download=data_cfg.get("download", False),
        indices=None,
        base_dataset=test_source,
    )

    loader_kwargs = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(data_cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    class_names = trainval_source.get_class_names()
    if not class_names:
        class_names = [str(index) for index in range(NUM_CLASSES)]
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "dataset_source": detect_pet_source(data_cfg["root"]),
        "class_names": class_names,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
