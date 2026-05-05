from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets as tv_datasets

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None


def _normalized_split_name(split):
    if split == "trainval":
        return "train"
    return split


def _collect_parquet_files(root, split):
    root = Path(root).expanduser()
    split = _normalized_split_name(split)
    patterns = [f"{split}-*.parquet"]
    if split == "val":
        patterns.append("validation-*.parquet")

    matches = []
    seen = set()
    search_roots = [root, root / "data"]

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for pattern in patterns:
            for path in sorted(search_root.glob(pattern)):
                if ".cache" in path.parts:
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                matches.append(path)
                seen.add(resolved)

    if matches:
        return matches

    if not root.exists():
        return []

    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if ".cache" in path.parts:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            matches.append(path)
            seen.add(resolved)
    return matches


def has_local_hf_pet_dataset(root):
    train_files = _collect_parquet_files(root, "train")
    test_files = _collect_parquet_files(root, "test")
    return bool(train_files) and bool(test_files)


def detect_pet_source(root):
    if has_local_hf_pet_dataset(root):
        return "hf_parquet"
    return "torchvision"


def _pick_first(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _to_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.array(image)).convert("RGB")


def normalize_segmentation_mask(mask):
    mask_array = np.array(mask, dtype=np.int64)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    unique_values = np.unique(mask_array)
    if unique_values.size == 0:
        return mask_array

    if unique_values.min() >= 1 and unique_values.max() <= 3:
        return mask_array - 1
    if unique_values.min() >= 0 and unique_values.max() <= 2:
        return mask_array
    return np.clip(mask_array, 0, 2)


class TorchvisionPetSource:
    def __init__(self, root, split, target, download=False):
        self.root = str(root)
        self.split = split
        self.target = target
        self.dataset = tv_datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=target,
            transform=None,
            download=download,
        )

    def __len__(self):
        return len(self.dataset)

    def get_image(self, index):
        image, _ = self.dataset[index]
        return _to_pil_rgb(image)

    def get_label(self, index):
        _, target = self.dataset[index]
        return int(target)

    def get_mask_array(self, index):
        _, mask = self.dataset[index]
        return normalize_segmentation_mask(mask)

    def get_image_id(self, index):
        image_paths = getattr(self.dataset, "_images", None)
        if image_paths is None:
            return f"{self.split}_{index:05d}"
        return Path(image_paths[index]).stem

    def get_labels(self):
        labels = getattr(self.dataset, "_labels", None)
        if labels is not None:
            return [int(label) for label in labels]
        return [self.get_label(index) for index in range(len(self))]

    def get_class_names(self):
        classes = getattr(self.dataset, "classes", None)
        if classes is None:
            labels = self.get_labels()
            if not labels:
                return []
            return [str(index) for index in range(max(labels) + 1)]
        return list(classes)


class HFPetSource:
    def __init__(self, root, split, target):
        if hf_load_dataset is None:
            raise ImportError("datasets is required to read local Hugging Face parquet datasets.")

        self.root = Path(root).expanduser().resolve()
        self.split = _normalized_split_name(split)
        self.target = target
        self.dataset = _load_hf_split(self.root, self.split)
        self.image_column = _pick_first(self.dataset.column_names, ["image"])
        self.label_column = _pick_first(
            self.dataset.column_names,
            ["label", "labels", "category", "class_id"],
        )
        self.mask_column = _pick_first(
            self.dataset.column_names,
            ["segmentation_mask", "trimap", "mask"],
        )
        self.image_id_column = _pick_first(
            self.dataset.column_names,
            ["image_id", "image_name", "id", "file_name"],
        )

        if self.image_column is None:
            raise KeyError(f"Could not find image column under {self.root}.")
        if self.target == "category" and self.label_column is None:
            raise KeyError(f"Could not find label column under {self.root}.")
        if self.target == "segmentation" and self.mask_column is None:
            raise KeyError(f"Could not find segmentation mask column under {self.root}.")

    def __len__(self):
        return len(self.dataset)

    def _row(self, index):
        return self.dataset[index]

    def get_image(self, index):
        row = self._row(index)
        return _to_pil_rgb(row[self.image_column])

    def get_label(self, index):
        row = self._row(index)
        return int(row[self.label_column])

    def get_mask_array(self, index):
        row = self._row(index)
        return normalize_segmentation_mask(row[self.mask_column])

    def get_image_id(self, index):
        row = self._row(index)
        if self.image_id_column is not None:
            value = row[self.image_id_column]
            if value not in (None, ""):
                return str(value)
        return f"{self.split}_{index:05d}"

    def get_labels(self):
        return [int(label) for label in self.dataset[self.label_column]]

    def get_class_names(self):
        feature = self.dataset.features.get(self.label_column)
        names = getattr(feature, "names", None)
        if names:
            return list(names)
        labels = self.get_labels()
        if not labels:
            return []
        return [str(index) for index in range(max(labels) + 1)]


def _load_hf_split(root, split):
    data_files = {}

    train_files = _collect_parquet_files(root, "train")
    if train_files:
        data_files["train"] = [str(path) for path in train_files]

    val_files = _collect_parquet_files(root, "val")
    if val_files:
        data_files["val"] = [str(path) for path in val_files]

    test_files = _collect_parquet_files(root, "test")
    if test_files:
        data_files["test"] = [str(path) for path in test_files]

    if split not in data_files:
        raise FileNotFoundError(f"Could not find split={split} parquet files under {root}.")

    return hf_load_dataset("parquet", data_files=data_files, split=split)


def build_pet_source(root, split, target, download=False):
    if detect_pet_source(root) == "hf_parquet":
        return HFPetSource(root=root, split=split, target=target)
    return TorchvisionPetSource(root=root, split=split, target=target, download=download)


def get_image_id(base_dataset, base_index):
    if hasattr(base_dataset, "get_image_id"):
        return base_dataset.get_image_id(base_index)

    image_paths = getattr(base_dataset, "_images", None)
    if image_paths is None:
        return str(base_index)
    return Path(image_paths[base_index]).stem


def get_category_labels(root, download=False):
    dataset = build_pet_source(
        root=root,
        split="trainval",
        target="category",
        download=download,
    )
    return dataset.get_labels()


def get_class_names(root, download=False):
    dataset = build_pet_source(
        root=root,
        split="trainval",
        target="category",
        download=download,
    )
    return dataset.get_class_names()


def split_trainval_indices(root, val_ratio, seed, download=False):
    labels = get_category_labels(root=root, download=download)
    indices = list(range(len(labels)))
    return train_test_split(
        indices,
        test_size=float(val_ratio),
        random_state=int(seed),
        shuffle=True,
        stratify=labels,
    )
