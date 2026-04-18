import numpy as np
import random
from datasets import load_dataset


def load_data(dataset_name="fashion_mnist"):
    dataset = load_dataset(dataset_name, cache_dir="./data")
    return dataset["train"], dataset["test"]

def _extract_xy(split):
    images = []
    labels = []
    for item in split:
        images.append(np.asarray(item["image"], dtype=np.float32))
        labels.append(item["label"])
    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def preprocess_data(data, transform=None, flatten=True):
    data = np.asarray(data, dtype=np.float32) / 255.0
    if flatten:
        data = data.reshape(data.shape[0], -1)
    if transform:
        data = transform(data)
    return data


def apply_preprocess_mode(images, mode="baseline", seed=42):
    mode = (mode or "baseline").lower()
    images = np.asarray(images, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)

    if mode in ("baseline", "none"):
        return images

    if mode in ("flip", "flip_mask"):
        flip_flags = rng.random(images.shape[0]) < 0.5
        images[flip_flags] = images[flip_flags, :, ::-1]

    if mode in ("mask", "flip_mask"):
        mask = rng.random(images.shape) < 0.1
        images[mask] = 0.0

    if mode not in ("baseline", "none", "flip", "mask", "flip_mask"):
        raise ValueError(f"Unsupported preprocess mode: {mode}")
    return images


class Transform:
    @classmethod
    def normalize(cls, data):
        return np.asarray(data, dtype=np.float32) / 255.0

    @classmethod
    def to_tensor(cls, data):
        return np.asarray(data)

    @classmethod
    def to_one_hot(cls, data):
        return np.eye(10)[data]

    @classmethod
    def random_flip(cls, data, p=0.5):
        data2 = data.copy()
        for i in range(len(data2)):
            if random.random() < p:
                data2[i] = data2[i, :, ::-1]
        return data2

    @classmethod
    def random_mask(cls, data, p=0.5):
        data2 = data.copy()
        mask = np.random.rand(*data.shape) < p
        data2[mask] = 0
        return data2

    @classmethod
    def compose(cls, *transforms):
        def composed_transform(data):
            for transform in transforms:
                data = transform(data)
            return data
        return composed_transform

class MnistDataset:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index+self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_index += self.batch_size
        X_batch = np.stack([item[0] for item in batch], axis=0)
        y_batch = np.asarray([item[1] for item in batch], dtype=np.int64)
        return X_batch, y_batch


def build_numpy_splits(dataset_name="fashion_mnist", preprocess_mode="baseline", seed=42):
    train_split, test_split = load_data(dataset_name)
    X_train_all, y_train_all = _extract_xy(train_split)
    X_test, y_test = _extract_xy(test_split)
    X_train_all = apply_preprocess_mode(X_train_all, mode=preprocess_mode, seed=seed)
    X_train_all = preprocess_data(X_train_all, flatten=True)
    X_test = preprocess_data(X_test, flatten=True)
    return X_train_all, y_train_all, X_test, y_test
