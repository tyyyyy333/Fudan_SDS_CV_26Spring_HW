import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def log_info(message):
    print(message, flush=True)


def ensure_dir(path):
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise TypeError(f"Expected a dict-like YAML config from {path}.")
    return content


def save_json(payload, path):
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_jsonl(rows, path):
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deep_update(base, overrides):
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_profile_config(config):
    merged = deepcopy(config)
    selected_profile = merged.get("profile", "high_score")
    profiles = merged.pop("profiles", {})
    if selected_profile in profiles:
        merged = deep_update(merged, profiles[selected_profile])
    merged["profile"] = selected_profile
    return merged


def save_yaml(payload, path):
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested="cuda"):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def prepare_run(config, default_device="cuda"):
    config = resolve_profile_config(config)
    set_seed(int(config.get("seed", 42)))
    output_dir = ensure_dir(config["output_dir"])
    device = resolve_device(config.get("device", default_device))
    return config, output_dir, device


def build_warmup_cosine_scheduler(
    optimizer,
    total_epochs,
    warmup_epochs=0,
    warmup_start_factor=0.1,
    min_lr_ratio=0.0,
):
    import torch

    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs, 1),
            eta_min=min_lr_ratio,
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=min_lr_ratio,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def build_scheduler(optimizer, config, base_lr):
    import torch

    scheduler_cfg = config.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "none")
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return build_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_epochs=int(config["train"]["epochs"]),
            warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 0)),
            warmup_start_factor=float(scheduler_cfg.get("warmup_start_factor", 0.1)),
            min_lr_ratio=base_lr * float(scheduler_cfg.get("min_lr_ratio", 0.0)),
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
