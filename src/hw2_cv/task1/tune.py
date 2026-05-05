from pathlib import Path

import optuna

from hw2_cv.task1.train import run_training_with_options
from hw2_cv.utils import deep_update, ensure_dir, log_info, save_json, save_yaml


def _variant_family(variant):
    if variant.startswith("swin") or variant.startswith("vit"):
        return "timm"
    return "resnet"


def _suggest_value(trial, name, spec):
    choices = spec.get("choices")
    if choices is not None:
        return trial.suggest_categorical(name, choices)

    low = spec["low"]
    high = spec["high"]
    step = spec.get("step")
    log = bool(spec.get("log", False))

    if isinstance(low, int) and isinstance(high, int) and step is None and not log:
        return trial.suggest_int(name, low, high)
    if isinstance(low, int) and isinstance(high, int) and step is not None and not log:
        return trial.suggest_int(name, low, high, step=int(step))
    return trial.suggest_float(name, low, high, step=step, log=log)


def _drop_empty(value):
    if not isinstance(value, dict):
        return value

    cleaned = {}
    for key, item in value.items():
        if isinstance(item, dict):
            item = _drop_empty(item)
            if item:
                cleaned[key] = item
        elif item is not None:
            cleaned[key] = item
    return cleaned


def _build_trial_overrides(trial, base_config):
    tuning_cfg = base_config.get("tuning", {})
    search_space = tuning_cfg.get("search_space", {})

    overrides = {
        "data": {
            "augmentation": {
                "randaugment": {},
                "mixup": {},
                "cutmix": {},
                "random_erasing": {},
            }
        },
        "model": {},
        "optimizer": {},
        "train": {
            "early_stopping": {
                "enabled": False,
            }
        },
        "evaluation": {},
    }

    variant_spec = search_space.get("variant")
    if variant_spec:
        variant = _suggest_value(trial, "variant", variant_spec)
        overrides["model"]["variant"] = variant
        overrides["model"]["family"] = _variant_family(variant)

    image_size_spec = search_space.get("image_size")
    if image_size_spec:
        overrides["data"]["image_size"] = int(_suggest_value(trial, "image_size", image_size_spec))

    batch_size_spec = search_space.get("batch_size")
    if batch_size_spec:
        overrides["data"]["batch_size"] = int(_suggest_value(trial, "batch_size", batch_size_spec))

    dropout_spec = search_space.get("dropout")
    if dropout_spec:
        overrides["model"]["dropout"] = float(_suggest_value(trial, "dropout", dropout_spec))

    head_lr_spec = search_space.get("head_lr")
    head_lr = None
    if head_lr_spec:
        head_lr = float(_suggest_value(trial, "head_lr", head_lr_spec))
        overrides["optimizer"]["head_lr"] = head_lr

    optimizer_name_spec = search_space.get("optimizer_name")
    if optimizer_name_spec:
        overrides["optimizer"]["name"] = str(_suggest_value(trial, "optimizer_name", optimizer_name_spec))

    backbone_ratio_spec = search_space.get("backbone_lr_ratio")
    if backbone_ratio_spec and head_lr is not None:
        backbone_ratio = float(_suggest_value(trial, "backbone_lr_ratio", backbone_ratio_spec))
        overrides["optimizer"]["backbone_lr"] = head_lr * backbone_ratio

    freeze_spec = search_space.get("freeze_backbone_epochs")
    if freeze_spec:
        overrides["train"]["freeze_backbone_epochs"] = int(
            _suggest_value(trial, "freeze_backbone_epochs", freeze_spec)
        )

    epochs_spec = search_space.get("epochs")
    if epochs_spec:
        overrides["train"]["epochs"] = int(_suggest_value(trial, "epochs", epochs_spec))

    label_smoothing_spec = search_space.get("label_smoothing")
    if label_smoothing_spec:
        overrides["train"]["label_smoothing"] = float(
            _suggest_value(trial, "label_smoothing", label_smoothing_spec)
        )

    randaugment_enabled_spec = search_space.get("randaugment_enabled")
    randaugment_enabled = None
    if randaugment_enabled_spec:
        randaugment_enabled = bool(_suggest_value(trial, "randaugment_enabled", randaugment_enabled_spec))
        overrides["data"]["augmentation"]["randaugment"]["enabled"] = randaugment_enabled

    randaugment_magnitude_spec = search_space.get("randaugment_magnitude")
    if randaugment_magnitude_spec and randaugment_enabled:
        overrides["data"]["augmentation"]["randaugment"]["magnitude"] = int(
            _suggest_value(trial, "randaugment_magnitude", randaugment_magnitude_spec)
        )

    crop_scale_min_spec = search_space.get("crop_scale_min")
    if crop_scale_min_spec:
        crop_scale_min = float(_suggest_value(trial, "crop_scale_min", crop_scale_min_spec))
        overrides["data"]["augmentation"]["crop_scale"] = [crop_scale_min, 1.0]

    random_erasing_enabled_spec = search_space.get("random_erasing_enabled")
    random_erasing_enabled = None
    if random_erasing_enabled_spec:
        random_erasing_enabled = bool(
            _suggest_value(trial, "random_erasing_enabled", random_erasing_enabled_spec)
        )
        overrides["data"]["augmentation"]["random_erasing"]["enabled"] = random_erasing_enabled

    random_erasing_prob_spec = search_space.get("random_erasing_probability")
    if random_erasing_prob_spec and random_erasing_enabled:
        overrides["data"]["augmentation"]["random_erasing"]["probability"] = float(
            _suggest_value(trial, "random_erasing_probability", random_erasing_prob_spec)
        )

    mixup_alpha_spec = search_space.get("mixup_alpha")
    if mixup_alpha_spec:
        overrides["data"]["augmentation"]["mixup"]["alpha"] = float(
            _suggest_value(trial, "mixup_alpha", mixup_alpha_spec)
        )

    cutmix_alpha_spec = search_space.get("cutmix_alpha")
    if cutmix_alpha_spec:
        overrides["data"]["augmentation"]["cutmix"]["alpha"] = float(
            _suggest_value(trial, "cutmix_alpha", cutmix_alpha_spec)
        )

    return _drop_empty(overrides)


def _make_sampler(tuning_cfg, seed):
    sampler_cfg = tuning_cfg.get("sampler", {})
    sampler_name = sampler_cfg.get("name", "tpe")
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=bool(sampler_cfg.get("multivariate", True)),
            n_startup_trials=int(sampler_cfg.get("n_startup_trials", 5)),
        )
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: {sampler_name}")


def _make_pruner(tuning_cfg):
    pruner_cfg = tuning_cfg.get("pruner", {})
    if not bool(pruner_cfg.get("enabled", True)):
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 5)),
    )


def run_tuning(config):
    tuning_cfg = config.get("tuning", {})
    if not tuning_cfg:
        raise ValueError("Task 1 tuning config is missing.")

    study_output_dir = ensure_dir(tuning_cfg.get("output_dir", "outputs/task1/tuning"))
    base_output_dir = Path(config["output_dir"])
    seed = int(config.get("seed", 42))
    study_name = tuning_cfg.get("study_name", "task1_high_score")
    storage = tuning_cfg.get("storage")
    if not storage:
        storage = None

    sampler = _make_sampler(tuning_cfg, seed)
    pruner = _make_pruner(tuning_cfg)
    study = optuna.create_study(
        study_name=study_name,
        direction=tuning_cfg.get("direction", "maximize"),
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    log_info(
        f"[task1-tune] study={study_name} | trials={tuning_cfg.get('n_trials', 20)} | output={study_output_dir}"
    )

    def objective(trial):
        trial_overrides = _build_trial_overrides(trial, config)
        trial_config = deep_update(config, trial_overrides)
        trial_config["output_dir"] = str(study_output_dir / f"trial_{trial.number:04d}")
        trial_config["export_predictions"] = False
        trial_config["train"] = deep_update(
            trial_config.get("train", {}),
            {
                "early_stopping": {
                    "enabled": False,
                }
            },
        )

        def epoch_callback(epoch, current_score, **_):
            trial.report(float(current_score), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        summary = run_training_with_options(
            trial_config,
            run_test=False,
            write_reports=False,
            epoch_callback=epoch_callback,
        )
        trial.set_user_attr("output_dir", summary["output_dir"])
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr("best_val_acc", summary["best_val_acc"])
        trial.set_user_attr("overrides", trial_overrides)
        return float(summary["best_val_acc"])

    timeout_hours = tuning_cfg.get("timeout_hours")
    timeout_seconds = None if timeout_hours in (None, "") else int(float(timeout_hours) * 3600)
    study.optimize(
        objective,
        n_trials=int(tuning_cfg.get("n_trials", 20)),
        timeout=timeout_seconds,
    )

    trials_summary = []
    for trial in study.trials:
        trials_summary.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
                "output_dir": trial.user_attrs.get("output_dir"),
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "best_val_acc": trial.user_attrs.get("best_val_acc"),
            }
        )

    best_overrides = study.best_trial.user_attrs.get("overrides", {})
    best_config = deep_update(config, best_overrides)
    best_config["output_dir"] = str(base_output_dir.parent / tuning_cfg.get("best_run_name", "best_from_tuning"))

    summary = {
        "study_name": study.study_name,
        "direction": tuning_cfg.get("direction", "maximize"),
        "trial_count": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_output_dir": study.best_trial.user_attrs.get("output_dir"),
        "best_config_path": str(study_output_dir / "best_config.yaml"),
    }

    save_json(summary, study_output_dir / "study_summary.json")
    save_json(trials_summary, study_output_dir / "trials_summary.json")
    save_yaml(best_config, study_output_dir / "best_config.yaml")
    log_info(
        f"[task1-tune] done | best_trial={study.best_trial.number} | best_val_acc={study.best_value:.4f}"
    )
    return summary
