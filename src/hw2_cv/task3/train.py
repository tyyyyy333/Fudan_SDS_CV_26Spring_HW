import torch

from hw2_cv.runner import run_supervised_training
from hw2_cv.task3.data import NUM_CLASSES, build_dataloaders, estimate_class_weights
from hw2_cv.task3.engine import evaluate, train_one_epoch
from hw2_cv.task3.losses import build_criterion
from hw2_cv.task3.models import build_model
from hw2_cv.task3.visualize import export_prediction_samples
from hw2_cv.utils import (
    build_scheduler,
    prepare_run,
    save_json,
)


def _build_optimizer(model, config):
    optimizer_cfg = config["optimizer"]
    optimizer_name = optimizer_cfg.get("name", "adamw")
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _history_row(epoch, train_metrics, val_metrics, optimizer):
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_miou": train_metrics["miou"],
        "train_pixel_acc": train_metrics["pixel_acc"],
        "train_classwise_iou": train_metrics["classwise_iou"],
        "val_loss": val_metrics["loss"],
        "val_miou": val_metrics["miou"],
        "val_pixel_acc": val_metrics["pixel_acc"],
        "val_classwise_iou": val_metrics["classwise_iou"],
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def run_training(config):
    config, output_dir, device = prepare_run(config)

    if config.get("save_best_by", "val_miou") != "val_miou":
        raise ValueError("Task 3 only supports save_best_by=val_miou.")

    data = build_dataloaders(config)
    auto_class_weights = bool(config["loss"].get("auto_class_weights", False))
    class_weights = config["loss"].get("class_weights")
    if auto_class_weights and class_weights is None:
        class_weights = estimate_class_weights(data["train_dataset"], num_classes=NUM_CLASSES)

    model = build_model(config).to(device)
    criterion = build_criterion(
        config,
        num_classes=NUM_CLASSES,
        class_weights_override=class_weights,
    ).to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, float(config["optimizer"]["lr"]))

    result = run_supervised_training(
        config=config,
        output_dir=output_dir,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        test_loader=data["test_loader"],
        train_one_epoch=train_one_epoch,
        evaluate=evaluate,
        score_name="miou",
        checkpoint_score_name="val_miou",
        history_row=_history_row,
        train_kwargs={"num_classes": NUM_CLASSES},
        val_kwargs={"num_classes": NUM_CLASSES, "collect_predictions": False},
        test_kwargs={
            "num_classes": NUM_CLASSES,
            "collect_predictions": bool(config.get("export_predictions", True)),
            "max_predictions": int(config.get("evaluation", {}).get("prediction_sample_count", 8)),
        },
        run_name=f"task3/{config['loss']['name']}/{config['profile']}",
    )
    history = result["history"]
    best_epoch = result["best_epoch"]
    best_score = result["best_score"]
    test_metrics = result["test_metrics"]

    predictions = test_metrics.pop("predictions", [])
    exported_prediction_paths = []
    if bool(config.get("export_predictions", True)) and predictions:
        exported_prediction_paths = export_prediction_samples(
            predictions=predictions,
            output_dir=output_dir / "prediction_exports",
            mean=data["normalize_mean"],
            std=data["normalize_std"],
            sample_count=int(config.get("evaluation", {}).get("prediction_sample_count", 8)),
        )

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "profile": config["profile"],
        "dataset_source": data["dataset_source"],
        "loss_name": config["loss"]["name"],
        "class_weights": class_weights,
        "best_epoch": best_epoch,
        "best_val_miou": best_score,
        "test_miou": test_metrics["miou"],
        "test_classwise_iou": test_metrics["classwise_iou"],
        "test_pixel_acc": test_metrics["pixel_acc"],
        "test_loss": test_metrics["loss"],
        "checkpoint": str(output_dir / "best.pt"),
        "exported_prediction_paths": exported_prediction_paths,
    }

    save_json(history, output_dir / "history.json")
    save_json(summary, output_dir / "summary.json")
    return summary
