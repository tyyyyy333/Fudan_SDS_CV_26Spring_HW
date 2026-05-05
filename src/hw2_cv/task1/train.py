from torch import nn

from hw2_cv.runner import run_supervised_training
from hw2_cv.task1.data import NUM_CLASSES, build_dataloaders
from hw2_cv.task1.engine import evaluate, train_one_epoch
from hw2_cv.task1.models import build_model, build_optimizer, set_backbone_trainable
from hw2_cv.task1.reporting import classwise_accuracy, confusion_matrix, top_misclassified_samples
from hw2_cv.utils import (
    build_scheduler,
    prepare_run,
    save_json,
)


def _build_mixup_fn(config):
    augmentation_cfg = config["data"].get("augmentation", {})
    mixup_cfg = augmentation_cfg.get("mixup", {})
    cutmix_cfg = augmentation_cfg.get("cutmix", {})

    mixup_alpha = float(mixup_cfg.get("alpha", 0.0))
    cutmix_alpha = float(cutmix_cfg.get("alpha", 0.0))
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return None

    from timm.data import Mixup

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=float(mixup_cfg.get("probability", 1.0)),
        switch_prob=float(cutmix_cfg.get("switch_probability", 0.5)),
        mode=str(mixup_cfg.get("mode", "batch")),
        label_smoothing=float(config["train"].get("label_smoothing", 0.0)),
        num_classes=NUM_CLASSES,
    )


def _history_row(epoch, train_metrics, val_metrics, optimizer):
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
        "head_lr": float(optimizer.param_groups[0]["lr"]),
        "backbone_lr": float(optimizer.param_groups[-1]["lr"]),
    }


def run_training(config):
    return run_training_with_options(config)


def run_training_with_options(config, run_test=True, write_reports=True, epoch_callback=None):
    config, output_dir, device = prepare_run(config)

    if config.get("save_best_by", "val_acc") != "val_acc":
        raise ValueError("Task 1 only supports save_best_by=val_acc.")

    data = build_dataloaders(config)
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
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        test_loader=data["test_loader"],
        train_one_epoch=train_one_epoch,
        evaluate=evaluate,
        score_name="acc",
        checkpoint_score_name="val_acc",
        history_row=_history_row,
        train_kwargs={"mixup_fn": mixup_fn},
        val_kwargs={"collect_outputs": False},
        test_kwargs={
            "collect_outputs": True,
            "tta_horizontal_flip": bool(config.get("evaluation", {}).get("tta_horizontal_flip", False)),
        },
        before_epoch=before_epoch,
        run_name=f"task1/{config['model']['variant']}/{config['profile']}",
        epoch_callback=epoch_callback,
        run_test=run_test,
    )
    history = result["history"]
    best_epoch = result["best_epoch"]
    best_value = result["best_score"]
    test_metrics = result["test_metrics"]

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "profile": config["profile"],
        "dataset_source": data["dataset_source"],
        "family": config["model"]["family"],
        "variant": config["model"]["variant"],
        "pretrained": bool(config["model"].get("pretrained", True)),
        "image_size": int(config["data"]["image_size"]),
        "batch_size": int(config["data"]["batch_size"]),
        "dropout": float(config["model"].get("dropout", 0.0)),
        "optimizer_name": str(config["optimizer"].get("name", "adamw")),
        "epochs": int(config["train"]["epochs"]),
        "freeze_backbone_epochs": freeze_epochs,
        "label_smoothing": float(config["train"].get("label_smoothing", 0.0)),
        "early_stopping_enabled": bool(config["train"].get("early_stopping", {}).get("enabled", False)),
        "early_stopping_patience": int(config["train"].get("early_stopping", {}).get("patience", 0)),
        "head_lr": float(config["optimizer"]["head_lr"]),
        "backbone_lr": float(config["optimizer"]["backbone_lr"]),
        "randaugment_enabled": bool(
            config["data"].get("augmentation", {}).get("randaugment", {}).get("enabled", False)
        ),
        "mixup_alpha": float(config["data"].get("augmentation", {}).get("mixup", {}).get("alpha", 0.0)),
        "cutmix_alpha": float(config["data"].get("augmentation", {}).get("cutmix", {}).get("alpha", 0.0)),
        "tta_horizontal_flip": bool(config.get("evaluation", {}).get("tta_horizontal_flip", False)),
        "best_epoch": best_epoch,
        "best_val_acc": best_value,
        "test_acc": None if test_metrics is None else test_metrics["acc"],
        "test_loss": None if test_metrics is None else test_metrics["loss"],
        "checkpoint": str(output_dir / "best.pt"),
    }

    save_json(history, output_dir / "history.json")
    save_json(summary, output_dir / "summary.json")
    if test_metrics is not None and write_reports:
        records = test_metrics.pop("records")
        targets = [record["target"] for record in records]
        predictions = [record["prediction"] for record in records]
        confusion = confusion_matrix(targets, predictions, num_classes=NUM_CLASSES)
        class_accuracy = classwise_accuracy(confusion, data["class_names"])
        misclassified = top_misclassified_samples(
            records,
            class_names=data["class_names"],
            top_k=int(config.get("evaluation", {}).get("top_k_misclassified", 30)),
        )
        save_json(confusion, output_dir / "confusion_matrix.json")
        save_json(class_accuracy, output_dir / "class_accuracy.json")
        if bool(config.get("export_predictions", True)):
            save_json(misclassified, output_dir / "misclassified_samples.json")
    return summary
