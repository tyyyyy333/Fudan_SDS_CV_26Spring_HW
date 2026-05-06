import copy
from contextlib import nullcontext

import torch

from hw2_cv.utils import log_info


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_supervised_training(
    config,
    output_dir,
    device,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    train_one_epoch,
    evaluate,
    score_name,
    checkpoint_score_name,
    history_row,
    train_kwargs=None,
    val_kwargs=None,
    test_kwargs=None,
    before_epoch=None,
    run_name=None,
    epoch_callback=None,
    run_test=True,
):
    trainCallKwargs = train_kwargs or {}
    valCallKwargs = val_kwargs or {}
    testCallKwargs = test_kwargs or {}

    ampEnabled = bool(config["train"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=ampEnabled and device.type == "cuda")

    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    history = []
    epochTotal = int(config["train"]["epochs"])
    resolvedRunName = run_name or str(output_dir.name)

    early_stopping_cfg = config["train"].get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_patience = int(early_stopping_cfg.get("patience", 10))
    epochs_without_improvement = 0

    log_info(
        f"[train] {resolvedRunName} | device={device} | epochs={epochTotal} | output={output_dir}"
    )

    for epoch in range(1, epochTotal + 1):
        if before_epoch is not None:
            before_epoch(epoch, model)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            log_interval=int(config["train"].get("log_interval", 10)),
            amp=ampEnabled,
            **trainCallKwargs,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp=ampEnabled,
            stage=f"val {epoch}",
            **valCallKwargs,
        )

        if scheduler is not None:
            scheduler.step()

        history.append(history_row(epoch, train_metrics, val_metrics, optimizer))

        current_score = float(val_metrics[score_name])
        log_info(
            f"[epoch {epoch}/{epochTotal}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_{score_name}={current_score:.4f}"
        )
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": config,
                    "epoch": epoch,
                    checkpoint_score_name: current_score,
                },
                output_dir / "best.pt",
            )
            epochs_without_improvement = 0
            log_info(
                f"[best] {resolvedRunName} | epoch={epoch} | {checkpoint_score_name}={current_score:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if epoch_callback is not None:
            epoch_callback(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                current_score=current_score,
                best_score=best_score,
                best_epoch=best_epoch,
            )

        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            log_info(
                f"[early-stop] {resolvedRunName} | epoch={epoch} | patience={early_stopping_patience}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if not run_test:
        log_info(
            f"[done] {resolved_run_name} | best_epoch={best_epoch} | best_{score_name}={best_score:.4f}"
        )
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "test_metrics": None,
        }

    log_info(f"[test] {resolvedRunName} | evaluating best checkpoint")
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp=ampEnabled,
        stage="test",
        **testCallKwargs,
    )
    log_info(
        f"[done] {resolvedRunName} | best_epoch={best_epoch} | best_{score_name}={best_score:.4f} | "
        f"test_{score_name}={float(test_metrics[score_name]):.4f}"
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "test_metrics": test_metrics,
    }
