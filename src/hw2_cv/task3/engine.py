import torch
from tqdm import tqdm

from hw2_cv.runner import autocast_context
from hw2_cv.task3.metrics import SegmentationMetric

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    epoch,
    log_interval,
    amp,
    num_classes,
):
    model.train()
    metric = SegmentationMetric(num_classes=num_classes)
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = masks.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        metric.update(logits.detach(), masks.detach())

        if step % max(log_interval, 1) == 0:
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                miou=f"{metric.mean_iou():.4f}",
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "miou": metric.mean_iou(),
        "pixel_acc": metric.pixel_accuracy(),
        "classwise_iou": metric.classwise_iou(),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    amp,
    stage,
    num_classes,
    collect_predictions=False,
    max_predictions=None,
):
    model.eval()
    metric = SegmentationMetric(num_classes=num_classes)
    total_loss = 0.0
    total_samples = 0
    predictions_for_export = []

    progress = tqdm(loader, desc=stage, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        
        
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_size = masks.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        metric.update(logits, masks)

        if collect_predictions and (max_predictions is None or len(predictions_for_export) < max_predictions):
            predicted_masks = logits.argmax(dim=1)
            image_ids = batch["image_id"]
            indices = batch["index"].tolist()
            for position in range(batch_size):
                if max_predictions is not None and len(predictions_for_export) >= max_predictions:
                    break
                predictions_for_export.append(
                    {
                        "image_id": image_ids[position],
                        "index": int(indices[position]),
                        "image": images[position].detach().cpu(),
                        "target_mask": masks[position].detach().cpu(),
                        "predicted_mask": predicted_masks[position].detach().cpu(),
                    }
                )

    result = {
        "loss": total_loss / max(total_samples, 1),
        "miou": metric.mean_iou(),
        "pixel_acc": metric.pixel_accuracy(),
        "classwise_iou": metric.classwise_iou(),
    }
    if collect_predictions:
        result["predictions"] = predictions_for_export
    return result
