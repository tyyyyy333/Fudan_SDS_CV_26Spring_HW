import torch
import torch.nn.functional as F
from tqdm import tqdm

from hw2_cv.runner import autocast_context


def _classification_loss(criterion, logits, targets):
    if targets.ndim == 2:
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()
    return criterion(logits, targets)


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
    mixup_fn=None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = _classification_loss(criterion, logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        target_labels = targets.argmax(dim=1) if targets.ndim == 2 else targets
        batch_size = target_labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == target_labels).sum().item()
        total_samples += batch_size

        if step % max(log_interval, 1) == 0:
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{total_correct / max(total_samples, 1):.4f}",
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    amp,
    stage,
    collect_outputs=False,
    tta_horizontal_flip=False,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    records = []

    progress = tqdm(loader, desc=stage, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast_context(device, amp):
            logits = model(images)
            if tta_horizontal_flip:
                flipped_images = torch.flip(images, dims=[3])
                flipped_logits = model(flipped_images)
                logits = 0.5 * (logits + flipped_logits)
            loss = _classification_loss(criterion, logits, targets)

        probabilities = F.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_samples += batch_size

        if collect_outputs:
            image_ids = batch["image_id"]
            indices = batch["index"].tolist()
            for position in range(batch_size):
                records.append(
                    {
                        "image_id": image_ids[position],
                        "index": int(indices[position]),
                        "target": int(targets[position].item()),
                        "prediction": int(predictions[position].item()),
                        "confidence": float(confidences[position].item()),
                    }
                )

    result = {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }
    if collect_outputs:
        result["records"] = records
    return result
