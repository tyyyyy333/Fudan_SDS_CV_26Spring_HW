import torch
import torch.nn.functional as F
from torch import nn


def _soft_targets(targets, num_classes, label_smoothing):
    if targets.ndim == 2:
        return targets.float()
    smoothing = float(label_smoothing)
    confidence = 1.0 - smoothing
    off_value = smoothing / num_classes
    target_probs = torch.full(
        (targets.size(0), num_classes),
        fill_value=off_value,
        dtype=torch.float32,
        device=targets.device,
    )
    target_probs.scatter_(1, targets.unsqueeze(1), confidence + off_value)
    return target_probs


class SoftTargetFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.supports_soft_targets = True

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        target_probs = _soft_targets(targets, num_classes, self.label_smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        focal_weight = (1.0 - probs).pow(self.gamma)
        return -(target_probs * focal_weight * log_probs).sum(dim=1).mean()


class MixedCrossEntropyFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, focal_weight=0.5, label_smoothing=0.0):
        super().__init__()
        self.focal = SoftTargetFocalLoss(gamma=gamma, label_smoothing=label_smoothing)
        self.focal_weight = float(focal_weight)
        self.label_smoothing = float(label_smoothing)
        self.supports_soft_targets = True

    def forward(self, logits, targets):
        if targets.ndim == 2:
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(targets * log_probs).sum(dim=1).mean()
        else:
            ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return (1.0 - self.focal_weight) * ce_loss + self.focal_weight * self.focal(logits, targets)


def build_criterion(config):
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "ce")
    label_smoothing = float(config["train"].get("label_smoothing", 0.0))

    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_name == "focal":
        return SoftTargetFocalLoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            label_smoothing=label_smoothing,
        )
    if loss_name == "ce_focal":
        return MixedCrossEntropyFocalLoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            focal_weight=float(loss_cfg.get("focal_weight", 0.5)),
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unsupported Task 1 loss: {loss_name}")
