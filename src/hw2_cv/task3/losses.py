import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probabilities * one_hot_targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + one_hot_targets.sum(dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        ce_weight=1.0,
        dice_weight=1.0,
        label_smoothing=0.0,
        class_weights=None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weight_tensor = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weight_tensor", weight_tensor, persistent=False)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weight_tensor,
            label_smoothing=label_smoothing,
        )
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        ce_value = self.ce_loss(logits, targets)
        dice_value = self.dice_loss(logits, targets)
        return self.ce_weight * ce_value + self.dice_weight * dice_value


def build_criterion(config, num_classes, class_weights_override=None):
    loss_cfg = config["loss"]
    loss_name = loss_cfg["name"]
    class_weights = class_weights_override if class_weights_override is not None else loss_cfg.get("class_weights")
    weight_tensor = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)

    if loss_name == "ce":
        return nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )
    if loss_name == "dice":
        return DiceLoss(num_classes=num_classes)
    if loss_name == "ce_dice":
        return CombinedSegmentationLoss(
            num_classes=num_classes,
            ce_weight=loss_cfg.get("ce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
            class_weights=class_weights,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")
