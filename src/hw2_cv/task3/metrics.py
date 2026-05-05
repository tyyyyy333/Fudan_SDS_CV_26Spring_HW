import torch


class SegmentationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, targets):
        predictions = logits.argmax(dim=1)
        predictions = predictions.view(-1).cpu()
        targets = targets.view(-1).cpu()

        valid_mask = (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        bins = self.num_classes * targets + predictions
        bincount = torch.bincount(bins, minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def pixel_accuracy(self):
        total = self.confusion_matrix.sum().item()
        if total == 0:
            return 0.0
        return self.confusion_matrix.diag().sum().item() / total

    def mean_iou(self):
        classwise_iou = self.classwise_iou()
        valid_iou = [value for value in classwise_iou if value is not None]
        if not valid_iou:
            return 0.0
        return float(sum(valid_iou) / len(valid_iou))

    def classwise_iou(self):
        true_positive = self.confusion_matrix.diag()
        false_positive = self.confusion_matrix.sum(dim=0) - true_positive
        false_negative = self.confusion_matrix.sum(dim=1) - true_positive
        denominator = true_positive + false_positive + false_negative
        results = []
        for class_index in range(self.num_classes):
            if denominator[class_index] <= 0:
                results.append(None)
            else:
                results.append((true_positive[class_index] / denominator[class_index]).item())
        return results
