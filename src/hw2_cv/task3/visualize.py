from pathlib import Path

import cv2
import numpy as np
import torch

from hw2_cv.utils import ensure_dir


PALETTE = np.array(
    [
        [255, 140, 0],
        [30, 30, 30],
        [70, 130, 180],
    ],
    dtype=np.uint8,
)


def _denormalize(image, mean, std):
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    image_np = image_np * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return image_np


def colorize_mask(mask):
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    return PALETTE[mask_np]


def overlay_mask(image, mask, alpha=0.45):
    return cv2.addWeighted(image, 1.0 - alpha, mask, alpha, 0.0)


def export_prediction_samples(predictions, output_dir, mean, std, sample_count):
    output_dir = ensure_dir(output_dir)
    saved_paths = []

    for record in predictions[:sample_count]:
        image_id = record["image_id"]
        image = _denormalize(record["image"], mean=mean, std=std)
        target_mask = colorize_mask(record["target_mask"])
        predicted_mask = colorize_mask(record["predicted_mask"])
        overlay = overlay_mask(image, predicted_mask)

        input_path = output_dir / f"{image_id}_input.png"
        target_path = output_dir / f"{image_id}_gt.png"
        prediction_path = output_dir / f"{image_id}_pred.png"
        overlay_path = output_dir / f"{image_id}_overlay.png"

        cv2.imwrite(str(input_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(target_path), cv2.cvtColor(target_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(prediction_path), cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        saved_paths.extend([str(input_path), str(target_path), str(prediction_path), str(overlay_path)])

    return saved_paths
