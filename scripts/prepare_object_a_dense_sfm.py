#!/usr/bin/env python3
"""Prepare all Object A video frames and supplemental photos for COLMAP."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def frame_metrics(path: Path) -> tuple[bool, float, float, float, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False, 0.0, 0.0, 1.0, 1.0
    return (
        True,
        float(cv2.Laplacian(image, cv2.CV_64F).var()),
        float(image.mean()),
        float(np.mean(image <= 5)),
        float(np.mean(image >= 250)),
    )


def link_video_frames(source: Path, output: Path, metrics_csv: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    accepted = 0
    for path in sorted(source.glob("frame_*.jpg")):
        decoded, sharpness, mean, black_ratio, white_ratio = frame_metrics(path)
        # Only reject corrupt or effectively blank frames. Motion blur and
        # geometry outliers are intentionally left for COLMAP/post-SfM filtering.
        keep = decoded and mean > 3.0 and mean < 252.0 and black_ratio < 0.98 and white_ratio < 0.98
        rows.append(
            [
                path.name,
                int(decoded),
                f"{sharpness:.6f}",
                f"{mean:.6f}",
                f"{black_ratio:.8f}",
                f"{white_ratio:.8f}",
                int(keep),
            ]
        )
        if keep:
            target = output / path.name
            target.hardlink_to(path.resolve())
            accepted += 1

    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "decoded", "laplacian_variance", "mean_gray", "black_ratio", "white_ratio", "pre_sfm_keep"]
        )
        writer.writerows(rows)
    return accepted


def prepare_photos(source: Path, output: Path, max_side: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        image.save(output / f"{path.stem}.jpg", quality=95, subsampling=0)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-frames", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--photo-max-side", type=int, default=1920)
    args = parser.parse_args()

    image_root = args.workspace / "images"
    shutil.rmtree(image_root, ignore_errors=True)
    image_root.mkdir(parents=True)
    accepted = link_video_frames(
        args.video_frames,
        image_root / "video",
        args.workspace / "frame_quality.csv",
    )
    photos = prepare_photos(args.supplement, image_root / "supplement", args.photo_max_side)
    print(f"video_frames_kept={accepted}")
    print(f"supplement_photos={photos}")
    print(f"image_root={image_root}")


if __name__ == "__main__":
    main()
