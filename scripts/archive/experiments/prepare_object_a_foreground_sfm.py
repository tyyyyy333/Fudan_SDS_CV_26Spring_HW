#!/usr/bin/env python3
"""Prepare foreground-first Object A images for object-centric SfM."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove


def remove_skin_and_keep_cup(rgba: Image.Image) -> Image.Image:
    arr = np.asarray(rgba).copy()
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    skin = (
        (r > 55)
        & (g > 25)
        & (b > 15)
        & (r > g + 7)
        & (r > b + 12)
        & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 18)
    )
    skin = cv2.morphologyEx(
        skin.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    )
    skin = cv2.dilate(skin, np.ones((7, 7), dtype=np.uint8), iterations=1).astype(bool)

    candidate = ((alpha > 24) & ~skin).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return rgba

    h, w = candidate.shape
    center = np.array([w * 0.5, h * 0.48])
    best_label = None
    best_score = -np.inf
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 0.002 * h * w:
            continue
        distance = np.linalg.norm((centroids[label] - center) / np.array([w, h]))
        score = np.log1p(area) - 3.0 * distance
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return rgba

    x, y, box_w, box_h, _ = stats[best_label]
    center_x = x + box_w / 2
    center_y = y + box_h / 2
    radius_x = max(box_w * 0.68, w * 0.08)
    radius_y = max(box_h * 0.68, h * 0.08)
    yy, xx = np.ogrid[:h, :w]
    cup_region = (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
        <= 1.0
    )
    arr[:, :, 3] = np.where(cup_region, alpha, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def composite_white(rgba: Image.Image) -> Image.Image:
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def process_group(
    paths: list[Path],
    output: Path,
    session,
    *,
    remove_skin: bool,
    max_side: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        rgba = remove(image, session=session).convert("RGBA")
        if remove_skin:
            rgba = remove_skin_and_keep_cup(rgba)
        composite_white(rgba).save(output / f"{path.stem}.png", format="PNG")
        if index % 10 == 0 or index == len(paths):
            print(f"prepared {index}/{len(paths)}: {output / f'{path.stem}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=Path("data/task1/object_a/images"))
    parser.add_argument(
        "--supplement-dir",
        type=Path,
        default=Path("data/object_A_pic_supplement"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/task1/object_a_foreground_sfm/input"),
    )
    parser.add_argument("--model", default="u2net")
    parser.add_argument("--video-max-side", type=int, default=960)
    parser.add_argument("--supplement-max-side", type=int, default=1600)
    args = parser.parse_args()

    video_paths = sorted(args.video_dir.glob("*"))
    supplement_paths = sorted(args.supplement_dir.glob("*"))
    if not video_paths or not supplement_paths:
        raise FileNotFoundError("Object A video frames or supplement images are missing")

    session = new_session(args.model)
    process_group(
        video_paths,
        args.output / "video",
        session,
        remove_skin=False,
        max_side=args.video_max_side,
    )
    process_group(
        supplement_paths,
        args.output / "supplement",
        session,
        remove_skin=True,
        max_side=args.supplement_max_side,
    )


if __name__ == "__main__":
    main()
