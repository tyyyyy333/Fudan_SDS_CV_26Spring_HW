#!/usr/bin/env python3
"""Select sharp, temporally distributed keyframes from an Object A video."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_contact(selected: list[tuple[int, float, Path]], output: Path) -> None:
    sample = np.linspace(0, len(selected) - 1, min(24, len(selected)), dtype=int)
    tile_w, tile_h, label_h, cols = 480, 270, 28, 4
    rows = math.ceil(len(sample) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for rank, selected_index in enumerate(sample):
        source_frame, score, path = selected[selected_index]
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = rank % cols * tile_w
        y = rank // cols * (tile_h + label_h)
        draw.text((x + 8, y + 6), f"source {source_frame:04d} sharp {score:.1f}", fill="black")
        canvas.paste(image, (x + (tile_w - image.width) // 2, y + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=360)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--contact", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < args.count:
        raise ValueError(f"Video has {frame_count} frames, fewer than requested {args.count}")

    shutil.rmtree(args.output, ignore_errors=True)
    args.output.mkdir(parents=True)
    edges = np.linspace(0, frame_count, args.count + 1, dtype=int)
    selected: list[tuple[int, float, Path]] = []

    # Decode sequentially once. Seeking for every candidate is much slower and
    # may return neighboring keyframes for variable-frame-rate phone videos.
    current_bin = 0
    best_frame = None
    best_number = -1
    best_score = -1.0
    frame_number = 0
    while current_bin < args.count:
        ok, frame = capture.read()
        if not ok:
            break
        while current_bin < args.count and frame_number >= edges[current_bin + 1]:
            if best_frame is None:
                raise RuntimeError(f"No decodable frame in temporal bin {current_bin}")
            destination = args.output / f"frame_{best_number:06d}.jpg"
            cv2.imwrite(str(destination), best_frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            selected.append((best_number, best_score, destination))
            current_bin += 1
            best_frame = None
            best_number = -1
            best_score = -1.0
        if current_bin >= args.count:
            break
        score = sharpness(frame)
        if score > best_score:
            best_frame = frame.copy()
            best_number = frame_number
            best_score = score
        frame_number += 1

    if current_bin < args.count and best_frame is not None:
        destination = args.output / f"frame_{best_number:06d}.jpg"
        cv2.imwrite(str(destination), best_frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        selected.append((best_number, best_score, destination))
        current_bin += 1
    capture.release()
    if len(selected) != args.count:
        raise RuntimeError(f"Selected {len(selected)} keyframes, expected {args.count}")

    csv_path = args.csv or args.output.parent / "keyframes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_index", "source_frame", "sharpness", "image"])
        for index, (source_frame, score, path) in enumerate(selected):
            writer.writerow([index, source_frame, f"{score:.6f}", path.name])

    contact_path = args.contact or args.output.parent / "keyframes_contact.jpg"
    save_contact(selected, contact_path)
    scores = np.asarray([score for _, score, _ in selected])
    print(f"video_frames={frame_count}")
    print(f"selected_keyframes={len(selected)}")
    print(f"sharpness_min={scores.min():.6f}")
    print(f"sharpness_median={np.median(scores):.6f}")
    print(f"sharpness_max={scores.max():.6f}")
    print(f"images={args.output}")
    print(f"csv={csv_path}")
    print(f"contact={contact_path}")


if __name__ == "__main__":
    main()
