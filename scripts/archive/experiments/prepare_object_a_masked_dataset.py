#!/usr/bin/env python3
"""Prepare an alpha-masked Object A COLMAP dataset for 2DGS retraining."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from rembg import new_session, remove


def prepare(src: Path, dst: Path, model: str):
    if not (src / "images").exists() or not (src / "sparse").exists():
        raise FileNotFoundError(f"Expected COLMAP dataset with images/ and sparse/: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "sparse").exists():
        shutil.rmtree(dst / "sparse")
    shutil.copytree(src / "sparse", dst / "sparse", symlinks=True)

    image_out = dst / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    session = new_session(model)
    inputs = sorted((src / "images").glob("*"))
    if not inputs:
        raise FileNotFoundError(f"No source images found in {src / 'images'}")

    for idx, path in enumerate(inputs, 1):
        img = Image.open(path).convert("RGB")
        rgba = remove(img, session=session).convert("RGBA")
        alpha = rgba.getchannel("A")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
        rgb_white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb_white.paste(rgba, mask=alpha)
        rgb_white.putalpha(alpha)
        out_path = image_out / path.name
        # Keep the COLMAP image basename unchanged. PIL readers detect PNG by
        # file signature even when the original extension is .jpg.
        rgb_white.save(out_path, format="PNG")
        if idx % 10 == 0 or idx == len(inputs):
            print(f"masked {idx}/{len(inputs)}: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/task1/object_a_undistorted"))
    parser.add_argument("--output", type=Path, default=Path("data/task1/object_a_masked_undistorted"))
    parser.add_argument("--model", default="u2net")
    args = parser.parse_args()
    prepare(args.source, args.output, args.model)


if __name__ == "__main__":
    main()
