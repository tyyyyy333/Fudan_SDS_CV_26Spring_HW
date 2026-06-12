#!/usr/bin/env python3
"""Create a white-background Object A dataset while preserving COLMAP names."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from rembg import new_session, remove


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="u2net")
    args = parser.parse_args()

    if not (args.source / "images").is_dir() or not (args.source / "sparse").is_dir():
        raise FileNotFoundError("Source must contain images/ and sparse/")

    args.output.mkdir(parents=True, exist_ok=True)
    output_sparse = args.output / "sparse"
    if not output_sparse.exists():
        shutil.copytree(args.source / "sparse", output_sparse)

    inputs = sorted(path for path in (args.source / "images").rglob("*") if path.is_file())
    session = new_session(args.model)
    completed = 0
    for index, path in enumerate(inputs, 1):
        relative = path.relative_to(args.source / "images")
        destination = args.output / "images" / relative
        if destination.exists():
            completed += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgba = remove(Image.open(path).convert("RGB"), session=session).convert("RGBA")
        alpha = rgba.getchannel("A")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.paste(rgba, mask=alpha)
        white.convert("RGB").save(destination, format="PNG", optimize=True)
        completed += 1
        if index % 10 == 0 or index == len(inputs):
            print(f"masked={completed}/{len(inputs)} latest={relative}", flush=True)


if __name__ == "__main__":
    main()
