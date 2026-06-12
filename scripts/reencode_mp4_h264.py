#!/usr/bin/env python3
"""Re-encode MP4 files to broadly compatible H.264/yuv420p."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from video_utils import get_ffmpeg


def reencode(input_path: Path, output_path: Path, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=output_path.stem + ".", suffix=".mp4", dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            get_ffmpeg(),
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            "-an",
            str(tmp_path),
        ]
        subprocess.run(cmd, check=True)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="MP4 files or directories to process.")
    parser.add_argument("--in-place", action="store_true", help="Replace input files instead of writing *_h264.mp4.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality. Lower is higher quality.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.mp4")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)

    for input_path in files:
        if args.in_place:
            output_path = input_path
        else:
            output_path = input_path.with_name(input_path.stem + "_h264.mp4")
        print(f"{input_path} -> {output_path}")
        reencode(input_path, output_path, args.crf)


if __name__ == "__main__":
    main()
