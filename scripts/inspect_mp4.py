#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an MP4 without ffprobe.")
    parser.add_argument("video", type=Path)
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(args.video)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stream = next(
        (line.strip() for line in result.stderr.splitlines() if "Video:" in line),
        "Video stream metadata unavailable",
    )
    print(f"path={args.video}")
    print(f"width={width} height={height} frames={frames} fps={fps:.3f}")
    print(stream)


if __name__ == "__main__":
    main()
