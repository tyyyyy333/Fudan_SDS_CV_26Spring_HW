"""Server-friendly video encoding helpers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def get_ffmpeg() -> str:
    ffmpeg = os.environ.get("FFMPEG_BINARY")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def encode_video_from_pattern(
    frame_pattern: Path,
    output: Path,
    fps: int,
    frames: int | None = None,
    crf: int = 18,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
    ]
    if frames is not None:
        cmd += ["-frames:v", str(frames)]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def encode_video_from_frames(
    frames: list[np.ndarray],
    output: Path,
    fps: int,
    crf: int = 18,
) -> None:
    if not frames:
        raise ValueError("No frames to encode")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hw3-video-") as tmp:
        frame_dir = Path(tmp)
        for idx, frame in enumerate(frames):
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 frame, got {frame.shape}")
            cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        encode_video_from_pattern(frame_dir / "%05d.jpg", output, fps, len(frames), crf=crf)
