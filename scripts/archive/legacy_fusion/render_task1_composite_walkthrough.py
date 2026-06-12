#!/usr/bin/env python3
"""Create a readable Task 1 walkthrough video from completed visual assets.

This is a presentation fallback for environments where Blender cannot render
and raw point-cloud videos are too dark/sparse to inspect. It composites the
completed A/B/C results over the reconstructed kitchen-sequence background and
adds a mild orbit/parallax motion.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from video_utils import encode_video_from_frames


ROOT = Path(__file__).resolve().parents[1]


def crop_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def make_soft_mask(img: Image.Image, mode: str) -> Image.Image:
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    if mode == "white":
        dist = np.maximum.reduce([255 - rgb[..., 0], 255 - rgb[..., 1], 255 - rgb[..., 2]])
        alpha = np.clip(dist * 2.8, 0, 255).astype(np.uint8)
    elif mode == "black":
        bright = np.maximum.reduce([rgb[..., 0], rgb[..., 1], rgb[..., 2]])
        alpha = np.clip((bright - 12) * 4.0, 0, 255).astype(np.uint8)
    else:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    mask = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(1.2))
    return mask


def extract_assets(fig_dir: Path):
    bg = Image.open(fig_dir / "task1_background_camera_views" / "render_01.png").convert("RGB")
    bg = crop_cover(bg, (1280, 720))
    bg = ImageEnhance.Brightness(bg).enhance(1.18)
    bg = ImageEnhance.Contrast(bg).enhance(1.04)

    a_sheet = Image.open(fig_dir / "task1_object_a_render.png").convert("RGB")
    aw, ah = a_sheet.size
    a = a_sheet.crop((aw // 3, ah // 2 + 16, aw * 2 // 3, ah - 10))
    a = ImageOps.contain(a, (310, 210), Image.Resampling.LANCZOS)

    b_sheet = Image.open(fig_dir / "task1_object_b_render.jpg").convert("RGB")
    b = b_sheet.crop((0, 150, 240, 390))
    b = ImageOps.contain(b, (240, 180), Image.Resampling.LANCZOS)

    c_sheet = Image.open(fig_dir / "task1_object_c_render.jpg").convert("RGB")
    c = c_sheet.crop((90, 225, 180, 315))
    c = ImageOps.contain(c, (240, 170), Image.Resampling.LANCZOS)
    c = ImageEnhance.Contrast(c).enhance(1.15)

    return bg, [
        ("A", a, make_soft_mask(a, "none"), (-330, 62), 1.05),
        ("B", b, make_soft_mask(b, "none"), (0, 78), 1.10),
        ("C", c, make_soft_mask(c, "white"), (330, 80), 1.35),
    ]


def paste_asset(canvas: Image.Image, asset: Image.Image, mask: Image.Image, center: tuple[float, float], scale: float):
    w = max(1, int(asset.width * scale))
    h = max(1, int(asset.height * scale))
    obj = asset.resize((w, h), Image.Resampling.LANCZOS)
    m = mask.resize((w, h), Image.Resampling.LANCZOS)
    shadow = Image.new("RGBA", (w + 36, h + 36), (0, 0, 0, 0))
    shadow_mask = m.filter(ImageFilter.GaussianBlur(10))
    shadow.paste((0, 0, 0, 90), (18, 18), shadow_mask)
    x = int(center[0] - w / 2)
    y = int(center[1] - h / 2)
    canvas.alpha_composite(shadow, (x - 18, y - 8))
    canvas.paste(obj.convert("RGBA"), (x, y), m)


def draw_floor_markers(frame: Image.Image):
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    cv = np.array(overlay)
    pts = np.array([[230, 582], [640, 520], [1050, 582], [1000, 632], [280, 632]], dtype=np.int32)
    cv2.polylines(cv, [pts], True, (35, 35, 35, 120), 2, cv2.LINE_AA)
    overlay = Image.fromarray(cv, "RGBA")
    frame.alpha_composite(overlay)


def render(args):
    fig_dir = ROOT / "docs" / "report_cvpr" / "figures"
    bg, assets = extract_assets(fig_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.contact.parent.mkdir(parents=True, exist_ok=True)

    contact_frames = []
    video_frames: list[np.ndarray] = []
    for i in range(args.frames):
        t = i / args.frames
        phase = 2 * math.pi * t
        bg_zoom = 1.06 + 0.025 * math.sin(phase)
        bg_w = int(1280 * bg_zoom)
        bg_h = int(720 * bg_zoom)
        bg_big = bg.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
        shift_x = int(34 * math.sin(phase))
        shift_y = int(12 * math.cos(phase))
        frame = bg_big.crop(((bg_w - 1280) // 2 + shift_x, (bg_h - 720) // 2 + shift_y,
                             (bg_w - 1280) // 2 + shift_x + 1280, (bg_h - 720) // 2 + shift_y + 720)).convert("RGBA")
        veil = Image.new("RGBA", frame.size, (255, 255, 255, 70))
        frame.alpha_composite(veil)
        draw_floor_markers(frame)

        for idx, (_, img, mask, base, scale) in enumerate(assets):
            parallax = 36 * math.sin(phase + idx * 0.8)
            bob = 10 * math.cos(phase + idx * 0.5)
            cx = 640 + base[0] + parallax
            cy = 455 + base[1] + bob
            local_scale = scale * (1.0 + 0.04 * math.sin(phase + idx))
            paste_asset(frame, img, mask, (cx, cy), local_scale)

        out = cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        video_frames.append(out)
        if i in {0, args.frames // 6, args.frames // 3, args.frames // 2, args.frames * 2 // 3, args.frames * 5 // 6}:
            contact_frames.append(out)
    encode_video_from_frames(video_frames, args.output, fps=30)

    small = [cv2.resize(f, (426, 240), interpolation=cv2.INTER_AREA) for f in contact_frames]
    rows = [np.concatenate(small[:3], axis=1), np.concatenate(small[3:6], axis=1)]
    cv2.imwrite(str(args.contact), np.concatenate(rows, axis=0))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.contact}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task1/fused_scene_readable_composite.avi")
    parser.add_argument("--contact", type=Path, default=ROOT / "outputs/task1/fused_scene_readable_composite_contact.jpg")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fourcc", default="MJPG")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
