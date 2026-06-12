from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = ROOT / "external" / "2d-gaussian-splatting"
sys.path.insert(0, str(GS_ROOT))
sys.path.insert(0, str(GS_ROOT / "submodules/simple-knn"))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--image-names", nargs="*", default=None)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--white-background", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    ns = parser.parse_args(
        [
            "--source_path",
            str(Path(args.source).resolve()),
            "--model_path",
            str(Path(args.model).resolve()),
            "--resolution",
            str(args.resolution),
        ]
        + (["--white_background"] if args.white_background else [])
    )
    dataset = model.extract(ns)
    pipe = pipeline.extract(ns)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if not cameras:
        raise RuntimeError(f"No cameras found for split {args.split}")

    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    if args.image_names:
        targets = set(args.image_names)
        chosen = [
            i
            for i, cam in enumerate(cameras)
            if getattr(cam, "image_name", "") in targets
        ]
        if not chosen:
            raise RuntimeError(f"No cameras matched image names: {sorted(targets)}")
    else:
        chosen = []
        i = args.start
        while len(chosen) < args.num_views and i < len(cameras):
            chosen.append(i)
            i += max(1, args.stride)
        if len(chosen) < args.num_views:
            chosen = list(range(min(args.num_views, len(cameras))))

    for rank, idx in enumerate(chosen):
        cam = cameras[idx]
        with torch.no_grad():
            pred = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
        torchvision.utils.save_image(pred, out / f"render_{rank:02d}.png")
        if getattr(cam, "original_image", None) is not None:
            torchvision.utils.save_image(cam.original_image.clamp(0, 1), out / f"gt_{rank:02d}.png")
        meta = out / f"view_{rank:02d}.txt"
        meta.write_text(f"split={args.split}\nindex={idx}\nimage_name={getattr(cam, 'image_name', '')}\n")


if __name__ == "__main__":
    main()
