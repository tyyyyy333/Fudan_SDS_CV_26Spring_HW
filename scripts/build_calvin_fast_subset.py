#!/usr/bin/env python3
"""Build a reproducible, fast-to-read CALVIN LeRobot subset.

The source ``huiwon/calvin_task_ABC_D`` dataset stores both camera streams in
compressed AV1 videos. This script samples complete episodes proportionally
across A/B/C/D and rewrites them either as decoded image features (the default
used by this project) or as videos with a user-selected codec. Image storage
uses more disk space but removes AV1 decoding from the ACT training hot path.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets import LeRobotDataset


AUTO_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "original_frame_idx",
}

SHARD_ENV_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def load_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta" / "info.json").read_text())


def load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files under {root}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def source_features(dataset: LeRobotDataset, *, store: str) -> dict[str, dict[str, Any]]:
    features = {}
    for key, value in dataset.features.items():
        if key in AUTO_KEYS:
            continue
        feature = dict(value)
        if store == "image" and feature.get("dtype") == "video":
            feature["dtype"] = "image"
            feature.pop("info", None)
        features[key] = feature
    return features


def frame_for_write(sample: dict[str, Any], feature_keys: set[str]) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for key in feature_keys:
        if key not in sample:
            continue
        value = sample[key]
        if key.startswith("observation.images.") and isinstance(value, torch.Tensor):
            if value.ndim == 3 and value.shape[0] == 3:
                value = value.permute(1, 2, 0).contiguous()
        elif key == "annotation.human.action.task_description" and isinstance(value, torch.Tensor):
            if value.ndim == 0:
                value = value.reshape(1)
        frame[key] = value
    frame["task"] = sample["task"]
    return frame


def select_episodes(
    episodes: pd.DataFrame,
    *,
    quota_frames: int,
    seed: int,
) -> tuple[list[int], int]:
    rows = episodes[["episode_index", "length"]].copy()
    items = [(int(r.episode_index), int(r.length)) for r in rows.itertuples(index=False)]
    rng = random.Random(seed)
    rng.shuffle(items)
    selected: list[int] = []
    total = 0
    for episode_idx, length in items:
        if selected and total + length > quota_frames:
            continue
        selected.append(episode_idx)
        total += length
        if total >= quota_frames:
            break
    if not selected and items:
        selected = [items[0][0]]
        total = items[0][1]
    return selected, total


def build_one_shard(
    *,
    src_root: Path,
    dst_root: Path,
    repo_id: str,
    selected_episodes: list[int],
    codec: str,
    crf: float,
    preset: str,
    gop: int,
    encoder_threads: int | None,
    store: str,
    image_writer_threads: int,
    tolerance_s: float,
    video_file_mb: int,
    data_file_mb: int,
) -> dict[str, Any]:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)

    src = LeRobotDataset(
        repo_id=repo_id,
        root=src_root,
        episodes=selected_episodes,
        video_backend="pyav",
        return_uint8=True,
        tolerance_s=tolerance_s,
    )
    features = source_features(src, store=store)
    feature_keys = set(features)
    encoder = None
    if store == "video":
        encoder = VideoEncoderConfig(
            vcodec=codec,
            crf=crf,
            preset=preset,
            g=gop,
            pix_fmt="yuv420p",
            video_backend="pyav",
        )
    dst = LeRobotDataset.create(
        repo_id=repo_id,
        root=dst_root,
        fps=src.fps,
        features=features,
        use_videos=store == "video",
        video_backend="pyav",
        camera_encoder=encoder,
        encoder_threads=encoder_threads,
        image_writer_threads=image_writer_threads,
        video_files_size_in_mb=video_file_mb,
        data_files_size_in_mb=data_file_mb,
        tolerance_s=tolerance_s,
    )

    selected_set = set(selected_episodes)
    written_episodes = 0
    written_frames = 0
    current_episode = None
    start = time.time()

    for i in range(len(src)):
        sample = src[i]
        episode_idx = int(sample["episode_index"])
        if episode_idx not in selected_set:
            continue
        if current_episode is None:
            current_episode = episode_idx
        if episode_idx != current_episode:
            dst.save_episode(parallel_encoding=True)
            written_episodes += 1
            current_episode = episode_idx
        dst.add_frame(frame_for_write(sample, feature_keys))
        written_frames += 1

    if dst.has_pending_frames():
        dst.save_episode(parallel_encoding=True)
        written_episodes += 1
    dst.finalize()

    elapsed = time.time() - start
    return {
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "repo_id": repo_id,
        "selected_episodes": selected_episodes,
        "written_episodes": written_episodes,
        "written_frames": written_frames,
        "size_gb": round(dir_size(dst_root) / 1024**3, 3),
        "elapsed_s": round(elapsed, 2),
        "codec": codec,
        "crf": crf,
        "preset": preset,
        "gop": gop,
        "store": store,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/calvin_hf/huiwon_calvin_task_ABC_D"))
    parser.add_argument("--output-root", type=Path, default=Path("data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D"))
    parser.add_argument("--target-raw-gb", type=float, default=40.0)
    parser.add_argument("--target-frames-total", type=int, default=None)
    parser.add_argument("--expected-output-gb", type=float, default=None)
    parser.add_argument("--seed", type=int, default=23300200022)
    parser.add_argument("--codec", default="h264")
    parser.add_argument("--crf", type=float, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--gop", type=int, default=2)
    parser.add_argument("--store", choices=["video", "image"], default="image")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--video-file-mb", type=int, default=512)
    parser.add_argument("--data-file-mb", type=int, default=100)
    parser.add_argument("--tolerance-s", type=float, default=0.01)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--only-shard", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument("--no-clean-output-root", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.no_clean_output_root:
        if not args.force:
            raise FileExistsError(f"{args.output_root} exists; pass --force to rebuild")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    available_gb = free_bytes(args.output_root) / 1024**3
    space_target_gb = args.expected_output_gb if args.expected_output_gb is not None else args.target_raw_gb
    if available_gb < space_target_gb + args.min_free_gb:
        raise RuntimeError(
            f"Only {available_gb:.1f} GB free at {args.output_root}; "
            f"expected_output_gb={space_target_gb:.1f} plus min_free_gb={args.min_free_gb:.1f} "
            "would risk filling the filesystem."
        )

    shard_roots = [args.source_root / f"calvin_task_ABC_D_lerobot_{i}_4" for i in range(4)]
    infos = [load_info(p) for p in shard_roots]
    frames = [int(info["total_frames"]) for info in infos]
    total_frames = sum(frames)

    # Raw uint8 RGB equivalent for two 256x256 cameras. This is the user's
    # requested "decoded" budget; actual H.264 size is recorded after encoding.
    bytes_per_frame = 256 * 256 * 3 * 2
    target_frames_total = args.target_frames_total
    if target_frames_total is None:
        target_frames_total = int(args.target_raw_gb * 1024**3 / bytes_per_frame)
    quotas = [max(1, round(target_frames_total * f / total_frames)) for f in frames]

    selected_shards = [args.only_shard] if args.only_shard is not None else list(range(4))

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "source_dataset": "huiwon/calvin_task_ABC_D",
        "source_dataset_url": "https://huggingface.co/datasets/huiwon/calvin_task_ABC_D",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "target_raw_gb": args.target_raw_gb,
        "target_frames_total_override": args.target_frames_total,
        "expected_output_gb": args.expected_output_gb,
        "available_gb_before_build": round(available_gb, 3),
        "min_free_gb": args.min_free_gb,
        "target_frames_total": target_frames_total,
        "bytes_per_frame_raw_uint8_two_cameras": bytes_per_frame,
        "ratio_basis": "current HF shard frame counts, sampled proportionally across A/B/C/D",
        "environment_mapping": {str(k): v for k, v in SHARD_ENV_MAP.items()},
        "codec": args.codec,
        "crf": args.crf,
        "preset": args.preset,
        "gop": args.gop,
        "store": args.store,
        "selected_shards": selected_shards,
        "shards": [],
    }

    for shard_idx, (src_root, quota) in enumerate(zip(shard_roots, quotas, strict=True)):
        if shard_idx not in selected_shards:
            continue
        episodes = load_episodes(src_root)
        selected, selected_frames = select_episodes(
            episodes,
            quota_frames=quota,
            seed=args.seed + shard_idx,
        )
        dst_root = args.output_root / f"calvin_task_ABC_D_lerobot_{shard_idx}_4"
        repo_id = f"local/calvin_task_ABC_D_lerobot_{shard_idx}_4"
        print(
            f"[shard {shard_idx}] selected {len(selected)} episodes, "
            f"{selected_frames} frames, quota {quota} frames",
            flush=True,
        )
        result = build_one_shard(
            src_root=src_root,
            dst_root=dst_root,
            repo_id=repo_id,
            selected_episodes=selected,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            gop=args.gop,
            encoder_threads=args.encoder_threads,
            store=args.store,
            image_writer_threads=args.image_writer_threads,
            tolerance_s=args.tolerance_s,
            video_file_mb=args.video_file_mb,
            data_file_mb=args.data_file_mb,
        )
        result["quota_frames"] = quota
        result["selected_frames_before_write"] = selected_frames
        result["environment"] = SHARD_ENV_MAP[shard_idx]
        result["seed"] = args.seed + shard_idx
        manifest["shards"].append(result)
        partial_name = "subset_manifest.partial.json" if args.only_shard is None else f"subset_manifest.shard_{shard_idx}.partial.json"
        (args.output_root / partial_name).write_text(json.dumps(manifest, indent=2) + "\n")

    manifest["actual_size_gb"] = round(dir_size(args.output_root) / 1024**3, 3)
    manifest["actual_frames"] = sum(s["written_frames"] for s in manifest["shards"])
    manifest_name = "subset_manifest.json" if args.only_shard is None else f"subset_manifest.shard_{args.only_shard}.json"
    (args.output_root / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
