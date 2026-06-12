"""CLI entry point — dispatches to pipeline steps for both tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .data import check_task1_assets, get_calvin_stats, validate_calvin_directory
from .download import (
    MIPNERF360_SCENES,
    download_calvin_command,
    download_mipnerf360_command,
    remove_background_rembg_command,
)
from .external import bootstrap_commands
from .runner import run_or_print
from .task1 import (
    build_colmap_command,
    build_magic123_command,
    build_render_scene_command,
    build_threestudio_command,
    build_train_2dgs_command,
    check_assets,
    convert_mesh_to_gs_ply,
    convert_simple_ply_to_gs_ply,
    extract_frames_command,
    fuse_scene_from_manifest,
    make_scene_manifest,
    threestudio_export_mesh_command,
    validate_capture,
)
from .task2 import (
    build_comparison_table,
    build_eval_act_command,
    build_train_act_command,
    build_wandb_config,
    calvin_data_stats,
    check_calvin_data,
    collect_act_metrics,
    make_calvin_subset,
    write_calvin_index,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hw3cv.cli")
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- setup / env ----
    bootstrap = subparsers.add_parser("bootstrap", help="Clone external repositories")
    _add_execution_flags(bootstrap)

    subparsers.add_parser("check-env", help="Check environment: tools, repos, GPU")

    # ---- download ----
    dl = subparsers.add_parser("download")
    dl_sub = dl.add_subparsers(dest="download_command", required=True)

    dl_calvin = dl_sub.add_parser("calvin", help="Download CALVIN dataset")
    dl_calvin.add_argument("--output", type=Path, default=Path("data/calvin"))
    dl_calvin.add_argument("--split", choices=["debug", "D", "ABC", "ABCD"], default="ABCD")
    _add_execution_flags(dl_calvin)

    dl_mip = dl_sub.add_parser("mipnerf360", help="Download Mip-NeRF 360 scene")
    dl_mip.add_argument("--scene", choices=list(MIPNERF360_SCENES), default="garden")
    dl_mip.add_argument("--output", type=Path, default=Path("data/background"))
    _add_execution_flags(dl_mip)

    # ---- task1 ----
    t1 = subparsers.add_parser("task1")
    t1_sub = t1.add_subparsers(dest="task1_command", required=True)

    # Pipeline steps
    t1_extract = t1_sub.add_parser("extract-frames", help="Extract frames from video")
    t1_extract.add_argument("--video", type=Path, required=True)
    t1_extract.add_argument("--output", type=Path, required=True)
    t1_extract.add_argument("--fps", type=int, default=2)
    _add_execution_flags(t1_extract)

    t1_colmap = t1_sub.add_parser("colmap", help="Run COLMAP SfM")
    t1_colmap.add_argument("--images", type=Path, required=True)
    t1_colmap.add_argument("--output", type=Path, required=True)
    t1_colmap.add_argument("--quality", default="medium")
    _add_execution_flags(t1_colmap)

    t1_train = t1_sub.add_parser("train-2dgs", help="Train 2DGS (external)")
    t1_train.add_argument("--target", choices=["object_a", "background"], required=True)
    _add_execution_flags(t1_train)

    t1_threestudio = t1_sub.add_parser("threestudio", help="Run threestudio text-to-3D")
    _add_execution_flags(t1_threestudio)

    t1_magic123 = t1_sub.add_parser("magic123", help="Run Magic123 image-to-3D")
    _add_execution_flags(t1_magic123)

    # Post-processing
    t1_convert = t1_sub.add_parser("convert-mesh", help="Convert AIGC mesh → Gaussian PLY")
    t1_convert.add_argument("--input", type=Path, required=True, help="OBJ mesh path")
    t1_convert.add_argument("--output", type=Path, required=True, help="Output PLY path")
    t1_convert.add_argument("--samples", type=int, default=200000)

    t1_convert_ply = t1_sub.add_parser("convert-ply", help="Convert simple xyz/rgb PLY → Gaussian PLY")
    t1_convert_ply.add_argument("--input", type=Path, required=True, help="Input simple PLY path")
    t1_convert_ply.add_argument("--output", type=Path, required=True, help="Output Gaussian PLY path")

    # threestudio post-processing
    t1_export = t1_sub.add_parser("export-mesh", help="Export threestudio model → OBJ mesh")
    t1_export.add_argument("--trial-dir", type=Path, required=True, help="threestudio trial_dir")
    t1_export.add_argument("--output", type=Path, required=True, help="Output OBJ path")
    _add_execution_flags(t1_export)

    # Background removal for object_c
    t1_rmbg = t1_sub.add_parser("remove-bg", help="Remove image background (for object_c)")
    t1_rmbg.add_argument("--input", type=Path, required=True)
    t1_rmbg.add_argument("--output", type=Path, required=True)
    _add_execution_flags(t1_rmbg)

    t1_manifest = t1_sub.add_parser("make-scene-manifest", help="Generate Blender scene manifest")
    t1_fuse = t1_sub.add_parser("fuse-scene", help="Merge all assets → unified Gaussian PLY")
    t1_fuse.add_argument("--output", type=Path, default=None)

    t1_render = t1_sub.add_parser("render-scene", help="Render fused scene (Blender)")
    _add_execution_flags(t1_render)

    # Data checks
    t1_sub.add_parser("check-assets", help="Check task1 output assets")
    t1_capture = t1_sub.add_parser("check-capture", help="Validate image capture dir")
    t1_capture.add_argument("--target", choices=["object_a", "background"], required=True)

    # ---- task2 ----
    t2 = subparsers.add_parser("task2")
    t2_sub = t2.add_subparsers(dest="task2_command", required=True)

    t2_index = t2_sub.add_parser("index-calvin", help="Build CALVIN experiment index")
    t2_index.add_argument("--exp", choices=["single_b", "abc_to_d"], required=True)
    t2_index.add_argument("--calvin-root", type=Path, default=None)

    t2_subset = t2_sub.add_parser("make-calvin-subset", help="Create a smaller linked CALVIN subset")
    t2_subset.add_argument("--source", type=Path, required=True)
    t2_subset.add_argument("--output", type=Path, default=Path("data/calvin_subset"))
    t2_subset.add_argument("--exp", choices=["single_b", "abc_to_d"], default="single_b")
    t2_subset.add_argument("--max-train-gb", type=float, default=35.0)
    t2_subset.add_argument("--max-eval-gb", type=float, default=5.0)
    t2_subset.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")

    t2_train = t2_sub.add_parser("train-act", help="Train ACT (LeRobot)")
    t2_train.add_argument("--exp", choices=["single_b", "abc_to_d"], required=True)
    t2_train.add_argument("--calvin-root", type=Path, default=None)
    t2_train.add_argument("--dataset-root", type=Path, default=None,
                          help="Local LeRobot v3 dataset root containing meta/, data/, and videos/")
    t2_train.add_argument("--repo-id", default=None,
                          help="LeRobot dataset id label, e.g. local/calvin_single_b")
    t2_train.add_argument("--steps", type=int, default=None,
                          help="Override training steps for LeRobot")
    _add_execution_flags(t2_train)

    t2_eval = t2_sub.add_parser("eval-act", help="Evaluate ACT (LeRobot)")
    t2_eval.add_argument("--exp", choices=["single_b", "abc_to_d"], required=True)
    t2_eval.add_argument("--calvin-root", type=Path, default=None)
    _add_execution_flags(t2_eval)

    t2_check = t2_sub.add_parser("check-calvin", help="Check CALVIN data")
    t2_check.add_argument("--calvin-root", type=Path, default=None)
    t2_stats = t2_sub.add_parser("calvin-stats", help="CALVIN episode stats")
    t2_stats.add_argument("--calvin-root", type=Path, default=None)

    t2_compare = t2_sub.add_parser("compare", help="Compare single_b vs abc_to_d results")
    t2_compare.add_argument("--exp", nargs="*", default=None)

    t2_wandb = t2_sub.add_parser("wandb-config", help="Print WandB config")
    t2_wandb.add_argument("--exp", choices=["single_b", "abc_to_d"], required=True)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "bootstrap":
        return _bootstrap(cfg, args)
    if args.command == "check-env":
        return _check_env(cfg)
    if args.command == "download":
        return _download(args)
    if args.command == "task1":
        return _task1(cfg, args)
    if args.command == "task2":
        return _task2(cfg, args)
    return 2


# ---- bootstrap ----

def _bootstrap(cfg, args) -> int:
    for cmd in bootstrap_commands(cfg.external_dir):
        r = run_or_print(cmd, execute=args.execute)
        if r.returncode != 0:
            return r.returncode
    return 0


# ---- task1 dispatch ----

def _task1(cfg, args) -> int:
    cmd = args.task1_command

    if cmd == "extract-frames":
        command = extract_frames_command(args.video, args.output, args.fps)
        return run_or_print(command, execute=args.execute).returncode

    if cmd == "colmap":
        command = build_colmap_command(args.images, args.output, args.quality)
        return run_or_print(command, execute=args.execute).returncode

    if cmd == "train-2dgs":
        command = build_train_2dgs_command(cfg, args.target)
        cwd = cfg.external_dir / "2d-gaussian-splatting"
        return run_or_print(command, execute=args.execute, cwd=cwd).returncode

    if cmd == "threestudio":
        command = build_threestudio_command(cfg)
        cwd = cfg.external_dir / "threestudio"
        return run_or_print(command, execute=args.execute, cwd=cwd).returncode

    if cmd == "magic123":
        command = build_magic123_command(cfg)
        cwd = cfg.external_dir / "Magic123"
        return run_or_print(command, execute=args.execute, cwd=cwd).returncode

    if cmd == "export-mesh":
        command = threestudio_export_mesh_command(args.trial_dir, args.output)
        cwd = cfg.external_dir / "threestudio"
        return run_or_print(command, execute=args.execute, cwd=cwd).returncode

    if cmd == "remove-bg":
        command = remove_background_rembg_command(args.input, args.output)
        return run_or_print(command, execute=args.execute).returncode

    if cmd == "convert-mesh":
        out = convert_mesh_to_gs_ply(args.input, args.output, n_samples=args.samples)
        print(f"Converted → {out}")
        return 0

    if cmd == "convert-ply":
        out = convert_simple_ply_to_gs_ply(args.input, args.output)
        print(f"Converted → {out}")
        return 0

    if cmd == "make-scene-manifest":
        path = make_scene_manifest(cfg)
        print(path)
        return 0

    if cmd == "fuse-scene":
        fuse_scene_from_manifest(cfg, fused_ply_output=args.output)
        return 0

    if cmd == "render-scene":
        command = build_render_scene_command(cfg)
        return run_or_print(command, execute=args.execute).returncode

    if cmd == "check-assets":
        status = check_assets(cfg)
        for name, ok in status.items():
            print(f"  {'[OK]' if ok else '[MISSING]'} {name}")
        return 0 if all(status.values()) else 1

    if cmd == "check-capture":
        result = validate_capture(cfg, args.target)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("valid") else 1

    return 2


# ---- task2 dispatch ----

def _task2(cfg, args) -> int:
    cmd = args.task2_command

    if cmd == "index-calvin":
        path = write_calvin_index(cfg, args.exp, calvin_root=args.calvin_root)
        print(path)
        return 0

    if cmd == "make-calvin-subset":
        experiment = cfg.task2.experiments[args.exp]
        summary = make_calvin_subset(
            args.source,
            args.output,
            train_envs=experiment.train_envs,
            eval_env=experiment.eval_env,
            max_train_gb=args.max_train_gb,
            max_eval_gb=args.max_eval_gb,
            link_mode=args.link_mode,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if cmd == "train-act":
        index_path = write_calvin_index(cfg, args.exp, calvin_root=args.calvin_root)
        command = build_train_act_command(
            cfg,
            args.exp,
            index_path,
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            steps=args.steps,
        )
        return run_or_print(command, execute=args.execute,
                           cwd=cfg.external_dir / "lerobot").returncode

    if cmd == "eval-act":
        index_path = write_calvin_index(cfg, args.exp, calvin_root=args.calvin_root)
        command = build_eval_act_command(cfg, args.exp, index_path)
        return run_or_print(command, execute=args.execute,
                           cwd=cfg.external_dir / "lerobot").returncode

    if cmd == "check-calvin":
        status = check_calvin_data(cfg, calvin_root=args.calvin_root)
        for env, episodes in status.items():
            print(f"  {env}: {len(episodes)} episodes")
        return 0

    if cmd == "calvin-stats":
        stats = calvin_data_stats(cfg, calvin_root=args.calvin_root)
        print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
        return 0

    if cmd == "compare":
        metrics = collect_act_metrics(cfg.outputs.task2, args.exp if args.exp else None)
        print(build_comparison_table(metrics))
        return 0

    if cmd == "wandb-config":
        wb = build_wandb_config(cfg, args.exp)
        print(json.dumps(wb, indent=2, ensure_ascii=False))
        return 0

    return 2


# ---- check-env ----

def _check_env(cfg) -> int:
    """Check that required tools, repos, and GPU are available."""
    import shutil
    import torch

    all_ok = True

    def _check(name: str, ok: bool, hint: str = "") -> None:
        nonlocal all_ok
        mark = "[OK]" if ok else "[MISSING]"
        print(f"  {mark} {name}")
        if not ok:
            all_ok = False
            if hint:
                print(f"       → {hint}")

    print("Environment check:")
    _check("git", shutil.which("git") is not None, "Install git")
    _check("conda", shutil.which("conda") is not None, "Install Miniconda/Anaconda")
    _check("colmap", shutil.which("colmap") is not None, "Install COLMAP: https://colmap.github.io/")
    _check("blender", shutil.which("blender") is not None, "Install Blender: https://www.blender.org/")
    _check("ffmpeg", shutil.which("ffmpeg") is not None, "Install ffmpeg")
    _check("GPU (CUDA/MPS)", torch.cuda.is_available() or torch.backends.mps.is_available(),
           "GPU recommended; CPU-only will be slow but works")
    _check("2DGS repo", (cfg.external_dir / "2d-gaussian-splatting").is_dir(),
           "Run: python -m hw3cv.cli bootstrap --execute")
    _check("threestudio repo", (cfg.external_dir / "threestudio").is_dir(),
           "Run: python -m hw3cv.cli bootstrap --execute")
    _check("Magic123 repo", (cfg.external_dir / "Magic123").is_dir(),
           "Run: python -m hw3cv.cli bootstrap --execute")
    _check("LeRobot repo", (cfg.external_dir / "lerobot").is_dir(),
           "Run: python -m hw3cv.cli bootstrap --execute")
    _check("CALVIN repo", (cfg.external_dir / "calvin").is_dir(),
           "Run: python -m hw3cv.cli bootstrap --execute")

    return 0 if all_ok else 1


# ---- download ----

def _download(args) -> int:
    cmd = args.download_command

    if cmd == "calvin":
        command = download_calvin_command(args.output, args.split)
        return run_or_print(command, execute=args.execute).returncode

    if cmd == "mipnerf360":
        command = download_mipnerf360_command(args.output, args.scene)
        return run_or_print(command, execute=args.execute).returncode

    return 2


# ---- helpers ----

def _add_execution_flags(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")


if __name__ == "__main__":
    sys.exit(main())
