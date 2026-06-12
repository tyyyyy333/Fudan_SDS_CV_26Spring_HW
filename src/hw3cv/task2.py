"""Task 2 pipeline: LeRobot ACT on CALVIN — cross-environment generalization.

This module orchestrates LeRobot's ACT implementation; it does NOT
reimplement the algorithm.  Responsibilities:

  1. Build CALVIN experiment indices that LeRobot can consume.
  2. Generate LeRobot-compatible training configurations.
  3. Build train/eval shell commands with correct hyperparameters.
  4. Collect evaluation metrics and produce comparison tables.

The two experiments are:
  - single_b:  train on environment B only, evaluate on D.
  - abc_to_d:  train on environments A+B+C, evaluate on D.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import HW3Config
from .data import collect_calvin_episodes, get_calvin_stats, validate_calvin_directory


# ======================================================================
# CALVIN index (LeRobot dataset format)
# ======================================================================

def index_calvin_experiment(
    config: HW3Config,
    experiment_name: str,
    calvin_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Build a LeRobot-compatible experiment index dict.

    The index maps episode paths to environments, which LeRobot's dataset
    loader uses to assemble training/validation sets.
    """
    experiment = config.task2.experiments[experiment_name]
    root = calvin_root or config.task2.calvin_root
    return {
        "experiment": experiment_name,
        "calvin_root": str(root),
        "train_envs": experiment.train_envs,
        "eval_env": experiment.eval_env,
        "train_episodes": collect_calvin_episodes(root, experiment.train_envs),
        "eval_episodes": collect_calvin_episodes(root, [experiment.eval_env]),
    }


def write_calvin_index(
    config: HW3Config,
    experiment_name: str,
    output_path: Optional[Path] = None,
    calvin_root: Optional[Path] = None,
) -> Path:
    """Write a CALVIN experiment index to JSON."""
    experiment = config.task2.experiments[experiment_name]
    path = output_path or experiment.output / "calvin_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = index_calvin_experiment(config, experiment_name, calvin_root=calvin_root)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def make_calvin_subset(
    source_root: Path,
    output_root: Path,
    *,
    train_envs: List[str],
    eval_env: str = "D",
    max_train_gb: float = 35.0,
    max_eval_gb: float = 5.0,
    link_mode: str = "symlink",
) -> Dict[str, object]:
    """Create a smaller CALVIN tree with A/B/C/D episode links.

    This expects a source tree already organized as ``source_root/{A,B,C,D}``.
    Official CALVIN zip files are monolithic, so this helper is for reducing an
    already extracted or shared dataset into a smaller training/eval subset.
    """
    if link_mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError("link_mode must be symlink, hardlink, or copy")

    def _budget_envs(envs: List[str], max_gb: float) -> Tuple[List[Tuple[str, Path]], int]:
        budget = int(max_gb * 1024**3)
        selected: List[Tuple[str, Path]] = []
        total = 0
        for env in envs:
            env_dir = source_root / env
            if not env_dir.is_dir():
                continue
            for path in sorted(env_dir.rglob("*.npz")):
                size = path.stat().st_size
                if selected and total + size > budget:
                    return selected, total
                selected.append((env, path))
                total += size
        return selected, total

    train_items, train_bytes = _budget_envs(train_envs, max_train_gb)
    eval_items, eval_bytes = _budget_envs([eval_env], max_eval_gb)
    output_root.mkdir(parents=True, exist_ok=True)

    def _link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            return
        if link_mode == "symlink":
            dst.symlink_to(src.resolve())
        elif link_mode == "hardlink":
            os.link(src, dst)
        else:
            import shutil
            shutil.copy2(src, dst)

    for env, src in train_items + eval_items:
        rel = src.relative_to(source_root / env)
        _link(src, output_root / env / rel)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "train_envs": train_envs,
        "eval_env": eval_env,
        "train_episodes": len(train_items),
        "eval_episodes": len(eval_items),
        "train_size_gb": round(train_bytes / 1024**3, 2),
        "eval_size_gb": round(eval_bytes / 1024**3, 2),
        "link_mode": link_mode,
    }
    (output_root / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


# ======================================================================
# Data validation
# ======================================================================

def check_calvin_data(config: HW3Config,
                      calvin_root: Optional[Path] = None) -> Dict[str, List[str]]:
    return validate_calvin_directory(calvin_root or config.task2.calvin_root)


def calvin_data_stats(config: HW3Config,
                      calvin_root: Optional[Path] = None) -> Dict[str, object]:
    return get_calvin_stats(calvin_root or config.task2.calvin_root)


# ======================================================================
# LeRobot training configuration
# ======================================================================

def build_act_training_config(
    config: HW3Config,
    experiment_name: str,
    index_path: Path,
) -> Dict[str, Any]:
    """Generate a LeRobot-compatible ACT training configuration dict.

    This can be saved as YAML or passed as Hydra overrides on the CLI.
    """
    experiment = config.task2.experiments[experiment_name]
    act = config.task2.act

    return {
        "policy": "act",
        "dataset": {
            "index_path": str(index_path),
        },
        "training": {
            "batch_size": act.batch_size,
            "lr": act.learning_rate,
            "epochs": act.epochs,
            "output_dir": str(experiment.output),
        },
        "policy_cfg": {
            "chunk_size": act.chunk_size,
            "n_encoder_layers": 4,
            "n_decoder_layers": 4,
            "hidden_dim": 512,
            "n_heads": 8,
            "dim_feedforward": 2048,
            "dropout": 0.1,
        },
    }


# ======================================================================
# LeRobot shell commands
# ======================================================================

def build_train_act_command(
    config: HW3Config,
    experiment_name: str,
    index_path: Path,
    *,
    dataset_root: Optional[Path] = None,
    repo_id: Optional[str] = None,
    steps: Optional[int] = None,
) -> List[str]:
    """Build the LeRobot ACT training CLI command."""
    experiment = config.task2.experiments[experiment_name]
    act = config.task2.act
    dataset_repo_id = repo_id or f"local/calvin_{experiment_name}"
    train_steps = steps or act.epochs * 1000
    output_dir = experiment.output
    if dataset_root is None:
        dataset_root = index_path.parent

    return [
        "conda", "run", "-n", "hw3t2", "lerobot-train",
        "--policy.type", "act",
        "--dataset.repo_id", dataset_repo_id,
        "--dataset.root", str(dataset_root),
        "--dataset.video_backend", "pyav",
        "--output_dir", str(output_dir),
        "--job_name", f"hw3_{experiment_name}",
        "--policy.device", "cuda",
        "--batch_size", str(act.batch_size),
        "--steps", str(train_steps),
        "--policy.optimizer_lr", str(act.learning_rate),
        "--policy.chunk_size", str(act.chunk_size),
        "--policy.n_action_steps", str(act.chunk_size),
        "--policy.push_to_hub", "false",
        "--policy.repo_id", f"hw3_{experiment_name}_act",
        "--num_workers", "4",
        "--tolerance_s", "0.01",
        "--save_freq", "1000",
        "--log_freq", "20",
        "--wandb.enable", "false",
    ]


def build_eval_act_command(
    config: HW3Config,
    experiment_name: str,
    index_path: Path,
) -> List[str]:
    """Build the LeRobot ACT evaluation CLI command."""
    experiment = config.task2.experiments[experiment_name]
    return [
        "python", "-m", "lerobot.scripts.eval",
        "policy=act",
        f"dataset.index_path={index_path}",
        f"checkpoint={experiment.output / 'checkpoints' / 'best'}",
        f"metrics_path={experiment.output / 'eval_metrics.json'}",
    ]


# ======================================================================
# Result collection and comparison
# ======================================================================

def collect_act_metrics(outputs_task2: Path,
                        experiments: Optional[List[str]] = None
                        ) -> Dict[str, Optional[Dict[str, Any]]]:
    """Collect evaluation metrics for named experiments.

    Returns:
        Dict mapping experiment name → metrics dict, or None if not found.
    """
    if experiments is None:
        experiments = ["single_b", "abc_to_d"]

    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for exp_name in experiments:
        metrics_path = outputs_task2 / exp_name / "eval_metrics.json"
        if metrics_path.exists():
            results[exp_name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            results[exp_name] = None
    return results


def build_comparison_table(metrics: Dict[str, Optional[Dict[str, Any]]],
                           training_summaries: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
                           ) -> str:
    """Produce a Markdown comparison table for the experiment report.

    Args:
        metrics: result of collect_act_metrics().
        training_summaries: optional training summaries with loss curves.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        "## Task 2 — ACT Cross-Environment Generalization",
        "",
        "### Evaluation on environment D (zero-shot)",
        "",
        "| Experiment | Train Envs | Success Rate | Action L1 Loss |",
        "|------------|------------|-------------|----------------|",
    ]
    env_map = {"single_b": "B", "abc_to_d": "A, B, C"}

    for exp_name, m in metrics.items():
        train_envs = env_map.get(exp_name, "?")
        if m is not None:
            sr = f"{m.get('success_rate', '—')}"
            l1 = f"{m.get('avg_action_l1', '—')}"
        else:
            sr = "— (not run yet)"
            l1 = "— (not run yet)"
        lines.append(f"| {exp_name} | {train_envs} | {sr} | {l1} |")

    lines.append("")
    lines.append("### Training convergence")
    lines.append("")

    if training_summaries:
        lines.append("| Experiment | Final Train L1 | Best Val L1 | Epochs |")
        lines.append("|------------|---------------|-------------|--------|")
        for exp_name, s in training_summaries.items():
            if s is not None:
                train_l1 = f"{s.get('final_train_loss', '—'):.4f}"
                val_l1 = f"{s['best_val_loss']:.4f}" if s.get('best_val_loss') else "—"
                ep = s.get('epochs', '—')
            else:
                train_l1 = "—"
                val_l1 = "—"
                ep = "—"
            lines.append(f"| {exp_name} | {train_l1} | {val_l1} | {ep} |")
        lines.append("")

    lines.append(
        "> Plotted with WandB/SwanLab — see report Figures X–Y for loss curves."
    )
    return "\n".join(lines)


# ======================================================================
# WandB configuration helper
# ======================================================================

def build_wandb_config(config: HW3Config, experiment_name: str) -> Dict[str, Any]:
    """Generate a WandB-compatible config dict for experiment tracking."""
    experiment = config.task2.experiments[experiment_name]
    act = config.task2.act
    return {
        "project": "cv-hw3",
        "group": "task2-act",
        "name": f"hw3_{experiment_name}",
        "config": {
            "experiment": experiment_name,
            "train_envs": experiment.train_envs,
            "eval_env": experiment.eval_env,
            "policy": "ACT",
            "batch_size": act.batch_size,
            "learning_rate": act.learning_rate,
            "epochs": act.epochs,
            "chunk_size": act.chunk_size,
        },
    }
