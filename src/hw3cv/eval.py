"""Evaluation metric utilities for HW3 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Task 2: ACT evaluation
# ---------------------------------------------------------------------------

def load_eval_metrics(path: Path) -> Dict[str, Any]:
    """Load eval_metrics.json from an ACT evaluation run."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarise_act_results(
    outputs_task2: Path,
    experiments: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect eval metrics for named experiments.

    Returns a dict keyed by experiment name with extracted success rate,
    action L1 loss, and raw metrics.
    """
    if experiments is None:
        experiments = ["single_b", "abc_to_d"]

    summary: Dict[str, Dict[str, Any]] = {}
    for exp_name in experiments:
        metrics_path = outputs_task2 / exp_name / "eval_metrics.json"
        if metrics_path.exists():
            raw = load_eval_metrics(metrics_path)
            summary[exp_name] = {
                "success_rate": raw.get("success_rate"),
                "avg_action_l1": raw.get("avg_action_l1"),
                "raw": raw,
            }
        else:
            summary[exp_name] = {"success_rate": None, "avg_action_l1": None, "raw": None}
    return summary


def format_act_comparison(summary: Dict[str, Dict[str, Any]]) -> str:
    """Produce a human-readable comparison table of ACT experiment results."""
    lines = [
        f"{'Experiment':<14} {'Success Rate':>14} {'Action L1':>12}",
        "-" * 42,
    ]
    for exp_name, metrics in summary.items():
        sr = f"{metrics['success_rate']:.4f}" if metrics["success_rate"] is not None else "N/A"
        l1 = f"{metrics['avg_action_l1']:.4f}" if metrics["avg_action_l1"] is not None else "N/A"
        lines.append(f"{exp_name:<14} {sr:>14} {l1:>12}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 1: asset metrics
# ---------------------------------------------------------------------------

def collect_task1_timing(outputs_task1: Path) -> Dict[str, Optional[float]]:
    """Try to read timing files from each asset pipeline.

    Looks for `timing.json` sidecars written by the generate / train steps.
    Returns a dict of asset name → elapsed seconds (or None if missing).
    """
    assets = {
        "object_a": outputs_task1 / "object_a_2dgs" / "timing.json",
        "object_b": outputs_task1 / "object_b" / "timing.json",
        "object_c": outputs_task1 / "object_c" / "timing.json",
        "background": outputs_task1 / "background_2dgs" / "timing.json",
    }
    timing: Dict[str, Optional[float]] = {}
    for name, path in assets.items():
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            timing[name] = data.get("elapsed_seconds") if isinstance(data, dict) else None
        else:
            timing[name] = None
    return timing


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def build_experiment_table(
    task1_timing: Dict[str, Optional[float]],
    act_summary: Dict[str, Dict[str, Any]],
    act_config: Dict[str, object],
) -> str:
    """Produce a Markdown-ready experiment summary table for the report."""
    lines = ["## Experiment Summary", ""]

    lines.append("### Task 1 — Asset Generation")
    lines.append("")
    lines.append("| Asset | Method | Elapsed (s) |")
    lines.append("|-------|--------|-------------|")
    method_map = {
        "object_a": "Multi-view 2DGS",
        "object_b": "Text-to-3D (threestudio)",
        "object_c": "Image-to-3D (Magic123)",
        "background": "2DGS Reconstruction",
    }
    for name, method in method_map.items():
        elapsed = task1_timing.get(name)
        elapsed_str = f"{elapsed:.0f}" if elapsed else "—"
        lines.append(f"| {name} | {method} | {elapsed_str} |")
    lines.append("")

    lines.append("### Task 2 — ACT Generalisation")
    lines.append("")
    lines.append(f"| Config | Value |")
    lines.append(f"|--------|-------|")
    for key, val in act_config.items():
        lines.append(f"| {key} | {val} |")
    lines.append("")

    lines.append("| Experiment | Success Rate | Action L1 |")
    lines.append("|------------|-------------|-----------|")
    for exp_name, metrics in act_summary.items():
        sr = f"{metrics['success_rate']:.4f}" if metrics["success_rate"] is not None else "—"
        l1 = f"{metrics['avg_action_l1']:.4f}" if metrics["avg_action_l1"] is not None else "—"
        lines.append(f"| {exp_name} | {sr} | {l1} |")

    return "\n".join(lines)
