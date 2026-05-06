import argparse
import json
from pathlib import Path

from hw2_cv.utils import deep_update, load_yaml, log_info, resolve_profile_config


def parse_config_args(description, help_text="Path to YAML config."):
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument("--config", type=str, required=True, help=help_text)
    return arg_parser.parse_args()


def load_run_config(path):
    return resolve_profile_config(load_yaml(path))


def print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _experiment_overrides(experiment_cfg):
    return {
        field_name: field_value
        for field_name, field_value in experiment_cfg.items()
        if field_name not in {"tag", "meta"}
    }


def run_sweep(base_config, default_experiments, run_training):
    rootOutputDir = Path(base_config["output_dir"])
    experiments = base_config.get("sweep", {}).get("experiments", default_experiments)

    all_summaries = []
    experimentCount = len(experiments)
    log_info(f"[sweep] start {experimentCount} experiments")

    for experiment_idx, experiment_cfg in enumerate(experiments, start=1):
        runTag = experiment_cfg["tag"]
        experimentMeta = experiment_cfg.get("meta", {})
        log_info(f"[sweep] ({experiment_idx}/{experimentCount}) start: {runTag}")
        merged_config = deep_update(base_config, _experiment_overrides(experiment_cfg))
        merged_config["output_dir"] = str(rootOutputDir.parent / runTag)
        summary = run_training(merged_config)
        summary["tag"] = runTag
        if experimentMeta:
            summary["meta"] = experimentMeta
        all_summaries.append(summary)
        log_info(f"[sweep] ({experiment_idx}/{experimentCount}) done: {runTag}")

    log_info("[sweep] finished")
    return all_summaries
