import argparse
import json
from pathlib import Path

from hw2_cv.utils import deep_update, load_yaml, log_info, resolve_profile_config


def parse_config_args(description, help_text="Path to YAML config."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help=help_text)
    return parser.parse_args()


def load_run_config(path):
    return resolve_profile_config(load_yaml(path))


def print_json(payload):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_sweep(base_config, default_experiments, run_training):
    base_output_dir = Path(base_config["output_dir"])
    experiments = base_config.get("sweep", {}).get("experiments", default_experiments)

    all_summaries = []
    total = len(experiments)
    log_info(f"[sweep] start {total} experiments")

    for index, experiment in enumerate(experiments, start=1):
        tag = experiment["tag"]
        meta = experiment.get("meta", {})
        log_info(f"[sweep] ({index}/{total}) start: {tag}")
        overrides = {
            key: value
            for key, value in experiment.items()
            if key not in {"tag", "meta"}
        }
        config = deep_update(base_config, overrides)
        config["output_dir"] = str(base_output_dir.parent / tag)
        summary = run_training(config)
        summary["tag"] = tag
        if meta:
            summary["meta"] = meta
        all_summaries.append(summary)
        log_info(f"[sweep] ({index}/{total}) done: {tag}")

    log_info("[sweep] finished")
    return all_summaries
