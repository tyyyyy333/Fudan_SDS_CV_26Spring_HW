import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.cli import load_run_config, parse_config_args, print_json, run_sweep
from hw2_cv.task3.train import run_training


DEFAULT_EXPERIMENTS = [
    {"tag": "ce_only", "loss": {"name": "ce"}},
    {"tag": "dice_only", "loss": {"name": "dice"}},
    {"tag": "ce_dice", "loss": {"name": "ce_dice"}},
]


def parse_args():
    return parse_config_args("Run Task 3 loss comparison experiments.", "Base YAML config.")


def main():
    args = parse_args()
    base_config = load_run_config(args.config)
    all_summaries = run_sweep(base_config, DEFAULT_EXPERIMENTS, run_training)
    print_json(all_summaries)


if __name__ == "__main__":
    main()
