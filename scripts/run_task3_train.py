import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.cli import load_run_config, parse_config_args, print_json
from hw2_cv.task3.train import run_training


def parse_args():
    return parse_config_args("Run Task 3 training.")


def main():
    args = parse_args()
    config = load_run_config(args.config)
    summary = run_training(config)
    print_json(summary)


if __name__ == "__main__":
    main()
