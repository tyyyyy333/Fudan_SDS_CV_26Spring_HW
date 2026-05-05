import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.cli import load_run_config, parse_config_args, print_json
from hw2_cv.task2.track import run_tracking


def parse_args():
    return parse_config_args("Run Task 2 tracking, occlusion analysis, and line counting.")


def main():
    args = parse_args()
    config = load_run_config(args.config)
    summary = run_tracking(config)
    print_json(summary)


if __name__ == "__main__":
    main()
