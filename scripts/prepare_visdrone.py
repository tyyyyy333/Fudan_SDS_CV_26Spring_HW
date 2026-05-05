import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw2_cv.task2.visdrone import convert_visdrone_to_yolo


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format.")
    parser.add_argument("--raw-root", type=str, required=True, help="Original VisDrone root.")
    parser.add_argument("--output-root", type=str, required=True, help="Converted YOLO root.")
    parser.add_argument("--data-yaml", type=str, default=None, help="Optional path to write the YOLO data yaml.")
    parser.add_argument(
        "--link-mode",
        type=str,
        default="copy",
        choices=("copy", "hardlink", "symlink"),
        help="How to place images into the converted dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = convert_visdrone_to_yolo(
        Path(args.raw_root),
        Path(args.output_root),
        link_mode=args.link_mode,
        data_yaml_path=None if args.data_yaml is None else Path(args.data_yaml),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
