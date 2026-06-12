import subprocess
import sys


def test_cli_bootstrap_dry_run_prints_bootstrap_command():
    result = subprocess.run(
        [sys.executable, "-m", "hw3cv.cli", "bootstrap", "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": "src"},
    )

    assert "mkdir -p" in result.stdout


def test_cli_task1_train_2dgs_dry_run_prints_command():
    result = subprocess.run(
        [sys.executable, "-m", "hw3cv.cli", "task1", "train-2dgs", "--target", "background", "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": "src"},
    )

    assert "python train.py -s" in result.stdout
    assert "kitchen_2dgs" in result.stdout
