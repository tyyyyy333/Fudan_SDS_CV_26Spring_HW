"""Tests for hw3cv.eval module."""

import json

from hw3cv.eval import (
    collect_task1_timing,
    format_act_comparison,
    load_eval_metrics,
    summarise_act_results,
)


class TestLoadEvalMetrics:
    def test_loads_valid_json(self, tmp_path):
        metrics_path = tmp_path / "eval_metrics.json"
        metrics_path.write_text(
            json.dumps({"success_rate": 0.72, "avg_action_l1": 0.034}),
            encoding="utf-8",
        )
        data = load_eval_metrics(metrics_path)
        assert data["success_rate"] == 0.72

    def test_raises_when_missing(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_eval_metrics(tmp_path / "does_not_exist.json")


class TestSummariseAct:
    def test_collects_existing_experiments(self, tmp_path):
        out = tmp_path
        for exp in ["single_b", "abc_to_d"]:
            (out / exp).mkdir()
            (out / exp / "eval_metrics.json").write_text(
                json.dumps({"success_rate": 0.5, "avg_action_l1": 0.02}),
                encoding="utf-8",
            )

        summary = summarise_act_results(out)

        assert summary["single_b"]["success_rate"] == 0.5
        assert summary["abc_to_d"]["success_rate"] == 0.5

    def test_missing_experiments_report_none(self, tmp_path):
        summary = summarise_act_results(tmp_path, experiments=["single_b"])

        assert summary["single_b"]["success_rate"] is None


class TestFormatComparison:
    def test_formats_both_experiments(self):
        summary = {
            "single_b": {"success_rate": 0.65, "avg_action_l1": 0.031},
            "abc_to_d": {"success_rate": 0.41, "avg_action_l1": 0.048},
        }
        output = format_act_comparison(summary)
        assert "single_b" in output
        assert "abc_to_d" in output
        assert "0.6500" in output
        assert "0.0480" in output

    def test_handles_none_values(self):
        summary = {"single_b": {"success_rate": None, "avg_action_l1": None}}
        output = format_act_comparison(summary)
        assert "N/A" in output


class TestTask1Timing:
    def test_collects_timing_files(self, tmp_path):
        (tmp_path / "object_a_2dgs").mkdir()
        (tmp_path / "object_a_2dgs" / "timing.json").write_text(
            json.dumps({"elapsed_seconds": 1234.5}), encoding="utf-8",
        )

        timing = collect_task1_timing(tmp_path)

        assert timing["object_a"] == 1234.5
        assert timing["object_b"] is None

    def test_handles_non_dict_timing(self, tmp_path):
        (tmp_path / "object_a_2dgs").mkdir()
        (tmp_path / "object_a_2dgs" / "timing.json").write_text("[]", encoding="utf-8")

        timing = collect_task1_timing(tmp_path)
        assert timing["object_a"] is None
