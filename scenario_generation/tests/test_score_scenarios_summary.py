"""Tests for the segment reader, per-group summary and log dict of ``score_scenarios``."""

import json
import math

from scenario_generation.score_scenarios import segment_rows, summarize, summary_log_dict


def _row(passed, values, metric="centerline", reason=""):
    return {"metric": metric, "passed": passed, "values": values, "reason": reason}


def test_summarize_means_over_scored_rows_only():
    rows = [
        _row(True, {"average_lateral_error_m": 0.2, "threshold_m": 1.0}),
        _row(False, {"average_lateral_error_m": 0.6, "threshold_m": 1.0}),
        # Not applicable: partial values must not leak into the means.
        _row(None, {"average_lateral_error_m": 9.0}, reason="anchor not reached"),
    ]
    s = summarize(rows)
    assert (s["n_anchors"], s["n_scored"], s["n_pass"], s["n_fail"]) == (3, 2, 1, 1)
    assert s["n_not_applicable"] == 1
    assert s["pass_rate"] == 0.5
    assert s["success_rate_percent"] == 50.0
    assert s["not_applicable_reasons"] == {"anchor not reached": 1}
    assert math.isclose(s["values"]["average_lateral_error_m"]["mean"], 0.4)
    assert s["values"]["average_lateral_error_m"]["n"] == 2
    assert s["values"]["threshold_m"] == {"mean": 1.0, "n": 2, "n_nonfinite": 0}
    json.dumps(s, allow_nan=False)


def test_summarize_skips_nonfinite_and_missing_keys():
    rows = [
        _row(True, {"min_clearance_m": math.inf, "progress_m": 3.0}),
        _row(True, {"min_clearance_m": 2.0}),
        _row(False, {"min_clearance_m": math.nan}),
    ]
    v = summarize(rows)["values"]
    assert v["min_clearance_m"] == {"mean": 2.0, "n": 1, "n_nonfinite": 2}
    assert v["progress_m"] == {"mean": 3.0, "n": 1, "n_nonfinite": 0}

    only_inf = summarize([_row(True, {"min_clearance_m": math.inf})])["values"]
    assert only_inf["min_clearance_m"] == {"mean": None, "n": 0, "n_nonfinite": 1}


def test_summarize_nothing_scored():
    s = summarize([_row(None, {"progress_m": 1.0}, metric="none", reason="missing trace")])
    assert s["pass_rate"] is None
    assert s["success_rate_percent"] is None
    assert s["values"] == {}


def test_summary_log_dict_keys_and_drops():
    summary = {
        "centerline": summarize(
            [
                _row(True, {"average_lateral_error_m": 0.2, "threshold_m": 1.0}),
                _row(True, {"average_lateral_error_m": 0.4, "min_clearance_m": math.inf}),
            ]
        ),
        "vehicle_yield": summarize([_row(None, {}, metric="none", reason="missing trace")]),
    }
    log = summary_log_dict(summary)
    p = "scenario_based_closed_loop/centerline"
    q = "scenario_based_closed_loop/vehicle_yield"
    # threshold_m is a parameter echo, min_clearance_m has no finite value and the
    # unscored group has no success rate: all dropped.
    assert set(log) == {
        f"{p}/success_rate_percent",
        f"{p}/n_anchors",
        f"{p}/n_scored",
        f"{p}/n_not_applicable",
        f"{p}/average_lateral_error_m",
        f"{q}/n_anchors",
        f"{q}/n_scored",
        f"{q}/n_not_applicable",
    }
    assert log[f"{p}/success_rate_percent"] == 100.0
    assert math.isclose(log[f"{p}/average_lateral_error_m"], 0.3)
    assert log[f"{q}/n_not_applicable"] == 1.0
    assert all(isinstance(v, float) for v in log.values())


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_segment_rows_reads_the_merged_file_once_next_to_ddp_shards(tmp_path):
    a = {"route": "r0", "segment": [0, 10]}
    b = {"route": "r1", "segment": [0, 10]}
    _write_rows(tmp_path / "segments_0.jsonl", [a])
    _write_rows(tmp_path / "segments_1.jsonl", [b])
    _write_rows(tmp_path / "segments.jsonl", [a, b])
    assert segment_rows(tmp_path) == [a, b]


def test_segment_rows_falls_back_to_the_shards_without_a_merged_file(tmp_path):
    a = {"route": "r0", "segment": [0, 10]}
    b = {"route": "r1", "segment": [0, 10]}
    _write_rows(tmp_path / "segments_0.jsonl", [a])
    _write_rows(tmp_path / "segments_1.jsonl", [b])
    assert segment_rows(tmp_path) == [a, b]
