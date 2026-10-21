"""A case that never wrote a row still counts, as a non-pass, in the suite summary."""

import json
from pathlib import Path

import numpy as np

from scenario_generation.scenario_sim_metrics import build_segment_row
from scenario_generation.scenario_sim_suite_summary import summarize_suite


def _row(result_kind: str) -> dict:
    return build_segment_row(
        n_steps_run=3,
        terminated="goal",
        result_kind=result_kind,
        clearances=[5.0, 4.0, 6.0],
        collisions=[False, False, False],
        rb_dists=np.array([2.0, 2.5, 3.0]),
        accels=np.array([0.0, -0.5, -0.2]),
        near_miss_thresh=0.5,
        strong_brake_mps2=-2.5,
        progress_m=12.0,
    )


def _case(run: Path, key: str, row: dict | None) -> None:
    (run / key).mkdir()
    if row is not None:
        (run / key / "row.json").write_text(json.dumps(row))


def _run(tmp_path: Path) -> Path:
    # pass, fail, timeout (no row), crash (no row)
    _case(tmp_path, "a", _row("Pass"))
    _case(tmp_path, "b", _row("Failure"))
    _case(tmp_path, "c", None)
    _case(tmp_path, "d", None)
    (tmp_path / "work.tsv").write_text("".join(f"{k}\t{k}.xosc\n" for k in "abcd"))
    (tmp_path / "timeouts.txt").write_text("c TIMED_OUT after 1800s\n")
    return tmp_path


def test_denominator_is_what_was_submitted(tmp_path):
    summary, rows = summarize_suite(_run(tmp_path))

    assert len(rows) == 2
    assert summary["n_submitted"] == 4
    assert summary["n_rows"] == 2
    assert summary["n_timeout"] == 1
    assert summary["n_pass"] == 1
    assert summary["pass_rate"] == 0.25


def test_references_this_path_lacks_are_null_not_zero(tmp_path):
    summary, _ = summarize_suite(_run(tmp_path))

    assert summary["mean_route_completion"] is None
    assert summary["mean_gt_deviation_m"] is None


def test_a_run_with_no_rows_still_reports_its_counts(tmp_path):
    _case(tmp_path, "a", None)
    (tmp_path / "work.tsv").write_text("a\ta.xosc\n")

    summary, rows = summarize_suite(tmp_path)

    assert rows == []
    assert summary["n_submitted"] == 1
    assert summary["pass_rate"] == 0.0
