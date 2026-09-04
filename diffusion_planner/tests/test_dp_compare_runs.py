from pathlib import Path

import pytest
from diffusion_planner.data_pipeline.validation.compare_runs import (
    Result,
    band,
    compare,
    main,
    plateau_stats,
)

HEADER = "epoch\ttrain_loss\tvalid_loss_ego\n"


def _write(tmp_path: Path, name: str, pairs: list[tuple[int, float, float]]) -> Path:
    p = tmp_path / name
    p.write_text(HEADER + "".join(f"{e}\t{a}\t{b}\n" for e, a, b in pairs))
    return p


def test_plateau_stats_selects_epoch_range_inclusive():
    rows = [
        {"epoch": "1", "train_loss": "9.0"},
        {"epoch": "2", "train_loss": "1.0"},
        {"epoch": "3", "train_loss": "3.0"},
    ]
    mean, sd = plateau_stats(rows, "train_loss", (2, 3))
    assert mean == pytest.approx(2.0)
    assert sd == pytest.approx(1.0)


def test_band_is_symmetric_k_sigma():
    assert band(10.0, 0.5, k=2.0) == pytest.approx((9.0, 11.0))


def test_compare_passes_when_candidate_mean_inside_band(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.2, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.05, 5.0), (2, 1.05, 5.0), (3, 1.1, 5.0)])
    results = compare(candidate, baseline, ["train_loss"], (1, 3))
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].column == "train_loss"


def test_compare_fails_when_candidate_mean_outside_band(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 9.0, 5.0), (2, 9.0, 5.0), (3, 9.0, 5.0)])
    results = compare(candidate, baseline, ["train_loss"], (1, 3))
    assert results[0].passed is False


def test_compare_raises_when_epoch_range_absent_from_a_file(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.0, 5.0)])
    with pytest.raises(ValueError, match="no rows"):
        compare(candidate, baseline, ["train_loss"], (70, 80))


def test_compare_raises_on_unknown_column(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0)])
    with pytest.raises(ValueError, match="column"):
        compare(candidate, baseline, ["nonexistent"], (1, 2))


def test_zero_variance_baseline_still_yields_a_usable_band(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 2.0, 5.0), (2, 2.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 2.0, 5.0), (2, 2.0, 5.0)])
    results = compare(candidate, baseline, ["train_loss"], (1, 2))
    assert results[0].passed is True


def test_main_returns_zero_when_every_metric_passes(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.2, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.05, 5.0), (2, 1.05, 5.0), (3, 1.1, 5.0)])
    code = main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--columns",
            "train_loss",
            "--from-epoch",
            "1",
            "--to-epoch",
            "3",
        ]
    )
    assert code == 0


def test_main_returns_one_when_a_metric_falls_outside_the_band(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 9.0, 5.0), (2, 9.0, 5.0), (3, 9.0, 5.0)])
    code = main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--columns",
            "train_loss",
            "--from-epoch",
            "1",
            "--to-epoch",
            "3",
        ]
    )
    assert code == 1


def test_main_reports_error_and_exits_one_on_unknown_column(tmp_path, capsys):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0)])
    code = main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--columns",
            "nonexistent",
            "--from-epoch",
            "1",
            "--to-epoch",
            "2",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err
