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


def _write_raw(tmp_path: Path, name: str, header: str, lines: list[str]) -> Path:
    p = tmp_path / name
    p.write_text(header + "".join(line + "\n" for line in lines))
    return p


def test_plateau_stats_selects_epoch_range_inclusive():
    rows = [
        {"epoch": "1", "train_loss": "9.0"},
        {"epoch": "2", "train_loss": "1.0"},
        {"epoch": "3", "train_loss": "3.0"},
    ]
    stats = plateau_stats(rows, "train_loss", (2, 3))
    assert stats.mean == pytest.approx(2.0)
    assert stats.sd == pytest.approx(1.0)
    assert stats.n == 2
    assert (stats.epoch_lo, stats.epoch_hi) == (2, 3)


def test_band_is_symmetric_k_sigma():
    assert band(10.0, 0.5, k=2.0) == pytest.approx((9.0, 11.0))


def test_band_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k"):
        band(10.0, 0.5, k=0.0)
    with pytest.raises(ValueError, match="k"):
        band(10.0, 0.5, k=-2.0)


def test_compare_passes_when_candidate_mean_inside_band(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.2, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.05, 5.0), (2, 1.05, 5.0), (3, 1.1, 5.0)])
    results = compare(candidate, baseline, ["train_loss"], (1, 3))
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].column == "train_loss"
    assert results[0].n == 3
    assert (results[0].epoch_lo, results[0].epoch_hi) == (1, 3)
    assert results[0].candidate_sd == pytest.approx(
        __import__("statistics").pstdev([1.05, 1.05, 1.1])
    )


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


# --- CRITICAL 1: a candidate missing plateau epochs, or carrying an empty/NaN cell in an
# in-range row, must NEVER be averaged down to a silent PASS. ---


def test_compare_raises_when_candidate_missing_plateau_epochs(tmp_path):
    """Baseline logged the full 1..21 plateau; the candidate logged only 3 of those 21 epochs
    (e.g. training was killed early). Averaging the candidate's 3 rows and comparing that mean
    to the baseline's 21-row mean must be refused outright, not silently produce a PASS."""
    baseline_pairs = [(e, 1.0, 5.0) for e in range(1, 22)]
    baseline = _write(tmp_path, "base.tsv", baseline_pairs)
    # Candidate mean is deliberately identical to the baseline's so that, absent the coverage
    # check, this would trivially PASS.
    candidate_pairs = [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.0, 5.0)]
    candidate = _write(tmp_path, "cand.tsv", candidate_pairs)
    with pytest.raises(ValueError) as exc_info:
        compare(candidate, baseline, ["train_loss"], (1, 21))
    msg = str(exc_info.value)
    assert "3" in msg and "21" in msg  # names both counts


def test_compare_raises_when_candidate_epoch_range_is_narrower(tmp_path):
    """Same row COUNT, but the candidate's rows land on a different sub-range of epochs than
    the baseline's — coverage must match on epoch min/max too, not just on count."""
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.0, 5.0), (1, 1.0, 5.0), (2, 1.0, 5.0)])
    # Note: duplicate "epoch 1" row above keeps candidate row count == 3 while its epoch range
    # (1..2) differs from baseline's (1..3).
    with pytest.raises(ValueError, match="epoch"):
        compare(candidate, baseline, ["train_loss"], (1, 3))


def test_compare_raises_on_empty_cell_in_candidate_range(tmp_path):
    """`pandas.DataFrame(...).to_csv(sep='\\t')` renders NaN as an empty string. An empty cell
    for a requested column, on an in-range row, must raise — never be silently dropped and
    averaged around."""
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.0, 5.0)])
    candidate = _write_raw(
        tmp_path,
        "cand.tsv",
        HEADER,
        ["1\t1.0\t5.0", "2\t\t5.0", "3\t1.0\t5.0"],  # epoch 2's train_loss cell is empty
    )
    with pytest.raises(ValueError) as exc_info:
        compare(candidate, baseline, ["train_loss"], (1, 3))
    assert "2" in str(exc_info.value)  # names the offending epoch


def test_compare_raises_on_empty_cell_in_baseline_range(tmp_path):
    baseline = _write_raw(
        tmp_path,
        "base.tsv",
        HEADER,
        ["1\t1.0\t5.0", "2\t\t5.0", "3\t1.0\t5.0"],
    )
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.0, 5.0)])
    with pytest.raises(ValueError) as exc_info:
        compare(candidate, baseline, ["train_loss"], (1, 3))
    assert "2" in str(exc_info.value)


def test_plateau_stats_raises_naming_epoch_on_empty_cell():
    rows = [
        {"epoch": "1", "train_loss": "1.0"},
        {"epoch": "2", "train_loss": ""},
        {"epoch": "3", "train_loss": "1.0"},
    ]
    with pytest.raises(ValueError, match="2"):
        plateau_stats(rows, "train_loss", (1, 3))


def test_plateau_stats_raises_on_ragged_missing_column_not_typeerror():
    """A short row (fewer fields than the header) makes csv.DictReader fill the missing
    trailing field with None, not "". This must still surface as a clear ValueError, never a
    raw TypeError from float(None)."""
    rows = [
        {"epoch": "1", "train_loss": "1.0"},
        {"epoch": "2", "train_loss": None},
        {"epoch": "3", "train_loss": "1.0"},
    ]
    with pytest.raises(ValueError, match="2"):
        plateau_stats(rows, "train_loss", (1, 3))


def test_plateau_stats_raises_on_non_numeric_cell():
    rows = [
        {"epoch": "1", "train_loss": "1.0"},
        {"epoch": "2", "train_loss": "not-a-number"},
    ]
    with pytest.raises(ValueError, match="2"):
        plateau_stats(rows, "train_loss", (1, 2))


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


def test_main_returns_one_when_candidate_is_missing_plateau_epochs(tmp_path, capsys):
    """The headline requirement: an incomplete candidate must never print PASS."""
    baseline_pairs = [(e, 1.0, 5.0) for e in range(1, 22)]
    baseline = _write(tmp_path, "base.tsv", baseline_pairs)
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.0, 5.0)])
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
            "21",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "PASS" not in captured.out
    assert captured.err.startswith("error: ")


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


def test_main_reports_clean_error_when_candidate_path_is_a_directory(tmp_path, capsys):
    """Passing a directory instead of a TSV file used to raise a raw IsADirectoryError out of
    main() (IsADirectoryError is an OSError, previously uncaught here)."""
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0)])
    a_directory = tmp_path / "not_a_file"
    a_directory.mkdir()
    code = main(
        [
            "--candidate",
            str(a_directory),
            "--baseline",
            str(baseline),
            "--columns",
            "train_loss",
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


def test_main_reports_clean_error_on_non_positive_k(tmp_path, capsys):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0)])
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
            "2",
            "--k",
            "-1",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")


def test_main_prints_row_count_epoch_range_and_candidate_sd(tmp_path, capsys):
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
    out = capsys.readouterr().out
    assert "n=3" in out
    assert "1" in out and "3" in out  # epoch range printed somewhere
    assert out.count("sd=") >= 2  # both baseline sd and candidate sd are printed


def test_result_exposes_candidate_sd_n_and_epoch_range(tmp_path):
    baseline = _write(tmp_path, "base.tsv", [(1, 1.0, 5.0), (2, 1.0, 5.0), (3, 1.2, 5.0)])
    candidate = _write(tmp_path, "cand.tsv", [(1, 1.05, 5.0), (2, 1.05, 5.0), (3, 1.1, 5.0)])
    results = compare(candidate, baseline, ["train_loss"], (1, 3))
    r = results[0]
    assert isinstance(r, Result)
    assert hasattr(r, "candidate_sd")
    assert hasattr(r, "n")
    assert hasattr(r, "epoch_lo") and hasattr(r, "epoch_hi")
