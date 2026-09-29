"""Coverage for ``scenario_generation.trajectory_colormap``, in particular the 4 metrics
(``centerline``, ``turn_indicator``, ``deviation_collision``, ``collision_rear``) added
alongside the ``rollout.jsonl`` fields those metrics are derived from."""

import json
from pathlib import Path

import pytest

from scenario_generation.trajectory_colormap import (
    METRIC_CHOICES,
    _risk_and_ticks,
    render_trajectory_colormap,
    render_trajectory_colormaps,
)

_NEW_METRICS = ("centerline", "turn_indicator", "deviation_collision", "collision_rear")


def _write_rollout(png_dir: Path, rows: list[dict]) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    with (png_dir / "rollout.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _full_row(k: int, **overrides) -> dict:
    row = {
        "k": k,
        "ego": [float(k), 0.0],
        "yaw": 0.0,
        "speed": 3.0,
        "clearance_m": 5.0,
        "collision": False,
        "collision_rear": False,
        "rb_dist_m": 2.0,
        "red_light_violation": False,
        "gt_deviation_m": 0.1,
        "deviation_collision": False,
        "centerline_dist_m": 0.3,
        "turn_indicator_pred": 0,
        "turn_indicator_gt": 0,
    }
    row.update(overrides)
    return row


def test_render_trajectory_colormaps_covers_every_metric_choice(tmp_path: Path):
    """Every entry in METRIC_CHOICES (old + new) renders when the trace has every field."""
    png_dir = tmp_path / "seg"
    rows = [_full_row(k) for k in range(5)]
    _write_rollout(png_dir, rows)

    out_dir = tmp_path / "out"
    rendered = render_trajectory_colormaps(png_dir, out_dir, "seg0_0_5")

    assert set(rendered) == set(METRIC_CHOICES)
    for path in rendered.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_new_metrics_skipped_on_old_style_rollout(tmp_path: Path):
    """A rollout.jsonl predating these fields renders the 7 old metrics and skips the 4 new
    ones, instead of erroring -- same contract _METRIC_TRACE_KEYS already gives old runs."""
    png_dir = tmp_path / "seg"
    old_rows = [
        {
            "k": k,
            "ego": [float(k), 0.0],
            "yaw": 0.0,
            "speed": 3.0,
            "clearance_m": 5.0,
            "collision": False,
            "rb_dist_m": 2.0,
            "red_light_violation": False,
            "gt_deviation_m": 0.1,
        }
        for k in range(5)
    ]
    _write_rollout(png_dir, old_rows)

    rendered = render_trajectory_colormaps(png_dir, tmp_path / "out", "seg0_0_5")

    assert set(rendered) == set(METRIC_CHOICES) - set(_NEW_METRICS)
    for metric in _NEW_METRICS:
        assert render_trajectory_colormap(png_dir, tmp_path / f"solo_{metric}.png", metric=metric) is None


def test_centerline_risk_scaling_and_missing_value(tmp_path: Path):
    rows = [
        _full_row(0, centerline_dist_m=0.0),
        _full_row(1, centerline_dist_m=2.0),  # == centerline_thresh_m -> cap/2 -> risk 0.5
        _full_row(2, centerline_dist_m=None),  # unmeasured -> treated as 0.0, not worst-case
        _full_row(3, centerline_dist_m=10.0),  # far past the cap -> clamped to 1.0
    ]
    risk, ticks, labels = _risk_and_ticks(rows, "centerline", near_miss_thresh=0.5, centerline_thresh_m=2.0)

    assert risk[0] == pytest.approx(0.0)
    assert risk[1] == pytest.approx(0.5)
    assert risk[2] == pytest.approx(0.0)
    assert risk[3] == pytest.approx(1.0)
    assert ticks == [0.0, 0.33, 0.66, 1.0]
    assert labels[0] == "0.00m"


def test_turn_indicator_risk_flags_mismatch_only(tmp_path: Path):
    rows = [
        _full_row(0, turn_indicator_pred=1, turn_indicator_gt=1),
        _full_row(1, turn_indicator_pred=2, turn_indicator_gt=1),
        _full_row(2, turn_indicator_pred=0, turn_indicator_gt=0),
    ]
    risk, ticks, labels = _risk_and_ticks(rows, "turn_indicator", near_miss_thresh=0.5)

    assert list(risk) == [0.0, 1.0, 0.0]
    assert ticks == [0.0, 1.0]
    assert labels == ["indicator matches GT", "indicator mismatch"]


@pytest.mark.parametrize(
    "metric,key,labels",
    [
        ("deviation_collision", "deviation_collision", ["no deviation collision", "deviation collision"]),
        ("collision_rear", "collision_rear", ["no rear collision", "rear collision"]),
    ],
)
def test_binary_new_metrics_map_truthy_to_one(metric, key, labels):
    rows = [_full_row(0, **{key: False}), _full_row(1, **{key: True})]
    risk, ticks, tick_labels = _risk_and_ticks(rows, metric, near_miss_thresh=0.5)

    assert list(risk) == [0.0, 1.0]
    assert ticks == [0.0, 1.0]
    assert tick_labels == labels
