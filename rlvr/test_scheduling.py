"""Unit tests for GRPOConfig per-epoch scheduling."""

from __future__ import annotations

import pytest

from rlvr.grpo_config import GRPOConfig


def test_linear_schedule():
    c = GRPOConfig(schedules={"w_progress": {"type": "linear", "start": 3.0, "end": 10.0}})
    assert c.get_scheduled_value("w_progress", 1, 20) == 3.0
    assert c.get_scheduled_value("w_progress", 20, 20) == 10.0
    mid = c.get_scheduled_value("w_progress", 11, 20)
    assert 6.0 < mid < 7.0  # ~6.68


def test_cosine_schedule():
    c = GRPOConfig(schedules={"w_progress": {"type": "cosine", "start": 3.0, "end": 10.0}})
    assert c.get_scheduled_value("w_progress", 1, 20) == pytest.approx(3.0)
    assert c.get_scheduled_value("w_progress", 20, 20) == pytest.approx(10.0)
    mid = c.get_scheduled_value("w_progress", 11, 20)
    # Cosine is slower than linear at midpoint
    assert 5.5 < mid < 7.5


def test_step_schedule():
    c = GRPOConfig(schedules={
        "w_progress": {"type": "step", "start": 3.0, "end": 10.0, "warmup_fraction": 0.5},
    })
    # Before warmup: start
    assert c.get_scheduled_value("w_progress", 1, 20) == 3.0
    assert c.get_scheduled_value("w_progress", 10, 20) == 3.0
    # After warmup: end
    assert c.get_scheduled_value("w_progress", 11, 20) == 10.0
    assert c.get_scheduled_value("w_progress", 20, 20) == 10.0


def test_constant_schedule():
    c = GRPOConfig(schedules={"w_progress": {"type": "constant", "start": 5.0, "end": 10.0}})
    assert c.get_scheduled_value("w_progress", 1, 20) == 5.0
    assert c.get_scheduled_value("w_progress", 20, 20) == 5.0


def test_total_epochs_one():
    c = GRPOConfig(schedules={"w_progress": {"type": "linear", "start": 3.0, "end": 10.0}})
    assert c.get_scheduled_value("w_progress", 1, 1) == 3.0


def test_no_schedule_returns_none():
    c = GRPOConfig(schedules={"w_progress": {"type": "linear", "start": 3.0, "end": 10.0}})
    assert c.get_scheduled_value("nonexistent", 1, 20) is None


def test_get_all_scheduled_values():
    c = GRPOConfig(schedules={
        "w_progress": {"type": "linear", "start": 3.0, "end": 10.0},
        "longitudinal_eta": {"type": "linear", "start": 0.0, "end": 1.0},
    })
    vals = c.get_all_scheduled_values(1, 20)
    assert "w_progress" in vals
    assert "longitudinal_eta" in vals
    assert vals["w_progress"] == pytest.approx(3.0)
    assert vals["longitudinal_eta"] == pytest.approx(0.0)


def test_step_warmup_boundary():
    c = GRPOConfig(schedules={
        "x": {"type": "step", "start": 1.0, "end": 2.0, "warmup_fraction": 0.3},
    })
    # progress at ep7/20 = 6/19 ≈ 0.316 > 0.3 → end
    assert c.get_scheduled_value("x", 7, 20) == 2.0
    # progress at ep6/20 = 5/19 ≈ 0.263 < 0.3 → start
    assert c.get_scheduled_value("x", 6, 20) == 1.0


def test_invalid_warmup_fraction():
    c = GRPOConfig(schedules={
        "x": {"type": "step", "start": 1.0, "end": 2.0, "warmup_fraction": 1.5},
    })
    with pytest.raises(ValueError, match="warmup_fraction"):
        c.get_scheduled_value("x", 1, 20)


def test_peak_schedule():
    c = GRPOConfig(schedules={
        "x": {"type": "peak", "start": 0.0, "end": 0.0, "peak": 0.3, "peak_fraction": 0.5},
    })
    # Endpoints
    assert c.get_scheduled_value("x", 1, 20) == pytest.approx(0.0)
    assert c.get_scheduled_value("x", 20, 20) == pytest.approx(0.0)
    # Peak at midpoint (ep11, progress=10/19≈0.526 > 0.5 → descending)
    mid = c.get_scheduled_value("x", 10, 20)  # progress=9/19≈0.474 < 0.5 → ascending
    assert 0.25 < mid < 0.31
    # Exactly at peak_fraction
    # ep11: progress = 10/19 ≈ 0.526 → just past peak, should be close to 0.3
    val_at_peak = c.get_scheduled_value("x", 11, 20)
    assert 0.25 < val_at_peak < 0.31


def test_peak_asymmetric():
    c = GRPOConfig(schedules={
        "x": {"type": "peak", "start": 0.0, "end": 0.1, "peak": 0.5, "peak_fraction": 0.3},
    })
    assert c.get_scheduled_value("x", 1, 20) == pytest.approx(0.0)
    assert c.get_scheduled_value("x", 20, 20) == pytest.approx(0.1)
    # Early peak means fast ramp up, slow ramp down
    ep4 = c.get_scheduled_value("x", 4, 20)  # progress=3/19≈0.158 < 0.3 → ascending
    assert ep4 > 0.2


def test_peak_invalid_fraction():
    c = GRPOConfig(schedules={
        "x": {"type": "peak", "start": 0.0, "end": 0.0, "peak": 0.3, "peak_fraction": 0.0},
    })
    with pytest.raises(ValueError, match="peak_fraction"):
        c.get_scheduled_value("x", 1, 20)


def test_invalid_schedule_type():
    c = GRPOConfig(schedules={"x": {"type": "invalid", "start": 1.0, "end": 2.0}})
    with pytest.raises(ValueError, match="Unknown schedule type"):
        c.get_scheduled_value("x", 1, 20)


def test_json_roundtrip(tmp_path):
    c = GRPOConfig(schedules={
        "w_progress": {"type": "cosine", "start": 3.0, "end": 10.0},
        "longitudinal_eta": {"type": "linear", "start": 0.0, "end": 0.5},
    })
    path = tmp_path / "test_sched_roundtrip.json"
    c.to_json(str(path))
    c2 = GRPOConfig.from_json(str(path))
    assert c2.schedules == c.schedules
    assert c2.get_scheduled_value("w_progress", 20, 20) == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
