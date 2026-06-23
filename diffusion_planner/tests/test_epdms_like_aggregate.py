"""Unit tests for the single [0,1] EPDMS-like aggregate (``planner_metrics.epdms_like``)."""

from __future__ import annotations

import pytest
import torch

from planner_metrics import EPDMSLikeConfig, epdms_like_aggregate, gt_path_length


def _subscores():
    """Four synthetic scenes, batched as ``compute_subscores_batch`` would return.

    scene 0: perfect            -> score == 1.0
    scene 1: at-fault collision -> NC gate zeros it
    scene 2: degraded quality   -> gates pass, quality computable by hand
    scene 3: drivable-area exit -> DAC gate zeros it
    """
    f = torch.float32
    return {
        "ttc": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=f),
        "progress": torch.tensor([10.0, 10.0, 5.0, 10.0], dtype=f),
        "comfort": torch.tensor([0.0, 0.0, -5.0, 0.0], dtype=f),  # -mean|jerk|
        "centerline": torch.tensor([0.0, 0.0, -4.0, 0.0], dtype=f),  # -lane_usage^2
        "red_light": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=f),
        "rb_crossing_gate": torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=f),
        "kinematic_gate": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=f),
        "collision_step": [None, 42, None, None],
    }, torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=f)  # gt_progress


def test_score_in_unit_interval_and_known_values():
    subs, gt = _subscores()
    cfg = EPDMSLikeConfig()  # defaults: jerk_bound=2, lane_usage_bound=1; w=5/5/2/5
    score, comp = epdms_like_aggregate(subs, gt, cfg)

    assert score.shape == (4,)
    assert torch.all(score >= 0.0) and torch.all(score <= 1.0)

    # scene 0: everything perfect -> 1.0
    assert score[0].item() == pytest.approx(1.0, abs=1e-6)

    # scene 1: collision -> NC gate hard-zeros regardless of quality
    assert score[1].item() == 0.0
    assert comp["gate_nc"][1].item() == 0.0

    # scene 3: drivable-area violation -> DAC gate hard-zeros
    assert score[3].item() == 0.0
    assert comp["gate_dac"][3].item() == 0.0

    # scene 2: gates all pass; quality by hand.
    #   ttc_q=1, progress_q=5/10=0.5, comfort_q=0 (|jerk|5 > 2), lane_q=0 (usage 2 > 1)
    #   quality = (5*1 + 5*0.5 + 2*0 + 5*0) / (5+5+2+5) = 7.5/17
    assert comp["q_progress"][2].item() == pytest.approx(0.5, abs=1e-6)
    assert comp["q_comfort"][2].item() == 0.0
    assert comp["q_lane"][2].item() == 0.0
    assert score[2].item() == pytest.approx(7.5 / 17.0, abs=1e-5)


def test_comfort_and_lane_thresholds_are_binary():
    subs, gt = _subscores()
    # Loosen bounds so scene 2 now passes comfort (|jerk|=5) and lane (usage=2).
    cfg = EPDMSLikeConfig(jerk_bound=10.0, lane_usage_bound=3.0)
    score, comp = epdms_like_aggregate(subs, gt, cfg)
    assert comp["q_comfort"][2].item() == 1.0
    assert comp["q_lane"][2].item() == 1.0
    # quality now = (5*1 + 5*0.5 + 2*1 + 5*1) / 17 = 14.5/17
    assert score[2].item() == pytest.approx(14.5 / 17.0, abs=1e-5)


def test_red_light_violation_gates_score():
    subs, gt = _subscores()
    subs["red_light"] = torch.tensor([0.0, 0.0, 0.0, -10.0], dtype=torch.float32)
    # also clear the DAC violation on scene 3 to isolate the TLC gate
    subs["rb_crossing_gate"] = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    score, comp = epdms_like_aggregate(subs, gt)
    assert comp["gate_tlc"][3].item() == 0.0
    assert score[3].item() == 0.0


def test_gt_path_length_straight_line():
    # 5 points spaced 2 m apart on a straight line -> length 8 m.
    t = torch.arange(5, dtype=torch.float32) * 2.0
    traj = torch.stack([t, torch.zeros_like(t)], dim=-1).unsqueeze(0)  # (1, 5, 2)
    assert gt_path_length(traj)[0].item() == pytest.approx(8.0, abs=1e-6)
