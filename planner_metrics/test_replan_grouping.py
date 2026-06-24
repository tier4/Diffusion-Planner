"""Tests for scenario grouping of the replan-consistency driver (path-only, no data)."""

from __future__ import annotations


def test_group_frames_by_scenario_groups_and_sorts():
    from planner_metrics.replan_consistency import group_frames_by_scenario

    paths = [
        "/d/2026-02-25/10-24-06/10-24-06_0000000000000037.npz",
        "/d/2026-02-25/10-24-06/10-24-06_0000000000000031.npz",
        "/d/2026-02-25/10-24-06/10-24-06_0000000000000033.npz",
        "/d/2026-02-25/09-58-54/09-58-54_0000000000000039.npz",
        "/d/2026-02-25/09-58-54/09-58-54_0000000000000031.npz",
    ]
    groups = group_frames_by_scenario(paths)
    assert set(groups.keys()) == {
        "/d/2026-02-25/10-24-06",
        "/d/2026-02-25/09-58-54",
    }
    # sorted ascending by frame index, value is (index, path)
    idxs = [i for i, _ in groups["/d/2026-02-25/10-24-06"]]
    assert idxs == [31, 33, 37]
    idxs2 = [i for i, _ in groups["/d/2026-02-25/09-58-54"]]
    assert idxs2 == [31, 39]


def test_group_frames_single_frame_scenario_kept():
    from planner_metrics.replan_consistency import group_frames_by_scenario

    groups = group_frames_by_scenario(["/d/sess/sess_0000000000000005.npz"])
    assert groups == {"/d/sess": [(5, "/d/sess/sess_0000000000000005.npz")]}


def test_consecutive_pairs_yields_adjacent_frames():
    from planner_metrics.replan_consistency import consecutive_frame_pairs

    paths = [
        "/d/s/s_0000000000000031.npz",
        "/d/s/s_0000000000000033.npz",
        "/d/s/s_0000000000000035.npz",
    ]
    pairs = list(consecutive_frame_pairs(paths))
    # (idx_a, path_a, idx_b, path_b, frame_gap)
    assert pairs == [
        (31, "/d/s/s_0000000000000031.npz", 33, "/d/s/s_0000000000000033.npz", 2),
        (33, "/d/s/s_0000000000000033.npz", 35, "/d/s/s_0000000000000035.npz", 2),
    ]


def test_consecutive_pairs_no_cross_scenario():
    from planner_metrics.replan_consistency import consecutive_frame_pairs

    paths = [
        "/d/s1/s1_0000000000000031.npz",
        "/d/s2/s2_0000000000000031.npz",
    ]
    # different scenarios -> no pair
    assert list(consecutive_frame_pairs(paths)) == []


# --- real dataset convention: {session}/{HH-MM-SS}_{scene}_{frame}.npz ---
def test_parse_two_field_groups_by_session_and_scene():
    from planner_metrics.replan_consistency import group_frames_by_scenario

    paths = [
        "/d/2026-04-08/14-40-01/14-40-01_00000000_00000034.npz",
        "/d/2026-04-08/14-40-01/14-40-01_00000000_00000031.npz",
        "/d/2026-04-08/14-40-01/14-40-01_00000001_00000031.npz",  # different scene
    ]
    groups = group_frames_by_scenario(paths)
    # scene is part of the group key -> scene 0 and scene 1 are separate
    assert set(groups.keys()) == {
        "/d/2026-04-08/14-40-01#00000000",
        "/d/2026-04-08/14-40-01#00000001",
    }
    assert [i for i, _ in groups["/d/2026-04-08/14-40-01#00000000"]] == [31, 34]


def test_consecutive_pairs_modal_step_skips_gaps():
    from planner_metrics.replan_consistency import consecutive_frame_pairs

    # within one scene: a step-3 run (31,34,37), then a gap, then another run (1114,1117)
    paths = [
        "/d/s/s_00000000_00000031.npz",
        "/d/s/s_00000000_00000034.npz",
        "/d/s/s_00000000_00000037.npz",
        "/d/s/s_00000000_00001114.npz",
        "/d/s/s_00000000_00001117.npz",
    ]
    pairs = list(consecutive_frame_pairs(paths))
    gaps = {(a, b) for a, _, b, _, _ in pairs}
    # only modal-step (3) adjacent pairs; the 37->1114 jump is NOT a pair
    assert gaps == {(31, 34), (34, 37), (1114, 1117)}
    # frame_gap reported is the step (3) -> equals g (trajectory steps)
    assert all(gap == 3 for *_, gap in pairs)


# --- driver data-layer helpers (pure; validated against real GT separately) ---
def test_ego_future_to_4col():
    import math

    import numpy as np
    import torch

    from planner_metrics.replan_consistency import ego_future_to_4col

    arr = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, math.pi / 2]], dtype=np.float32)
    out = ego_future_to_4col(arr)  # (2, 4) x,y,cos,sin
    assert out.shape == (2, 4)
    assert torch.allclose(out[:, :2], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.allclose(out[0, 2:], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[1, 2:], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(out[:, 2] ** 2 + out[:, 3] ** 2, torch.ones(2), atol=1e-6)


def test_ego_future_to_4col_passthrough_when_already_4():
    import numpy as np
    import torch

    from planner_metrics.replan_consistency import ego_future_to_4col

    arr = np.zeros((5, 4), dtype=np.float32)
    arr[:, 2] = 1.0
    out = ego_future_to_4col(arr)
    assert out.shape == (5, 4)
    assert torch.allclose(out, torch.from_numpy(arr))


def test_inter_frame_transform_returns_step_g_minus_1():
    import math

    import torch

    from planner_metrics.replan_consistency import inter_frame_transform

    fut = torch.zeros(10, 4)
    fut[:, 2] = 1.0
    fut[3] = torch.tensor([1.5, 0.5, math.cos(0.2), math.sin(0.2)])
    rel_pos, rel_h = inter_frame_transform(fut, g=4)  # frame at step g-1 = 3
    assert torch.allclose(rel_pos, torch.tensor([1.5, 0.5]))
    assert abs(rel_h.item() - 0.2) < 1e-6
