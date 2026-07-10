from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import rlvr.autoresearch.tools.eval_det_avoidance as eval_det_avoidance
from planner_metrics.aggregate import compute_subscores_batch, compute_subscores_scene_batch
from rlvr.autoresearch.tools.classify_scene_failures import (
    _DEFAULT_THRESHOLD_CONFIG,
    _apply_scene_thresholds,
    _classify_det,
    _load_npz_data,
    _load_scene_thresholds,
    _merge_output_dirs,
    _prediction_path_for_scene,
    _prepare_scoring_data,
    _save_prediction_batch,
    _saved_prediction_trajectory,
    _stack_scene_data,
    _write_outputs,
    classify_loaded_scene,
    classify_loaded_scenes_batch,
)
from rlvr.reward import RewardConfig

T = 80


def _ego_shape() -> torch.Tensor:
    return torch.tensor([[0.0, 2.0, 2.0]], dtype=torch.float32)


def _moving_collision_data_3col() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = torch.arange(T, dtype=torch.float32)
    ego = torch.stack(
        [
            1.0 + 0.2 * t,
            torch.zeros(T),
            torch.ones(T),
            torch.zeros(T),
        ],
        dim=-1,
    ).unsqueeze(0)

    neighbor = torch.zeros(1, 1, T, 3)
    neighbor[..., 0] = 100.0
    neighbor[..., 2] = 0.0
    neighbor[0, 0, 30, 0] = ego[0, 30, 0]
    neighbor[0, 0, 30, 1] = 0.0

    past = torch.zeros(1, 1, 21, 11)
    past[0, 0, -1, 6] = 2.0
    past[0, 0, -1, 7] = 2.0

    data = {
        "ego_shape": _ego_shape(),
        "neighbor_agents_future": neighbor,
        "neighbor_agents_past": past,
    }
    return ego, data


def _clone_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in data.items()}


def _rear_end_collision_data_3col() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ego = torch.zeros(1, T, 4, dtype=torch.float32)
    ego[0, :, 0] = torch.arange(T, dtype=torch.float32) * 0.5
    ego[0, :, 2] = 1.0

    neighbor = torch.zeros(1, 1, T, 3)
    neighbor[0, 0, :, 0] = ego[0, :, 0] - 3.0
    neighbor[0, 0, :, 2] = 0.0

    past = torch.zeros(1, 1, 21, 11)
    past[0, 0, -1, 0] = -3.0
    past[0, 0, -1, 2] = 1.0
    past[0, 0, -1, 6] = 2.0
    past[0, 0, -1, 7] = 4.5

    data = {
        "ego_shape": _ego_shape(),
        "neighbor_agents_future": neighbor,
        "neighbor_agents_past": past,
    }
    return ego, data


def test_classify_scene_failures_converts_3col_future_and_flags_moving_collision():
    ego, data = _moving_collision_data_3col()

    row = classify_loaded_scene(
        "/tmp/moving_collision.npz",
        ego,
        data,
        RewardConfig(),
        moving_collision_thresh=0.2,
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )

    assert "moving_collision" in row["labels"]
    assert row["moving_collision_step"] == 30
    assert row["moving_argmin_t"] == 30
    assert row["moving_argmin_neighbor"] == 0
    assert row["moving_min_dist"] < 0.0
    assert row["ttc_first_collision_step"] == 30
    assert "static_collision" in row
    assert "static_min_dist" in row
    assert "static_collision_step" in row
    assert "static_neighbor_count" in row


def test_compute_subscores_scene_batch_matches_per_scene_scoring():
    ego, data = _moving_collision_data_3col()
    clean_candidate = ego.clone()
    clean_candidate[..., 1] = 10.0
    candidates = torch.cat([ego, clean_candidate], dim=0)
    datas = [_prepare_scoring_data(data), _prepare_scoring_data(_clone_data(data))]

    batched = compute_subscores_scene_batch(
        candidates.unsqueeze(0).repeat(2, 1, 1, 1),
        _stack_scene_data(datas),
        RewardConfig(),
    )
    single = compute_subscores_batch(candidates, datas[0], RewardConfig())

    assert torch.allclose(batched["safety"][0], single["safety"])
    assert torch.allclose(batched["ttc"][1], single["ttc"])
    assert batched["collision_step"][0] == single["collision_step"]
    assert batched["ttc_first_collision_steps"][1] == single["ttc_first_collision_steps"]


def test_classify_loaded_scenes_batch_handles_multiple_scenes_one_trajectory_each():
    ego, data = _moving_collision_data_3col()
    clean_ego = ego.clone()
    clean_ego[..., 1] = 10.0
    ego_trajs = torch.stack([ego, clean_ego], dim=0)

    rows = classify_loaded_scenes_batch(
        ["/tmp/a.npz", "/tmp/b.npz"],
        ego_trajs,
        [_clone_data(data), _clone_data(data)],
        RewardConfig(),
        moving_collision_thresh=0.2,
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )

    assert len(rows) == 2
    assert [row["candidate_index"] for row in rows] == [0, 0]
    assert "moving_collision" in rows[0]["labels"]
    assert rows[0]["moving_collision_step"] == 30
    assert rows[1]["labels"] == ["clean"]


def test_classify_scene_failures_counts_rear_end_collision_when_enabled():
    ego, data = _rear_end_collision_data_3col()

    row = classify_loaded_scene(
        "/tmp/rear_end_collision.npz",
        ego,
        data,
        RewardConfig(ignore_rear_end_collisions=False),
        moving_collision_thresh=0.2,
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )

    assert "moving_collision" in row["labels"]
    assert row["moving_collision_step"] == 0


def test_classify_scene_failures_suppresses_rear_end_collision_by_default():
    # Under the DEFAULT config (ignore_rear_end_collisions=True) the shared gated
    # rule SUPPRESSES a rear-end collision (NPC hits the ego from behind — not the
    # ego's fault), matching compute_safety_score_batch / the reward definition.
    # It is only labelled moving_collision when --count_rear_end_collisions flips
    # ignore_rear_end_collisions=False (see the _when_enabled test above). This
    # replaces the old raw clearance<=thresh rule, which had no rear-end
    # suppression and drifted from the reward.
    ego, data = _rear_end_collision_data_3col()

    row = classify_loaded_scene(
        "/tmp/rear_end_collision.npz",
        ego,
        data,
        RewardConfig(),
        moving_collision_thresh=0.2,
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )

    assert "moving_collision" not in row["labels"]
    assert row["moving_collision_step"] is None


def test_classify_scene_failures_writes_null_rb_min_dist_without_borders(tmp_path):
    ego = torch.zeros(1, T, 4)
    ego[..., 2] = 1.0
    data = {
        "ego_shape": _ego_shape(),
        "line_strings": torch.zeros(1, 0, 20, 4),
    }

    row = classify_loaded_scene(
        "/tmp/no_road_borders.npz",
        ego,
        data,
        RewardConfig(),
        moving_collision_thresh=0.2,
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )
    _write_outputs(
        [row],
        [],
        tmp_path,
        {"moving_collision_thresh": 0.2, "moving_near_thresh": 1.0},
    )

    written = (tmp_path / "classified_scenes.jsonl").read_text()
    assert "Infinity" not in written
    assert "NaN" not in written
    parsed = json.loads(written)
    assert parsed["rb_min_dist"] is None
    assert json.loads((tmp_path / "summary.json").read_text())["n_errors"] == 0


def test_classify_scene_failures_writes_training_path_lists(tmp_path):
    rows = [
        {
            "scene_path": "/tmp/a.npz",
            "labels": ["moving_collision", "road_border_crossing"],
            "moving_min_dist": float("inf"),
        },
        {"scene_path": "/tmp/a.npz", "labels": ["clean"]},
        {"scene_path": "/tmp/b.npz", "labels": ["clean"]},
        {"scene_path": "/tmp/c.npz", "labels": ["moving_collision"]},
    ]

    _write_outputs(
        rows,
        [],
        tmp_path,
        {"moving_collision_thresh": 0.2, "moving_near_thresh": 1.0},
    )

    assert json.loads((tmp_path / "lists" / "moving_collision.json").read_text()) == [
        "/tmp/a.npz",
        "/tmp/c.npz",
    ]
    assert json.loads((tmp_path / "lists" / "all_flagged.json").read_text()) == [
        "/tmp/a.npz",
        "/tmp/c.npz",
    ]
    assert json.loads((tmp_path / "lists" / "clean.json").read_text()) == ["/tmp/b.npz"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["label_counts"]["moving_collision"] == 2
    assert summary["label_counts"]["clean"] == 2
    first_row = json.loads((tmp_path / "classified_scenes.jsonl").read_text().splitlines()[0])
    assert first_row["moving_min_dist"] is None


def test_scene_failure_threshold_config_uses_requested_defaults():
    thresholds = _load_scene_thresholds(_DEFAULT_THRESHOLD_CONFIG)

    assert thresholds == {
        "moving_collision_thresh": 0.2,
        "moving_near_thresh": 0.7,
        "static_near_thresh": 0.5,
        "rb_near_thresh": 0.2,
        "expert_disagreement_wait_speed_mps": 0.5,
        "expert_disagreement_wait_progress_m": 1.0,
        "expert_disagreement_forward_progress_gap_m": 2.0,
        "expert_disagreement_lag_progress_gap_m": 3.0,
        "expert_disagreement_moving_speed_mps": 1.0,
        "sc_cross_thresh": 0.2,
        "rb_cross_thresh": 0.2,
    }


def test_scene_failure_thresholds_override_reward_config():
    class Args:
        threshold_config = _DEFAULT_THRESHOLD_CONFIG
        moving_collision_thresh = None
        moving_near_thresh = None
        static_near_thresh = None
        rb_near_thresh = None
        expert_disagreement_wait_speed_mps = None
        expert_disagreement_wait_progress_m = None
        expert_disagreement_forward_progress_gap_m = None
        expert_disagreement_lag_progress_gap_m = None
        expert_disagreement_moving_speed_mps = None
        sc_cross_thresh = None
        rb_cross_thresh = None

    config = RewardConfig(rb_cross_thresh=0.45, rb_near_thresh=0.45, sc_near_thresh=0.4)
    thresholds = _apply_scene_thresholds(config, Args())

    assert thresholds["moving_collision_thresh"] == 0.2
    assert thresholds["moving_near_thresh"] == 0.7
    assert thresholds["static_near_thresh"] == 0.5
    assert thresholds["rb_near_thresh"] == 0.2
    assert thresholds["sc_cross_thresh"] == 0.2
    assert thresholds["rb_cross_thresh"] == 0.2
    assert config.rb_cross_thresh == 0.2
    assert config.rb_near_thresh == 0.2
    assert config.sc_near_thresh == 0.5


def test_merge_output_dirs_rejects_threshold_mismatch(tmp_path):
    shard = tmp_path / "shard"
    _write_outputs(
        [{"scene_path": "/tmp/a.npz", "labels": ["moving_near_miss"]}],
        [],
        shard,
        {"moving_collision_thresh": 0.2, "moving_near_thresh": 1.0},
    )

    try:
        _merge_output_dirs(
            [shard],
            tmp_path / "merged",
            {"moving_collision_thresh": 0.2, "moving_near_thresh": 0.7},
        )
    except ValueError as exc:
        assert "do not match requested merge thresholds" in str(exc)
    else:
        raise AssertionError("expected threshold mismatch to fail")


def test_saved_prediction_trajectory_extracts_ego_from_agent_major_npz(tmp_path):
    pred = torch.zeros(3, T, 4).numpy()
    pred[0, :, 0] = 1.5
    pred[1, :, 0] = 9.0
    pred_path = tmp_path / "prediction00000000.npz"

    np.savez(pred_path, prediction=pred, turn_indicator=0)

    ego = _saved_prediction_trajectory(pred_path, torch.device("cpu"))

    assert ego.shape == (1, T, 4)
    assert torch.allclose(ego[0, :, 0], torch.full((T,), 1.5))


def test_load_npz_data_preserves_nonzero_delay(tmp_path):
    scene_path = tmp_path / "scene.npz"
    np.savez(scene_path, ego_shape=np.array([2.79, 4.34, 1.70]), delay=np.array(4))

    data = _load_npz_data(scene_path, torch.device("cpu"))

    assert data["delay"].dtype == torch.long
    assert data["delay"].shape == (1,)
    assert int(data["delay"].item()) == 4


def test_prediction_path_for_scene_supports_flat_and_mirrored_layouts(tmp_path):
    scene_path = "/data/root/dataset/train/date/time/frame_000123.npz"
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    flat = flat_dir / "prediction00000007.npz"
    flat.write_bytes(b"")
    assert _prediction_path_for_scene(flat_dir, scene_path, 7) == flat

    mirrored_dir = tmp_path / "mirrored"
    mirrored = mirrored_dir / "dataset/train/date/time/frame_000123.npz"
    mirrored.parent.mkdir(parents=True)
    mirrored.write_bytes(b"")
    assert (
        _prediction_path_for_scene(
            mirrored_dir,
            scene_path,
            7,
            prediction_scene_root=tmp_path / "missing",
        )
        == mirrored
    )


def test_save_prediction_batch_writes_valid_predictor_layout(tmp_path):
    scene_path = tmp_path / "dataset" / "valid" / "2026-01-15" / "13-49-19" / "frame_000123.npz"
    predictions_dir = tmp_path / "predictions"
    pred = torch.zeros(1, 2, T, 4)
    pred[0, 0, :, 0] = 1.25
    pred[0, 1, :, 1] = 9.0
    turn = torch.tensor([2], dtype=torch.long)

    [saved_path] = _save_prediction_batch(predictions_dir, [str(scene_path)], pred, turn)

    assert (
        saved_path
        == predictions_dir / "dataset" / "valid" / "2026-01-15" / "13-49-19" / "frame_000123.npz"
    )
    with np.load(saved_path) as saved:
        assert saved["prediction"].shape == (2, T, 4)
        assert np.allclose(saved["prediction"][0, :, 0], 1.25)
        assert int(saved["turn_indicator"]) == 2

    ego = _saved_prediction_trajectory(saved_path, torch.device("cpu"))
    assert ego.shape == (1, T, 4)
    assert torch.allclose(ego[0, :, 0], torch.full((T,), 1.25))


def test_classify_det_can_save_compatible_predictions(monkeypatch, tmp_path):
    _, data = _moving_collision_data_3col()
    full_prediction = torch.zeros(1, 2, T, 4)
    full_prediction[0, 0, :, 0] = 3.0
    full_prediction[0, 1, :, 1] = 7.0
    ego_prediction = full_prediction[:, 0]

    monkeypatch.setattr(
        "rlvr.autoresearch.tools.classify_scene_failures._load_npz_data",
        lambda *_args, **_kwargs: _clone_data(data),
    )
    monkeypatch.setattr(
        eval_det_avoidance, "load_model", lambda *_args, **_kwargs: (object(), object())
    )
    monkeypatch.setattr(
        eval_det_avoidance,
        "det_inference_batched",
        lambda *_args, **_kwargs: (ego_prediction, full_prediction, torch.tensor([1])),
    )

    scene_path = str(
        tmp_path / "dataset" / "valid" / "2026-01-15" / "13-49-19" / "frame_000123.npz"
    )
    args = SimpleNamespace(
        model_path="/tmp/model.pth",
        batch_size=4,
        moving_collision_thresh=0.2,
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        save_predictions_dir=str(tmp_path / "saved_predictions"),
    )

    rows, errors = _classify_det([scene_path], RewardConfig(), args, torch.device("cpu"))

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["trajectory_source"] == "det"
    saved_path = Path(rows[0]["prediction_path"])
    assert saved_path.exists()
    with np.load(saved_path) as saved:
        assert saved["prediction"].shape == (2, T, 4)
        assert np.allclose(saved["prediction"][0, :, 0], 3.0)
        assert int(saved["turn_indicator"]) == 1


def test_classify_scene_flags_moving_collision_at_clearance_threshold():
    t = torch.arange(T, dtype=torch.float32)
    ego = torch.stack([0.5 * t, torch.zeros(T), torch.ones(T), torch.zeros(T)], dim=-1).unsqueeze(0)

    neighbor = torch.zeros(1, 1, T, 3)
    neighbor[0, 0, :, 0] = ego[0, :, 0] + 4.52
    neighbor[0, 0, :, 2] = 0.0

    past = torch.zeros(1, 1, 21, 11)
    past[0, 0, -1, 0] = 4.52
    past[0, 0, -1, 2] = 1.0
    past[0, 0, -1, 6] = 2.0
    past[0, 0, -1, 7] = 4.5

    data = {
        "ego_shape": torch.tensor([[2.79, 4.34, 1.70]], dtype=torch.float32),
        "neighbor_agents_future": neighbor,
        "neighbor_agents_past": past,
    }

    row = classify_loaded_scene(
        "/tmp/threshold_collision.npz",
        ego,
        data,
        RewardConfig(ignore_rear_end_collisions=False),
        moving_collision_thresh=0.2,
        moving_near_thresh=0.7,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )

    assert "moving_collision" in row["labels"]
    assert row["moving_collision_step"] == 0


# --- moving-collision distance-band definition (clearance <= moving_collision_thresh) ---
# The moving-collision detector counts a contact when the EXACT closest-point
# clearance is within the threshold band (default 0.2 m), matching the static
# path's sc_cross_thresh — NOT strict OBB overlap. These tests pin that a
# genuinely NON-overlapping neighbor (positive clearance, but <= 0.2 m) counts,
# a farther one does not, and rear-end / low-speed gating still apply.

from planner_metrics.subscores import compute_ego_neighbor_signed_clearance
from rlvr.autoresearch.tools.classify_scene_failures import _moving_collision_step_gated


def _forward_pair(gap_x: float, nsteps: int = 5, behind: bool = False):
    """Ego (2x2) + one neighbor (2x2), both moving forward together at 2.5 m/s.

    Returns (ego_trajs (1,T,4), ego_shape (3,), nf (1,T,4), ns (1,2), nv (1,T)).
    With both boxes 2 m long and centered, along-x clearance ~= |gap_x| - 2.
    """
    T = nsteps
    ego_x = torch.arange(T, dtype=torch.float32) * 0.25  # 2.5 m/s at dt=0.1
    ego = torch.zeros(1, T, 4)
    ego[0, :, 0] = ego_x
    ego[0, :, 2] = 1.0  # heading +x (cos=1, sin=0)
    off = -gap_x if behind else gap_x
    nf = torch.zeros(1, T, 4)
    nf[0, :, 0] = ego_x + off
    nf[0, :, 2] = 1.0
    ego_shape = torch.tensor([0.0, 2.0, 2.0], dtype=torch.float32)  # wb=0, L=2, W=2
    ns = torch.tensor([[2.0, 2.0]], dtype=torch.float32)  # width, length
    nv = torch.ones(1, T, dtype=torch.bool)
    return ego, ego_shape, nf, ns, nv


def _min_clearance(ego, ego_shape, nf, ns, nv) -> float:
    d = compute_ego_neighbor_signed_clearance(ego, ego_shape, nf, ns, nv)
    return float(d.min().item())


def test_moving_collision_counts_within_band_without_overlap():
    # gap 2.1 m centers -> ~0.1 m clearance: NON-overlapping but inside 0.2 m band.
    ego, es, nf, ns, nv = _forward_pair(2.1)
    clr = _min_clearance(ego, es, nf, ns, nv)
    assert clr > 0.0, f"expected non-overlapping (clr>0), got {clr:.3f}"
    assert clr <= 0.2, f"expected within 0.2 m band, got {clr:.3f}"
    cfg = RewardConfig()
    step = _moving_collision_step_gated(ego, es, nf, ns, nv, cfg, 0.2)
    assert step is not None, "within-band non-overlapping contact must count as a collision"
    # And the strict-overlap definition would NOT have flagged it:
    assert _moving_collision_step_gated(ego, es, nf, ns, nv, cfg, 0.0) is None


def test_moving_collision_ignores_outside_band():
    # gap 2.6 m centers -> ~0.6 m clearance: outside the 0.2 m band.
    ego, es, nf, ns, nv = _forward_pair(2.6)
    clr = _min_clearance(ego, es, nf, ns, nv)
    assert clr > 0.2, f"expected clearance > band, got {clr:.3f}"
    step = _moving_collision_step_gated(ego, es, nf, ns, nv, RewardConfig(), 0.2)
    assert step is None


def test_moving_collision_rear_end_suppressed_unless_counted():
    # Neighbor 0.1 m behind the ego, moving with it.
    ego, es, nf, ns, nv = _forward_pair(2.1, behind=True)
    clr = _min_clearance(ego, es, nf, ns, nv)
    assert 0.0 < clr <= 0.2
    # Default suppresses rear-ends -> not the ego's fault -> no collision.
    assert (
        _moving_collision_step_gated(
            ego, es, nf, ns, nv, RewardConfig(ignore_rear_end_collisions=True), 0.2
        )
        is None
    )
    # With --count_rear_end_collisions (ignore=False) it IS counted.
    assert (
        _moving_collision_step_gated(
            ego, es, nf, ns, nv, RewardConfig(ignore_rear_end_collisions=False), 0.2
        )
        is not None
    )


def test_moving_collision_low_speed_suppressed():
    # In-band ahead contact, but ego is nearly stationary (<1 m/s) -> queued
    # traffic, not a collision (low-speed gate, T>=2).
    ego, es, nf, ns, nv = _forward_pair(2.1)
    ego[0, :, 0] = torch.arange(ego.shape[1], dtype=torch.float32) * 0.02  # 0.2 m/s
    nf[0, :, 0] = ego[0, :, 0] + 2.1
    step = _moving_collision_step_gated(ego, es, nf, ns, nv, RewardConfig(), 0.2)
    assert step is None
