import json
from types import SimpleNamespace

import numpy as np
import pytest

from planner_metrics.pdms_navsim import CollisionType
from scenario_generation import eval_windows as ew
from scenario_generation.metrics.at_fault import (
    at_fault_block,
    classify_collision_step,
    classify_new_collisions,
)
from scenario_generation.reproducer_rollout import _event_count

EGO_SHAPE = np.array([2.8, 4.8, 1.9], dtype=np.float32)  # wheelbase, length, width


# --------------------------------------------------------------------------- #
# window planning
# --------------------------------------------------------------------------- #
def test_fixed_windows_split_and_merge_short_tail():
    assert ew.plan_fixed_windows(900, 300) == [(0, 300), (300, 600), (600, 900)]
    # 20-frame remainder (< min_tail 50) merges into the previous window.
    assert ew.plan_fixed_windows(620, 300, min_tail=50) == [(0, 300), (300, 620)]
    # A long enough remainder stays its own window.
    assert ew.plan_fixed_windows(680, 300, min_tail=50) == [(0, 300), (300, 600), (600, 680)]
    assert ew.plan_fixed_windows(0, 300) == []
    assert ew.plan_fixed_windows(100, 300) == [(0, 100)]


def test_fixed_windows_with_stride_overlap():
    assert ew.plan_fixed_windows(700, 300, stride=200) == [
        (0, 300),
        (200, 500),
        (400, 700),
    ]


def test_anchor_windows_clip_and_drop_out_of_range():
    assert ew.plan_anchor_windows(1000, [500, 50, 995, -3, 1200], 100, 100) == [
        (400, 600),
        (0, 150),
        (895, 1000),
    ]


def test_resolve_route_anchors_exact_then_unique_substring():
    anchors = {"2026-06-16_10-27-57": [1, 2], "2026-06-30_11-06-16_00000000": [3]}
    key = "2231_odaiba_2026-06-16_10-27-57_00000000"
    assert ew.resolve_route_anchors(anchors, key) == [1, 2]
    assert ew.resolve_route_anchors(anchors, "other") == []
    assert ew.resolve_route_anchors({key: [9]}, key) == [9]
    with pytest.raises(ValueError, match="make them unique"):
        ew.resolve_route_anchors({"2026-06-16": [1], "10-27-57": [2]}, key)


def test_route_windows_uses_seconds_for_fixed_and_anchor_units():
    cfg = ew.WindowConfig(mode="fixed", window_len_s=3.0, min_tail_s=0.5)
    assert ew.route_windows(cfg, "r", 65) == [(0, 30), (30, 60), (60, 65)]
    cfg = ew.WindowConfig(
        mode="anchor", anchors={"r": [5.0]}, anchor_unit="sec", anchor_pre_s=1.0, anchor_post_s=2.0
    )
    assert ew.route_windows(cfg, "r", 100) == [(40, 70)]


def test_config_validation():
    with pytest.raises(ValueError, match="anchor mode requires"):
        ew.WindowConfig(mode="anchor").validate()
    with pytest.raises(ValueError, match="weights"):
        ew.WindowConfig(w_progress=0, w_lon=0, w_lat=0).validate()
    with pytest.raises(ValueError, match="coverage_abort_m"):
        ew.WindowConfig(coverage_abort_m=0).validate()


# --------------------------------------------------------------------------- #
# divergence / progress
# --------------------------------------------------------------------------- #
def _straight_poses(n: int, spacing: float = 1.0) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) * spacing
    return np.column_stack([x, np.zeros(n), np.zeros(n)])


def test_divergence_separates_lon_and_lat_on_a_straight_route():
    poses = _straight_poses(200)
    lo, hi = 50, 100
    # Live ego lags 2 m behind the recorded ego and sits 0.5 m to the left.
    live = np.column_stack([poses[lo:hi, 0] - 2.0, np.full(hi - lo, 0.5)])
    d = ew.window_divergence(poses, lo, hi, live, margin_frames=10)
    assert d["n_steps"] == hi - lo
    assert d["ade_lon_m"] == pytest.approx(2.0)
    assert d["ade_lat_m"] == pytest.approx(0.5)
    assert d["recorded_progress_m"] == pytest.approx(49.0)
    assert d["realized_progress_m"] == pytest.approx(47.0)
    assert d["progress_ratio"] == pytest.approx(47.0 / 49.0)


def test_divergence_clips_progress_and_handles_short_rollouts():
    poses = _straight_poses(200)
    lo, hi = 50, 100
    live = np.column_stack([poses[lo:hi, 0] + 10.0, np.zeros(hi - lo)])
    d = ew.window_divergence(poses, lo, hi, live, margin_frames=20)
    assert d["progress_ratio"] == 1.0  # ran ahead: clipped
    # A rollout cut short (diverged) only scores the ticks it ran.
    d = ew.window_divergence(poses, lo, hi, live[:10], margin_frames=20)
    assert d["n_steps"] == 10
    d = ew.window_divergence(poses, lo, hi, live[:0], margin_frames=20)
    assert d["n_steps"] == 0 and d["progress_ratio"] == 0.0


# --------------------------------------------------------------------------- #
# gates / composite
# --------------------------------------------------------------------------- #
def _metrics(*, terminated="max_steps", at_fault=0, rb_steps=0):
    return {
        "terminated": terminated,
        "at_fault": {"steps": at_fault, "count": at_fault, "by_type_steps": {}},
        "road_border": {"collision_steps": rb_steps},
    }


def _div(ratio=1.0, lon=0.0, lat=0.0, recorded=30.0):
    return {
        "progress_ratio": ratio,
        "ade_lon_m": lon,
        "ade_lat_m": lat,
        "recorded_progress_m": recorded,
    }


def test_score_is_graded_when_every_gate_passes():
    cfg = ew.WindowConfig()
    v = ew.score_window(_metrics(), _div(ratio=0.8, lon=3.0, lat=0.3), cfg)
    assert v["gate"] == 1
    assert v["score_lon"] == pytest.approx(0.9)
    assert v["score_lat"] == pytest.approx(0.9)
    assert v["graded"] == pytest.approx(0.5 * 0.8 + 0.25 * 0.9 + 0.25 * 0.9)
    assert v["score"] == pytest.approx(v["graded"])


@pytest.mark.parametrize(
    "metrics, div, gate",
    [
        (_metrics(terminated="diverged"), _div(), "coverage"),
        (_metrics(at_fault=1), _div(), "nc"),
        (_metrics(rb_steps=1), _div(), "offroad"),
        (_metrics(), _div(ratio=0.1), "progress"),
    ],
)
def test_each_gate_zeroes_the_score(metrics, div, gate):
    v = ew.score_window(metrics, div, ew.WindowConfig())
    assert v["gates"][gate] == 0
    assert v["gate"] == 0 and v["score"] == 0.0
    assert v["graded"] > 0  # graded survives for inspection


def test_low_recorded_progress_window_passes_progress_gate():
    v = ew.score_window(_metrics(), _div(ratio=0.0, recorded=2.0), ew.WindowConfig())
    assert v["low_recorded_progress"] is True
    assert v["gates"]["progress"] == 1 and v["progress_ratio"] == 1.0


def test_score_requires_at_fault_block():
    with pytest.raises(ValueError, match="at_fault"):
        ew.score_window({"terminated": "max_steps", "road_border": {}}, _div(), ew.WindowConfig())


def _epdms(score, *, invalid=False, reason="window_end", nc=1.0, bucket="main"):
    return {
        "score": score,
        "invalid": invalid,
        "bucket": bucket,
        "valid_ticks": 150,
        "valid_span": {"reason": reason, "tick": 150},
        "multiplicative": 1.0 if score else 0.0,
        "weighted": score if score else 0.5,
        "terms": {
            "nc": nc,
            "dac": 1.0,
            "ddc": 1.0,
            "tlc": 1.0,
            "mp": 1.0,
            "ep": 1.0,
            "ttc": 1.0,
            "sl": 1.0,
            "comfort": 1.0,
            "lk": 1.0,
        },
        "detail": {
            "progress_ratio": 1.0,
            "ade_lon_m": 1.0,
            "ade_lat_m": 0.1,
            "route_adherence_frac": 1.0,
            "tl_measured_frac": 0.5,
            "divergence_flag_ticks": 0,
        },
    }


def _row(route, score, **kw):
    return {
        "route": route,
        "terminated": "max_steps",
        "object": {"collision_count": 0},
        "road_border": {"collision_steps": 0},
        "at_fault": {"count": 0},
        "epdms": _epdms(score, **kw),
    }


def test_summary_macro_vs_micro_and_invalid_windows():
    rows = [
        _row("a", 1.0),
        _row("a", 0.0, nc=0.0),
        _row("b", 1.0),
        _row("b", None, invalid=True, reason="ghost_contact"),
    ]
    s = ew.summarize_windows(rows)
    assert s["n_windows"] == 4 and s["n_routes"] == 2 and s["n_valid"] == 3
    assert s["invalid_windows"] == 1
    assert s["valid_span_reasons"] == {"window_end": 3, "ghost_contact": 1}
    assert s["score_micro"] == pytest.approx(2 / 3)
    assert s["score_macro"] == pytest.approx((0.5 + 1.0) / 2)
    assert s["zero_windows"]["nc"] == 1
    assert s["term_means"]["nc"] == pytest.approx(2 / 3)
    rows.append(_row("a", 0.2, bucket="recorded_stop"))
    s = ew.summarize_windows(rows)
    assert s["n_valid"] == 3 and s["buckets"]["recorded_stop"]["n_windows"] == 1
    s = ew.summarize_windows(rows, include_recorded_stop=True)
    assert s["n_valid"] == 4


# --------------------------------------------------------------------------- #
# at-fault classification
# --------------------------------------------------------------------------- #
def _neighbor(x, y, heading=0.0, vx=0.0, vy=0.0, w=1.8, length=4.5):
    row = np.zeros(11, dtype=np.float32)
    row[:8] = [x, y, np.cos(heading), np.sin(heading), vx, vy, w, length]
    row[8] = 1.0
    return row


def _scene(*rows):
    nb = np.zeros((320, 11), dtype=np.float32)
    for i, r in enumerate(rows):
        nb[i] = r
    return nb


def test_front_collision_into_stopped_track_is_at_fault():
    # Ego center is 0.5*wheelbase ahead of the rear axle; a stopped car 3 m ahead overlaps.
    fault, types, _slots = classify_collision_step(
        _scene(_neighbor(3.0, 0.0)), EGO_SHAPE, 3.0, "cpu"
    )
    assert fault is True
    assert types == [int(CollisionType.STOPPED_TRACK_COLLISION)]


def test_active_front_collision_is_at_fault():
    fault, types, _slots = classify_collision_step(
        _scene(_neighbor(3.0, 0.0, vx=1.0)), EGO_SHAPE, 3.0, "cpu"
    )
    assert fault is True
    assert types == [int(CollisionType.ACTIVE_FRONT_COLLISION)]


def test_stopped_ego_hit_is_not_at_fault():
    fault, types, _slots = classify_collision_step(
        _scene(_neighbor(3.0, 0.0, vx=1.0)), EGO_SHAPE, 0.0, "cpu"
    )
    assert fault is False
    assert types == [int(CollisionType.STOPPED_EGO_COLLISION)]


def test_rear_end_by_follower_is_not_at_fault():
    # Moving neighbor overlapping the ego from behind (ego moving forward).
    fault, types, _slots = classify_collision_step(
        _scene(_neighbor(-2.5, 0.0, vx=5.0)), EGO_SHAPE, 3.0, "cpu"
    )
    assert fault is False
    assert types == [int(CollisionType.ACTIVE_REAR_COLLISION)]


def test_lateral_contact_is_lenient_and_mixed_scene_reports_all_types():
    lateral = _neighbor(1.0, 1.7, vx=1.0)  # side-by-side moving car overlapping laterally
    fault, types, _slots = classify_collision_step(_scene(lateral), EGO_SHAPE, 3.0, "cpu")
    assert fault is False
    assert types == [int(CollisionType.ACTIVE_LATERAL_COLLISION)]
    fault, types, _slots = classify_collision_step(
        _scene(lateral, _neighbor(3.0, 0.0)), EGO_SHAPE, 3.0, "cpu"
    )
    assert fault is True
    assert sorted(types) == sorted(
        [int(CollisionType.ACTIVE_LATERAL_COLLISION), int(CollisionType.STOPPED_TRACK_COLLISION)]
    )


def test_no_collision_and_empty_scene():
    assert classify_collision_step(_scene(_neighbor(20.0, 0.0)), EGO_SHAPE, 3.0, "cpu") == (
        False,
        [],
        [],
    )
    assert classify_collision_step(_scene(), EGO_SHAPE, 3.0, "cpu") == (False, [], [])


def test_collision_set_matches_score_object_step():
    from scenario_generation.metrics.object import score_object_step

    rng = np.random.default_rng(0)
    for _ in range(30):
        rows = [
            _neighbor(
                rng.uniform(-8, 8),
                rng.uniform(-4, 4),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-3, 3),
                rng.uniform(-3, 3),
            )
            for _ in range(4)
        ]
        scene = _scene(*rows)
        _, col, _ = score_object_step(scene, EGO_SHAPE, "cpu")
        _, types, _ = classify_collision_step(scene, EGO_SHAPE, 2.0, "cpu")
        assert bool(types) == bool(col)


def test_at_fault_block_counts_events_and_types():
    mask = np.array([0, 1, 1, 0, 0, 0, 0, 1], dtype=bool)
    types = [[], [2], [2], [], [], [], [], [1, 4]]
    block = at_fault_block(mask, types, _event_count)
    assert block["steps"] == 3
    assert block["count"] == 2
    assert block["by_type_steps"]["ACTIVE_FRONT_COLLISION"] == 2
    assert block["by_type_steps"]["STOPPED_TRACK_COLLISION"] == 1
    assert block["by_type_steps"]["ACTIVE_LATERAL_COLLISION"] == 1


# --------------------------------------------------------------------------- #
# driver contract (mocked rollout)
# --------------------------------------------------------------------------- #
def test_run_windowed_eval_pins_window_contract(tmp_path, monkeypatch):
    n = 130
    poses = _straight_poses(n)

    class _TL:
        def __init__(self, *_a, **_k):
            self.poses = poses
            self.frame_indices = np.arange(n)

        def __len__(self):
            return n

        def npz(self, idx):
            return {"ego_shape": np.array([2.8, 4.8, 1.9], dtype=np.float32)}

    calls = []

    def fake_render(model, model_args, tl, lo, hi, out_dir, **kw):
        calls.append((lo, hi, kw))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "rollout.jsonl", "w") as f:
            f.write(json.dumps({"event": "start"}) + "\n")
            for k in range(hi - lo):
                f.write(
                    json.dumps(
                        {"k": k, "ego": [float(poses[lo + k, 0]), 0.2], "yaw": 0.0, "speed": 1.0}
                    )
                    + "\n"
                )
        return {
            "segment": [lo, hi],
            "n_steps_run": hi - lo,
            "terminated": "max_steps",
            "route_completion": 1.0,
            "mean_gt_deviation_m": 0.2,
            "progress_m": float(hi - lo - 1),
            "object": {"collision_count": 0, "collision_steps": 0, "_tdigest": {"x": 1}},
            "road_border": {"collision_steps": 0, "collision_count": 0},
            "red_light_violation": {"steps": 0, "count": 0},
            "strong_brake": {"count": 0, "steps": 0},
            "reproducer": {},
            "at_fault": {"steps": 0, "count": 0, "by_type_steps": {}},
        }

    monkeypatch.setattr(ew, "render_segment", fake_render)
    seen = []

    def fake_epdms(tl, lo, hi, rows, ego_shape, cfg, **kw):
        seen.append((lo, hi, len(rows), tuple(ego_shape), cfg))
        return _epdms(0.75)

    monkeypatch.setattr(ew, "score_window_epdms", fake_epdms)
    monkeypatch.setattr(ew, "RouteTimeline", _TL)
    monkeypatch.setattr(
        ew, "enumerate_multi_root_routes", lambda _root: ({"routeA": ["p"]}, {"routeA": "d"})
    )
    cfg = ew.WindowConfig(mode="fixed", window_len_s=5.0, min_tail_s=2.0)
    summary = ew.run_windowed_eval(
        SimpleNamespace(),
        SimpleNamespace(),
        "root",
        tmp_path,
        cfg=cfg,
        render_kwargs={"device": "cpu", "draw_every": 8, "unstick_after": 300, "k_lag": 3},
        verbose=False,
    )
    assert [(lo, hi) for lo, hi, _ in calls] == [(0, 50), (50, 100), (100, 130)]
    kw = calls[0][2]
    assert kw["draw_every"] is None and kw["unstick_after"] == 0
    assert kw["goal_reach_m"] == 0.0 and kw["timeline_progress_mode"] == "clock"
    assert kw["at_fault_scoring"] is True and kw["abort_deviation_m"] == 100.0
    assert kw["k_lag"] == 3  # caller knobs survive
    rows = [json.loads(line) for line in open(tmp_path / "windows.jsonl")]
    assert len(rows) == 3 and "_tdigest" not in rows[0]["object"]
    assert rows[0]["v2"]["score"] == pytest.approx(1.0 - 0.25 * 0.2 / 3.0)
    assert rows[0]["epdms"]["score"] == 0.75
    assert [(lo, hi, n) for lo, hi, n, _, _ in seen] == [(0, 50, 50), (50, 100, 50), (100, 130, 30)]
    assert seen[0][4].min_valid_s == cfg.min_valid_s
    assert summary["n_windows"] == 3 and summary["score_micro"] == pytest.approx(0.75)
    assert (tmp_path / "windows_summary.json").is_file()


def test_rear_end_track_is_not_reclassified_when_it_ends_up_ahead():
    collided: dict = {}
    uuids = ["follower"] + [""] * 319
    # Tick 1: replayed follower hits the ego from behind -> ACTIVE_REAR, not at fault.
    fault, types, new = classify_new_collisions(
        _scene(_neighbor(-2.5, 0.0, vx=5.0)),
        EGO_SHAPE,
        3.0,
        "cpu",
        slot_uuids=uuids,
        collided=collided,
    )
    assert fault is False and types == [int(CollisionType.ACTIVE_REAR_COLLISION)]
    assert new == ["follower"] and collided == {
        "follower": int(CollisionType.ACTIVE_REAR_COLLISION)
    }
    # Tick 2: the same track has pushed through and now overlaps ahead of the ego.
    fault, types, new = classify_new_collisions(
        _scene(_neighbor(3.0, 0.0, vx=5.0)),
        EGO_SHAPE,
        3.0,
        "cpu",
        slot_uuids=uuids,
        collided=collided,
    )
    assert fault is False and types == [] and new == []
    # A different track hit head-on is still classified and at fault.
    uuids2 = ["follower", "lead"] + [""] * 318
    fault, types, new = classify_new_collisions(
        _scene(_neighbor(3.0, 0.0, vx=5.0), _neighbor(3.0, 0.0)),
        EGO_SHAPE,
        3.0,
        "cpu",
        slot_uuids=uuids2,
        collided=collided,
    )
    assert (
        fault is True and types == [int(CollisionType.STOPPED_TRACK_COLLISION)] and new == ["lead"]
    )
    # No UUID list: slots stand in for tracks.
    collided2: dict = {}
    classify_new_collisions(
        _scene(_neighbor(3.0, 0.0)), EGO_SHAPE, 3.0, "cpu", slot_uuids=None, collided=collided2
    )
    assert collided2 == {"slot0": int(CollisionType.STOPPED_TRACK_COLLISION)}


def test_at_fault_block_reports_dedup_and_hard_brake_counts():
    block = at_fault_block(
        np.array([0, 1], dtype=bool),
        [[], [2]],
        _event_count,
        collided={"a": 2, "b": 3},
        rear_under_hard_brake=1,
    )
    assert block["collided_tracks"] == 2 and block["rear_under_hard_brake_tracks"] == 1


def test_rescore_windows_reweights_from_stored_terms():
    e = _epdms(1.0)
    e["terms"]["clearance"] = 0.0
    e["weights"] = {"ep": 5.0, "ttc": 5.0, "sl": 4.0, "comfort": 2.0, "lk": 2.0}
    rows = [{"route": "a", "epdms": e}]
    out = ew.rescore_windows(rows, weights={"clearance": 2.0})
    assert out[0]["epdms"]["score"] == pytest.approx(18 / 20)
    assert rows[0]["epdms"]["score"] == 1.0  # input untouched
    out = ew.rescore_windows(rows, weights={"clearance": 0.0})
    assert out[0]["epdms"]["score"] == 1.0
