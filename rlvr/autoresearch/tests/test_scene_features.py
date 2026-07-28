"""Unit tests for the standalone scene_features library.

Synthetic tensors only (no model, no data files); a minimal NPZ is written to a
tmp path for the end-to-end entry point. Verifies the numeric features, the
path-relevant interaction flags, and the label taxonomy.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from rlvr.autoresearch import scene_features as sf


# --------------------------------------------------------------------------- #
# helpers to build synthetic scenes
# --------------------------------------------------------------------------- #
def _straight_ego_future(n: int = 80, step: float = 0.5) -> np.ndarray:
    """Ego drives straight along +x (ego frame): [x, y, yaw]."""
    fut = np.zeros((n, 3), np.float32)
    fut[:, 0] = np.arange(n, dtype=np.float32) * step
    return fut


def _turning_ego_future(total_yaw_deg: float, n: int = 80) -> np.ndarray:
    """Ego future with a smooth heading ramp to total_yaw_deg (signed)."""
    fut = np.zeros((n, 3), np.float32)
    yaw = np.linspace(0.0, np.radians(total_yaw_deg), n).astype(np.float32)
    fut[:, 2] = yaw
    # integrate a unit-ish speed path so x/y move (not required for yaw metric)
    dx = np.cos(yaw) * 0.5
    dy = np.sin(yaw) * 0.5
    fut[:, 0] = np.cumsum(dx)
    fut[:, 1] = np.cumsum(dy)
    return fut


def _neighbor_row(x, y, *, is_veh=0, is_ped=0, is_bike=0):
    """One neighbor feature row (11 cols): [x,y,cos,sin,vx,vy,w,l,veh,ped,bike]."""
    return [x, y, 1.0, 0.0, 0.0, 0.0, 0.6, 1.0, is_veh, is_ped, is_bike]


def _neighbors(rows, n_slots=8, t=2):
    """Pack neighbor rows into (n_slots, t, 11); unused slots stay zero (padding)."""
    nap = np.zeros((n_slots, t, 11), np.float32)
    for i, r in enumerate(rows):
        nap[i, :, :] = np.array(r, np.float32)
    return nap


def _ego_state(speed_ms: float) -> np.ndarray:
    s = np.zeros(10, np.float32)
    s[2] = 1.0  # cos heading
    s[4] = speed_ms  # vx
    return s


# --------------------------------------------------------------------------- #
# geometry reuse
# --------------------------------------------------------------------------- #
def test_min_dist_to_polyline_straight():
    poly = np.array([[0.0, 0.0], [40.0, 0.0]], np.float32)  # along +x
    pts = np.array([[5.0, 3.0], [10.0, -1.0], [20.0, 0.0]], np.float32)
    d = sf.min_dist_to_polyline(pts, poly)
    assert np.allclose(d, [3.0, 1.0, 0.0], atol=1e-4)


def test_batch_min_dist_matches_canonical_per_scene():
    """The batched GPU helper must equal the canonical per-scene reuse (the basis
    for GPU==CPU identity in the extractor)."""
    import torch

    rng = np.random.default_rng(0)
    B, Q, P = 5, 7, 80
    dists_ref = np.zeros((B, Q), np.float32)
    pts = np.zeros((B, Q, 2), np.float32)
    p1 = np.zeros((B, P - 1, 2), np.float32)
    p2 = np.zeros((B, P - 1, 2), np.float32)
    for b in range(B):
        path = np.cumsum(rng.normal(0, 1, size=(P, 2)).astype(np.float32), axis=0)
        points = rng.normal(0, 5, size=(Q, 2)).astype(np.float32)
        pts[b] = points
        p1[b], p2[b] = path[:-1], path[1:]
        dists_ref[b] = sf.min_dist_to_polyline(points, path)  # canonical per-scene
    out = sf.batch_min_dist_to_paths(
        torch.from_numpy(pts), torch.from_numpy(p1), torch.from_numpy(p2)
    ).numpy()
    assert np.allclose(out, dists_ref, atol=1e-4)


def test_min_dist_to_polyline_empty_and_degenerate():
    poly = np.array([[0.0, 0.0], [10.0, 0.0]], np.float32)
    assert sf.min_dist_to_polyline(np.zeros((0, 2), np.float32), poly).shape == (0,)
    # degenerate polyline (single vertex) -> point-to-point
    d = sf.min_dist_to_polyline(
        np.array([[3.0, 4.0]], np.float32), np.array([[0.0, 0.0]], np.float32)
    )
    assert np.isclose(d[0], 5.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# scalar features / label bins
# --------------------------------------------------------------------------- #
def test_speed_bin_boundaries():
    bins = (0.5, 3.0, 8.0)
    assert sf.speed_bin(0.0, bins) == "stopped"
    assert sf.speed_bin(2.9, bins) == "low"
    assert sf.speed_bin(5.0, bins) == "mid"
    assert sf.speed_bin(12.0, bins) == "high"


def test_maneuver_sign():
    assert sf.maneuver(5.0, 15.0) == "straight"
    assert sf.maneuver(40.0, 15.0) == "turn-L"
    assert sf.maneuver(-40.0, 15.0) == "turn-R"


def test_signed_total_yaw_matches_maneuver():
    assert sf.signed_total_yaw_deg(_turning_ego_future(60.0)) > 15.0
    assert sf.signed_total_yaw_deg(_turning_ego_future(-60.0)) < -15.0
    assert abs(sf.signed_total_yaw_deg(_straight_ego_future())) < 1.0


def _to_4col_ego(ego3: np.ndarray) -> np.ndarray:
    """[x,y,yaw] -> canonical [x,y,cos(yaw),sin(yaw)]."""
    yaw = ego3[:, 2]
    out = np.zeros((ego3.shape[0], 4), np.float32)
    out[:, 0:2] = ego3[:, 0:2]
    out[:, 2] = np.cos(yaw)
    out[:, 3] = np.sin(yaw)
    return out


def test_signed_yaw_3col_4col_equivalent():
    # the same turn must classify identically whether ego future is 3-col [x,y,yaw]
    # or canonical 4-col [x,y,cos,sin]; reading col 2 on 4-col would flip the sign.
    for deg in (60.0, -60.0, 20.0, -20.0):
        ego3 = _turning_ego_future(deg)
        y3 = sf.signed_total_yaw_deg(ego3)
        y4 = sf.signed_total_yaw_deg(_to_4col_ego(ego3))
        assert abs(y3 - y4) < 1e-3, (deg, y3, y4)
        assert sf.maneuver(y3, 15.0) == sf.maneuver(y4, 15.0)
    # a left turn stays turn-L in both widths (regression for the col-2-as-cos bug)
    ego3 = _turning_ego_future(60.0)
    assert sf.maneuver(sf.signed_total_yaw_deg(_to_4col_ego(ego3)), 15.0) == "turn-L"


def test_signed_yaw_bad_width_fails_loud():
    with pytest.raises(ValueError):
        sf.signed_total_yaw_deg(np.zeros((10, 5), np.float32))


def test_unsigned_yaw_3col_4col_equivalent():
    # peak_abs_yaw_deg (unsigned total) must decode width consistently too — the
    # legacy compute_gt_stats read col 2 as radians and gave ~28.6 for a 60° 4-col turn.
    for deg in (60.0, -60.0, 35.0):
        ego3 = _turning_ego_future(deg)
        u3 = sf.unsigned_total_yaw_deg(ego3)
        u4 = sf.unsigned_total_yaw_deg(_to_4col_ego(ego3))
        assert abs(u3 - u4) < 1e-3, (deg, u3, u4)
        assert abs(u3 - abs(deg)) < 1.0  # ~|deg| total heading change


def test_session_key_injective_on_ambiguous_dirs():
    # a plain "_".join of the last two path parts is NOT injective:
    # /A/a_b/c and /B/a/b_c would both collapse to "a_b_c". session_key must not.
    ka = sf.session_key("/A/a_b/c/run_00000000_00000000.npz")
    kb = sf.session_key("/B/a/b_c/run_00000000_00000000.npz")
    assert ka != kb, (ka, kb)
    # and contig_relpath (build side) uses the same key -> consistent + injective
    from rlvr.autoresearch.tools.build_small_dataset import contig_relpath

    assert (
        contig_relpath("/A/a_b/c/run_00000000_00000000.npz").split(os.sep)[0]
        != contig_relpath("/B/a/b_c/run_00000000_00000000.npz").split(os.sep)[0]
    )


def test_bag_and_frame_distinguishes_sessions(tmp_path):
    # identical filename under two different session dirs must NOT collide.
    # Paths are built from tmp_path (self-contained; bag_and_frame only parses
    # the string, so the files need not exist).
    a = tmp_path / "sessA" / "seg" / "run_00000000_00000131.npz"
    b = tmp_path / "sessB" / "seg" / "run_00000000_00000131.npz"
    c = tmp_path / "sessA" / "seg" / "run_00000000_00000132.npz"
    ba, fa = sf.bag_and_frame(str(a))
    bb, fb = sf.bag_and_frame(str(b))
    bc, fc = sf.bag_and_frame(str(c))
    assert ba != bb, (ba, bb)
    assert fa == fb == 131
    # same session dir + prefix -> same bag, consecutive frame
    assert bc == ba and fc == 132


def test_interaction_priority_ped_first():
    # a lead vehicle AND a relevant pedestrian both present -> has-ped wins
    assert (
        sf.interaction(n_interacting=10, has_lead=True, has_pedestrian=True, dense_count=4)
        == "has-ped"
    )
    assert (
        sf.interaction(n_interacting=1, has_lead=True, has_pedestrian=False, dense_count=4)
        == "has-lead"
    )
    assert (
        sf.interaction(n_interacting=5, has_lead=False, has_pedestrian=False, dense_count=4)
        == "dense"
    )
    assert (
        sf.interaction(n_interacting=0, has_lead=False, has_pedestrian=False, dense_count=4)
        == "none"
    )


# --------------------------------------------------------------------------- #
# path-relevant neighbor flags — the fix
# --------------------------------------------------------------------------- #
def test_pedestrian_near_path_flagged():
    ego_future = _straight_ego_future()  # along +x
    nap = _neighbors([_neighbor_row(10.0, 1.0, is_ped=1)])  # 1 m off the path
    out = sf.neighbor_stats(
        nap,
        ego_future,
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_pedestrian"] is True
    assert out["nearest_ped_path_m"] < 6.0


def test_pedestrian_far_from_path_not_flagged():
    ego_future = _straight_ego_future()
    nap = _neighbors([_neighbor_row(10.0, 25.0, is_ped=1)])  # 25 m off the path
    out = sf.neighbor_stats(
        nap,
        ego_future,
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_pedestrian"] is False
    assert out["nearest_ped_path_m"] > 6.0  # present but far -> recorded, not flagged


def test_lead_vehicle_front_cone():
    ego_future = _straight_ego_future()
    nap = _neighbors([_neighbor_row(15.0, 0.0, is_veh=1)])  # directly ahead
    out = sf.neighbor_stats(
        nap,
        ego_future,
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_lead"] is True
    assert out["nearest_ped_path_m"] is None  # no pedestrians present


def test_lead_requires_vehicle_type():
    # a pedestrian directly ahead is NOT a lead (lead is vehicle-typed) -> has-ped
    out = sf.neighbor_stats(
        _neighbors([_neighbor_row(15.0, 0.0, is_ped=1)]),
        _straight_ego_future(),
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_lead"] is False
    assert out["has_pedestrian"] is True


def test_lead_on_curved_path_caught():
    # ego turns left; a vehicle sitting on the curved path ahead is laterally OFF the
    # straight +x axis (a t=0 front cone would miss it) but ON the path corridor.
    ego = _turning_ego_future(80.0)
    midpt = ego[40, :2]
    assert abs(float(midpt[1])) > 2.0  # genuinely outside a straight |y|<=2 cone
    out = sf.neighbor_stats(
        _neighbors([_neighbor_row(float(midpt[0]), float(midpt[1]), is_veh=1)]),
        ego,
        radius_m=60.0,
        lead_range_m=60.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_lead"] is True


def test_lead_off_path_not_flagged():
    # vehicle ahead in x but far laterally from the (straight) path -> not a lead
    out = sf.neighbor_stats(
        _neighbors([_neighbor_row(15.0, 10.0, is_veh=1)]),
        _straight_ego_future(),
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_lead"] is False


def test_no_active_neighbors():
    out = sf.neighbor_stats(
        np.zeros((8, 2, 11), np.float32),
        _straight_ego_future(),
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["neighbor_count"] == 0
    assert out["n_interacting"] == 0
    assert out["has_lead"] is False
    assert out["has_pedestrian"] is False


# --------------------------------------------------------------------------- #
# end-to-end entry point
# --------------------------------------------------------------------------- #
def _write_npz(path, ego_future, nap, speed_ms=5.0):
    np.savez(
        path,
        ego_current_state=_ego_state(speed_ms),
        ego_agent_future=ego_future.astype(np.float32),
        neighbor_agents_past=nap.astype(np.float32),
        static_objects=np.zeros((5, 10), np.float32),
        goal_pose=np.array([20.0, 0.0, 0.0], np.float32),
        ego_shape=np.array([4.76, 7.24, 2.29], np.float32),
    )


def test_require_fields_presence_contract():
    # _require_fields checks field-NAME presence (the model's input contract); shapes
    # are validated separately by the canonical loader, so no shape literals here.
    from rlvr.autoresearch.tools.build_small_dataset import _REQUIRED_FIELDS, _require_fields

    full = {k: np.zeros((1,), np.float32) for k in _REQUIRED_FIELDS}  # names only, dummy values
    _require_fields(full, "ok")  # no raise
    for drop in ("ego_agent_past", "goal_pose", "neighbor_agents_future"):
        d = dict(full)
        del d[drop]
        with pytest.raises(ValueError):
            _require_fields(d, f"missing-{drop}")


def test_extract_scene_features_normalizes_path_identity(tmp_path):
    # the same source scene reached via different spellings (absolute vs a relative
    # alias) must resolve to ONE physical identity in row["npz_path"], so it can't
    # alias into both train and val (realpath dedup).
    p = tmp_path / "sub" / "scene_00000007.npz"
    p.parent.mkdir(parents=True)
    _write_npz(p, _straight_ego_future(), _neighbors([_neighbor_row(10.0, 1.0, is_ped=1)]))
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        rel = os.path.join("sub", "scene_00000007.npz")
        r_abs = sf.extract_scene_features(str(p), with_sidecar=False)
        r_rel = sf.extract_scene_features(rel, with_sidecar=False)
    finally:
        os.chdir(cwd)
    assert r_abs["npz_path"] == r_rel["npz_path"] == os.path.realpath(str(p))
    assert r_abs["bag"] == r_rel["bag"]


def test_extract_scene_features_end_to_end(tmp_path):
    p = tmp_path / "scene_00000042.npz"
    _write_npz(
        p, _turning_ego_future(50.0), _neighbors([_neighbor_row(8.0, 1.5, is_ped=1)]), speed_ms=5.0
    )
    row = sf.extract_scene_features(str(p), with_sidecar=False)
    # every declared column present
    assert set(row) == set(sf.FEATURE_COLUMNS)
    assert row["bag"].endswith("scene")  # bag id = <session>/<prefix>; prefix is "scene"
    assert row["frame"] == 42
    assert row["speed_bin"] == "mid"
    assert row["maneuver"] == "turn-L"
    assert row["interaction"] == "has-ped"
    assert row["label"] == "mid|turn-L|has-ped"
    assert row["has_pedestrian"] is True


def _axis_len(a) -> int:
    """Concrete length for one expected axis: exact int as-is, first of a one-of
    tuple, 1 for an unconstrained (None) axis."""
    if a is None:
        return 1
    if isinstance(a, tuple):
        return a[0]
    return a


def _valid_canonical_scene() -> dict:
    """A minimal scene carrying every required field at its canonical shape — the
    fixture the shared schema validator must accept. Uses the real per-axis contract
    (NOT None->1), so it can't mask a too-loose axis."""
    exp = sf._canonical_expected_shapes()
    return {k: np.zeros(tuple(_axis_len(a) for a in shp), np.float32) for k, shp in exp.items()}


def test_validate_canonical_scene_accepts_valid():
    sf.validate_canonical_scene(_valid_canonical_scene(), ctx="valid")


def test_validate_canonical_scene_required_fields_match_builder():
    # single source of truth: the builder's required list IS the validator's keys
    from rlvr.autoresearch.tools.build_small_dataset import _REQUIRED_FIELDS

    assert set(_REQUIRED_FIELDS) == set(sf.canonical_required_fields())


def test_validate_canonical_scene_rejects_scalar_fields():
    # a scalar in ANY required field (the encoder later slices it) must fail loud
    for field in ("lanes", "static_objects", "turn_indicators", "ego_current_state", "goal_pose"):
        d = _valid_canonical_scene()
        d[field] = np.float32(0.0)
        with pytest.raises(ValueError, match=field):
            sf.validate_canonical_scene(d, ctx="scalar")


def test_validate_canonical_scene_rejects_wrong_fixed_axis():
    d = _valid_canonical_scene()
    d["neighbor_agents_past"] = np.zeros((100, 31, 11), np.float32)  # slots != 320
    with pytest.raises(ValueError, match="neighbor_agents_past"):
        sf.validate_canonical_scene(d, ctx="slots")
    d = _valid_canonical_scene()
    d["ego_agent_future"] = np.zeros((80, 2), np.float32)  # width != 3
    with pytest.raises(ValueError, match="ego_agent_future"):
        sf.validate_canonical_scene(d, ctx="width")


def test_validate_canonical_scene_rejects_missing_field():
    d = _valid_canonical_scene()
    del d["lanes"]
    with pytest.raises(ValueError, match="missing required"):
        sf.validate_canonical_scene(d, ctx="missing")


def test_validate_canonical_scene_accepts_both_ego_pose_widths():
    # repo supports raw 3-col AND converted 4-col ego/goal poses (heading_to_cos_sin
    # is idempotent for 4-col) — both must validate.
    for w in (3, 4):
        d = _valid_canonical_scene()
        d["ego_agent_future"] = np.zeros((80, w), np.float32)
        d["ego_agent_past"] = np.zeros((d["ego_agent_past"].shape[0], w), np.float32)
        d["goal_pose"] = np.zeros((w,), np.float32)
        sf.validate_canonical_scene(d, ctx=f"ego-width-{w}")
    # a width the model can't consume still fails
    d = _valid_canonical_scene()
    d["ego_agent_future"] = np.zeros((80, 5), np.float32)
    with pytest.raises(ValueError, match="ego_agent_future"):
        sf.validate_canonical_scene(d, ctx="ego-width-5")


def test_validate_canonical_scene_map_point_widths_exact():
    # polygon/line-string trailing widths and turn-indicator length are fixed by the
    # encoder contract — a wrong width must NOT slip through as "variable".
    for field, bad in (
        ("polygons", (10, 40, 1)),
        ("line_strings", (60, 20, 1)),
        ("turn_indicators", (1,)),
    ):
        d = _valid_canonical_scene()
        d[field] = np.zeros(bad, np.float32)
        with pytest.raises(ValueError, match=field):
            sf.validate_canonical_scene(d, ctx=f"bad-{field}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
