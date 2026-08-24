"""Synthetic straight-road checks for the EPDMS-style window scorer."""

import math

import numpy as np
import pytest

from scenario_generation import window_metrics as wm

EGO_SHAPE = np.array([2.8, 4.8, 1.9])  # wheelbase, length, width
N = 400  # recorded frames, 1 m apart along +x (10 m/s at 10 Hz)
LIMIT = 12.0


def _route_lane(lat_shift=0.0, tl_state=4, x0=-40.0, x1=120.0):
    """One straight route lane in the ego frame: centreline y=lat_shift, half width 1.75 m."""
    lane = np.zeros((20, 33), dtype=np.float32)
    xs = np.linspace(x0, x1, 20)
    lane[:, 0] = xs
    lane[:, 1] = lat_shift
    lane[:, 2] = xs[1] - xs[0]
    lane[:, 4:6] = [0.0, 1.75]
    lane[:, 6:8] = [0.0, -1.75]
    lane[:, 8 + tl_state] = 1.0
    return lane


class FakeTL:
    """Recorded ego drives +x at 10 m/s; frames carry a straight route lane."""

    def __init__(self, *, lane_shift_frames=(), stop_frames=(), agents=None, limit=LIMIT):
        self.poses = np.column_stack([np.arange(N, dtype=float), np.zeros(N), np.zeros(N)])
        self.speeds = np.full(N, 10.0)
        for i in stop_frames:
            self.speeds[i] = 0.0
        self.frame_indices = np.arange(N)
        self.lane_shift_frames = set(lane_shift_frames)
        self.agents = agents or {}
        self.limit = limit

    def __len__(self):
        return N

    def nearest(self, xy):
        return int(np.clip(round(float(xy[0])), 0, N - 1))

    def npz(self, idx):
        # In frame idx the recorded ego is at x=idx; a "lane shift" frame puts the
        # centreline 1.2 m to the left of the recorded ego (the human is off-centre).
        shift = -1.2 if idx in self.lane_shift_frames else 0.0
        route = np.zeros((25, 20, 33), dtype=np.float32)
        route[0] = _route_lane(lat_shift=shift)
        nb = np.zeros((320, 31, 11), dtype=np.float32)
        for k, (dx, dy, vx) in enumerate(self.agents.get(idx, [])):
            nb[k, -1, :8] = [dx, dy, 1.0, 0.0, vx, 0.0, 1.8, 4.5]
        return {
            "route_lanes": route,
            "route_lanes_speed_limit": np.array([[self.limit]] + [[0.0]] * 24, dtype=np.float32),
            "route_lanes_has_speed_limit": np.array([[True]] + [[False]] * 24),
            "line_strings": np.zeros((60, 20, 4), dtype=np.float32),
            "polygons": np.zeros((10, 40, 3), dtype=np.float32),
            "neighbor_agents_past": nb,
            "ego_shape": EGO_SHAPE.astype(np.float32),
        }


def _rows(lo, hi, *, x=None, y=0.0, speed=10.0, extra=None):
    rows = []
    for k in range(hi - lo):
        xk = float(lo + k) if x is None else float(x[k])
        row = {
            "k": k,
            "ego": [xk, float(y)],
            "yaw": 0.0,
            "speed": float(speed),
            "clearance_m": 20.0,
            "collision": False,
            "rb_dist_m": 3.0,
            "red_light_violation": False,
            "gt_deviation_m": 0.0,
            "collision_types": [],
            "at_fault": False,
        }
        if extra:
            row.update(extra(k))
        rows.append(row)
    return rows


LO, HI = 100, 250


def test_perfect_tracking_scores_one_except_speed_limit_term():
    tl = FakeTL()
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI), EGO_SHAPE, wm.ScoreConfig())
    t = out["terms"]
    for k in ("nc", "dac", "ddc", "tlc", "mp", "ep", "ttc", "comfort", "lk"):
        assert t[k] == 1.0, k
    assert t["sl"] == pytest.approx(1.0)  # 10 m/s under a 12 m/s limit
    assert out["score"] == pytest.approx(1.0)
    assert out["valid_ticks"] == HI - LO and not out["invalid"]
    assert out["detail"]["route_adherence_frac"] == 1.0
    assert out["detail"]["tl_measured_frac"] == 0.0


def test_speed_limit_term_follows_nuplan_overspeed():
    tl = FakeTL(limit=8.0)  # ego at 10 m/s -> 2 m/s over
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI), EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["sl"] == pytest.approx(1.0 - 2.0 / wm.SPEED_LIMIT_MAX_OVERSPEED_MPS)
    assert out["detail"]["overspeed_max_mps"] == pytest.approx(2.0)


def test_lane_keeping_fails_when_off_centre_and_human_was_centred():
    tl = FakeTL()
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, y=1.0), EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["lk"] == 0.0
    assert out["terms"]["ddc"] == 1.0  # still inside the 3.5 m wide lane


def test_lane_keeping_exempt_in_recorded_avoidance_zone_with_margins():
    # Human was off-centre on frames 160..180 (x=160..180); the ego is off-centre on x in
    # [140, 190] -- earlier and wider than the human -- and is still exempt thanks to the
    # 15 m back / 5 m forward zone.  Beyond the zone the violation counts.
    tl = FakeTL(lane_shift_frames=range(160, 181))
    y = np.zeros(HI - LO)
    for k in range(HI - LO):
        if 145 <= LO + k <= 185:
            y[k] = 1.0
    rows = _rows(LO, HI, y=0.0)
    for k, row in enumerate(rows):
        row["ego"][1] = float(y[k])
    out = wm.score_window_epdms(tl, LO, HI, rows, EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["lk"] == 1.0
    assert 0.0 < out["detail"]["lk_exempt_recorded_off_frac"] < 1.0
    # A 3 s violation 40 m after the zone is not covered.
    for k, row in enumerate(rows):
        row["ego"][1] = 1.0 if 225 <= LO + k <= 255 else float(y[k])
    out = wm.score_window_epdms(tl, LO, HI, rows, EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["lk"] == 0.0


def test_lane_keeping_exempt_when_stopped_obstacle_ahead_on_route():
    agents = {idx: [(15.0, 0.0, 0.0)] for idx in range(LO, HI)}  # parked car 15 m ahead
    tl = FakeTL(agents=agents)
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, y=1.0), EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["lk"] == 1.0
    assert out["detail"]["lk_exempt_obstacle_frac"] == 1.0


def test_ddc_penalises_driving_outside_route_lanes_unless_human_did():
    tl = FakeTL()
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, y=5.0), EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["ddc"] == 0.0 and out["detail"]["route_adherence_frac"] == 0.0
    # The human also left the route lanes (bus bay) on the same stretch -> exempt.
    tl = FakeTL(lane_shift_frames=range(LO - 20, HI + 20))
    tl.npz = _shifted_far(tl)
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, y=5.0), EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["ddc"] == 1.0
    assert out["detail"]["ddc_exempt_recorded_off_route_frac"] == 1.0


def _shifted_far(tl):
    base = tl.npz

    def npz(idx):
        d = base(idx)
        if idx in tl.lane_shift_frames:
            d["route_lanes"][0] = _route_lane(lat_shift=-5.0)  # human 5 m off the lane
        return d

    return npz


def test_recorded_operational_stop_is_tagged():
    stop = range(150, 200)  # human stopped 5 s in a bay 5 m off the lane
    tl = FakeTL(lane_shift_frames=stop, stop_frames=stop)
    tl.npz = _shifted_far(tl)
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI), EGO_SHAPE, wm.ScoreConfig())
    assert out["recorded_stop"] is True
    tl = FakeTL()
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI), EGO_SHAPE, wm.ScoreConfig())
    assert out["recorded_stop"] is False


def test_ghost_contact_truncates_and_short_span_is_invalid():
    def extra(k):
        return {"collision_types": [3], "at_fault": False} if k == 20 else {}

    tl = FakeTL()
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, extra=extra), EGO_SHAPE, wm.ScoreConfig())
    assert out["valid_ticks"] == 20 and out["valid_span"]["reason"] == "ghost_contact"
    assert out["invalid"] is True
    cfg = wm.ScoreConfig(truncate_on_ghost_contact=False)
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, extra=extra), EGO_SHAPE, cfg)
    assert out["valid_ticks"] == HI - LO and out["terms"]["nc"] == 1.0


def test_at_fault_and_red_light_zero_multiplicative_terms():
    def extra(k):
        return (
            {"collision_types": [2], "at_fault": True, "red_light_violation": True}
            if k == 60
            else {}
        )

    tl = FakeTL()
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, extra=extra), EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["nc"] == 0.0 and out["terms"]["tlc"] == 0.0
    assert out["score"] == 0.0 and out["weighted"] > 0.0


def test_not_making_progress_and_low_recorded_progress():
    tl = FakeTL()
    x = np.full(HI - LO, float(LO))  # ego parked at the window start
    out = wm.score_window_epdms(
        tl, LO, HI, _rows(LO, HI, x=x, speed=0.0), EGO_SHAPE, wm.ScoreConfig()
    )
    assert out["terms"]["ep"] == 0.0 and out["terms"]["mp"] == 0.0
    # Human barely moved (red light wait): no progress demanded.
    tl.poses[LO:HI, 0] = float(LO)
    tl.speeds[LO:HI] = 0.0
    out = wm.score_window_epdms(
        tl, LO, HI, _rows(LO, HI, x=x, speed=0.0), EGO_SHAPE, wm.ScoreConfig()
    )
    assert out["terms"]["ep"] == 1.0 and out["terms"]["mp"] == 1.0
    assert out["detail"]["low_recorded_progress"] is True


def test_ttc_infraction_with_stopped_agent_ahead():
    agents = {idx: [(6.0, 0.0, 0.0)] for idx in range(LO, HI)}  # stopped car 6 m ahead
    tl = FakeTL(agents=agents)
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI), EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["ttc"] == 0.0


def test_comfort_fails_on_hard_braking_and_clearance_term_is_optional():
    tl = FakeTL()
    x = np.array([LO + 10 * k * 0.1 if k < 50 else LO + 50.0 for k in range(HI - LO)])
    rows = _rows(LO, HI, x=x, speed=10.0)
    for k, r in enumerate(rows):
        r["speed"] = 10.0 if k < 50 else 0.0
        r["clearance_m"] = 0.2
    out = wm.score_window_epdms(tl, LO, HI, rows, EGO_SHAPE, wm.ScoreConfig())
    assert out["terms"]["comfort"] == 0.0 and out["detail"]["comfort_failed"]
    assert out["terms"]["clearance"] == pytest.approx(0.4)
    assert "clearance" not in out["weights"]
    out2 = wm.score_window_epdms(tl, LO, HI, rows, EGO_SHAPE, wm.ScoreConfig(clearance_weight=2.0))
    assert out2["weights"]["clearance"] == 2.0 and out2["weighted"] < out["weighted"]


def test_map_terms_follow_a_lagging_ego():
    # Route lanes in a frame span x-40..x+120 around the recorded ego; an ego 60 m behind
    # the recorded one would fall off the truth frame's map but not off the nearest frame's.
    tl = FakeTL()
    x = np.arange(LO, HI, dtype=float) - 60.0
    out = wm.score_window_epdms(tl, LO, HI, _rows(LO, HI, x=x), EGO_SHAPE, wm.ScoreConfig())
    assert out["detail"]["route_adherence_frac"] == 1.0
    assert out["detail"]["map_frame_follows_ego_frac"] == 1.0
    assert (
        out["terms"]["ep"] == pytest.approx(1.0 - 60.0 / 149.0, abs=1e-2)
        or out["terms"]["ep"] < 1.0
    )
    assert math.isclose(out["detail"]["lon_max_behind_m"], 60.0, abs_tol=1e-6)
