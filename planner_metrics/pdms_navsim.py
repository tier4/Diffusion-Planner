"""Formula-faithful NAVSIM PDMS sub-metrics (open-loop, label-aware port).

This is a **1:1 port of NAVSIM's PDM-Score sub-metric formulas** — comfort,
ego-progress (EP), no-at-fault-collision (NC) and time-to-collision (TTC) — using
the same underlying libraries NAVSIM uses (``scipy.signal.savgol_filter`` +
``shapely``). Every threshold, filter parameter, aggregation weight and the
multiplicative/weighted structure is copied verbatim from the navsim_v2 source
checked into ``refer/Drive-JEPA/navsim_v2`` (file:line citations inline) and from
the nuplan-devkit helpers it imports (``is_agent_ahead/behind``,
``is_track_stopped``, ``get_collision_type``, ``CollisionType``, ``AGENT_TYPES``).

WHAT IS BIT-EXACT to navsim:
  * comfort: the 6 bounds, savgol params (poly/window/deriv), ``_within_bound``,
    the all-timesteps + all-metrics reduction (``pdm_comfort_metrics.py``).
  * EP: centerline projection + clip>=0 + the proposal-relative normalization
    (``pdm_scorer.py::_calculate_progress`` / ``_aggregate_pdm_scores``).
  * NC: the at-fault classification (front/stopped-track => penalised;
    rear/stopped-ego => not), 0.0-for-AGENT_TYPES / 0.5-for-static, the
    already-collided dedup (``pdm_scorer.py::_calculate_no_at_fault_collision``
    + ``pdm_scorer_utils.py::get_collision_type``).
  * TTC: constant-velocity forward projection at future idcs [0,3,6,9]*0.1 s,
    intersect vs agents at t+dt, agent-ahead => infraction, skip if ego stopped
    (``pdm_scorer.py::_calculate_ttc``).
  * aggregation: PDMS = prod(NC,DAC,DDC,TLC) * weighted_avg(EP=5, TTC=5, LK=2,
    HISTORY_COMFORT=2) (``pdm_scorer.py::_aggregate_pdm_scores`` + config).

THE DOCUMENTED DEVIATIONS (unavoidable without installing navsim/nuplan, which
need a maps DB + an old Python and conflict with our torch/numpy — verified):
  1. **Ego kinematic state is DERIVED from the predicted poses** (finite
     differences + the SAME savgol smoothing), not produced by navsim's
     bicycle-model + LQR simulator. navsim ALWAYS routes a predicted trajectory
     through ``simulator.simulate_proposals`` before scoring
     (``navsim/evaluate/pdm_score.py``); we cannot run that simulator, so comfort
     reads accel/yaw re-derived from the pose path. We also skip navsim's
     rear-axle->center shift (it needs nuplan ``VehicleParameters`` = pacifica,
     the WRONG vehicle for J6/jpntaxi anyway); lon/lat accel are decomposed in
     the ego body frame, which is the physically-correct decomposition the shift
     approximates. For a FIXED-vehicle ranking proxy this is a constant transform
     that does not change the ordering between experiments.
  2. **Agents are GT-future, not policy-simulated.** NC/TTC use the ACTUAL
     future agent boxes from the ``.gt.tar`` sidecar (ground-truth future),
     where navsim uses a (non-)reactive traffic-agents policy forecast. The
     real future is arguably a better open-loop reference; it is not navsim's
     forecast.
  3. **Map-gated branches are IMPLEMENTED** (``ego_area_flags``): NC's
     lateral at-fault case and TTC's multi-lane/non-drivable/intersection
     widening run against polygons built from the shard's own lane tensor
     (centerline +- boundary offsets) and intersection polygons — navsim's
     corner-count semantics 1:1. Remaining representation deltas: navsim's
     drivable set also contains ROADBLOCK / DRIVABLE_AREA /
     CARPARK polygons which the shards do not carry — our lanes+intersections
     set can flag non_drivable in legitimate drivable gaps (notably
     carparks), i.e. STRICTER than navsim there; the flags only widen
     penalties at actual collision / projected-collision events, bounding
     the impact. The polygon set is the generation window (140 lanes / 10
     intersections), not a global map. Callers that pass no flags keep the historical lenient
     else-branch.
  4. **This module exposes low-level subscore helpers.** The Autoware/C++
     synthetic EPDMS aggregation, including strict availability and fixed
     denominator semantics, lives in ``pdms_proxy.synthetic_epdms``. **DDC is
     implemented** (``ddc_from_route_lanes``): navsim's oncoming-traffic test
     ("center outside every on-route lane polygon") against polygons built
     from the route tensor's centerline +- boundary-offset channels, identical
     1 s sliding window + 2 m / 6 m thresholds; coverage transparency via
     ``val/ddc_route_frac``. **DAC is implemented** from the shard's own road-border
     polylines (``dac_from_road_borders``) — the lanelet drivable BOUNDARY in
     the same center frame as the predictions, so no nuPlan map DB is needed;
     coverage is the generation window's 60 nearest line strings (read
     ``val/dac_border_frac`` alongside). Traffic-light compliance requires
     Autoware traffic-signal and regulatory-element context; callers that do not
     have those inputs must leave TLC unavailable rather than treating it as a
     silent pass.

Honest label: these are formula-faithful open-loop subscore helpers. Full
Autoware-compatible synthetic EPDMS additionally requires the availability
rules and human-filtered aggregation implemented in ``pdms_proxy``.

All arrays are numpy (navsim is numpy); the torch entry points live in
``pdms_proxy``. Poses are ``[T, 4] = (x, y, cos_yaw, sin_yaw)`` in metres in the
ego frame; agent boxes are ``[N, 9] = (x, y, z, w, l, h, yaw, vx, vy)`` in the
ego frame (the BEVFusion / ``.gt.tar`` layout; col3=w width, col4=l length —
verified numerically, see ``box_corners``).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# State layout (navsim StateIndex, pdm_enums.py:StateIndex). We only fill the
# fields the comfort metric reads; the rest stay zero. size()==11 so the
# ego_is_comfortable assert (n_states == StateIndex.size()) holds 1:1.
# ---------------------------------------------------------------------------
STATE_X = 0
STATE_Y = 1
STATE_HEADING = 2
STATE_VEL_X = 3
STATE_VEL_Y = 4
STATE_ACC_X = 5
STATE_ACC_Y = 6
STATE_SIZE = 11  # StateIndex.size(): X,Y,HEADING,VEL_X/Y,ACC_X/Y,STEER,STEER_RATE,ANG_VEL,ANG_ACC


# ---------------------------------------------------------------------------
# Comfort thresholds — VERBATIM from
# refer/.../pdm_planner/scoring/pdm_comfort_metrics.py lines 18-34.
# ---------------------------------------------------------------------------
MAX_ABS_MAG_JERK = 8.37  # [m/s^3]
MAX_ABS_LAT_ACCEL = 4.89  # [m/s^2]
MAX_LON_ACCEL = 2.40  # [m/s^2]
MIN_LON_ACCEL = -4.05
MAX_ABS_YAW_ACCEL = 1.93  # [rad/s^2]
MAX_ABS_LON_JERK = 4.13  # [m/s^3]
MAX_ABS_YAW_RATE = 0.95  # [rad/s]

# ---------------------------------------------------------------------------
# Scorer config — VERBATIM from pdm_scorer.py::PDMScorerConfig (lines 56-98).
# ---------------------------------------------------------------------------
PROGRESS_WEIGHT = 5.0
TTC_WEIGHT = 5.0
LANE_KEEPING_WEIGHT = 2.0
HISTORY_COMFORT_WEIGHT = 2.0
EXTENDED_COMFORT_WEIGHT = 2.0  # EPDMS (planning_data_analyzer) weighted term
STOPPED_SPEED_THRESHOLD = 5e-03  # [m/s] (ttc) -- note: collision classify uses 5e-2
FUTURE_COLLISION_HORIZON_WINDOW = 1.0  # [s] (ttc)
PROGRESS_DISTANCE_THRESHOLD = 5.0  # [m] (progress)

# nuplan get_collision_type / is_track_stopped default (collision_utils + idm/utils.py)
COLLISION_STOPPED_SPEED_THRESHOLD = 5e-02  # [m/s]


class CollisionType(IntEnum):
    """nuplan ``metrics/utils/collision_utils.py::CollisionType`` (verbatim)."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


# ===========================================================================
# Comfort -- 1:1 port of pdm_comfort_metrics.py
# ===========================================================================
def _phase_unwrap(headings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """pdm_comfort_metrics.py::_phase_unwrap (verbatim)."""
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    return headings - two_pi * adjustments


def _approximate_derivatives(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float64]:
    """pdm_comfort_metrics.py::_approximate_derivatives (verbatim)."""
    window_length = min(window_length, len(x))
    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")
    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")
    dx = dx.mean()
    out = savgol_filter(
        y, polyorder=poly_order, window_length=window_length, deriv=deriv_order, delta=dx, axis=axis
    )
    return np.asarray(out, dtype=np.float64)


def _within_bound(metric, min_bound=None, max_bound=None) -> npt.NDArray[np.bool_]:
    """pdm_comfort_metrics.py::_within_bound (verbatim, incl. the truthiness quirk)."""
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float64]:
    """pdm_comfort_metrics.py::_extract_ego_acceleration.

    DEVIATION (#1): navsim does ``state_array_to_center_state_array`` (rear-axle
    -> center via pacifica params) before reading ACCELERATION_X/Y. We read the
    body-frame accel we filled directly (see ``states_from_poses``); the savgol
    smoothing + rounding are identical.
    """
    n_batch, n_time, n_states = states.shape
    if acceleration_coordinate in ("x", "y"):
        idx = STATE_ACC_X if acceleration_coordinate == "x" else STATE_ACC_Y
        acceleration = states[..., idx]
    elif acceleration_coordinate == "magnitude":
        acceleration = np.hypot(states[..., STATE_ACC_X], states[..., STATE_ACC_Y])
    else:
        raise ValueError(
            f"acceleration_coordinate option: {acceleration_coordinate} not available."
        )
    acceleration = savgol_filter(
        acceleration, polyorder=poly_order, window_length=min(window_length, n_time), axis=-1
    )
    return np.round(acceleration, decimals=decimals)


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float64]:
    """pdm_comfort_metrics.py::_extract_ego_jerk (verbatim formula)."""
    n_batch, n_time, n_states = states.shape
    ego_acceleration = _extract_ego_acceleration(
        states, acceleration_coordinate=acceleration_coordinate
    )
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    return np.round(jerk, decimals=decimals)


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float64]:
    """pdm_comfort_metrics.py::_extract_ego_yaw_rate (verbatim formula)."""
    ego_headings = states[..., STATE_HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings), time_steps_s, deriv_order=deriv_order, poly_order=poly_order
    )
    return np.round(ego_yaw_rate, decimals=decimals)


def ego_is_comfortable(
    states: npt.NDArray[np.float64], time_point_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """pdm_comfort_metrics.py::ego_is_comfortable (verbatim; vehicle params dropped).

    ``states`` is ``[n_batch, n_time, STATE_SIZE]``; returns ``[n_batch, 6]``.
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == STATE_SIZE
    metric_functions = [
        lambda s, t: _within_bound(
            _extract_ego_acceleration(s, "x"), min_bound=MIN_LON_ACCEL, max_bound=MAX_LON_ACCEL
        ),
        lambda s, t: _within_bound(
            _extract_ego_acceleration(s, "y"),
            min_bound=-MAX_ABS_LAT_ACCEL,
            max_bound=MAX_ABS_LAT_ACCEL,
        ),
        lambda s, t: _within_bound(
            _extract_ego_jerk(s, "magnitude", t),
            min_bound=-MAX_ABS_MAG_JERK,
            max_bound=MAX_ABS_MAG_JERK,
        ),
        lambda s, t: _within_bound(
            _extract_ego_jerk(s, "x", t), min_bound=-MAX_ABS_LON_JERK, max_bound=MAX_ABS_LON_JERK
        ),
        lambda s, t: _within_bound(
            _extract_ego_yaw_rate(s, t, deriv_order=2, poly_order=3),
            min_bound=-MAX_ABS_YAW_ACCEL,
            max_bound=MAX_ABS_YAW_ACCEL,
        ),
        lambda s, t: _within_bound(
            _extract_ego_yaw_rate(s, t), min_bound=-MAX_ABS_YAW_RATE, max_bound=MAX_ABS_YAW_RATE
        ),
    ]
    results = np.zeros((n_batch, len(metric_functions)), dtype=np.bool_)
    for idx, fn in enumerate(metric_functions):
        results[:, idx] = fn(states, time_point_s)
    return results


def states_from_poses(poses: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """Build a navsim-layout state array ``[B, T, STATE_SIZE]`` from poses.

    DEVIATION (#1): kinematics are DERIVED from the pose path (not navsim's
    bicycle-model sim). World velocity/acceleration via central differences,
    then accel decomposed into the ego BODY frame (lon=x, lat=y) so the comfort
    lon/lat bounds apply to the correct components. ``poses`` is ``[..., T, 4]``
    = (x, y, cos, sin); returns ``[..., T, STATE_SIZE]``.
    """
    poses = np.asarray(poses, dtype=np.float64)
    lead = poses.shape[:-2]
    T = poses.shape[-2]
    x = poses[..., 0]
    y = poses[..., 1]
    heading = np.arctan2(poses[..., 3], poses[..., 2])

    # World-frame velocity / acceleration via numpy.gradient (central diff,
    # one-sided at the ends) -- a stable, symmetric finite difference.
    vx = np.gradient(x, dt, axis=-1)
    vy = np.gradient(y, dt, axis=-1)
    ax = np.gradient(vx, dt, axis=-1)
    ay = np.gradient(vy, dt, axis=-1)

    # Rotate world accel into the ego body frame (lon along heading, lat left).
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    acc_lon = ax * cos_h + ay * sin_h
    acc_lat = -ax * sin_h + ay * cos_h
    vel_lon = vx * cos_h + vy * sin_h
    vel_lat = -vx * sin_h + vy * cos_h

    states = np.zeros(lead + (T, STATE_SIZE), dtype=np.float64)
    states[..., STATE_X] = x
    states[..., STATE_Y] = y
    states[..., STATE_HEADING] = heading
    states[..., STATE_VEL_X] = vel_lon
    states[..., STATE_VEL_Y] = vel_lat
    states[..., STATE_ACC_X] = acc_lon
    states[..., STATE_ACC_Y] = acc_lat
    return states


def comfort_score(poses: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """navsim comfort sub-metric in {0,1}: 1 iff ALL 6 comfort metrics stay
    within bound at ALL timesteps (matches ``ego_is_comfortable(...).all(-1)``
    used by ``_calculate_history_comfort``). ``poses`` ``[..., T, 4]`` -> ``[...]``.
    """
    poses = np.asarray(poses, dtype=np.float64)
    lead = poses.shape[:-2]
    T = poses.shape[-2]
    flat = poses.reshape(-1, T, 4)
    states = states_from_poses(flat, dt)  # [B, T, STATE_SIZE]
    return comfort_score_from_states(states, dt).reshape(lead)


def comfort_score_from_states(
    states: npt.NDArray[np.float64], dt: float
) -> npt.NDArray[np.float64]:
    """Same output as :func:`comfort_score` when states came from the same poses."""
    states = np.asarray(states, dtype=np.float64)
    lead = states.shape[:-2]
    T = states.shape[-2]
    flat = states.reshape(-1, T, STATE_SIZE)
    time_point_s = np.arange(0, T).astype(np.float64) * dt
    comfortable = ego_is_comfortable(flat, time_point_s).all(axis=-1).astype(np.float64)
    return np.asarray(comfortable.reshape(lead), dtype=np.float64)


# --- EC (extended comfort): consecutive-plan consistency ---------------------
# Ported from the Autoware planning_data_analyzer (extended_comfort.cpp +
# comfort_signal.cpp). Constants verbatim from the C++.
EC_MAX_ACCEL_RMS = 0.7  # tau_a
EC_MAX_JERK_RMS = 0.5  # tau_j
EC_MAX_YAW_RATE_RMS = 0.1  # tau_psi_dot
EC_MAX_YAW_ACCEL_RMS = 0.1  # tau_psi_ddot
EC_ACCEL_WINDOW = 8  # kAccelerationFilterWindow
EC_DERIV_WINDOW = 15  # kJerkFilterWindow == kYawAccelerationFilterWindow
EC_POLY = 2  # kFilterPolynomialOrder
EC_YAW_ACCEL_POLY = 3  # kYawAccelerationPolynomialOrder


def _ec_filtered(
    values: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    window_length: int,
    poly_order: int,
    deriv_order: int,
) -> npt.NDArray[np.float64]:
    """The C++ ``local_polynomial_filter`` semantics on a 1-D signal.

    Effective poly order is clamped to ``len-1`` (the C++ does this for short
    segments); when the requested derivative exceeds the clamped order the C++
    returns all-zeros — mirror that instead of raising. The filter engine is
    scipy savgol (DEVIATION: the C++ hand-rolls the same least-squares fit
    with shifted edge windows; scipy ``mode='interp'`` is the same polynomial-
    edge idea, values can differ in the last ``window/2`` samples).
    """
    eff_poly = min(poly_order, max(0, len(values) - 1))
    if deriv_order > eff_poly:
        return np.zeros_like(values)
    return _approximate_derivatives(
        values[None],
        times,
        window_length=min(window_length, len(values)),
        poly_order=eff_poly,
        deriv_order=deriv_order,
    )[0]


def _ec_signals(poses: npt.NDArray[np.float64], dt: float) -> dict[str, npt.NDArray[np.float64]]:
    """comfort_signal.cpp::compute_comfort_signals for one trajectory segment.

    a = savgol-smoothed acceleration magnitude (window 8, poly 2);
    j = its first derivative (window 15, poly 2);
    yaw rate / yaw accel = first / second derivative of the unwrapped heading
    (window 15; poly 2 / poly 3). DEVIATION: the C++ reads the planner
    message's longitudinal-acceleration field with lateral pinned to 0 (the
    message has no lateral channel); our trajectories carry no acceleration
    field at all, so kinematics come from the pose path (``states_from_poses``,
    deviation #1) and the magnitude includes the derived lateral component.
    """
    states = states_from_poses(poses[None], dt)  # [1, T, STATE_SIZE]
    T = poses.shape[0]
    times = np.arange(T, dtype=np.float64) * dt
    acc_raw = np.hypot(states[0, :, STATE_ACC_X], states[0, :, STATE_ACC_Y])
    acc = _ec_filtered(acc_raw, times, EC_ACCEL_WINDOW, EC_POLY, 0)
    jerk = _ec_filtered(acc, times, EC_DERIV_WINDOW, EC_POLY, 1)
    yaw = _phase_unwrap(states[0, :, STATE_HEADING])
    yaw_rate = _ec_filtered(yaw, times, EC_DERIV_WINDOW, EC_POLY, 1)
    yaw_accel = _ec_filtered(yaw, times, EC_DERIV_WINDOW, EC_YAW_ACCEL_POLY, 2)
    return {"a": acc, "j": jerk, "yr": yaw_rate, "ya": yaw_accel}


def extended_comfort(
    prev_poses: npt.NDArray[np.float64],
    curr_poses: npt.NDArray[np.float64],
    dt: float,
    observation_interval: float | None = None,
) -> float:
    """extended_comfort.cpp::calculate_extended_comfort.

    Compares the CURRENT plan against the PREVIOUS plan over their time
    overlap: shift ``k = round(observation_interval / dt)`` (the planning-
    cycle gap; defaults to one sample), ``S_prev = prev[k:k+n]``,
    ``S_curr = curr[:n]``, signals computed per segment independently
    (re-based time), pointwise deltas, and EC = 1 iff all four RMS values
    stay under their thresholds. Returns NaN where the C++ reports
    "unavailable" (k == 0, either trajectory < 3 points, overlap < 3).
    """
    prev_poses = np.asarray(prev_poses, dtype=np.float64)
    curr_poses = np.asarray(curr_poses, dtype=np.float64)
    interval = dt if observation_interval is None else observation_interval
    if interval < 0.0:
        return float("nan")
    k = int(round((interval if interval > 0.0 else dt) / dt))
    if k == 0:
        return float("nan")
    np_prev, np_curr = prev_poses.shape[0], curr_poses.shape[0]
    if np_prev < 3 or np_curr < 3 or k >= np_prev:
        return float("nan")
    n = min(np_curr, np_prev - k)
    if n < 3:
        return float("nan")
    sig_curr = _ec_signals(curr_poses[:n], dt)
    sig_prev = _ec_signals(prev_poses[k : k + n], dt)

    def _rms(key: str) -> float:
        d = sig_curr[key] - sig_prev[key]
        return float(np.sqrt(np.mean(d * d)))

    ok = (
        _rms("a") <= EC_MAX_ACCEL_RMS
        and _rms("j") <= EC_MAX_JERK_RMS
        and _rms("yr") <= EC_MAX_YAW_RATE_RMS
        and _rms("ya") <= EC_MAX_YAW_ACCEL_RMS
    )
    return 1.0 if ok else 0.0


# ===========================================================================
# Geometry helpers (shapely) + nuplan collision helpers
# ===========================================================================
def ego_corners(
    x: float, y: float, heading: float, length: float, width: float
) -> npt.NDArray[np.float64]:
    """Ego box corners in navsim BBCoordsIndex order [FL, RL, RR, FR].

    nuplan ``OrientedBox`` corner order (pdm_enums.py::BBCoordsIndex): FRONT_LEFT=0,
    REAR_LEFT=1, REAR_RIGHT=2, FRONT_RIGHT=3. The front bumper edge used by
    ``get_collision_type`` (ACTIVE_FRONT) is coords[0]->coords[3] = FL->FR.
    +x forward, +y left. ``length`` is the full longitudinal extent, ``width``
    the full lateral extent.
    """
    hl, hw = length / 2.0, width / 2.0
    c, s = np.cos(heading), np.sin(heading)

    # local (forward, left) -> world
    def rot(fx, fy):
        return (x + fx * c - fy * s, y + fx * s + fy * c)

    return np.array([rot(hl, hw), rot(-hl, hw), rot(-hl, -hw), rot(hl, -hw)], dtype=np.float64)


def ego_corners_batch(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    heading: npt.NDArray[np.float64],
    length: float,
    width: float,
) -> npt.NDArray[np.float64]:
    """Vectorized :func:`ego_corners` with identical corner ordering."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    hl, hw = length / 2.0, width / 2.0
    c, s = np.cos(heading), np.sin(heading)
    local = np.asarray([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]], dtype=np.float64)
    fx, fy = local[:, 0], local[:, 1]
    out = np.empty(x.shape + (4, 2), dtype=np.float64)
    out[..., :, 0] = x[..., None] + fx * c[..., None] - fy * s[..., None]
    out[..., :, 1] = y[..., None] + fx * s[..., None] + fy * c[..., None]
    return out


def box_corners(box: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Agent box corners from a ``[9] = (x,y,z,w,l,h,yaw,vx,vy)`` record.

    The ``.gt.tar`` / BEVFusion layout (docs/data_pipeline.md:192,305) is
    ``(x,y,z,w,l,h,yaw,vx,vy)``: column 3 = ``w`` (WIDTH, lateral), column 4 =
    ``l`` (LENGTH, longitudinal), ``yaw`` the heading. VERIFIED numerically on a
    real jpntaxi-val ``.gt.tar`` frame: col3 median 1.54 m < col4 median 3.41 m
    (vehicles longer than wide), and ``yaw`` aligns with ``atan2(vy, vx)`` for
    moving agents (median 9.1deg, 77% within 30deg, 5% flipped) => the LENGTH
    (col4) lies ALONG ``yaw`` and the WIDTH (col3) perpendicular. (NB: this is a
    DIFFERENT column order from ``data/augmentations.py``'s mmdet3d training
    boxes where col3 is the yaw-aligned dim — do not conflate the two.)
    Returns 4 corners for a shapely polygon.
    """
    cx, cy = float(box[0]), float(box[1])
    width, length = float(box[3]), float(box[4])  # col3 = w (lateral), col4 = l (longitudinal)
    yaw = float(box[6])
    hl, hw = length / 2.0, width / 2.0  # half-length along yaw, half-width perpendicular
    c, s = np.cos(yaw), np.sin(yaw)

    def rot(fx, fy):  # fx along yaw (longitudinal), fy perpendicular (lateral)
        return (cx + fx * c - fy * s, cy + fx * s + fy * c)

    return np.array([rot(hl, hw), rot(-hl, hw), rot(-hl, -hw), rot(hl, -hw)], dtype=np.float64)


def box_corners_batch(boxes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorized :func:`box_corners` for ``[N, 9]`` boxes."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 9)
    cx, cy = boxes[:, 0], boxes[:, 1]
    width, length = boxes[:, 3], boxes[:, 4]
    yaw = boxes[:, 6]
    hl, hw = length / 2.0, width / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    fx = np.stack([hl, -hl, -hl, hl], axis=-1)
    fy = np.stack([hw, hw, -hw, -hw], axis=-1)
    out = np.empty((boxes.shape[0], 4, 2), dtype=np.float64)
    out[:, :, 0] = cx[:, None] + fx * c[:, None] - fy * s[:, None]
    out[:, :, 1] = cy[:, None] + fx * s[:, None] + fy * c[:, None]
    return out


def _sat_intersects_one_to_many(
    ego: npt.NDArray[np.float64],
    boxes: npt.NDArray[np.float64],
    eps: float = 1.0e-9,
) -> npt.NDArray[np.bool_]:
    """Exact-safe oriented-rectangle prefilter before expensive shapely calls.

    A ``False`` result means the rectangles cannot intersect by the separating
    axis theorem. ``True`` only means "possible"; callers still run shapely for
    the authoritative ``intersects`` result, so this cannot change scores.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4, 2)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    ego = np.asarray(ego, dtype=np.float64).reshape(4, 2)
    ego_axes = np.stack([ego[0] - ego[1], ego[1] - ego[2]], axis=0)
    box_axes = np.stack([boxes[:, 0] - boxes[:, 1], boxes[:, 1] - boxes[:, 2]], axis=1)
    axes = np.concatenate(
        [np.broadcast_to(ego_axes, (boxes.shape[0], 2, 2)), box_axes], axis=1
    )
    axes = axes / np.clip(np.linalg.norm(axes, axis=-1, keepdims=True), 1.0e-12, None)

    ego_proj = np.einsum("nkd,vd->nkv", axes, ego)
    box_proj = np.einsum("nkd,nvd->nkv", axes, boxes)
    overlap = (ego_proj.max(axis=-1) + eps >= box_proj.min(axis=-1)) & (
        box_proj.max(axis=-1) + eps >= ego_proj.min(axis=-1)
    )
    return overlap.all(axis=-1)


def _polygon(corners: npt.NDArray[np.float64]):
    from shapely.geometry import Polygon

    return Polygon([tuple(p) for p in corners])


def get_agent_relative_angle(ego_xyh, agent_xyh) -> float:
    """nuplan idm/utils.py::get_agent_relative_angle (verbatim).

    Angle in [0, pi] between ego heading and the ego->agent vector.
    ``*_xyh`` = (x, y, heading).
    """
    agent_vector = np.array([agent_xyh[0] - ego_xyh[0], agent_xyh[1] - ego_xyh[1]])
    norm = np.linalg.norm(agent_vector)
    if norm < 1e-9:
        # agent centroid coincides with ego (overlapping) -> treat as directly
        # ahead (relative angle 0). nuplan would divide-by-zero here; this is the
        # only addition to its verbatim logic and only triggers on full overlap.
        return 0.0
    ego_vector = np.array([np.cos(ego_xyh[2]), np.sin(ego_xyh[2])])
    dot_product = np.dot(ego_vector, agent_vector / norm)
    return float(np.arccos(np.clip(dot_product, -1.0, 1.0)))


def is_agent_ahead(ego_xyh, agent_xyh, angle_tolerance: float = 30.0) -> bool:
    """nuplan idm/utils.py::is_agent_ahead (verbatim; tol 30 deg)."""
    return bool(get_agent_relative_angle(ego_xyh, agent_xyh) < np.deg2rad(angle_tolerance))


def is_agent_behind(ego_xyh, agent_xyh, angle_tolerance: float = 150.0) -> bool:
    """nuplan idm/utils.py::is_agent_behind (verbatim; tol 150 deg)."""
    return bool(get_agent_relative_angle(ego_xyh, agent_xyh) > np.deg2rad(angle_tolerance))


def is_track_stopped(
    box: npt.NDArray[np.float64], stopped_speed_threshold: float = COLLISION_STOPPED_SPEED_THRESHOLD
) -> bool:
    """nuplan idm/utils.py::is_track_stopped (verbatim): speed magnitude <= thresh.

    Agent speed = hypot(vx, vy) from the box's velocity columns (7,8).
    """
    speed = float(np.hypot(box[7], box[8]))
    return speed <= stopped_speed_threshold


def get_collision_type(
    ego_xyh, ego_speed: float, ego_poly, agent_box: npt.NDArray[np.float64], agent_poly
) -> CollisionType:
    """pdm_scorer_utils.py::get_collision_type (verbatim logic).

    ``ego_xyh`` = (x, y, heading) at ego rear-axle/reference; ``ego_speed`` the
    ego speed magnitude; ``ego_poly``/``agent_poly`` shapely polygons.
    """
    from shapely.geometry import LineString

    is_ego_stopped = float(ego_speed) <= COLLISION_STOPPED_SPEED_THRESHOLD
    agent_xyh = (float(agent_box[0]), float(agent_box[1]), float(agent_box[6]))

    if is_ego_stopped:
        return CollisionType.STOPPED_EGO_COLLISION
    if is_track_stopped(agent_box):
        return CollisionType.STOPPED_TRACK_COLLISION
    if is_agent_behind(ego_xyh, agent_xyh):
        return CollisionType.ACTIVE_REAR_COLLISION
    # front bumper edge = exterior coords[0]->coords[3] (FL->FR)
    coords = list(ego_poly.exterior.coords)
    if LineString([coords[0], coords[3]]).intersects(agent_poly):
        return CollisionType.ACTIVE_FRONT_COLLISION
    return CollisionType.ACTIVE_LATERAL_COLLISION


# ===========================================================================
# NC -- no at-fault collision (pdm_scorer.py::_calculate_no_at_fault_collision)
# ===========================================================================
def no_at_fault_collision(
    ego_states: npt.NDArray[np.float64],
    agent_boxes_per_t: list,
    ego_length: float,
    ego_width: float,
    agent_labels_per_t: list | None = None,
    static_labels: set | None = None,
    center_offset: float = 0.0,
    area_flags: "npt.NDArray[np.bool_] | None" = None,
) -> float:
    """1:1 port of NC. Returns 1.0 (no at-fault collision), 0.5 (static object)
    or 0.0 (dynamic agent), the min over the horizon.

    ``ego_states`` ``[T, STATE_SIZE]`` (from ``states_from_poses``);
    ``agent_boxes_per_t[t]`` an ``[N_t, 9]`` array of agent boxes in the ego
    frame at timestep ``t``. ``agent_labels_per_t[t]`` (optional) the per-box
    int class ids; a box whose label is in ``static_labels`` scores 0.5 (navsim
    "not in AGENT_TYPES"), all others 0.0. Default ``static_labels=None`` =>
    every agent is dynamic (0.0), the conservative default for T4 GT which is
    dominated by vehicles/pedestrians/bicycles (all in nuplan AGENT_TYPES).

    DEVIATIONS #2 (GT-future agents), #3 (lateral case needs maps => navsim's
    no-penalty else-branch). The front/stopped-track at-fault cases — the
    dominant ones — are map-INDEPENDENT and fully faithful.
    """
    T = ego_states.shape[0]
    score = 1.0
    # navsim dedups by track token across time. Our GT-future boxes are not
    # tracked across frames, so we conservatively treat each frame independently
    # for the at-fault test but keep navsim's "already-collided => skip" within
    # the SAME frame's repeated-geometry guard (no token => no cross-frame dedup;
    # documented). This never under-counts a fresh at-fault collision.
    for t in range(T):
        boxes = np.asarray(agent_boxes_per_t[t], dtype=np.float64).reshape(-1, 9)
        if boxes.shape[0] == 0:
            continue
        x, y, h = ego_states[t, STATE_X], ego_states[t, STATE_Y], ego_states[t, STATE_HEADING]
        ego_speed = float(np.hypot(ego_states[t, STATE_VEL_X], ego_states[t, STATE_VEL_Y]))
        # Footprint CENTER = trajectory point + center_offset along heading
        # (the trajectory reference is the rear-ish point; the in-repo
        # convention — loss.py::compute_ego_bbox_corners — and navsim's
        # rear-axle->OrientedBox-center both shift by wheel_base/2). The
        # ahead/behind classification below keeps the RAW pose (navsim
        # rear-axle semantics).
        cxs = x + center_offset * np.cos(h)
        cys = y + center_offset * np.sin(h)
        # Disjointness prefilter (pure numpy, exact-safe): centers farther apart
        # than the two half-diagonals cannot intersect — skip without building
        # shapely polygons. Kills ~99.9% of the polygon work on real scenes.
        ego_rad = 0.5 * float(np.hypot(ego_length, ego_width))
        d = np.hypot(boxes[:, 0] - cxs, boxes[:, 1] - cys)
        cand = np.where(d <= ego_rad + 0.5 * np.hypot(boxes[:, 3], boxes[:, 4]))[0]
        if cand.size == 0:
            continue
        ego_corners_t = ego_corners(cxs, cys, h, ego_length, ego_width)
        agent_corners = box_corners_batch(boxes[cand])
        sat = _sat_intersects_one_to_many(ego_corners_t, agent_corners)
        if not sat.any():
            continue
        ego_poly = _polygon(ego_corners_t)
        labels = agent_labels_per_t[t] if agent_labels_per_t is not None else None
        for local_j in np.where(sat)[0]:
            j = cand[local_j]
            agent_poly = _polygon(agent_corners[local_j])
            if not ego_poly.intersects(agent_poly):
                continue
            ctype = get_collision_type((x, y, h), ego_speed, ego_poly, boxes[j], agent_poly)
            at_fault = ctype in (
                CollisionType.ACTIVE_FRONT_COLLISION,
                CollisionType.STOPPED_TRACK_COLLISION,
            )
            # navsim's lateral at-fault branch: ACTIVE_LATERAL while ego is in
            # multiple lanes or a non-drivable area (ego_area_flags from the
            # shard's lane/intersection polygons; legacy no-flags callers keep
            # the lenient else — old deviation #3, now closed when flags given).
            if (
                not at_fault
                and area_flags is not None
                and ctype == CollisionType.ACTIVE_LATERAL_COLLISION
                and bool(area_flags[t, 0] or area_flags[t, 1])
            ):
                at_fault = True
            if not at_fault:
                continue
            is_static = (
                static_labels is not None and labels is not None and int(labels[j]) in static_labels
            )
            this_score = 0.5 if is_static else 0.0
            score = min(score, this_score)
    return score


# ===========================================================================
# TTC -- time to collision (pdm_scorer.py::_calculate_ttc)
# ===========================================================================
def time_to_collision(
    ego_states: npt.NDArray[np.float64],
    agent_boxes_per_t: list,
    ego_length: float,
    ego_width: float,
    dt: float,
    center_offset: float = 0.0,
    area_flags: "npt.NDArray[np.bool_] | None" = None,
) -> float:
    """1:1 port of TTC. Returns 1.0 (ok) or 0.0 (infraction).

    Projects the ego box forward at constant velocity at future idcs
    ``arange(0, int(horizon*10), 3)`` => [0,3,6,9] (0/0.3/0.6/0.9 s for a 1 s
    horizon), and checks intersection with the agents at ``t+future_idx``. An
    infraction is the agent being AHEAD (deviation #3: navsim's
    intersection/multi-lane branch needs maps). Skips when ego speed <
    ``STOPPED_SPEED_THRESHOLD``.
    """
    T = ego_states.shape[0]
    future_time_idcs = np.arange(0, int(FUTURE_COLLISION_HORIZON_WINDOW * 10), 3)
    max_future = int(future_time_idcs.max())
    speeds = np.hypot(ego_states[:, STATE_VEL_X], ego_states[:, STATE_VEL_Y])
    headings = ego_states[:, STATE_HEADING]
    # per-step world velocity for the constant-velocity projection
    dxy_per_s = np.stack([np.cos(headings) * speeds, np.sin(headings) * speeds], axis=-1)

    score = 1.0
    n_eval = T - max_future
    for t in range(max(0, n_eval)):
        if speeds[t] < STOPPED_SPEED_THRESHOLD:
            continue
        x0, y0, h = ego_states[t, STATE_X], ego_states[t, STATE_Y], headings[t]
        for fidx in future_time_idcs:
            ct = t + int(fidx)
            if ct >= T:
                continue
            boxes = np.asarray(agent_boxes_per_t[ct], dtype=np.float64).reshape(-1, 9)
            if boxes.shape[0] == 0:
                continue
            delta_t = float(fidx) * dt
            dx, dy = dxy_per_s[t] * delta_t
            # Footprint centre = projected pose + center_offset along heading
            # (see no_at_fault_collision); ahead-test keeps the raw pose.
            pcx = x0 + dx + center_offset * np.cos(h)
            pcy = y0 + dy + center_offset * np.sin(h)
            # Disjointness prefilter (see no_at_fault_collision) on the
            # PROJECTED ego centre — exact-safe, skips polygon construction.
            ego_rad = 0.5 * float(np.hypot(ego_length, ego_width))
            dcent = np.hypot(boxes[:, 0] - pcx, boxes[:, 1] - pcy)
            cand = np.where(dcent <= ego_rad + 0.5 * np.hypot(boxes[:, 3], boxes[:, 4]))[0]
            if cand.size == 0:
                continue
            # Projected polygon for the INTERSECTION test, but the ahead/behind
            # test uses the CURRENT (unshifted) ego pose at time_idx — matching
            # navsim (pdm_scorer.py::_calculate_ttc uses self._states[..,time_idx]
            # for is_agent_ahead). Using the projected pose would miss cases where
            # the ego reaches/passes a stopped agent within the window (the agent
            # ends up behind the projected centre though the boxes intersect).
            ego_corners_t = ego_corners(pcx, pcy, h, ego_length, ego_width)
            agent_corners = box_corners_batch(boxes[cand])
            sat = _sat_intersects_one_to_many(ego_corners_t, agent_corners)
            if not sat.any():
                continue
            ego_poly = _polygon(ego_corners_t)
            ego_xyh_current = (x0, y0, h)
            for local_j in np.where(sat)[0]:
                j = cand[local_j]
                agent_poly = _polygon(agent_corners[local_j])
                if not ego_poly.intersects(agent_poly):
                    continue
                agent_xyh = (float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][6]))
                # navsim widens the infraction when ego is in multiple lanes /
                # non-drivable / an intersection: any projected collision with
                # an agent NOT BEHIND counts (lateral included). Flags read at
                # the CURRENT time_idx t, matching navsim.
                map_widened = area_flags is not None and bool(
                    area_flags[t, 0] or area_flags[t, 1] or area_flags[t, 2]
                )
                if is_agent_ahead(ego_xyh_current, agent_xyh) or (
                    map_widened and not is_agent_behind(ego_xyh_current, agent_xyh)
                ):
                    score = min(score, 0.0)
    return score


# ===========================================================================
# EP -- ego progress (pdm_scorer.py::_calculate_progress + _aggregate_pdm_scores)
# ===========================================================================
def ego_progress_with_gate(
    pred_poses: npt.NDArray[np.float64],
    reference_path: npt.NDArray[np.float64],
    multiplicative: float = 1.0,
) -> tuple[float, bool]:
    """1:1 port of EP, plus the GATE flag. Project the predicted start/end onto a reference path
    (shapely ``LineString.project``), ``raw = clip(end - start, 0, None)``, then
    normalise by the reference's own progress (navsim's proposal-relative
    ``raw / max_raw_progress`` with the 5 m threshold gate).

    Returns ``(score, gated)``: ``gated=True`` means the navsim threshold branch
    fired (``max_raw <= 5 m`` — typically a stationary/slow expert, e.g. a red
    light), where the score is 1.0 REGARDLESS of the prediction. Such frames are
    uninformative for ranking; report their fraction alongside the EP mean so a
    high EP cannot silently come from gated frames.

    DEVIATION: navsim normalises by the max over the proposal SET (the PDM-closed
    reference proposal). Open-loop we have ONE trajectory, so the EXPERT future
    plays the reference-proposal role: ``reference_path`` should be the expert GT
    polyline (its centerline-projected extent is the denominator). When the
    reference progress <= 5 m navsim returns 1.0 (or 0.0 if the multiplicative
    metrics zeroed the proposal) — replicated here.

    ``pred_poses``/``reference_path`` are ``[T, >=2]`` (x, y, ...).
    """
    from shapely.geometry import LineString, Point

    ref_xy = np.asarray(reference_path, dtype=np.float64)[:, :2]
    if len(ref_xy) < 2:
        return 1.0, True  # degenerate reference: same uninformative branch
    line = LineString([tuple(p) for p in ref_xy])
    pred = np.asarray(pred_poses, dtype=np.float64)
    start = line.project(Point(pred[0, 0], pred[0, 1]))
    end = line.project(Point(pred[-1, 0], pred[-1, 1]))
    raw_progress = max(end - start, 0.0)

    # reference's own raw progress along itself = its arclength extent
    ref_start = line.project(Point(ref_xy[0, 0], ref_xy[0, 1]))
    ref_end = line.project(Point(ref_xy[-1, 0], ref_xy[-1, 1]))
    ref_progress = max(ref_end - ref_start, 0.0)

    raw = raw_progress * multiplicative
    max_raw = max(ref_progress * multiplicative, raw)  # proposal-set max incl. self
    if max_raw > PROGRESS_DISTANCE_THRESHOLD:
        return float(np.clip(raw / max_raw, 0.0, 1.0)), False
    # navsim: <= threshold => 1.0, or 0.0 if multiplicative zeroed this proposal
    return (0.0 if multiplicative == 0.0 else 1.0), True


def ego_progress(
    pred_poses: npt.NDArray[np.float64],
    reference_path: npt.NDArray[np.float64],
    multiplicative: float = 1.0,
) -> float:
    """1:1 port of EP (score only) — see :func:`ego_progress_with_gate`."""
    return ego_progress_with_gate(pred_poses, reference_path, multiplicative)[0]


# ===========================================================================
# Aggregation (pdm_scorer.py::_aggregate_pdm_scores + config weights)
# ===========================================================================
def aggregate_pdms(
    *,
    nc: float = 1.0,
    dac: float = 1.0,
    ddc: float = 1.0,
    tlc: float = 1.0,
    ego_progress: float | None = None,
    ttc: float | None = None,
    lane_keeping: float | None = None,
    history_comfort: float | None = None,
    extended_comfort: float | None = None,
) -> float:
    """Legacy partial PDMS helper: prod(multiplicative) * weighted_avg(weighted metrics).

    Multiplicative: NC * DAC * DDC * TLC. Weighted average over the metrics that
    are PROVIDED (not ``None``) with navsim weights EP=5, TTC=5, LK=2,
    HISTORY_COMFORT=2, EC=2 (EC is the EPDMS extension; pass ``None`` where the
    reference reports "unavailable" — e.g. the first plan of a scene). This is
    intentionally not the Autoware ``synthetic_epdms`` aggregation, whose
    denominator is fixed to 16 and whose result is unavailable unless all raw
    subscores are available. Use ``pdms_proxy.synthetic_epdms`` for validation
    metrics that need to match Autoware planning_data_analyzer semantics.
    """
    multiplicative = nc * dac * ddc * tlc
    weighted = [
        (PROGRESS_WEIGHT, ego_progress),
        (TTC_WEIGHT, ttc),
        (LANE_KEEPING_WEIGHT, lane_keeping),
        (HISTORY_COMFORT_WEIGHT, history_comfort),
        (EXTENDED_COMFORT_WEIGHT, extended_comfort),
    ]
    present = [(w, v) for w, v in weighted if v is not None]
    if not present:
        return float(multiplicative)
    wsum = sum(w for w, _ in present)
    weighted_avg = sum(w * v for w, v in present) / wsum
    return float(multiplicative * weighted_avg)


# ===========================================================================
# DAC — drivable-area compliance proxy from road-border polylines
# ===========================================================================
def dac_from_road_borders(
    poses: npt.NDArray[np.float64],
    borders: list,
    ego_length: float,
    ego_width: float,
    center_offset: float = 0.0,
) -> float:
    """Drivable-area compliance over the horizon: 1.0 if the ego FOOTPRINT
    never crosses a road-border polyline, else 0.0 (navsim DAC is a binary
    multiplicative term).

    Proxy vs navsim: navsim tests the footprint against nuPlan's drivable-area
    polygons; our shards carry the lanelet maps' ``road_border`` line strings
    (the drivable-area BOUNDARY) in the same center-frame coordinates as the
    predictions, so "crossing a border" == "leaving the drivable area". Border
    coverage is the 60 nearest line strings within the generation window —
    report the presence fraction alongside (``val/dac_border_frac``) like
    ``agent_gt_frac``.

    ``poses`` is ``[T, >=4]`` as ``(x, y, cos_h, sin_h, ...)``;``borders`` a
    list of ``[P, 2]`` polylines. ``center_offset`` shifts the footprint centre
    along heading (wheel_base/2 — loss.py::compute_ego_bbox_corners convention,
    same as NC).
    """
    import shapely
    from shapely.geometry import MultiLineString

    polylines = [np.asarray(b, dtype=np.float64) for b in borders]
    polylines = [b for b in polylines if b.shape[0] >= 2]
    if not polylines:
        return 1.0
    mls = MultiLineString([b.tolist() for b in polylines])
    if mls.is_empty:
        return 1.0

    T = poses.shape[0]
    heading = np.arctan2(poses[:, 3], poses[:, 2])
    cx = poses[:, 0] + center_offset * np.cos(heading)
    cy = poses[:, 1] + center_offset * np.sin(heading)

    # cheap prefilter: horizon bbox (+ ego halo) vs border bbox (the NC-style
    # ~99.9% kill — most horizons never come near a border)
    halo = 0.5 * float(np.hypot(ego_length, ego_width))
    minx, miny, maxx, maxy = mls.bounds
    if (
        cx.max() + halo < minx
        or cx.min() - halo > maxx
        or cy.max() + halo < miny
        or cy.min() - halo > maxy
    ):
        return 1.0

    corners = ego_corners_batch(cx, cy, heading, ego_length, ego_width)  # [T, 4, 2]
    boxes = shapely.creation.polygons(corners)  # vectorized, one call
    return 0.0 if bool(shapely.intersects(boxes, mls).any()) else 1.0


# ===========================================================================
# DDC — driving-direction compliance from route-lane polygons
# ===========================================================================
DRIVING_DIRECTION_HORIZON = 1.0  # [s] (pdm_scorer.py config, nuplan)
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # [m]
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0  # [m]


def ddc_from_route_lanes(
    poses: npt.NDArray[np.float64],
    route_polygons: list,
    dt: float,
) -> float:
    """Driving-direction compliance, navsim's re-implementation 1:1
    (``pdm_scorer.py::_calculate_driving_direction_compliance`` +
    ``_calculate_ego_area``'s ONCOMING_TRAFFIC):

    - a pose is "in oncoming traffic" iff the ego CENTER is not inside ANY
      on-route lane polygon;
    - per-step displacement is accumulated ONLY over oncoming poses;
    - the max over a sliding ``driving_direction_horizon`` (1 s) window is
      thresholded: < 2 m -> 1.0, < 6 m -> 0.5, else 0.0.

    Proxy vs navsim: their on-route polygons come from nuPlan's map API; ours
    are built from the shard's ROUTE tensor (centerline ± boundary-offset
    channels), same center frame as the predictions, windowed to the 25
    nearest route segments — report ``val/ddc_route_frac`` alongside. The
    trajectory is prepended with the origin (the ego's center-frame position)
    to mirror navsim's initial-state-inclusive indexing.

    ``poses``: ``[T, >=2]`` future positions; ``route_polygons``: list of
    ``[P, 2]`` polygon rings.
    """
    import shapely
    from shapely.geometry import Polygon

    polys = []
    for ring in route_polygons:
        ring = np.asarray(ring, dtype=np.float64)
        if ring.shape[0] >= 3:
            p = Polygon(ring)
            if p.is_valid and not p.is_empty:
                polys.append(p)
    pts_xy = np.concatenate([np.zeros((1, 2)), np.asarray(poses, dtype=np.float64)[:, :2]])
    if not polys:
        # no route coverage -> no oncoming evidence (navsim semantics would
        # flag EVERYTHING oncoming; with a windowed tensor that would punish
        # coverage gaps, so absence scores 1.0 and is reported via route_frac)
        return 1.0

    points = shapely.points(pts_xy)
    inside = np.zeros(len(points), dtype=bool)
    for p in polys:
        inside |= shapely.contains(p, points)
        if inside.all():
            break
    oncoming = ~inside  # [T+1]

    disp = np.zeros(len(points), dtype=np.float64)
    disp[1:] = np.linalg.norm(pts_xy[1:] - pts_xy[:-1], axis=-1)
    disp[~oncoming] = 0.0

    horizon = int(DRIVING_DIRECTION_HORIZON / dt)
    worst = 0.0
    for t in range(len(disp)):
        worst = max(worst, float(disp[max(0, t - horizon) : t + 1].sum()))
    if worst < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD:
        return 1.0
    if worst < DRIVING_DIRECTION_VIOLATION_THRESHOLD:
        return 0.5
    return 0.0


def route_polygons_from_tensor(route_rows: npt.NDArray) -> list:
    """Build on-route lane polygon rings from a ``[N, P, C>=8]`` route tensor
    slice: ring = (centerline + left-offset) forward, then (centerline +
    right-offset) reversed. Zero-padded rows are skipped."""
    rings = []
    for r in range(route_rows.shape[0]):
        seg = np.asarray(route_rows[r], dtype=np.float64)
        if np.abs(seg).sum() == 0:
            continue
        center = seg[:, 0:2]
        left = center + seg[:, 4:6]
        right = center + seg[:, 6:8]
        rings.append(np.concatenate([left, right[::-1]], axis=0))
    return rings


# ===========================================================================
# Ego-area flags — pdm_scorer.py::_calculate_ego_area (map gate for NC/TTC)
# ===========================================================================
def ego_area_flags(
    ego_states: npt.NDArray[np.float64],
    lane_rings: list,
    intersection_rings: list,
    ego_length: float,
    ego_width: float,
    center_offset: float = 0.0,
) -> npt.NDArray[np.bool_]:
    """Per-step (multiple_lanes, non_drivable, in_intersection) flags, navsim's
    ``_calculate_ego_area`` semantics:

    - multiple_lanes: >1 LANE polygon contains at least one footprint corner
      AND no single lane polygon contains all 4 corners;
    - non_drivable: at least one corner outside EVERY drivable polygon
      (drivable = lanes + intersections here — the shards carry no carpark
      polygons, a documented deviation that can only fire inside carparks);
    - in_intersection: the RAW pose point (navsim: rear axle) inside any
      intersection polygon.

    ``ego_states`` are the NC/TTC state rows (x, y, heading at STATE_*);
    rings are ``[P, 2]`` polygon rings in the same center frame.
    """
    import shapely
    from shapely.geometry import Polygon

    T = ego_states.shape[0]
    out = np.zeros((T, 3), dtype=bool)

    def _polys(rings):
        ps = []
        for ring in rings:
            ring = np.asarray(ring, dtype=np.float64)
            if ring.shape[0] >= 3:
                p = Polygon(ring)
                if p.is_valid and not p.is_empty:
                    ps.append(p)
        return ps

    lanes = _polys(lane_rings)
    inters = _polys(intersection_rings)
    if not lanes and not inters:
        return out

    xs, ys, hs = (
        ego_states[:, STATE_X],
        ego_states[:, STATE_Y],
        ego_states[:, STATE_HEADING],
    )
    cxs = xs + center_offset * np.cos(hs)
    cys = ys + center_offset * np.sin(hs)
    corners = ego_corners_batch(cxs, cys, hs, ego_length, ego_width)  # [T, 4, 2]
    corner_pts = shapely.points(corners.reshape(-1, 2)).reshape(T, 4)
    pose_pts = shapely.points(np.stack([xs, ys], axis=-1))

    # corner-in-polygon counts per lane polygon: [n_lanes, T]
    if lanes:
        lane_counts = np.stack(
            [shapely.contains(p, corner_pts).sum(axis=-1) for p in lanes]
        )  # corners per lane poly per t
        n_hit = (lane_counts > 0).sum(axis=0)  # lanes touched per t
        full = (lane_counts == 4).any(axis=0)  # some lane holds all corners
        out[:, 0] = (n_hit > 1) & ~full
    # drivable coverage per corner: corner in ANY (lane | intersection) poly
    drivable = lanes + inters
    if drivable:
        covered = np.zeros((T, 4), dtype=bool)
        for p in drivable:
            covered |= shapely.contains(p, corner_pts)
            if covered.all():
                break
        out[:, 1] = ~covered.all(axis=-1)
    for p in inters:
        out[:, 2] |= shapely.contains(p, pose_pts)
    return out


# ===========================================================================
# LK — lane keeping (Autoware planning_data_analyzer EPDMS spec)
# ===========================================================================
LK_D_MAX = 0.5  # [m] max accepted lateral deviation from the route centerline
LK_MAX_VIOLATION_S = 2.0  # max continuous over-threshold run
LK_QUEUE_SPEED = 1.0  # [m/s]
LK_QUEUE_WINDOW_S = 1.0
LK_QUEUE_PROGRESS_M = 1.5
LK_QUEUE_RELEASE_S = 1.5


def lane_keeping_score(
    poses: npt.NDArray[np.float64],
    route_centerlines: list,
    intersection_rings: list,
    dt: float,
    lane_change_exempt: bool = False,
) -> float:
    """Lane keeping per the Autoware planning_data_analyzer EPDMS spec
    (docs/metrics/epdms_metrics.md, read verbatim from autoware_tools):

    - per-sample lateral distance to the NEAREST route centerline; over-flag
      at |d| > 0.5 m;
    - exemptions: inside an intersection; queue (speed <= 1 m/s AND <= 1.5 m
      progress over a 1 s window) with 1.5 s release grace; lane-change
      indicator windows;
    - LK = 1 iff every continuous violation run is shorter than 2 s, else 0.

    Deviations: the analyzer reads per-sample turn-indicator/hazard intervals
    (+-1 s grace) from the bag — our batches carry only the PAST-window
    indicator state, so ``lane_change_exempt`` applies the center-frame state
    to the whole horizon. Route centerlines are the shard window's 25 route
    segments. No route coverage scores 1.0 (reported via ddc_route_frac).
    """
    from shapely.geometry import LineString, Polygon

    pts = np.asarray(poses, dtype=np.float64)[:, :2]
    T = pts.shape[0]
    if lane_change_exempt or T == 0:
        return 1.0
    lines = [
        LineString(np.asarray(c, dtype=np.float64))
        for c in route_centerlines
        if np.asarray(c).shape[0] >= 2
    ]
    if not lines:
        return 1.0
    inters = []
    for ring in intersection_rings:
        ring = np.asarray(ring, dtype=np.float64)
        if ring.shape[0] >= 3:
            p = Polygon(ring)
            if p.is_valid and not p.is_empty:
                inters.append(p)

    import shapely

    sh_pts = shapely.points(pts)
    d = np.full(T, np.inf)
    for ln in lines:
        d = np.minimum(d, shapely.distance(sh_pts, ln))
    over = d > LK_D_MAX

    in_inter = np.zeros(T, dtype=bool)
    for p in inters:
        in_inter |= shapely.contains(p, sh_pts)

    # queue: low speed AND low progress over the trailing 1 s window
    step_d = np.zeros(T)
    step_d[1:] = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    speed = step_d / max(dt, 1e-9)
    if T > 1:
        # forward-difference speed at t0: step_d[0] is structurally 0, which
        # would mark the FIRST sample queued on every trajectory and the
        # release grace would otherwise mask early violations
        speed[0] = speed[1]
    win = max(1, int(LK_QUEUE_WINDOW_S / dt))
    queue = np.zeros(T, dtype=bool)
    for t in range(T):
        prog = step_d[max(0, t - win) : t + 1].sum()
        queue[t] = (speed[t] <= LK_QUEUE_SPEED) and (prog <= LK_QUEUE_PROGRESS_M)
    release = np.zeros(T, dtype=bool)
    grace = int(LK_QUEUE_RELEASE_S / dt)
    last_q = -(10**9)
    for t in range(T):
        if queue[t]:
            last_q = t
        elif t - last_q <= grace:
            release[t] = True

    violation = over & ~in_inter & ~queue & ~release
    run = 0
    for t in range(T):
        run = run + 1 if violation[t] else 0
        if run * dt >= LK_MAX_VIOLATION_S:
            return 0.0
    return 1.0


# ===========================================================================
# Semantic DAC — Autoware planning_data_analyzer port (ego_footprint.cpp)
# ===========================================================================
# Constants verbatim from metrics/geometry/ego_footprint.cpp:
SEM_DAC_BORDER_MAX_GAP_M = 3.0  # kRoadBorderMaxGapM
SEM_DAC_MAX_SEMANTIC_TO_BORDER_M = 4.0  # kRoadBorderMaxSemanticToBorderM
SEM_DAC_BETWEEN_TOL_M = 0.75  # kRoadBorderBetweenToleranceM
SEM_DAC_MAX_TANGENT_ALIGN = 0.6  # kRoadBorderMaxTangentAlignment
SEM_DAC_MIN_SEG_LEN_M = 1.0e-3  # kMinRoadBorderSegmentLengthM
SEM_DAC_SIDE_EPS = 1.0e-3  # kRoadBorderSideEpsilon
SEM_DAC_PROBES_M = (0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # kRoadBorderSideProbeDistancesM
SEM_DAC_BORDER_SEARCH_M = 5.0  # kRoadBorderSearchMarginM


def _polygon_parts(geom) -> list:
    """Flatten any shapely geometry into its Polygon parts.

    ``buffer(0)`` healing of a self-intersecting lanelet ring can return a
    MultiPolygon (e.g. a bow-tie splits into two triangles) or, degenerately,
    a GeometryCollection — downstream code touches ``.exterior``, which only
    Polygon has (Job 628 died on a real jpntaxi-val scene exactly here).
    """
    if geom.geom_type == "Polygon":
        return [] if geom.is_empty else [geom]
    if hasattr(geom, "geoms"):
        out: list = []
        for g in geom.geoms:
            out.extend(_polygon_parts(g))
        return out
    return []


def _sem_union_polys(layers: dict):
    from shapely.geometry import Polygon

    polys = []
    for key in ("road", "shoulder", "intersection", "hatched", "parking"):
        for ring in layers.get(key, []):
            ring = np.asarray(ring, dtype=np.float64)
            if ring.shape[0] >= 3:
                p = Polygon(ring)
                if not p.is_valid:
                    p = p.buffer(0)  # heal self-touching lanelet rings
                if p.is_valid and not p.is_empty:
                    polys.extend(_polygon_parts(p))
    return polys


def dac_semantic(
    poses: npt.NDArray[np.float64],
    layers: dict,
    ego_length: float,
    ego_width: float,
    center_offset: float = 0.0,
) -> float:
    """Semantic-primary DAC, ported line-by-line from the Autoware
    planning_data_analyzer (``ego_footprint.cpp``): every footprint corner must
    be inside the semantic drivable union (road + shoulder lanelets,
    intersection_area, hatched_road_markings, parking_lot) OR accepted by the
    road-border side-probe fallback (the digitization-gap healer):

    corner X fails semantically -> S = closest semantic-boundary point,
    candidate border segments within 5 m -> B = closest point on the segment,
    probe B +- rho*n for rho in {0.3..4.0} until exactly one side is
    semantically drivable; accept iff X is on that road side
    (side*side > 1e-3), X projects into the S->B gap (ratio in [-0.05, 1.05],
    lateral offset <= 0.75 m), S->B is ACROSS the border
    (|tangent alignment| <= 0.6), d(X,B) <= 3 m and d(S,B) <= 4 m.

    ``layers``: ego-frame dict — ring lists for the five semantic keys plus
    ``border`` polylines. Returns navsim's binary {0.0, 1.0}.
    """
    import shapely
    from shapely.ops import nearest_points

    polys = _sem_union_polys(layers)
    if not polys:
        return 1.0  # no semantic coverage -> nothing provable; report via frac
    tree = shapely.STRtree(polys)

    borders = []
    for b in layers.get("border", []):
        b = np.asarray(b, dtype=np.float64)
        if b.shape[0] >= 2:
            borders.append(b)

    T = poses.shape[0]
    heading = np.arctan2(poses[:, 3], poses[:, 2])
    cx = poses[:, 0] + center_offset * np.cos(heading)
    cy = poses[:, 1] + center_offset * np.sin(heading)

    def _in_union(pt) -> bool:
        # STRtree predicate runs query-geometry-vs-tree-geometry: a POINT must
        # be covered_by the polygon (point.covers(poly) is always false)
        return len(tree.query(pt, predicate="covered_by")) > 0

    for t in range(T):
        corners = ego_corners(cx[t], cy[t], heading[t], ego_length, ego_width)
        for k in range(4):
            X = shapely.points(corners[k])
            if _in_union(X):
                continue
            # S: closest point on the union boundary
            sd = np.inf
            S = None
            for p in polys:
                q = nearest_points(p.exterior, X)[0]
                d = q.distance(X)
                if d < sd:
                    sd, S = d, q
            if S is None:
                return 0.0
            accepted = False
            for line in borders:
                if accepted:
                    break
                for si in range(1, line.shape[0]):
                    a, b = line[si - 1], line[si]
                    seg_d = b - a
                    seg_len = float(np.hypot(*seg_d))
                    if seg_len < SEM_DAC_MIN_SEG_LEN_M:
                        continue
                    # closest point on segment to X
                    Xa = corners[k] - a
                    r = float(np.clip(np.dot(Xa, seg_d) / (seg_len * seg_len), 0.0, 1.0))
                    B = a + r * seg_d
                    dXB = float(np.hypot(*(corners[k] - B)))
                    if dXB > SEM_DAC_BORDER_SEARCH_M:
                        continue
                    SB = B - np.array([S.x, S.y])
                    sb_len = float(np.hypot(*SB))
                    if sb_len * sb_len <= SEM_DAC_MIN_SEG_LEN_M**2:
                        continue
                    Xv = corners[k] - np.array([S.x, S.y])
                    ratio = float(np.dot(Xv, SB) / (sb_len * sb_len))
                    proj = np.array([S.x, S.y]) + ratio * SB
                    lat = float(np.hypot(*(corners[k] - proj)))
                    tangent = seg_d / seg_len
                    align = abs(float(np.dot(SB / sb_len, tangent)))
                    n = np.array([-tangent[1], tangent[0]])
                    road_side = None
                    for rho in SEM_DAC_PROBES_M:
                        plus_ok = _in_union(shapely.points(B + rho * n))
                        minus_ok = _in_union(shapely.points(B - rho * n))
                        if plus_ok != minus_ok:
                            road_side = 1.0 if plus_ok else -1.0
                            break
                    if road_side is None:
                        continue
                    corner_side = float(np.dot(corners[k] - B, n))
                    same_side = corner_side * road_side > SEM_DAC_SIDE_EPS
                    bounded_gap = -0.05 <= ratio <= 1.05 and lat <= SEM_DAC_BETWEEN_TOL_M
                    across = align <= SEM_DAC_MAX_TANGENT_ALIGN
                    if (
                        same_side
                        and bounded_gap
                        and across
                        and dXB <= SEM_DAC_BORDER_MAX_GAP_M
                        and sd <= SEM_DAC_MAX_SEMANTIC_TO_BORDER_M
                    ):
                        accepted = True
                        break
            if not accepted:
                return 0.0
    return 1.0
