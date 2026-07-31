"""Numpy core for the bridge-based state perturbation augmentation.

Ported from the reference implementation in ``trajectory_data_augmentation_test``
(trajectory_augmentation/core.py), trimmed to the parts needed at training time
and adapted to arbitrary past/future horizons taken from the data:

- quintic (C2) lateral offset profile with merge-point search (regula falsi)
- progress-based speed consistency after the merge
- bidirectional augmentation (reversed past segment, sign-flipped heading offset)
- constraint diagnostics: lateral accel, bridge speed gap, bridge jerk
- fast feasibility search: binary search over bridge times with a coarse-scan
  fallback for non-monotone cases
- history decorrelation: bridge-time randomization and a single sin^3 past bump
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DT = 0.1
MIN_BRIDGE_TIME_S = 0.1

# Fixed sampling ranges for the single random past-history bump.
BUMP_AMPLITUDE_RANGE_M = (0.05, 0.20)
BUMP_DURATION_RANGE_S = (1.0, 2.0)
BUMP_MIN_START_TIME_S = 0.3


@dataclass
class Trajectory2D:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray

    def slice(self, start: int, end: int | None = None) -> "Trajectory2D":
        return Trajectory2D(
            t=self.t[start:end].copy(),
            x=self.x[start:end].copy(),
            y=self.y[start:end].copy(),
            yaw=self.yaw[start:end].copy(),
        )


@dataclass
class Centerline:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass
class DensePath:
    sigma: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass
class LateralBump:
    """Lateral perturbation applied to the past history.

    Times are expressed in the reversed-past segment time axis: ``start_time_s``
    seconds before t0, extending further into the past by ``duration_s``.
    """

    start_time_s: float
    duration_s: float
    amplitude_m: float


@dataclass
class SegmentAugmentationResult:
    original_segment: Trajectory2D
    augmented_segment: Trajectory2D
    distance_profile: np.ndarray
    progress_profile: np.ndarray
    exact_speed_profile: np.ndarray
    connect_time_s: float
    merge_centerline_s: float
    merge_path_length_m: float
    connect_speed_scale: float


@dataclass
class BidirectionalAugmentationResult:
    original_full: Trajectory2D
    augmented_full: Trajectory2D
    past_segment: SegmentAugmentationResult
    future_segment: SegmentAugmentationResult
    current_index: int
    lateral_offset_m: float
    heading_offset_rad: float
    past_connect_time_s: float
    future_recover_time_s: float
    past_lateral_bump: LateralBump | None = None


@dataclass
class ConstraintDiagnostics:
    max_abs_augmented_lateral_accel_mps2: float
    max_bridge_speed_gap_mps: float
    max_abs_bridge_jerk_mps3: float
    lateral_accel_limit_mps2: float
    speed_gap_limit_mps: float
    jerk_limit_mps3: float
    lateral_accel_passes: bool
    speed_gap_passes: bool
    jerk_passes: bool
    passes: bool


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=float)
    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    s = np.zeros_like(x, dtype=float)
    s[1:] = np.cumsum(segment_lengths)
    return s


def strictly_increasing_param(param: np.ndarray) -> np.ndarray:
    fixed = np.asarray(param, dtype=float).copy()
    if len(fixed) < 2 or np.all(np.diff(fixed) > 0.0):
        return fixed
    for idx in range(1, len(fixed)):
        if fixed[idx] <= fixed[idx - 1]:
            fixed[idx] = fixed[idx - 1] + 1.0e-6
    return fixed


def heading_from_xy(x: np.ndarray, y: np.ndarray, param: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        if len(x) == 0:
            return np.array([], dtype=float)
        if len(x) == 1:
            return np.array([0.0], dtype=float)
        yaw = np.arctan2(y[1] - y[0], x[1] - x[0])
        return np.array([yaw, yaw], dtype=float)
    param = strictly_increasing_param(param)
    dx = np.gradient(x, param, edge_order=2)
    dy = np.gradient(y, param, edge_order=2)
    return np.unwrap(np.arctan2(dy, dx))


def curvature_from_xy(x: np.ndarray, y: np.ndarray, param: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x, dtype=float)
    param = strictly_increasing_param(param)
    dx = np.gradient(x, param, edge_order=2)
    dy = np.gradient(y, param, edge_order=2)
    ddx = np.gradient(dx, param, edge_order=2)
    ddy = np.gradient(dy, param, edge_order=2)
    denom = np.maximum(dx * dx + dy * dy, 1.0e-9) ** 1.5
    return (dx * ddy - dy * ddx) / denom


def compute_forward_yaw(
    x: np.ndarray, y: np.ndarray, fallback_yaw: np.ndarray | None = None
) -> np.ndarray:
    sigma = cumulative_distance(x, y)
    if len(x) >= 3 and sigma[-1] > 1.0e-6:
        return wrap_angle(heading_from_xy(x, y, sigma))
    if fallback_yaw is not None:
        return wrap_angle(np.asarray(fallback_yaw, dtype=float))
    return np.zeros_like(x, dtype=float)


def speed_from_progress(progress: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(time) < 3:
        return np.zeros_like(time, dtype=float)
    return np.gradient(progress, strictly_increasing_param(time), edge_order=2)


def acceleration_from_speed(speed: np.ndarray, time: np.ndarray) -> np.ndarray:
    return speed_from_progress(speed, time)


def jerk_from_acceleration(acceleration: np.ndarray, time: np.ndarray) -> np.ndarray:
    return speed_from_progress(acceleration, time)


def build_progress_time_lookup(
    progress: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    progress = np.asarray(progress, dtype=float)
    time = np.asarray(time, dtype=float)
    unique_progress, first_indices = np.unique(progress, return_index=True)
    return unique_progress, time[first_indices]


def build_centerline(segment: Trajectory2D, dense_ds: float = 0.05) -> Centerline:
    s_samples = cumulative_distance(segment.x, segment.y)
    dense_s = np.arange(0.0, s_samples[-1] + dense_ds * 0.5, dense_ds)
    yaw_samples = np.unwrap(segment.yaw)
    dense_x = np.interp(dense_s, s_samples, segment.x)
    dense_y = np.interp(dense_s, s_samples, segment.y)
    dense_yaw = np.interp(dense_s, s_samples, yaw_samples)
    return Centerline(s=dense_s, x=dense_x, y=dense_y, yaw=dense_yaw)


def sample_centerline(
    centerline: Centerline, s_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_clamped = np.clip(s_query, centerline.s[0], centerline.s[-1])
    x = np.interp(s_clamped, centerline.s, centerline.x)
    y = np.interp(s_clamped, centerline.s, centerline.y)
    yaw = np.interp(s_clamped, centerline.s, centerline.yaw)
    return x, y, yaw


def solve_lateral_profile_coeffs(
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float,
) -> np.ndarray:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    # Quintic profile in arc length satisfying l(0), l'(0), l''(0), l(L),
    # l'(L), l''(L). l'(0) controls the initial heading error.
    L = float(s_merge)
    a0 = float(lateral_offset_m)
    a1 = float(np.tan(heading_offset_rad))
    a2 = 0.0

    # Closed-form minimum-jerk coefficients for l(L) = l'(L) = l''(L) = 0.
    acc0 = 2.0 * a2
    a3 = (-20.0 * a0 - 12.0 * a1 * L - 3.0 * acc0 * L**2) / (2.0 * L**3)
    a4 = (30.0 * a0 + 16.0 * a1 * L + 3.0 * acc0 * L**2) / (2.0 * L**4)
    a5 = (-12.0 * a0 - 6.0 * a1 * L - acc0 * L**2) / (2.0 * L**5)
    return np.array([a0, a1, a2, a3, a4, a5], dtype=float)


def lateral_offset_profile(
    s: np.ndarray,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
) -> np.ndarray:
    coeffs = solve_lateral_profile_coeffs(
        s_merge=s_merge,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
    )
    result = np.full_like(np.asarray(s, dtype=float), coeffs[5])
    for coeff in coeffs[4::-1]:
        result = result * s + coeff
    return result


def _merge_path_positions(
    centerline: Centerline,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float,
    dense_ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    s_segment = np.arange(0.0, s_merge, dense_ds)
    if len(s_segment) == 0 or not np.isclose(s_segment[-1], s_merge):
        s_segment = np.append(s_segment, s_merge)

    base_x, base_y, base_yaw = sample_centerline(centerline, s_segment)
    offset = lateral_offset_profile(
        s_segment,
        s_merge,
        lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
    )
    x = base_x - offset * np.sin(base_yaw)
    y = base_y + offset * np.cos(base_yaw)
    return x, y


def build_merge_path(
    centerline: Centerline,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
) -> DensePath:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")
    x, y = _merge_path_positions(
        centerline, s_merge, lateral_offset_m, heading_offset_rad, dense_ds
    )
    sigma = cumulative_distance(x, y)
    yaw = heading_from_xy(x, y, sigma)
    return DensePath(sigma=sigma, x=x, y=y, yaw=yaw)


def merge_path_length(
    centerline: Centerline,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
) -> float:
    """Polyline length of the merge path, without building yaw or a DensePath."""
    x, y = _merge_path_positions(
        centerline, s_merge, lateral_offset_m, heading_offset_rad, dense_ds
    )
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def solve_merge_centerline_s(
    centerline: Centerline,
    distance_budget_m: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
    tol_m: float = 1.0e-3,
) -> tuple[float, DensePath]:
    if abs(lateral_offset_m) < 1.0e-9 and abs(heading_offset_rad) < 1.0e-9:
        merge_path = build_merge_path(
            centerline,
            distance_budget_m,
            lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        )
        return distance_budget_m, merge_path

    upper = min(distance_budget_m, centerline.s[-1])
    lower = min(max(dense_ds, abs(lateral_offset_m) * 0.25), upper * 0.5)

    def length_for(s_merge: float) -> float:
        return merge_path_length(
            centerline,
            s_merge,
            lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        )

    lower_length = length_for(lower)
    while lower > dense_ds * 1.0e-3 and lower_length >= distance_budget_m:
        lower *= 0.5
        lower_length = length_for(lower)

    upper_length = length_for(upper)
    if upper_length < distance_budget_m:
        raise RuntimeError("Unable to find a feasible merge point within the distance budget.")

    # length(s) is nearly linear in s, so regula falsi converges in a few
    # iterations where plain bisection needs ~log2(range / tol).
    for _ in range(30):
        if upper - lower < tol_m:
            break
        denominator = upper_length - lower_length
        if denominator <= 1.0e-12:
            mid = 0.5 * (lower + upper)
        else:
            mid = lower + (distance_budget_m - lower_length) * (upper - lower) / denominator
            mid = min(max(mid, lower + 0.05 * tol_m), upper - 0.05 * tol_m)
        mid_length = length_for(mid)
        if abs(mid_length - distance_budget_m) < 0.1 * tol_m:
            lower = upper = mid
            break
        if mid_length < distance_budget_m:
            lower, lower_length = mid, mid_length
        else:
            upper, upper_length = mid, mid_length

    s_merge = 0.5 * (lower + upper)
    merge_path = build_merge_path(
        centerline,
        s_merge,
        lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )
    return s_merge, merge_path


def plan_recovery_path(
    centerline: Centerline,
    distance_budget_m: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
) -> tuple[float, DensePath, float]:
    candidate_length = merge_path_length(
        centerline=centerline,
        s_merge=distance_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )
    if candidate_length <= distance_budget_m + 1.0e-3:
        candidate_path = build_merge_path(
            centerline=centerline,
            s_merge=distance_budget_m,
            lateral_offset_m=lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        )
        speed_scale = candidate_length / max(distance_budget_m, 1.0e-9)
        return distance_budget_m, candidate_path, speed_scale

    s_merge, merge_path = solve_merge_centerline_s(
        centerline=centerline,
        distance_budget_m=distance_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )
    return s_merge, merge_path, 1.0


def build_full_augmented_path(
    centerline: Centerline,
    merge_path: DensePath,
    s_merge: float,
    total_distance_m: float,
    dense_ds: float = 0.05,
) -> DensePath:
    continuation_s = np.arange(s_merge, total_distance_m, dense_ds)
    if len(continuation_s) == 0 or not np.isclose(continuation_s[-1], total_distance_m):
        continuation_s = np.append(continuation_s, total_distance_m)

    cont_x, cont_y, _ = sample_centerline(centerline, continuation_s)
    continuation_sigma = merge_path.sigma[-1] + (continuation_s - s_merge)

    full_sigma = np.concatenate((merge_path.sigma, continuation_sigma[1:]))
    full_x = np.concatenate((merge_path.x, cont_x[1:]))
    full_y = np.concatenate((merge_path.y, cont_y[1:]))
    full_yaw = heading_from_xy(full_x, full_y, full_sigma)
    return DensePath(sigma=full_sigma, x=full_x, y=full_y, yaw=full_yaw)


def sample_dense_path(
    path: DensePath, sigma_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(path.sigma) == 0:
        raise ValueError("DensePath must contain at least one sample.")
    sigma = strictly_increasing_param(path.sigma)
    sigma_clamped = np.clip(sigma_query, sigma[0], sigma[-1])
    x = np.interp(sigma_clamped, sigma, path.x)
    y = np.interp(sigma_clamped, sigma, path.y)
    yaw = wrap_angle(np.interp(sigma_clamped, sigma, np.unwrap(path.yaw)))
    return x, y, yaw


def lateral_bump_window(unit_value: np.ndarray) -> np.ndarray:
    # sin^3 is zero-valued with zero first and second derivatives at both
    # edges, so the bump joins the surrounding path with C2 continuity.
    u = np.clip(unit_value, 0.0, 1.0)
    return np.sin(np.pi * u) ** 3


def apply_lateral_bump_to_dense_path(
    path: DensePath,
    progress_profile: np.ndarray,
    segment_time: np.ndarray,
    bump: LateralBump,
    min_bump_length_m: float = 0.5,
) -> tuple[DensePath, np.ndarray]:
    old_sigma = strictly_increasing_param(path.sigma)
    bump_start_s = float(np.interp(bump.start_time_s, segment_time, progress_profile))
    bump_end_s = float(
        np.interp(bump.start_time_s + bump.duration_s, segment_time, progress_profile)
    )
    if bump_end_s - bump_start_s < min_bump_length_m:
        return path, progress_profile
    u = (old_sigma - bump_start_s) / (bump_end_s - bump_start_s)
    offset = bump.amplitude_m * lateral_bump_window(u)

    normal_x = -np.sin(path.yaw)
    normal_y = np.cos(path.yaw)
    new_x = path.x + offset * normal_x
    new_y = path.y + offset * normal_y
    new_sigma = cumulative_distance(new_x, new_y)
    new_yaw = heading_from_xy(new_x, new_y, new_sigma)
    # Progress values refer to the unperturbed arc length; remap them so the
    # vehicle stays aligned with GT along track and the extra bump length shows
    # up as a locally higher exact-arc speed.
    new_progress = np.interp(progress_profile, old_sigma, new_sigma)
    return DensePath(sigma=new_sigma, x=new_x, y=new_y, yaw=new_yaw), new_progress


def sample_past_history_bump(
    rng: np.random.Generator,
    past_horizon_s: float,
) -> LateralBump | None:
    # The bump may overlap the past bridge window; every sampled candidate is
    # re-checked against the bridge constraints, so only t0 needs protecting.
    available_s = past_horizon_s - BUMP_MIN_START_TIME_S
    if available_s < BUMP_DURATION_RANGE_S[0]:
        return None
    duration_s = float(
        rng.uniform(BUMP_DURATION_RANGE_S[0], min(BUMP_DURATION_RANGE_S[1], available_s))
    )
    start_time_s = float(rng.uniform(BUMP_MIN_START_TIME_S, past_horizon_s - duration_s))
    amplitude_m = float(rng.uniform(*BUMP_AMPLITUDE_RANGE_M) * rng.choice(np.array([-1.0, 1.0])))
    return LateralBump(
        start_time_s=start_time_s,
        duration_s=duration_s,
        amplitude_m=amplitude_m,
    )


def extract_future_segment(gt: Trajectory2D, current_index: int) -> Trajectory2D:
    future = gt.slice(current_index, None)
    return Trajectory2D(
        t=future.t - future.t[0],
        x=future.x.copy(),
        y=future.y.copy(),
        yaw=future.yaw.copy(),
    )


def extract_reversed_past_segment(gt: Trajectory2D, current_index: int) -> Trajectory2D:
    past = gt.slice(0, current_index + 1)
    return Trajectory2D(
        t=-past.t[::-1],
        x=past.x[::-1].copy(),
        y=past.y[::-1].copy(),
        yaw=past.yaw[::-1].copy(),
    )


def augment_directed_segment(
    segment: Trajectory2D,
    lateral_offset_m: float,
    heading_offset_rad: float,
    connect_time_s: float,
    dense_ds: float = 0.05,
    lateral_bump: LateralBump | None = None,
) -> SegmentAugmentationResult:
    if not 0.0 < connect_time_s <= segment.t[-1]:
        raise ValueError("connect_time_s must be within the segment horizon.")

    centerline = build_centerline(segment, dense_ds=dense_ds)
    distance_profile = cumulative_distance(segment.x, segment.y)
    progress_time_samples, progress_time_lookup = build_progress_time_lookup(
        distance_profile, segment.t
    )
    connect_budget_m = float(np.interp(connect_time_s, segment.t, distance_profile))
    total_distance_m = float(distance_profile[-1])
    s_merge, merge_path, connect_speed_scale = plan_recovery_path(
        centerline=centerline,
        distance_budget_m=connect_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )

    dense_full_path = build_full_augmented_path(
        centerline=centerline,
        merge_path=merge_path,
        s_merge=s_merge,
        total_distance_m=total_distance_m,
        dense_ds=dense_ds,
    )

    connect_mask = segment.t <= connect_time_s + 1.0e-9
    progress_profile = distance_profile.copy()
    if connect_budget_m > 1.0e-9:
        progress_profile[connect_mask] = distance_profile[connect_mask] * connect_speed_scale
    else:
        progress_profile[connect_mask] = 0.0

    post_indices = np.where(~connect_mask)[0]
    merge_path_length_m = float(merge_path.sigma[-1]) if len(merge_path.sigma) > 0 else 0.0
    if len(post_indices) > 0:
        merge_time_on_gt = float(np.interp(s_merge, progress_time_samples, progress_time_lookup))
        shifted_gt_time = merge_time_on_gt + (segment.t[post_indices] - connect_time_s)
        shifted_gt_time = np.clip(shifted_gt_time, segment.t[0], segment.t[-1])
        centerline_progress = np.interp(shifted_gt_time, segment.t, distance_profile)
        progress_profile[post_indices] = merge_path_length_m + (centerline_progress - s_merge)

    progress_profile = np.clip(progress_profile, 0.0, dense_full_path.sigma[-1])
    if lateral_bump is not None:
        dense_full_path, progress_profile = apply_lateral_bump_to_dense_path(
            dense_full_path,
            progress_profile,
            segment.t,
            lateral_bump,
        )
        progress_profile = np.clip(progress_profile, 0.0, dense_full_path.sigma[-1])
    query_x, query_y, directional_yaw = sample_dense_path(dense_full_path, progress_profile)
    exact_speed_profile = speed_from_progress(progress_profile, segment.t)
    augmented_segment = Trajectory2D(
        t=segment.t.copy(),
        x=query_x,
        y=query_y,
        yaw=directional_yaw,
    )

    return SegmentAugmentationResult(
        original_segment=segment,
        augmented_segment=augmented_segment,
        distance_profile=distance_profile,
        progress_profile=progress_profile,
        exact_speed_profile=exact_speed_profile,
        connect_time_s=connect_time_s,
        merge_centerline_s=s_merge,
        merge_path_length_m=merge_path_length_m,
        connect_speed_scale=connect_speed_scale,
    )


def augment_trajectory_bidirectional(
    gt: Trajectory2D,
    current_index: int,
    lateral_offset_m: float,
    heading_offset_rad: float,
    future_recover_time_s: float,
    past_connect_time_s: float,
    dense_ds: float = 0.05,
    past_lateral_bump: LateralBump | None = None,
) -> BidirectionalAugmentationResult:
    future_segment = extract_future_segment(gt, current_index)
    past_reverse_segment = extract_reversed_past_segment(gt, current_index)

    future_result = augment_directed_segment(
        segment=future_segment,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        connect_time_s=future_recover_time_s,
        dense_ds=dense_ds,
    )
    past_result = augment_directed_segment(
        segment=past_reverse_segment,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=-heading_offset_rad,
        connect_time_s=past_connect_time_s,
        dense_ds=dense_ds,
        lateral_bump=past_lateral_bump,
    )

    augmented_past_x = past_result.augmented_segment.x[::-1]
    augmented_past_y = past_result.augmented_segment.y[::-1]

    augmented_full_x = gt.x.copy()
    augmented_full_y = gt.y.copy()
    augmented_full_x[:current_index] = augmented_past_x[:-1]
    augmented_full_y[:current_index] = augmented_past_y[:-1]
    augmented_full_x[current_index:] = future_result.augmented_segment.x
    augmented_full_y[current_index:] = future_result.augmented_segment.y

    augmented_full_yaw = compute_forward_yaw(
        augmented_full_x, augmented_full_y, fallback_yaw=gt.yaw
    )
    augmented_full = Trajectory2D(
        t=gt.t.copy(),
        x=augmented_full_x,
        y=augmented_full_y,
        yaw=augmented_full_yaw,
    )

    return BidirectionalAugmentationResult(
        original_full=gt.slice(0, None),
        augmented_full=augmented_full,
        past_segment=past_result,
        future_segment=future_result,
        current_index=current_index,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        past_connect_time_s=past_connect_time_s,
        future_recover_time_s=future_recover_time_s,
        past_lateral_bump=past_lateral_bump,
    )


def _full_series_from_segments(past_series: np.ndarray, future_series: np.ndarray) -> np.ndarray:
    return np.concatenate((past_series[::-1][:-1], future_series))


def evaluate_constraints(
    result: BidirectionalAugmentationResult,
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
) -> ConstraintDiagnostics:
    past = result.past_segment
    future = result.future_segment

    past_gt_speed = speed_from_progress(past.distance_profile, past.original_segment.t)
    future_gt_speed = speed_from_progress(future.distance_profile, future.original_segment.t)
    augmented_arc_speed = _full_series_from_segments(
        past.exact_speed_profile, future.exact_speed_profile
    )

    augmented_sigma = cumulative_distance(result.augmented_full.x, result.augmented_full.y)
    augmented_curvature = curvature_from_xy(
        result.augmented_full.x, result.augmented_full.y, augmented_sigma
    )
    augmented_lateral_accel = augmented_arc_speed * augmented_arc_speed * augmented_curvature
    max_abs_augmented_lateral_accel = float(np.max(np.abs(augmented_lateral_accel)))

    speed_gap = _full_series_from_segments(
        np.abs(past.exact_speed_profile - past_gt_speed),
        np.abs(future.exact_speed_profile - future_gt_speed),
    )
    past_acceleration = acceleration_from_speed(past.exact_speed_profile, past.original_segment.t)
    past_jerk = jerk_from_acceleration(past_acceleration, past.original_segment.t)
    future_acceleration = acceleration_from_speed(
        future.exact_speed_profile, future.original_segment.t
    )
    future_jerk = jerk_from_acceleration(future_acceleration, future.original_segment.t)
    augmented_jerk = _full_series_from_segments(past_jerk, future_jerk)

    full_time = result.original_full.t
    bridge_mask = (full_time >= -result.past_connect_time_s - 1.0e-9) & (
        full_time <= result.future_recover_time_s + 1.0e-9
    )
    max_speed_gap = float(np.max(speed_gap[bridge_mask])) if np.any(bridge_mask) else 0.0
    max_abs_bridge_jerk = (
        float(np.max(np.abs(augmented_jerk[bridge_mask]))) if np.any(bridge_mask) else 0.0
    )

    lateral_accel_passes = max_abs_augmented_lateral_accel <= float(max_lateral_accel_mps2) + 1.0e-9
    speed_gap_passes = max_speed_gap <= float(max_bridge_speed_gap_mps) + 1.0e-9
    jerk_passes = max_abs_bridge_jerk <= float(max_bridge_jerk_mps3) + 1.0e-9

    return ConstraintDiagnostics(
        max_abs_augmented_lateral_accel_mps2=max_abs_augmented_lateral_accel,
        max_bridge_speed_gap_mps=max_speed_gap,
        max_abs_bridge_jerk_mps3=max_abs_bridge_jerk,
        lateral_accel_limit_mps2=float(max_lateral_accel_mps2),
        speed_gap_limit_mps=float(max_bridge_speed_gap_mps),
        jerk_limit_mps3=float(max_bridge_jerk_mps3),
        lateral_accel_passes=lateral_accel_passes,
        speed_gap_passes=speed_gap_passes,
        jerk_passes=jerk_passes,
        passes=lateral_accel_passes and speed_gap_passes and jerk_passes,
    )


def generate_time_candidates(
    start_s: float,
    end_s: float,
    step_s: float = DEFAULT_DT,
) -> list[float]:
    start_tick = int(np.ceil((start_s - 1.0e-9) / step_s))
    end_tick = int(np.floor((end_s + 1.0e-9) / step_s))
    return [tick * step_s for tick in range(start_tick, end_tick + 1)]


def search_min_passing_index(
    num_candidates: int,
    passes_at,
    coarse_stride: int = 5,
) -> int | None:
    """Find the smallest passing index, assuming feasibility is mostly monotone.

    Longer bridge times are usually gentler, so passing candidates form a
    suffix and a binary search finds the boundary in O(log n) evaluations. If
    the last candidate fails (a non-monotone feasibility pocket, or nothing
    feasible), fall back to a coarse linear scan refined locally.
    """
    if num_candidates <= 0:
        return None

    memo: dict[int, bool] = {}

    def passes(index: int) -> bool:
        if index not in memo:
            memo[index] = passes_at(index)
        return memo[index]

    if passes(num_candidates - 1):
        low = 0
        high = num_candidates - 1
        while low < high:
            mid = (low + high) // 2
            if passes(mid):
                high = mid
            else:
                low = mid + 1
        return high

    for coarse_index in range(0, num_candidates, coarse_stride):
        if passes(coarse_index):
            for index in range(max(0, coarse_index - coarse_stride + 1), coarse_index + 1):
                if passes(index):
                    return index
    return None


@dataclass
class FeasibleSearchOutcome:
    result: BidirectionalAugmentationResult
    diagnostics: ConstraintDiagnostics


def search_feasible_result(
    initial_result: BidirectionalAugmentationResult,
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
    max_future_recover_time_s: float,
    max_past_connect_time_s: float,
    search_step_s: float = DEFAULT_DT,
) -> FeasibleSearchOutcome | None:
    """Return the first feasible candidate, or None if nothing passes.

    If the initial result already passes it is returned as-is; otherwise the
    minimal feasible N (then minimal (M, N)) is searched on the step grid.
    """
    initial_diag = evaluate_constraints(
        initial_result,
        max_lateral_accel_mps2=max_lateral_accel_mps2,
        max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
        max_bridge_jerk_mps3=max_bridge_jerk_mps3,
    )
    if initial_diag.passes:
        return FeasibleSearchOutcome(result=initial_result, diagnostics=initial_diag)

    evaluated: dict[tuple[float, float], FeasibleSearchOutcome] = {}

    def evaluate_candidate(
        recover_time_s: float, past_connect_time_s: float
    ) -> FeasibleSearchOutcome:
        key = (round(recover_time_s, 6), round(past_connect_time_s, 6))
        if key not in evaluated:
            candidate_result = augment_trajectory_bidirectional(
                gt=initial_result.original_full,
                current_index=initial_result.current_index,
                lateral_offset_m=initial_result.lateral_offset_m,
                heading_offset_rad=initial_result.heading_offset_rad,
                future_recover_time_s=recover_time_s,
                past_connect_time_s=past_connect_time_s,
                past_lateral_bump=initial_result.past_lateral_bump,
            )
            candidate_diag = evaluate_constraints(
                candidate_result,
                max_lateral_accel_mps2=max_lateral_accel_mps2,
                max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
                max_bridge_jerk_mps3=max_bridge_jerk_mps3,
            )
            evaluated[key] = FeasibleSearchOutcome(
                result=candidate_result, diagnostics=candidate_diag
            )
        return evaluated[key]

    initial_n = initial_result.future_recover_time_s
    initial_m = initial_result.past_connect_time_s

    future_candidates = generate_time_candidates(
        start_s=initial_n + search_step_s,
        end_s=max_future_recover_time_s,
        step_s=search_step_s,
    )
    future_index = search_min_passing_index(
        len(future_candidates),
        lambda index: evaluate_candidate(future_candidates[index], initial_m).diagnostics.passes,
    )
    if future_index is not None:
        return evaluate_candidate(future_candidates[future_index], initial_m)

    past_candidates = generate_time_candidates(
        start_s=initial_m + search_step_s,
        end_s=max_past_connect_time_s,
        step_s=search_step_s,
    )
    future_with_past_candidates = generate_time_candidates(
        start_s=initial_n,
        end_s=max_future_recover_time_s,
        step_s=search_step_s,
    )
    if past_candidates and future_with_past_candidates:
        max_n = future_with_past_candidates[-1]
        past_index = search_min_passing_index(
            len(past_candidates),
            lambda index: evaluate_candidate(max_n, past_candidates[index]).diagnostics.passes,
        )
        if past_index is not None:
            candidate_m = past_candidates[past_index]
            future_index = search_min_passing_index(
                len(future_with_past_candidates),
                lambda index: (
                    evaluate_candidate(
                        future_with_past_candidates[index], candidate_m
                    ).diagnostics.passes
                ),
            )
            if future_index is not None:
                return evaluate_candidate(future_with_past_candidates[future_index], candidate_m)

    return None


def snap_time_to_grid(value_s: float, step_s: float = DEFAULT_DT) -> float:
    return round(value_s / step_s) * step_s


def randomize_bridge_times_result(
    rng: np.random.Generator,
    base_result: BidirectionalAugmentationResult,
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
    max_future_recover_time_s: float,
    max_past_connect_time_s: float,
    extra_time_range_s: float = 1.5,
    max_attempts: int = 10,
    step_s: float = DEFAULT_DT,
) -> FeasibleSearchOutcome | None:
    """Sample random feasible bridge times at or above the given minimums."""
    min_n = base_result.future_recover_time_s
    min_m = base_result.past_connect_time_s
    for _ in range(max_attempts):
        candidate_n = snap_time_to_grid(min_n + float(rng.uniform(0.0, extra_time_range_s)), step_s)
        candidate_m = snap_time_to_grid(min_m + float(rng.uniform(0.0, extra_time_range_s)), step_s)
        candidate_n = float(np.clip(candidate_n, min_n, max_future_recover_time_s))
        candidate_m = float(np.clip(candidate_m, min_m, max_past_connect_time_s))
        candidate_result = augment_trajectory_bidirectional(
            gt=base_result.original_full,
            current_index=base_result.current_index,
            lateral_offset_m=base_result.lateral_offset_m,
            heading_offset_rad=base_result.heading_offset_rad,
            future_recover_time_s=candidate_n,
            past_connect_time_s=candidate_m,
            past_lateral_bump=base_result.past_lateral_bump,
        )
        candidate_diag = evaluate_constraints(
            candidate_result,
            max_lateral_accel_mps2=max_lateral_accel_mps2,
            max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
            max_bridge_jerk_mps3=max_bridge_jerk_mps3,
        )
        if candidate_diag.passes:
            return FeasibleSearchOutcome(result=candidate_result, diagnostics=candidate_diag)
    return None


def apply_random_past_history_bump(
    rng: np.random.Generator,
    base_result: BidirectionalAugmentationResult,
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
    past_horizon_s: float,
    max_attempts: int = 20,
) -> FeasibleSearchOutcome | None:
    for _ in range(max_attempts):
        bump = sample_past_history_bump(rng, past_horizon_s=past_horizon_s)
        if bump is None:
            return None
        candidate_result = augment_trajectory_bidirectional(
            gt=base_result.original_full,
            current_index=base_result.current_index,
            lateral_offset_m=base_result.lateral_offset_m,
            heading_offset_rad=base_result.heading_offset_rad,
            future_recover_time_s=base_result.future_recover_time_s,
            past_connect_time_s=base_result.past_connect_time_s,
            past_lateral_bump=bump,
        )
        candidate_diag = evaluate_constraints(
            candidate_result,
            max_lateral_accel_mps2=max_lateral_accel_mps2,
            max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
            max_bridge_jerk_mps3=max_bridge_jerk_mps3,
        )
        if candidate_diag.passes:
            return FeasibleSearchOutcome(result=candidate_result, diagnostics=candidate_diag)
    return None
