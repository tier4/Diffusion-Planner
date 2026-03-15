from dataclasses import dataclass

import numpy as np
import torch

TIME_INTERVAL = 0.1
DENSE_SAMPLE_DS = 0.05


@dataclass
class SegmentAugmentationResultTorch:
    query_xy: torch.Tensor
    query_heading: torch.Tensor
    progress: torch.Tensor
    exact_speed: torch.Tensor
    distance_profile: torch.Tensor
    segment_time: torch.Tensor
    merge_centerline_s: float
    merge_path_length_m: float
    connect_distance_budget_m: float
    connect_speed_scale: float


@dataclass
class ConstraintDiagnosticsTorch:
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


@dataclass
class AugmentedSampleTorchResult:
    aug_current_state: torch.Tensor
    aug_past: torch.Tensor
    aug_future: torch.Tensor
    original_full_xy: torch.Tensor
    augmented_full_xy: torch.Tensor
    original_full_heading: torch.Tensor
    augmented_full_heading: torch.Tensor
    full_time: torch.Tensor
    past_segment: SegmentAugmentationResultTorch
    future_segment: SegmentAugmentationResultTorch
    past_connect_time_s: float
    future_recover_time_s: float


def vector_transform(vector, transform_mat, bias=None):
    """
    vector: (B, ..., 2)
    transform_mat: (B, 2, 2)
    bias: (B, ..., 2)
    """
    shape = vector.shape
    B = vector.shape[0]
    nexpand = vector.ndim - 2
    if bias is not None:
        vector = vector - bias.reshape(B, *([1] * nexpand), -1)
    vector = vector.reshape(B, -1, 2).permute(0, 2, 1)  # (B, 2, N1 * N2 ...)
    return torch.bmm(transform_mat, vector).permute(0, 2, 1).reshape(*shape)  # (B, ..., 2)


def heading_transform(heading, transform_mat):
    """
    heading: (B, ...)
    transform_mat: (B, 2, 2)
    """
    B = heading.shape[0]
    shape = heading.shape
    heading = heading.reshape(B, -1)
    transform_mat = transform_mat.reshape(B, 1, 2, 2)
    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0]
        + torch.sin(heading) * transform_mat[..., 1, 1],
        torch.cos(heading) * transform_mat[..., 0, 0]
        + torch.sin(heading) * transform_mat[..., 0, 1],
    ).reshape(*shape)


def normalize_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def cumulative_distance_torch(xy: torch.Tensor) -> torch.Tensor:
    if xy.shape[0] == 0:
        return torch.zeros(0, dtype=xy.dtype, device=xy.device)
    deltas = torch.diff(xy, dim=0)
    segment_lengths = torch.linalg.norm(deltas, dim=-1)
    distance = torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)
    distance[1:] = torch.cumsum(segment_lengths, dim=0)
    return distance


def strictly_increasing_param_torch(param: torch.Tensor) -> torch.Tensor:
    fixed = param.clone()
    for idx in range(1, fixed.shape[0]):
        if fixed[idx] <= fixed[idx - 1]:
            fixed[idx] = fixed[idx - 1] + 1.0e-6
    return fixed


def interp1d_torch(x: torch.Tensor, y: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    if x.numel() == 1:
        return y[0].expand_as(xq) if y.ndim == 1 else y[0].expand(xq.shape[0], *y.shape[1:])

    xq_clamped = torch.clamp(xq, min=x[0], max=x[-1])
    idx = torch.searchsorted(x, xq_clamped, right=False)
    idx = torch.clamp(idx, 1, x.shape[0] - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    weight = (xq_clamped - x0) / torch.clamp(x1 - x0, min=1.0e-6)

    if y.ndim == 1:
        y0 = y[idx - 1]
        y1 = y[idx]
        return y0 + weight * (y1 - y0)

    y0 = y[idx - 1]
    y1 = y[idx]
    return y0 + weight.unsqueeze(-1) * (y1 - y0)


def interp_heading_torch(
    x: torch.Tensor, heading: torch.Tensor, xq: torch.Tensor
) -> torch.Tensor:
    cos_interp = interp1d_torch(x, torch.cos(heading), xq)
    sin_interp = interp1d_torch(x, torch.sin(heading), xq)
    return torch.atan2(sin_interp, cos_interp)


def build_progress_time_lookup_torch(
    progress: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.ones_like(progress, dtype=torch.bool)
    if progress.shape[0] > 1:
        keep[1:] = progress[1:] > progress[:-1] + 1.0e-9
    return progress[keep], time[keep]


def heading_from_positions_torch(
    xy: torch.Tensor, fallback_heading: torch.Tensor | None = None
) -> torch.Tensor:
    num_points = xy.shape[0]
    if num_points == 0:
        return torch.zeros(0, dtype=xy.dtype, device=xy.device)
    if num_points == 1:
        if fallback_heading is not None:
            return fallback_heading[:1]
        return torch.zeros(1, dtype=xy.dtype, device=xy.device)

    heading = torch.zeros(num_points, dtype=xy.dtype, device=xy.device)
    first_delta = xy[1] - xy[0]
    last_delta = xy[-1] - xy[-2]
    heading[0] = torch.atan2(first_delta[1], first_delta[0])
    heading[-1] = torch.atan2(last_delta[1], last_delta[0])
    if num_points > 2:
        middle_delta = xy[2:] - xy[:-2]
        heading[1:-1] = torch.atan2(middle_delta[:, 1], middle_delta[:, 0])

    if fallback_heading is not None:
        delta_norm = torch.zeros(num_points, dtype=xy.dtype, device=xy.device)
        delta_norm[0] = torch.linalg.norm(first_delta)
        delta_norm[-1] = torch.linalg.norm(last_delta)
        if num_points > 2:
            delta_norm[1:-1] = torch.linalg.norm(middle_delta, dim=-1)
        heading = torch.where(delta_norm < 1.0e-6, fallback_heading, heading)

    return normalize_angle_torch(heading)


def speed_from_progress_torch(progress: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    if time.shape[0] < 3:
        return torch.zeros_like(time)
    time = strictly_increasing_param_torch(time)
    speed = torch.zeros_like(progress)
    speed[0] = (progress[1] - progress[0]) / (time[1] - time[0])
    speed[-1] = (progress[-1] - progress[-2]) / (time[-1] - time[-2])
    speed[1:-1] = (progress[2:] - progress[:-2]) / (time[2:] - time[:-2])
    return speed


def acceleration_from_speed_torch(speed: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    return speed_from_progress_torch(speed, time)


def jerk_from_acceleration_torch(acceleration: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    return speed_from_progress_torch(acceleration, time)


def curvature_from_xy_torch(xy: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    if xy.shape[0] < 3:
        return torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)
    param = strictly_increasing_param_torch(param)
    dx = torch.gradient(xy[:, 0], spacing=(param,), edge_order=2)[0]
    dy = torch.gradient(xy[:, 1], spacing=(param,), edge_order=2)[0]
    ddx = torch.gradient(dx, spacing=(param,), edge_order=2)[0]
    ddy = torch.gradient(dy, spacing=(param,), edge_order=2)[0]
    denom = torch.clamp(dx * dx + dy * dy, min=1.0e-9) ** 1.5
    return (dx * ddy - dy * ddx) / denom


def quintic_decay_torch(unit_s: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(unit_s, 0.0, 1.0)
    return 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5


def solve_lateral_profile_coeffs_torch(
    s_merge: torch.Tensor,
    lateral_offset: torch.Tensor,
    heading_offset: torch.Tensor,
) -> torch.Tensor:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    length = s_merge
    a0 = lateral_offset
    a1 = torch.tan(heading_offset)
    a2 = torch.zeros((), dtype=length.dtype, device=length.device)

    system = torch.stack(
        [
            torch.stack([length**3, length**4, length**5]),
            torch.stack([3.0 * length**2, 4.0 * length**3, 5.0 * length**4]),
            torch.stack([6.0 * length, 12.0 * length**2, 20.0 * length**3]),
        ]
    )
    rhs = torch.stack(
        [
            -(a0 + a1 * length + a2 * length**2),
            -(a1 + 2.0 * a2 * length),
            -(2.0 * a2),
        ]
    )
    a3_to_a5 = torch.linalg.solve(system, rhs)
    return torch.stack([a0, a1, a2, a3_to_a5[0], a3_to_a5[1], a3_to_a5[2]])


def lateral_offset_profile_torch(
    s: torch.Tensor,
    s_merge: torch.Tensor,
    lateral_offset: torch.Tensor,
    heading_offset: torch.Tensor,
) -> torch.Tensor:
    coeffs = solve_lateral_profile_coeffs_torch(
        s_merge=s_merge,
        lateral_offset=lateral_offset,
        heading_offset=heading_offset,
    )
    powers = torch.stack([s**idx for idx in range(6)], dim=-1)
    return powers @ coeffs


def sample_centerline_torch(
    center_s: torch.Tensor,
    center_xy: torch.Tensor,
    center_heading: torch.Tensor,
    query_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_xy = interp1d_torch(center_s, center_xy, query_s)
    base_heading = interp_heading_torch(center_s, center_heading, query_s)
    return base_xy, base_heading


def build_offset_path_torch(
    center_xy: torch.Tensor,
    center_heading: torch.Tensor,
    center_s: torch.Tensor,
    s_merge: torch.Tensor,
    lateral_offset: torch.Tensor,
    heading_offset: torch.Tensor,
    dense_ds: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = center_xy.device
    dtype = center_xy.dtype
    num_dense = max(int(torch.ceil(s_merge / dense_ds).item()), 1)
    dense_s = torch.linspace(0.0, float(s_merge.item()), num_dense + 1, device=device, dtype=dtype)
    base_xy, base_heading = sample_centerline_torch(center_s, center_xy, center_heading, dense_s)

    offset = lateral_offset_profile_torch(
        dense_s,
        torch.clamp(s_merge, min=1.0e-6),
        lateral_offset,
        heading_offset,
    )
    normals = torch.stack([-torch.sin(base_heading), torch.cos(base_heading)], dim=-1)
    path_xy = base_xy + offset.unsqueeze(-1) * normals
    path_sigma = cumulative_distance_torch(path_xy)
    path_heading = heading_from_positions_torch(path_xy, fallback_heading=base_heading)
    return path_xy, path_sigma, path_heading


def merge_path_length_torch(
    center_xy: torch.Tensor,
    center_heading: torch.Tensor,
    center_s: torch.Tensor,
    s_merge: torch.Tensor,
    lateral_offset: torch.Tensor,
    heading_offset: torch.Tensor,
    dense_ds: float,
) -> torch.Tensor:
    _, path_sigma, _ = build_offset_path_torch(
        center_xy, center_heading, center_s, s_merge, lateral_offset, heading_offset, dense_ds
    )
    return path_sigma[-1]


def solve_merge_centerline_s_torch(
    center_xy: torch.Tensor,
    center_heading: torch.Tensor,
    center_s: torch.Tensor,
    distance_budget: torch.Tensor,
    lateral_offset: torch.Tensor,
    heading_offset: torch.Tensor,
    dense_ds: float,
    tol_m: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = center_xy.device
    dtype = center_xy.dtype
    upper = torch.minimum(distance_budget, center_s[-1])
    lower = torch.minimum(
        torch.maximum(
            torch.tensor(dense_ds, device=device, dtype=dtype),
            torch.abs(lateral_offset) * 0.25,
        ),
        upper * 0.5,
    )

    while lower > dense_ds * 1.0e-3:
        length_lower = merge_path_length_torch(
            center_xy, center_heading, center_s, lower, lateral_offset, heading_offset, dense_ds
        )
        if length_lower < distance_budget:
            break
        lower = lower * 0.5

    upper_length = merge_path_length_torch(
        center_xy, center_heading, center_s, upper, lateral_offset, heading_offset, dense_ds
    )
    if upper_length < distance_budget:
        raise RuntimeError("Unable to find a feasible merge point within the distance budget.")

    for _ in range(50):
        mid = 0.5 * (lower + upper)
        mid_length = merge_path_length_torch(
            center_xy, center_heading, center_s, mid, lateral_offset, heading_offset, dense_ds
        )
        if mid_length < distance_budget:
            lower = mid
        else:
            upper = mid
        if upper - lower < tol_m:
            break

    s_merge = 0.5 * (lower + upper)
    path_xy, path_sigma, path_heading = build_offset_path_torch(
        center_xy, center_heading, center_s, s_merge, lateral_offset, heading_offset, dense_ds
    )
    return s_merge, path_xy, path_sigma, path_heading


def build_full_augmented_path_torch(
    center_xy: torch.Tensor,
    center_heading: torch.Tensor,
    center_s: torch.Tensor,
    merge_xy: torch.Tensor,
    merge_sigma: torch.Tensor,
    merge_heading: torch.Tensor,
    s_merge: torch.Tensor,
    total_distance_m: torch.Tensor,
    dense_ds: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = center_xy.device
    dtype = center_xy.dtype
    continuation_s = torch.arange(
        float(s_merge.item()),
        float(total_distance_m.item()),
        dense_ds,
        device=device,
        dtype=dtype,
    )
    if continuation_s.numel() == 0 or not torch.isclose(continuation_s[-1], total_distance_m):
        continuation_s = torch.cat([continuation_s, total_distance_m.unsqueeze(0)])

    continuation_xy, continuation_heading = sample_centerline_torch(
        center_s, center_xy, center_heading, continuation_s
    )
    continuation_sigma = merge_sigma[-1] + (continuation_s - s_merge)

    full_sigma = torch.cat([merge_sigma, continuation_sigma[1:]], dim=0)
    full_xy = torch.cat([merge_xy, continuation_xy[1:]], dim=0)
    fallback_heading = torch.cat([merge_heading, continuation_heading[1:]], dim=0)
    full_heading = heading_from_positions_torch(full_xy, fallback_heading=fallback_heading)
    return full_xy, full_sigma, full_heading


def sample_dense_path_torch(
    path_sigma: torch.Tensor,
    path_xy: torch.Tensor,
    path_heading: torch.Tensor,
    query_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if path_sigma.numel() == 0:
        raise ValueError("Dense path must contain at least one sample.")
    sigma = strictly_increasing_param_torch(path_sigma)
    sigma_query = torch.clamp(query_sigma, min=sigma[0], max=sigma[-1])
    xy = interp1d_torch(sigma, path_xy, sigma_query)
    heading = interp_heading_torch(sigma, path_heading, sigma_query)
    return xy, heading


def augment_segment_torch(
    segment_xy: torch.Tensor,
    segment_heading: torch.Tensor,
    segment_time: torch.Tensor,
    lateral_offset: torch.Tensor,
    heading_offset: torch.Tensor,
    connect_time_s: float,
    dense_ds: float,
) -> SegmentAugmentationResultTorch:
    center_s = cumulative_distance_torch(segment_xy)
    # Use GT heading directly so past-bridge generation keeps the same lateral
    # offset convention even when the past segment is traversed in reverse.
    center_heading = normalize_angle_torch(segment_heading)
    connect_time = torch.tensor(connect_time_s, dtype=segment_xy.dtype, device=segment_xy.device)
    connect_time = torch.clamp(connect_time, min=segment_time[0], max=segment_time[-1])
    distance_budget = interp1d_torch(segment_time, center_s, connect_time.unsqueeze(0))[0]
    progress_samples, progress_time_lookup = build_progress_time_lookup_torch(center_s, segment_time)
    total_distance = center_s[-1]

    path_xy_candidate, path_sigma_candidate, path_heading_candidate = build_offset_path_torch(
        center_xy=segment_xy,
        center_heading=center_heading,
        center_s=center_s,
        s_merge=distance_budget,
        lateral_offset=lateral_offset,
        heading_offset=heading_offset,
        dense_ds=dense_ds,
    )
    candidate_length = path_sigma_candidate[-1]

    if candidate_length <= distance_budget + 1.0e-3:
        s_merge = distance_budget
        merge_xy = path_xy_candidate
        merge_sigma = path_sigma_candidate
        merge_heading = path_heading_candidate
        speed_scale = candidate_length / torch.clamp(distance_budget, min=1.0e-6)
    else:
        s_merge, merge_xy, merge_sigma, merge_heading = solve_merge_centerline_s_torch(
            center_xy=segment_xy,
            center_heading=center_heading,
            center_s=center_s,
            distance_budget=distance_budget,
            lateral_offset=lateral_offset,
            heading_offset=heading_offset,
            dense_ds=dense_ds,
        )
        speed_scale = torch.tensor(1.0, dtype=segment_xy.dtype, device=segment_xy.device)

    dense_full_xy, dense_full_sigma, dense_full_heading = build_full_augmented_path_torch(
        center_xy=segment_xy,
        center_heading=center_heading,
        center_s=center_s,
        merge_xy=merge_xy,
        merge_sigma=merge_sigma,
        merge_heading=merge_heading,
        s_merge=s_merge,
        total_distance_m=total_distance,
        dense_ds=dense_ds,
    )

    connect_mask = segment_time <= connect_time + 1.0e-6
    progress = torch.zeros_like(segment_time)
    if distance_budget > 1.0e-9:
        progress[connect_mask] = speed_scale * center_s[connect_mask]
    else:
        progress[connect_mask] = 0.0

    continue_mask = ~connect_mask
    if torch.any(continue_mask):
        merge_time_on_gt = interp1d_torch(
            progress_samples, progress_time_lookup, s_merge.unsqueeze(0)
        )[0]
        shifted_gt_time = merge_time_on_gt + (segment_time[continue_mask] - connect_time)
        shifted_gt_time = torch.clamp(shifted_gt_time, min=segment_time[0], max=segment_time[-1])
        centerline_progress = interp1d_torch(segment_time, center_s, shifted_gt_time)
        progress[continue_mask] = merge_sigma[-1] + (centerline_progress - s_merge)

    progress = torch.clamp(progress, min=0.0, max=dense_full_sigma[-1])
    query_xy, query_heading = sample_dense_path_torch(
        dense_full_sigma, dense_full_xy, dense_full_heading, progress
    )
    exact_speed = speed_from_progress_torch(progress, segment_time)
    return SegmentAugmentationResultTorch(
        query_xy=query_xy,
        query_heading=query_heading,
        progress=progress,
        exact_speed=exact_speed,
        distance_profile=center_s,
        segment_time=segment_time,
        merge_centerline_s=float(s_merge.item()),
        merge_path_length_m=float(merge_sigma[-1].item()),
        connect_distance_budget_m=float(distance_budget.item()),
        connect_speed_scale=float(speed_scale.item()),
    )


def lateral_acceleration_from_speed_and_curvature_torch(
    speed: torch.Tensor, curvature: torch.Tensor
) -> torch.Tensor:
    return speed * speed * curvature


def generate_time_candidates(
    start_s: float,
    end_s: float,
    step_s: float = TIME_INTERVAL,
) -> list[float]:
    start_tick = int(np.ceil((start_s - 1.0e-9) / step_s))
    end_tick = int(np.floor((end_s + 1.0e-9) / step_s))
    return [tick * step_s for tick in range(start_tick, end_tick + 1)]


def segment_kinematics_torch(
    segment_result: SegmentAugmentationResultTorch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_speed = speed_from_progress_torch(
        segment_result.distance_profile, segment_result.segment_time
    )
    augmented_speed = segment_result.exact_speed
    gt_acceleration = acceleration_from_speed_torch(gt_speed, segment_result.segment_time)
    augmented_acceleration = acceleration_from_speed_torch(
        augmented_speed, segment_result.segment_time
    )
    gt_jerk = jerk_from_acceleration_torch(gt_acceleration, segment_result.segment_time)
    augmented_jerk = jerk_from_acceleration_torch(
        augmented_acceleration, segment_result.segment_time
    )
    return gt_speed, augmented_speed, gt_jerk, augmented_jerk


def bridge_constraint_series_torch(
    sample_result: AugmentedSampleTorchResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    past_gt_speed, past_aug_speed, past_gt_jerk, past_aug_jerk = segment_kinematics_torch(
        sample_result.past_segment
    )
    future_gt_speed, future_aug_speed, future_gt_jerk, future_aug_jerk = segment_kinematics_torch(
        sample_result.future_segment
    )
    bridge_speed_gap = torch.cat(
        [
            torch.abs(past_aug_speed - past_gt_speed).flip(0)[:-1],
            torch.abs(future_aug_speed - future_gt_speed),
        ],
        dim=0,
    )
    augmented_jerk = torch.cat(
        [past_aug_jerk.flip(0)[:-1], future_aug_jerk],
        dim=0,
    )
    gt_arc_speed = torch.cat([past_gt_speed.flip(0)[:-1], future_gt_speed], dim=0)
    augmented_arc_speed = torch.cat([past_aug_speed.flip(0)[:-1], future_aug_speed], dim=0)
    return gt_arc_speed, augmented_arc_speed, bridge_speed_gap, augmented_jerk


class StatePerturbation:
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory that
    reconnects to the original GT while respecting longitudinal progress and bridge feasibility limits.
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        wheel_base: float = 2.75,
        device: torch.device | str = "cpu",
        past_bridge_sec: float = 1.0,
        future_bridge_sec: float = 1.5,
        dense_sample_ds: float = DENSE_SAMPLE_DS,
        max_heading_offset_deg: float = 10.0,
        max_lateral_accel_mps2: float = 3.0,
        max_bridge_speed_gap_mps: float = 0.5,
        max_bridge_jerk_mps3: float = 5.0,
        adaptive_bridge_search: bool = True,
    ) -> None:
        """
        Initialize the augmentor.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param past_bridge_sec: duration used to connect the past trajectory to the perturbed state
        :param future_bridge_sec: duration used to reconnect the perturbed state to the GT future
        :param dense_sample_ds: dense sampling resolution used for path-length feasibility checks
        :param max_heading_offset_deg: maximum absolute initial heading perturbation in degrees
        :param max_lateral_accel_mps2: lateral acceleration feasibility limit
        :param max_bridge_speed_gap_mps: bridge speed-gap feasibility limit
        :param max_bridge_jerk_mps3: bridge jerk feasibility limit
        :param adaptive_bridge_search: if True, extend M/N to the first feasible candidate
        """
        self._augment_prob = augment_prob
        self._device = torch.device(device)
        heading_limit_rad = np.deg2rad(max_heading_offset_deg)
        lo = ([0.0, -0.75, -heading_limit_rad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],)
        hi = ([0.0, +0.75, +heading_limit_rad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],)
        self._low = torch.tensor(lo, dtype=torch.float32, device=self._device)
        self._high = torch.tensor(hi, dtype=torch.float32, device=self._device)
        self._wheel_base = wheel_base
        self._past_bridge_sec = past_bridge_sec
        self._future_bridge_sec = future_bridge_sec
        self._max_lateral_accel_mps2 = max_lateral_accel_mps2
        self._max_bridge_speed_gap_mps = max_bridge_speed_gap_mps
        self._max_bridge_jerk_mps3 = max_bridge_jerk_mps3
        self._adaptive_bridge_search = adaptive_bridge_search
        self.time_interval = TIME_INTERVAL
        self.dense_sample_ds = dense_sample_ds

    def __call__(self, inputs, ego_future, neighbors_future):
        aug_flag, aug_current_state, aug_ego_past, aug_ego_future = self.augment(inputs, ego_future)

        inputs["ego_current_state"][aug_flag] = aug_current_state[aug_flag]
        inputs["ego_agent_past"][aug_flag] = aug_ego_past[aug_flag]
        ego_future[aug_flag] = aug_ego_future[aug_flag]

        return self.centric_transform(inputs, ego_future, neighbors_future)

    def normalize_angle(self, angle: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_transform_matrix_batch(self, cur_state):
        processed_input = torch.column_stack(
            (
                cur_state[:, 2],  # cos
                cur_state[:, 3],  # sin
            )
        )

        reshaping_tensor = torch.tensor(
            [
                [1, 0, 0, 1],
                [0, 1, -1, 0],
            ],
            dtype=torch.float32,
        ).to(processed_input.device)
        return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)

    def _sample_lateral_offsets(self, batch_size: int) -> torch.Tensor:
        random_tensor = torch.rand(batch_size, device=self._device)
        return self._low[:, 1] + (self._high[:, 1] - self._low[:, 1]) * random_tensor

    def _sample_heading_offsets(self, batch_size: int) -> torch.Tensor:
        random_tensor = torch.rand(batch_size, device=self._device)
        return self._low[:, 2] + (self._high[:, 2] - self._low[:, 2]) * random_tensor

    def _state_to_heading(self, current_state: torch.Tensor) -> torch.Tensor:
        return torch.atan2(current_state[..., 3], current_state[..., 2])

    def _build_augmented_sample(
        self,
        ego_past: torch.Tensor,
        ego_current_state: torch.Tensor,
        ego_future: torch.Tensor,
        lateral_offset: torch.Tensor,
        heading_offset: torch.Tensor,
        past_connect_time_s: float,
        future_recover_time_s: float,
    ) -> AugmentedSampleTorchResult:
        """
        Rebuild ego past/current/future around a laterally perturbed current state while keeping
        the longitudinal progress aligned with the GT progress-time relation after merge.
        """
        dt = self.time_interval
        dtype = ego_past.dtype
        device = ego_past.device

        past_len = ego_past.shape[0]
        future_len = ego_future.shape[0]

        past_xy = ego_past[:, :2].clone()
        past_heading = torch.atan2(ego_past[:, 3], ego_past[:, 2]).clone()
        current_xy = ego_current_state[:2].clone()
        current_heading = self._state_to_heading(ego_current_state).clone()
        future_xy = ego_future[:, :2].clone()
        future_heading = ego_future[:, 2].clone()

        past_xy[-1] = current_xy
        past_heading[-1] = current_heading

        future_segment_xy = torch.cat([current_xy.unsqueeze(0), future_xy], dim=0)
        future_segment_heading = torch.cat([current_heading.unsqueeze(0), future_heading], dim=0)
        future_time = torch.arange(
            0.0, (future_len + 1) * dt, dt, dtype=dtype, device=device
        )

        past_segment_xy = torch.flip(past_xy, dims=[0])
        past_segment_heading = torch.flip(past_heading, dims=[0])
        past_time = torch.arange(0.0, past_len * dt, dt, dtype=dtype, device=device)

        future_result = augment_segment_torch(
            future_segment_xy,
            future_segment_heading,
            future_time,
            lateral_offset,
            heading_offset,
            min(future_recover_time_s, float(future_time[-1].item())),
            self.dense_sample_ds,
        )
        past_result = augment_segment_torch(
            past_segment_xy,
            past_segment_heading,
            past_time,
            lateral_offset,
            -heading_offset,
            min(past_connect_time_s, float(past_time[-1].item())),
            self.dense_sample_ds,
        )

        aug_past_xy = torch.flip(past_result.query_xy, dims=[0])
        full_xy = torch.cat([aug_past_xy[:-1], future_result.query_xy], dim=0)

        original_full_heading = torch.cat(
            [past_heading[:-1], current_heading.unsqueeze(0), future_heading], dim=0
        )
        original_full_xy = torch.cat([past_xy[:-1], future_segment_xy], dim=0)
        full_heading = heading_from_positions_torch(full_xy, fallback_heading=original_full_heading)

        current_index = past_len - 1
        aug_past_heading = full_heading[:past_len]
        aug_future_heading = full_heading[past_len:]

        aug_past_4d = torch.stack(
            [
                aug_past_xy[:, 0],
                aug_past_xy[:, 1],
                torch.cos(aug_past_heading),
                torch.sin(aug_past_heading),
            ],
            dim=-1,
        )
        aug_future_3d = torch.stack(
            [full_xy[past_len:, 0], full_xy[past_len:, 1], aug_future_heading], dim=-1
        )

        if current_index == 0:
            velocity = (full_xy[1] - full_xy[0]) / dt
            acceleration = torch.zeros(2, dtype=dtype, device=device)
            yaw_rate = normalize_angle_torch(full_heading[1] - full_heading[0]) / dt
        elif current_index == full_xy.shape[0] - 1:
            velocity = (full_xy[-1] - full_xy[-2]) / dt
            acceleration = torch.zeros(2, dtype=dtype, device=device)
            yaw_rate = normalize_angle_torch(full_heading[-1] - full_heading[-2]) / dt
        else:
            velocity = (full_xy[current_index + 1] - full_xy[current_index - 1]) / (2.0 * dt)
            acceleration = (
                full_xy[current_index + 1]
                - 2.0 * full_xy[current_index]
                + full_xy[current_index - 1]
            ) / (dt**2)
            yaw_rate = normalize_angle_torch(
                full_heading[current_index + 1] - full_heading[current_index - 1]
            ) / (2.0 * dt)

        speed = torch.linalg.norm(velocity)
        steering_angle = torch.tensor(0.0, dtype=dtype, device=device)
        if speed >= 0.2:
            steering_angle = torch.atan(yaw_rate * self._wheel_base / torch.abs(speed))
            steering_angle = torch.clamp(steering_angle, -2.0 / 3.0 * np.pi, 2.0 / 3.0 * np.pi)
        else:
            yaw_rate = torch.tensor(0.0, dtype=dtype, device=device)

        aug_current_state = ego_current_state.clone()
        aug_current_state[:2] = full_xy[current_index]
        aug_current_state[2] = torch.cos(full_heading[current_index])
        aug_current_state[3] = torch.sin(full_heading[current_index])
        aug_current_state[4:6] = velocity
        aug_current_state[6:8] = acceleration
        aug_current_state[8] = steering_angle
        aug_current_state[9] = yaw_rate

        full_time = torch.arange(
            -(past_len - 1) * dt,
            (future_len + 1) * dt,
            dt,
            dtype=dtype,
            device=device,
        )

        return AugmentedSampleTorchResult(
            aug_current_state=aug_current_state,
            aug_past=aug_past_4d,
            aug_future=aug_future_3d,
            original_full_xy=original_full_xy,
            augmented_full_xy=full_xy,
            original_full_heading=original_full_heading,
            augmented_full_heading=full_heading,
            full_time=full_time,
            past_segment=past_result,
            future_segment=future_result,
            past_connect_time_s=past_connect_time_s,
            future_recover_time_s=future_recover_time_s,
        )

    def _evaluate_constraints(
        self,
        sample_result: AugmentedSampleTorchResult,
    ) -> ConstraintDiagnosticsTorch:
        full_time = sample_result.full_time
        original_sigma = cumulative_distance_torch(sample_result.original_full_xy)
        augmented_sigma = cumulative_distance_torch(sample_result.augmented_full_xy)
        gt_arc_speed, augmented_arc_speed, bridge_speed_gap, augmented_jerk = (
            bridge_constraint_series_torch(sample_result)
        )

        gt_curvature = curvature_from_xy_torch(sample_result.original_full_xy, original_sigma)
        augmented_curvature = curvature_from_xy_torch(
            sample_result.augmented_full_xy, augmented_sigma
        )
        augmented_lateral_accel = lateral_acceleration_from_speed_and_curvature_torch(
            augmented_arc_speed, augmented_curvature
        )
        max_abs_augmented_lateral_accel = float(torch.max(torch.abs(augmented_lateral_accel)).item())

        bridge_time = full_time
        bridge_mask = (
            (bridge_time >= -sample_result.past_connect_time_s - 1.0e-9)
            & (bridge_time <= sample_result.future_recover_time_s + 1.0e-9)
        )

        if torch.any(bridge_mask):
            max_speed_gap = float(torch.max(bridge_speed_gap[bridge_mask]).item())
            max_abs_bridge_jerk = float(torch.max(torch.abs(augmented_jerk[bridge_mask])).item())
        else:
            max_speed_gap = 0.0
            max_abs_bridge_jerk = 0.0

        lateral_accel_passes = max_abs_augmented_lateral_accel <= self._max_lateral_accel_mps2 + 1.0e-9
        speed_gap_passes = max_speed_gap <= self._max_bridge_speed_gap_mps + 1.0e-9
        jerk_passes = max_abs_bridge_jerk <= self._max_bridge_jerk_mps3 + 1.0e-9

        return ConstraintDiagnosticsTorch(
            max_abs_augmented_lateral_accel_mps2=max_abs_augmented_lateral_accel,
            max_bridge_speed_gap_mps=max_speed_gap,
            max_abs_bridge_jerk_mps3=max_abs_bridge_jerk,
            lateral_accel_limit_mps2=self._max_lateral_accel_mps2,
            speed_gap_limit_mps=self._max_bridge_speed_gap_mps,
            jerk_limit_mps3=self._max_bridge_jerk_mps3,
            lateral_accel_passes=lateral_accel_passes,
            speed_gap_passes=speed_gap_passes,
            jerk_passes=jerk_passes,
            passes=lateral_accel_passes and speed_gap_passes and jerk_passes,
        )

    def _search_feasible_sample(
        self,
        ego_past: torch.Tensor,
        ego_current_state: torch.Tensor,
        ego_future: torch.Tensor,
        lateral_offset: torch.Tensor,
        heading_offset: torch.Tensor,
        initial_past_connect_time_s: float,
        initial_future_recover_time_s: float,
    ) -> AugmentedSampleTorchResult:
        def build_candidate(
            past_connect_time_s: float,
            future_recover_time_s: float,
        ) -> AugmentedSampleTorchResult:
            return self._build_augmented_sample(
                ego_past=ego_past,
                ego_current_state=ego_current_state,
                ego_future=ego_future,
                lateral_offset=lateral_offset,
                heading_offset=heading_offset,
                past_connect_time_s=past_connect_time_s,
                future_recover_time_s=future_recover_time_s,
            )

        initial_result = build_candidate(
            past_connect_time_s=initial_past_connect_time_s,
            future_recover_time_s=initial_future_recover_time_s,
        )
        initial_diag = self._evaluate_constraints(initial_result)
        if not self._adaptive_bridge_search or initial_diag.passes:
            return initial_result

        max_future_recover_time_s = float(ego_future.shape[0] * self.time_interval)
        max_past_connect_time_s = float((ego_past.shape[0] - 1) * self.time_interval)

        future_candidates = generate_time_candidates(
            start_s=initial_future_recover_time_s + self.time_interval,
            end_s=max_future_recover_time_s,
            step_s=self.time_interval,
        )
        for candidate_n in future_candidates:
            candidate_result = build_candidate(
                past_connect_time_s=initial_past_connect_time_s,
                future_recover_time_s=candidate_n,
            )
            if self._evaluate_constraints(candidate_result).passes:
                return candidate_result

        past_candidates = generate_time_candidates(
            start_s=initial_past_connect_time_s + self.time_interval,
            end_s=max_past_connect_time_s,
            step_s=self.time_interval,
        )
        future_with_past_candidates = generate_time_candidates(
            start_s=initial_future_recover_time_s,
            end_s=max_future_recover_time_s,
            step_s=self.time_interval,
        )
        for candidate_m in past_candidates:
            for candidate_n in future_with_past_candidates:
                candidate_result = build_candidate(
                    past_connect_time_s=candidate_m,
                    future_recover_time_s=candidate_n,
                )
                if self._evaluate_constraints(candidate_result).passes:
                    return candidate_result

        return initial_result

    def augment(self, inputs, ego_future):
        ego_current_state = inputs["ego_current_state"].clone()
        ego_agent_past = inputs["ego_agent_past"].clone()
        aug_ego_future = ego_future.clone()

        batch_size = ego_current_state.shape[0]
        lateral_offsets = self._sample_lateral_offsets(batch_size)
        heading_offsets = self._sample_heading_offsets(batch_size)
        valid_speed = torch.abs(ego_current_state[:, 4]) >= 2.0
        valid_offset = (torch.abs(lateral_offsets) > 1.0e-3) | (torch.abs(heading_offsets) > 1.0e-3)
        aug_flag = (
            (torch.rand(batch_size, device=self._device) < self._augment_prob)
            & valid_speed
            & valid_offset
        )

        for batch_index in torch.nonzero(aug_flag, as_tuple=False).flatten():
            best_result = self._search_feasible_sample(
                ego_past=ego_agent_past[batch_index],
                ego_current_state=ego_current_state[batch_index],
                ego_future=ego_future[batch_index],
                lateral_offset=lateral_offsets[batch_index],
                heading_offset=heading_offsets[batch_index],
                initial_past_connect_time_s=self._past_bridge_sec,
                initial_future_recover_time_s=self._future_bridge_sec,
            )
            ego_current_state[batch_index] = best_result.aug_current_state
            ego_agent_past[batch_index] = best_result.aug_past
            aug_ego_future[batch_index] = best_result.aug_future

        return aug_flag, ego_current_state, ego_agent_past, aug_ego_future

    def centric_transform(
        self,
        inputs: torch.Tensor,
        ego_future: torch.Tensor,
        neighbors_future: torch.Tensor,
    ):
        cur_state = inputs["ego_current_state"].clone()
        center_xy = cur_state[:, :2]
        transform_matrix = self.get_transform_matrix_batch(cur_state)

        # ego xy
        inputs["ego_current_state"][..., :2] = vector_transform(
            inputs["ego_current_state"][..., :2], transform_matrix, center_xy
        )
        # ego cos sin
        inputs["ego_current_state"][..., 2:4] = vector_transform(
            inputs["ego_current_state"][..., 2:4], transform_matrix
        )
        # ego vx, vy
        inputs["ego_current_state"][..., 4:6] = vector_transform(
            inputs["ego_current_state"][..., 4:6], transform_matrix
        )
        # ego ax, ay
        inputs["ego_current_state"][..., 6:8] = vector_transform(
            inputs["ego_current_state"][..., 6:8], transform_matrix
        )

        # ego past
        ego_past_mask = torch.sum(torch.ne(inputs["ego_agent_past"][..., :4], 0), dim=-1) == 0
        inputs["ego_agent_past"][..., :2] = vector_transform(
            inputs["ego_agent_past"][..., :2], transform_matrix, center_xy
        )
        inputs["ego_agent_past"][..., 2:4] = vector_transform(
            inputs["ego_agent_past"][..., 2:4], transform_matrix
        )
        inputs["ego_agent_past"][ego_past_mask] = 0.0

        # ego future xy
        ego_future[..., :2] = vector_transform(ego_future[..., :2], transform_matrix, center_xy)
        ego_future[..., 2] = heading_transform(ego_future[..., 2], transform_matrix)
        inputs["ego_agent_future"] = ego_future

        # neighbor past xy
        mask = torch.sum(torch.ne(inputs["neighbor_agents_past"][..., :6], 0), dim=-1) == 0
        inputs["neighbor_agents_past"][..., :2] = vector_transform(
            inputs["neighbor_agents_past"][..., :2], transform_matrix, center_xy
        )
        # neighbor past cos sin
        inputs["neighbor_agents_past"][..., 2:4] = vector_transform(
            inputs["neighbor_agents_past"][..., 2:4], transform_matrix
        )
        # neighbor past vx, vy
        inputs["neighbor_agents_past"][..., 4:6] = vector_transform(
            inputs["neighbor_agents_past"][..., 4:6], transform_matrix
        )
        inputs["neighbor_agents_past"][mask] = 0.0

        # neighbor future xy
        mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0
        neighbors_future[..., :2] = vector_transform(
            neighbors_future[..., :2], transform_matrix, center_xy
        )
        neighbors_future[..., 2] = heading_transform(neighbors_future[..., 2], transform_matrix)
        neighbors_future[mask] = 0.0

        # lanes
        mask = torch.sum(torch.ne(inputs["lanes"][..., :8], 0), dim=-1) == 0
        inputs["lanes"][..., :2] = vector_transform(
            inputs["lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["lanes"][..., 2:4] = vector_transform(inputs["lanes"][..., 2:4], transform_matrix)
        inputs["lanes"][..., 4:6] = vector_transform(inputs["lanes"][..., 4:6], transform_matrix)
        inputs["lanes"][..., 6:8] = vector_transform(inputs["lanes"][..., 6:8], transform_matrix)
        inputs["lanes"][mask] = 0.0

        # route_lanes
        mask = torch.sum(torch.ne(inputs["route_lanes"][..., :8], 0), dim=-1) == 0
        inputs["route_lanes"][..., :2] = vector_transform(
            inputs["route_lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["route_lanes"][..., 2:4] = vector_transform(
            inputs["route_lanes"][..., 2:4], transform_matrix
        )
        inputs["route_lanes"][..., 4:6] = vector_transform(
            inputs["route_lanes"][..., 4:6], transform_matrix
        )
        inputs["route_lanes"][..., 6:8] = vector_transform(
            inputs["route_lanes"][..., 6:8], transform_matrix
        )
        inputs["route_lanes"][mask] = 0.0

        # polygons
        mask = torch.sum(torch.ne(inputs["polygons"], 0), dim=-1) == 0
        inputs["polygons"][..., :2] = vector_transform(
            inputs["polygons"][..., :2], transform_matrix, center_xy
        )
        inputs["polygons"][mask] = 0.0

        # line_strings
        mask = torch.sum(torch.ne(inputs["line_strings"], 0), dim=-1) == 0
        inputs["line_strings"][..., :2] = vector_transform(
            inputs["line_strings"][..., :2], transform_matrix, center_xy
        )
        inputs["line_strings"][mask] = 0.0

        # static objects xy
        mask = torch.sum(torch.ne(inputs["static_objects"][..., :10], 0), dim=-1) == 0
        inputs["static_objects"][..., :2] = vector_transform(
            inputs["static_objects"][..., :2], transform_matrix, center_xy
        )
        # static objects cos sin
        inputs["static_objects"][..., 2:4] = vector_transform(
            inputs["static_objects"][..., 2:4], transform_matrix
        )
        inputs["static_objects"][mask] = 0.0

        return inputs, ego_future, neighbors_future


if __name__ == "__main__":
    import argparse
    from copy import deepcopy
    from pathlib import Path

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    from diffusion_planner.train_epoch import heading_to_cos_sin
    from diffusion_planner.utils.visualize_input import visualize_inputs

    parser = argparse.ArgumentParser()
    parser.add_argument("target_npz", type=Path)
    args = parser.parse_args()

    target_npz = args.target_npz

    save_dir = target_npz.parent.parent / "augmented"
    save_dir.mkdir(parents=True, exist_ok=True)

    loaded = np.load(target_npz)
    data = {}
    for key, value in loaded.items():
        if key == "token":
            continue
        data[key] = torch.tensor(value).unsqueeze(0)
        if key == "goal_pose" or key == "ego_agent_past":
            data[key] = heading_to_cos_sin(data[key])

    # Load future trajectories separately
    ego_future = torch.tensor(loaded["ego_agent_future"]).unsqueeze(0)
    neighbors_future = torch.tensor(loaded["neighbor_agents_future"]).unsqueeze(0)

    aug = StatePerturbation(augment_prob=1.0, device="cpu")

    # Save original data visualization with augmentation range rectangle
    original_save_path = save_dir / "original.png"
    fig, ax = plt.subplots(figsize=(10, 10))

    # Visualize inputs on the ax
    view_range = 20
    visualize_inputs(deepcopy(data), save_path=None, ax=ax, view_ranges=[view_range])

    # Get augmentation ranges from the aug object
    lo = aug._low.cpu().numpy()[0]  # Extract from tuple
    hi = aug._high.cpu().numpy()[0]  # Extract from tuple
    x_min, y_min = lo[0], lo[1]
    x_max, y_max = hi[0], hi[1]

    # Draw the augmentation range rectangle
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label="Augmentation Range",
    )
    ax.add_patch(rect)
    ax.legend()

    plt.tight_layout()
    plt.savefig(original_save_path, dpi=100)
    plt.close()

    trial_num = 10
    for i in range(trial_num):
        aug_data, aug_ego_future, aug_neighbors_future = aug(
            deepcopy(data), ego_future.clone(), neighbors_future.clone()
        )

        # Save augmented data to npz file
        data_dict = {}
        for key, value in aug_data.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.squeeze(0).detach().cpu().numpy()
            else:
                data_dict[key] = value

        # Add future trajectories with consistent naming
        data_dict["ego_agent_future"] = aug_ego_future.squeeze(0).detach().cpu().numpy()
        data_dict["neighbor_agents_future"] = (
            aug_neighbors_future.squeeze(0).detach().cpu().numpy()
        )
        aug_data["ego_agent_future"] = aug_ego_future
        aug_data["neighbor_agents_future"] = aug_neighbors_future

        # Save to npz file
        output_path = save_dir / f"augmented_{i:08d}.npz"
        np.savez(output_path, **data_dict)

        # Use deepcopy to avoid side effects from visualize_inputs
        visualize_inputs(
            deepcopy(aug_data), save_dir / f"augmented_{i:08d}.png", view_ranges=[view_range]
        )

    print(f"Augmented data saved: {trial_num} files to {save_dir}")
