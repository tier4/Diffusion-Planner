"""Bridge-based ego state perturbation augmentation.

The current ego pose is shifted laterally (and in heading) and reconnected to
the original GT trajectory with C2-continuous quintic bridges on both the past
and the future side, while keeping the longitudinal progress aligned with the
GT progress-time relation. Candidates are checked against lateral-acceleration,
bridge-speed-gap, and bridge-jerk limits.

To keep the model from shortcutting map-based recovery by extrapolating its
own history, two decorrelation mechanisms are applied on top of the minimal
feasible bridge times (see ``data_augmentation_bridge_core``):

- bridge-time randomization: random feasible (M, N) above the minimum
- past-history bump: one random sin^3 lateral bump on the conditioning side

The heavy lifting runs in numpy (``data_augmentation_bridge_core``); this
module only adapts the torch training batch format.
"""

from __future__ import annotations

import numpy as np
import torch

from diffusion_planner.utils import data_augmentation_bridge_core as bridge_core
from diffusion_planner.utils.data_augmentation_bridge_core import (
    MIN_BRIDGE_TIME_S,
    BidirectionalAugmentationResult,
    Trajectory2D,
)

TIME_INTERVAL = 0.1
DENSE_SAMPLE_DS = 0.05


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


class StatePerturbation:
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory
    that reconnects to the original GT while respecting longitudinal progress and bridge
    feasibility limits. Optionally randomizes the bridge times and injects a past-history bump to
    decorrelate the history from the recovery.
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        wheel_base: float = 2.75,
        device: torch.device | str = "cpu",
        past_bridge_sec: float = MIN_BRIDGE_TIME_S,
        future_bridge_sec: float = MIN_BRIDGE_TIME_S,
        dense_sample_ds: float = DENSE_SAMPLE_DS,
        max_lateral_offset_m: float = 0.75,
        max_heading_offset_deg: float = 10.0,
        max_lateral_accel_mps2: float = 3.0,
        max_bridge_speed_gap_mps: float = 0.5,
        max_bridge_jerk_mps3: float = 5.0,
        randomize_bridge_times: bool = True,
        bridge_time_extra_range_s: float = 1.5,
        past_bump: bool = True,
    ) -> None:
        """
        Initialize the augmentor.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param past_bridge_sec: initial past bridge time M; the minimal feasible M is searched
            upward from this value
        :param future_bridge_sec: initial future bridge time N; the minimal feasible N is
            searched upward from this value
        :param dense_sample_ds: dense sampling resolution used for the bridge paths
        :param max_lateral_offset_m: maximum absolute lateral perturbation in meters
        :param max_heading_offset_deg: maximum absolute initial heading perturbation in degrees
        :param max_lateral_accel_mps2: lateral acceleration feasibility limit
        :param max_bridge_speed_gap_mps: bridge speed-gap feasibility limit
        :param max_bridge_jerk_mps3: bridge jerk feasibility limit
        :param randomize_bridge_times: sample random feasible bridge times above the minimum so
            the same past maps to varied recoveries
        :param bridge_time_extra_range_s: upper range added on top of the minimal feasible M/N
            when randomizing bridge times
        :param past_bump: inject one random sin^3 lateral bump into the past history
        """
        self._augment_prob = augment_prob
        self._device = torch.device(device)
        heading_limit_rad = np.deg2rad(max_heading_offset_deg)
        lo = ([0.0, -max_lateral_offset_m, -heading_limit_rad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],)
        hi = ([0.0, +max_lateral_offset_m, +heading_limit_rad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],)
        self._low = torch.tensor(lo, dtype=torch.float32, device=self._device)
        self._high = torch.tensor(hi, dtype=torch.float32, device=self._device)
        self._wheel_base = wheel_base
        self._past_bridge_sec = past_bridge_sec
        self._future_bridge_sec = future_bridge_sec
        self._max_lateral_accel_mps2 = max_lateral_accel_mps2
        self._max_bridge_speed_gap_mps = max_bridge_speed_gap_mps
        self._max_bridge_jerk_mps3 = max_bridge_jerk_mps3
        self._randomize_bridge_times = randomize_bridge_times
        self._bridge_time_extra_range_s = bridge_time_extra_range_s
        self._past_bump = past_bump
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

    def _build_full_trajectory(
        self,
        ego_past: np.ndarray,
        ego_current_state: np.ndarray,
        ego_future: np.ndarray,
    ) -> tuple[Trajectory2D, int]:
        dt = self.time_interval
        past_len = ego_past.shape[0]
        future_len = ego_future.shape[0]

        past_x = ego_past[:, 0].astype(float).copy()
        past_y = ego_past[:, 1].astype(float).copy()
        past_heading = np.arctan2(ego_past[:, 3], ego_past[:, 2]).astype(float)
        # The last past row corresponds to the current frame; force exact agreement.
        past_x[-1] = float(ego_current_state[0])
        past_y[-1] = float(ego_current_state[1])
        past_heading[-1] = float(np.arctan2(ego_current_state[3], ego_current_state[2]))

        x = np.concatenate((past_x, ego_future[:, 0].astype(float)))
        y = np.concatenate((past_y, ego_future[:, 1].astype(float)))
        yaw = np.concatenate((past_heading, ego_future[:, 2].astype(float)))
        t = (np.arange(past_len + future_len, dtype=float) - (past_len - 1)) * dt
        return Trajectory2D(t=t, x=x, y=y, yaw=yaw), past_len - 1

    def _augment_single(
        self,
        gt: Trajectory2D,
        current_index: int,
        lateral_offset_m: float,
        heading_offset_rad: float,
        rng: np.random.Generator,
    ) -> BidirectionalAugmentationResult | None:
        """Run the full pipeline for one sample; None means no feasible candidate."""
        max_past_connect_time_s = float(-gt.t[0])
        max_future_recover_time_s = float(gt.t[-1])
        initial_m = min(max(self._past_bridge_sec, MIN_BRIDGE_TIME_S), max_past_connect_time_s)
        initial_n = min(max(self._future_bridge_sec, MIN_BRIDGE_TIME_S), max_future_recover_time_s)

        initial_result = bridge_core.augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            future_recover_time_s=initial_n,
            past_connect_time_s=initial_m,
            dense_ds=self.dense_sample_ds,
        )
        outcome = bridge_core.search_feasible_result(
            initial_result=initial_result,
            max_lateral_accel_mps2=self._max_lateral_accel_mps2,
            max_bridge_speed_gap_mps=self._max_bridge_speed_gap_mps,
            max_bridge_jerk_mps3=self._max_bridge_jerk_mps3,
            max_future_recover_time_s=max_future_recover_time_s,
            max_past_connect_time_s=max_past_connect_time_s,
        )
        if outcome is None:
            return None

        if self._randomize_bridge_times:
            randomized = bridge_core.randomize_bridge_times_result(
                rng,
                outcome.result,
                max_lateral_accel_mps2=self._max_lateral_accel_mps2,
                max_bridge_speed_gap_mps=self._max_bridge_speed_gap_mps,
                max_bridge_jerk_mps3=self._max_bridge_jerk_mps3,
                max_future_recover_time_s=max_future_recover_time_s,
                max_past_connect_time_s=max_past_connect_time_s,
                extra_time_range_s=self._bridge_time_extra_range_s,
            )
            if randomized is not None:
                outcome = randomized

        if self._past_bump:
            bumped = bridge_core.apply_random_past_history_bump(
                rng,
                outcome.result,
                max_lateral_accel_mps2=self._max_lateral_accel_mps2,
                max_bridge_speed_gap_mps=self._max_bridge_speed_gap_mps,
                max_bridge_jerk_mps3=self._max_bridge_jerk_mps3,
                past_horizon_s=max_past_connect_time_s,
            )
            if bumped is not None:
                outcome = bumped

        return outcome.result

    def _assemble_sample(
        self,
        result: BidirectionalAugmentationResult,
        ego_current_state: np.ndarray,
        past_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = self.time_interval
        full = result.augmented_full
        current_index = result.current_index

        aug_past = np.stack(
            (
                full.x[:past_len],
                full.y[:past_len],
                np.cos(full.yaw[:past_len]),
                np.sin(full.yaw[:past_len]),
            ),
            axis=-1,
        )
        aug_future = np.stack(
            (full.x[past_len:], full.y[past_len:], full.yaw[past_len:]),
            axis=-1,
        )

        xy = np.stack((full.x, full.y), axis=-1)
        if current_index == 0:
            velocity = (xy[1] - xy[0]) / dt
            acceleration = np.zeros(2, dtype=float)
            yaw_rate = bridge_core.wrap_angle(full.yaw[1] - full.yaw[0]) / dt
        elif current_index == xy.shape[0] - 1:
            velocity = (xy[-1] - xy[-2]) / dt
            acceleration = np.zeros(2, dtype=float)
            yaw_rate = bridge_core.wrap_angle(full.yaw[-1] - full.yaw[-2]) / dt
        else:
            velocity = (xy[current_index + 1] - xy[current_index - 1]) / (2.0 * dt)
            acceleration = (
                xy[current_index + 1] - 2.0 * xy[current_index] + xy[current_index - 1]
            ) / (dt**2)
            yaw_rate = bridge_core.wrap_angle(
                full.yaw[current_index + 1] - full.yaw[current_index - 1]
            ) / (2.0 * dt)

        speed = float(np.hypot(velocity[0], velocity[1]))
        if speed >= 0.2:
            steering_angle = float(
                np.clip(
                    np.arctan(yaw_rate * self._wheel_base / abs(speed)),
                    -2.0 / 3.0 * np.pi,
                    2.0 / 3.0 * np.pi,
                )
            )
        else:
            steering_angle = 0.0
            yaw_rate = 0.0

        aug_current_state = ego_current_state.astype(float).copy()
        aug_current_state[0] = full.x[current_index]
        aug_current_state[1] = full.y[current_index]
        aug_current_state[2] = np.cos(full.yaw[current_index])
        aug_current_state[3] = np.sin(full.yaw[current_index])
        aug_current_state[4:6] = velocity
        aug_current_state[6:8] = acceleration
        aug_current_state[8] = steering_angle
        aug_current_state[9] = float(yaw_rate)
        return aug_current_state, aug_past, aug_future

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
            # Derive the numpy seed from the torch RNG so global seeding stays effective.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            gt, current_index = self._build_full_trajectory(
                ego_past=ego_agent_past[batch_index].detach().cpu().numpy(),
                ego_current_state=ego_current_state[batch_index].detach().cpu().numpy(),
                ego_future=ego_future[batch_index].detach().cpu().numpy(),
            )
            lateral_offset_m = float(lateral_offsets[batch_index].item())
            heading_offset_rad = float(heading_offsets[batch_index].item())

            # When no feasible candidate exists for the sampled perturbation
            # (short history horizons can make large offsets infeasible),
            # retry with a halved offset instead of dropping the augmentation.
            result = None
            for _ in range(8):
                try:
                    result = self._augment_single(
                        gt=gt,
                        current_index=current_index,
                        lateral_offset_m=lateral_offset_m,
                        heading_offset_rad=heading_offset_rad,
                        rng=rng,
                    )
                except (RuntimeError, ValueError):
                    result = None
                if result is not None:
                    break
                lateral_offset_m *= 0.5
                heading_offset_rad *= 0.5
                if abs(lateral_offset_m) < 1.0e-3 and abs(heading_offset_rad) < 1.0e-3:
                    break
            if result is None:
                # Still no feasible candidate: keep the original (non-augmented) sample.
                aug_flag[batch_index] = False
                continue

            aug_current_state_np, aug_past_np, aug_future_np = self._assemble_sample(
                result=result,
                ego_current_state=ego_current_state[batch_index].detach().cpu().numpy(),
                past_len=current_index + 1,
            )
            dtype = ego_current_state.dtype
            ego_current_state[batch_index] = torch.from_numpy(aug_current_state_np).to(
                dtype=dtype, device=self._device
            )
            ego_agent_past[batch_index] = torch.from_numpy(aug_past_np).to(
                dtype=dtype, device=self._device
            )
            aug_ego_future[batch_index] = torch.from_numpy(aug_future_np).to(
                dtype=dtype, device=self._device
            )

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

        # goal pose (x, y, cos, sin)
        # Validity is decided from the position only: heading_to_cos_sin turns an
        # all-zero (x, y, heading) goal into (0, 0, 1, 0), so a full-width zero test
        # never fires and an absent goal would survive as a goal at the ego itself.
        mask = torch.sum(torch.ne(inputs["goal_pose"][..., :2], 0), dim=-1) == 0
        inputs["goal_pose"][..., :2] = vector_transform(
            inputs["goal_pose"][..., :2], transform_matrix, center_xy
        )
        inputs["goal_pose"][..., 2:4] = vector_transform(
            inputs["goal_pose"][..., 2:4], transform_matrix
        )
        inputs["goal_pose"][mask] = 0.0

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
