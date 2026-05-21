import numpy as np
import torch

from diffusion_planner.utils.unicycle_accel_curvature import smoothing_future_trajectory

NUM_REFINE = 20
TIME_INTERVAL = 0.1
EGO_LENGTH = 5.0
EGO_WIDTH = 2.0


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


def _cross2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """2D cross product along the last dimension: u × v = u.x*v.y - u.y*v.x"""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _rect_corners(rect: torch.Tensor) -> torch.Tensor:
    """
    rect: [B, 6] — (x, y, cos_h, sin_h, length, width)
    Returns [B, 4, 2] corner points.
    """
    B = rect.shape[0]
    xy, cos_h, sin_h, lw = rect[:, :2], rect[:, 2], rect[:, 3], rect[:, 4:]
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=1).reshape(B, 2, 2)
    signs = torch.tensor([[1.0, 1], [-1, 1], [-1, -1], [1, -1]], device=lw.device)
    local = torch.einsum("bj,ij->bij", lw / 2, signs)  # [B, 4, 2]
    local = torch.einsum("bij,bkj->bik", local, rot)   # [B, 4, 2]
    return xy[:, None, :] + local


def _sat_signed_distance(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    SAT signed distance between two rectangles.
    c1, c2: [B, 4, 2] corner points
    Returns [B] — negative means overlap.
    """
    nv = torch.stack(
        [c1[:, 0] - c1[:, 1], c1[:, 1] - c1[:, 2],
         c2[:, 0] - c2[:, 1], c2[:, 1] - c2[:, 2]],
        dim=1,
    )  # [B, 4, 2]
    nv = nv / torch.norm(nv, dim=2, keepdim=True).clamp(min=1e-6)
    p1 = torch.einsum("bij,bkj->bik", nv, c1)  # [B, 4, 4]
    p2 = torch.einsum("bij,bkj->bik", nv, c2)
    overlap = torch.cat(
        [p1.min(2).values - p2.max(2).values, p2.min(2).values - p1.max(2).values],
        dim=1,
    )  # [B, 8]
    is_overlap = (overlap < 0).all(dim=1)
    pos = torch.where(overlap < 0, torch.full_like(overlap, 1e5), overlap)
    return torch.where(is_overlap, overlap.max(1).values, pos.min(1).values)


def _segments_intersect_rect(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    rect_corners: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns [B] bool — True if any valid segment touches the rectangle.

    seg_start, seg_end: [B, N, 2]
    rect_corners:       [B, 4, 2]
    valid:              [B, N] bool — True for valid segments
    """
    hit = torch.zeros(seg_start.shape[:2], dtype=torch.bool, device=seg_start.device)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Proper segment–edge crossing: both pairs straddle each other's line
    for i, j in edges:
        C = rect_corners[:, i, :].unsqueeze(1)  # [B, 1, 2]
        D = rect_corners[:, j, :].unsqueeze(1)  # [B, 1, 2]
        AB = seg_end - seg_start                # [B, N, 2]
        CD = D - C                              # [B, 1, 2]
        hit = hit | (
            (_cross2d(AB, C - seg_start) * _cross2d(AB, D - seg_start) < 0)
            & (_cross2d(CD, seg_start - C) * _cross2d(CD, seg_end - C) < 0)
        )

    # Endpoint inside polygon: all edge cross products share the same sign
    for pt in (seg_start, seg_end):
        crosses = torch.stack(
            [
                _cross2d(
                    (rect_corners[:, j, :] - rect_corners[:, i, :]).unsqueeze(1),
                    pt - rect_corners[:, i, :].unsqueeze(1),
                )
                for i, j in edges
            ],
            dim=-1,
        )  # [B, N, 4]
        hit = hit | (crosses > 0).all(-1) | (crosses < 0).all(-1)

    if valid is not None:
        hit = hit & valid
    return hit.any(dim=1)  # [B]


class StatePerturbation:
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory that
    satisfies polynomial constraints.
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        wheel_base: float = 2.75,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Initialize the augmentor,
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        """
        self._augment_prob = augment_prob
        self._device = torch.device(device)
        lo = ([0.0, -0.75, -0.2, -1, -0.5, -0.2, -0.1, 0.0, 0.0],)
        hi = ([0.0, +0.75, +0.2, +1, +0.5, +0.2, +0.1, 0.0, 0.0],)
        self._low = torch.tensor(lo).to(self._device)
        self._high = torch.tensor(hi).to(self._device)
        self._wheel_base = wheel_base

        self.num_refine = NUM_REFINE
        self.time_interval = TIME_INTERVAL

        REFINE_HORIZON = NUM_REFINE * TIME_INTERVAL

        T = REFINE_HORIZON + TIME_INTERVAL
        self.coeff_matrix = torch.linalg.inv(
            torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [1, T, T**2, T**3, T**4, T**5],
                    [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                    [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
                ],
                device=device,
                dtype=torch.float32,
            )
        )
        self.t_matrix = torch.pow(
            torch.linspace(TIME_INTERVAL, REFINE_HORIZON, NUM_REFINE).unsqueeze(1),
            torch.arange(6).unsqueeze(0),
        ).to(device=device)  # shape (B, N+1)

    def __call__(self, inputs, ego_future, neighbors_future):
        aug_flag, aug_ego_current_state = self.augment(inputs)

        # Interpolate future trajectory
        interpolated_ego_future = self.interpolation_future_trajectory(
            aug_ego_current_state, ego_future
        )

        inputs["ego_current_state"][aug_flag] = aug_ego_current_state[aug_flag]
        ego_future[aug_flag] = interpolated_ego_future[aug_flag]

        return self.centric_transform(inputs, ego_future, neighbors_future)

    def augment(self, inputs):
        # Only aug current state
        ego_current_state = inputs["ego_current_state"].clone()

        B = ego_current_state.shape[0]
        aug_flag = (torch.rand(B) < self._augment_prob).bool().to(self._device) & ~(
            abs(ego_current_state[:, 4]) < 2.0
        )

        random_tensor = torch.rand(B, len(self._low)).to(self._device)
        scaled_random_tensor = self._low + (self._high - self._low) * random_tensor

        new_state = torch.zeros((B, 9), dtype=torch.float32).to(self._device)
        new_state[:, 3:] = ego_current_state[
            :, 4:10
        ]  # x, y, h is 0 because of ego-centric, update vx, vy, ax, ay, steering angle, yaw rate
        new_state = new_state + scaled_random_tensor
        new_state[:, 3] = torch.max(new_state[:, 3], torch.tensor(0.0, device=new_state.device))
        new_state[:, -1] = torch.clip(new_state[:, -1], -0.85, 0.85)

        ego_current_state[:, :2] = new_state[:, :2]
        ego_current_state[:, 2] = torch.cos(new_state[:, 2])
        ego_current_state[:, 3] = torch.sin(new_state[:, 2])
        ego_current_state[:, 4:8] = new_state[:, 3:7]
        ego_current_state[:, 8:10] = new_state[:, -2:]  # steering angle, yaw rate

        # update steering angle and yaw rate
        cur_velocity = ego_current_state[:, 4]
        yaw_rate = ego_current_state[:, 9]

        steering_angle = torch.zeros_like(cur_velocity)
        new_yaw_rate = torch.zeros_like(yaw_rate)

        mask = torch.abs(cur_velocity) < 0.2
        not_mask = ~mask
        steering_angle[not_mask] = torch.atan(
            yaw_rate[not_mask] * self._wheel_base / torch.abs(cur_velocity[not_mask])
        )
        steering_angle[not_mask] = torch.clamp(
            steering_angle[not_mask], -2 / 3 * np.pi, 2 / 3 * np.pi
        )
        new_yaw_rate[not_mask] = yaw_rate[not_mask]

        ego_current_state[:, 8] = steering_angle
        ego_current_state[:, 9] = new_yaw_rate

        # Discard augmentations that cause collisions
        collision = self._check_aug_validity(ego_current_state, inputs)
        aug_flag = aug_flag & ~collision

        return aug_flag, ego_current_state

    def _check_aug_validity(
        self, aug_ego_state: torch.Tensor, inputs: dict
    ) -> torch.Tensor:
        """
        Returns [B] bool — True where the augmented ego position is invalid.

        Invalid conditions:
          1. Ego polygon overlaps with a neighbour agent polygon.
          2. Ego polygon intersects a lane left or right boundary segment.
        """
        B = aug_ego_state.shape[0]
        device = aug_ego_state.device
        dtype = aug_ego_state.dtype

        # ego_shape = [wheelbase, length, width]; fall back to module constants if absent.
        if "ego_shape" in inputs:
            ego_length = inputs["ego_shape"][:, 1:2].to(device=device, dtype=dtype)  # [B, 1]
            ego_width = inputs["ego_shape"][:, 2:3].to(device=device, dtype=dtype)   # [B, 1]
        else:
            ego_length = torch.full((B, 1), EGO_LENGTH, device=device, dtype=dtype)
            ego_width = torch.full((B, 1), EGO_WIDTH, device=device, dtype=dtype)

        ego_rect = torch.cat(
            [aug_ego_state[:, :4], ego_length, ego_width],
            dim=-1,
        )  # [B, 6]
        ego_corners = _rect_corners(ego_rect)  # [B, 4, 2]

        collision = torch.zeros(B, dtype=torch.bool, device=device)

        # ── 1. Neighbour agent polygon collision ──────────────────────────────
        if "neighbor_agents_past" in inputs:
            nbr = inputs["neighbor_agents_past"][:, :, -1, :]  # [B, N, 11]
            N = nbr.shape[1]
            valid = torch.sum(torch.ne(nbr[:, :, :4], 0), dim=-1) > 0  # [B, N]
            if valid.any():
                # neighbor_agents_past layout: x,y,cos,sin (0:4), width (6), length (7)
                nbr_rect = torch.cat(
                    [nbr[:, :, :4], nbr[:, :, 7:8], nbr[:, :, 6:7]], dim=-1
                )  # [B, N, 6]  — (x,y,cos,sin,length,width)
                dists = _sat_signed_distance(
                    _rect_corners(ego_rect.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 6)),
                    _rect_corners(nbr_rect.reshape(B * N, 6)),
                ).reshape(B, N)
                collision = collision | ((dists < 0) & valid).any(dim=1)

        # ── 2. Lane boundary segment collision ───────────────────────────────
        if "lanes" in inputs:
            lanes = inputs["lanes"]  # [B, L, P, 33]
            left_offset = lanes[..., 4:6]  # [B, L, P, 2]
            right_offset = lanes[..., 6:8]  # [B, L, P, 2]

            # Absolute boundary positions
            left_pts = lanes[..., :2] + left_offset   # [B, L, P, 2]
            right_pts = lanes[..., :2] + right_offset  # [B, L, P, 2]

            # A waypoint is valid when its first 8 features are not all zero.
            # Additionally, only include a boundary side when its offset is
            # non-trivial; a near-zero offset means no boundary data.
            lane_valid = torch.sum(torch.ne(lanes[..., :8], 0), dim=-1) > 0  # [B, L, P]
            left_bound_valid = (torch.norm(left_offset, dim=-1) > 0.01) & lane_valid
            right_bound_valid = (torch.norm(right_offset, dim=-1) > 0.01) & lane_valid

            def _boundary_segs(pts, point_valid):
                s = pts[:, :, :-1, :].reshape(B, -1, 2)
                e = pts[:, :, 1:, :].reshape(B, -1, 2)
                v = (point_valid[:, :, :-1] & point_valid[:, :, 1:]).reshape(B, -1)
                return s, e, v

            ls, le, lv = _boundary_segs(left_pts, left_bound_valid)
            rs, re, rv = _boundary_segs(right_pts, right_bound_valid)

            collision = collision | _segments_intersect_rect(
                torch.cat([ls, rs], dim=1),
                torch.cat([le, re], dim=1),
                ego_corners,
                torch.cat([lv, rv], dim=1),
            )

        return collision

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

        # ego future xy
        ego_future[..., :2] = vector_transform(ego_future[..., :2], transform_matrix, center_xy)
        ego_future[..., 2] = heading_transform(ego_future[..., 2], transform_matrix)

        # ego past
        # inputs["ego_agent_past"][..., :2] = vector_transform(
        #     inputs["ego_agent_past"][..., :2], transform_matrix, center_xy
        # )
        # inputs["ego_agent_past"][..., 2:4] = vector_transform(
        #     inputs["ego_agent_past"][..., 2:4], transform_matrix
        # )

        ego_past4d = torch.cat(
            [
                inputs["ego_agent_past"][..., :2],  # x, y
                torch.cos(inputs["ego_agent_past"][..., 2:3]),  # cos
                torch.sin(inputs["ego_agent_past"][..., 2:3]),  # sin
            ],
            dim=-1,
        )
        ego_future4d = torch.cat(
            [
                ego_future[..., :2],  # x, y
                torch.cos(ego_future[..., 2:3]),  # cos
                torch.sin(ego_future[..., 2:3]),  # sin
            ],
            dim=-1,
        )

        ego_future4d = smoothing_future_trajectory(
            ego_past4d, inputs["ego_current_state"], ego_future4d
        )

        ego_future = torch.cat(
            [
                ego_future4d[..., :2],  # x, y
                torch.atan2(ego_future4d[..., 3], ego_future4d[..., 2]).unsqueeze(
                    -1
                ),  # heading from cos, sin
            ],
            dim=-1,
        )
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

    def interpolation_future_trajectory(self, aug_current_state, ego_future, keep_remaining=True):
        """
        refine future trajectory with quintic Hermite interpolation

        Args:
            aug_current_state: (B, 16) current state of the ego vehicle after augmentation
            ego_future:        (B, T, 3) future trajectory of the ego vehicle
            keep_remaining:    If True, keep the remaining trajectory after P frames (default: True)

        Returns:
            ego_future: refined future trajectory of the ego vehicle
        """

        P = self.num_refine
        dt = self.time_interval
        B = aug_current_state.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(B, -1, -1)
        A = self.coeff_matrix.unsqueeze(0).expand(B, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]

        x0, y0, theta0, v0, a0, omega0 = (
            aug_current_state[:, 0],
            aug_current_state[:, 1],
            torch.atan2(
                (ego_future[:, int(P / 2), 1] - aug_current_state[:, 1]),
                (ego_future[:, int(P / 2), 0] - aug_current_state[:, 0]),
            ),
            torch.norm(aug_current_state[:, 4:6], dim=-1),
            torch.norm(aug_current_state[:, 6:8], dim=-1),
            aug_current_state[:, 9],
        )

        xT, yT, thetaT, vT, aT, omegaT = (
            ego_future[:, P, 0],
            ego_future[:, P, 1],
            ego_future[:, P, 2],
            torch.norm(ego_future[:, P, :2] - ego_future[:, P - 1, :2], dim=-1) / dt,
            torch.norm(
                ego_future[:, P, :2] - 2 * ego_future[:, P - 1, :2] + ego_future[:, P - 2, :2],
                dim=-1,
            )
            / dt**2,
            self.normalize_angle(ego_future[:, P, 2] - ego_future[:, P - 1, 2]) / dt,
        )

        # Boundary conditions
        sx = torch.stack(
            [
                x0,
                v0 * torch.cos(theta0),
                a0 * torch.cos(theta0) - v0 * torch.sin(theta0) * omega0,
                xT,
                vT * torch.cos(thetaT),
                aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT,
            ],
            dim=-1,
        )

        sy = torch.stack(
            [
                y0,
                v0 * torch.sin(theta0),
                a0 * torch.sin(theta0) + v0 * torch.cos(theta0) * omega0,
                yT,
                vT * torch.sin(thetaT),
                aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT,
            ],
            dim=-1,
        )

        ax = A @ sx[:, :, None]  # B, 6, 1
        ay = A @ sy[:, :, None]  # B, 6, 1

        traj_x = M_t @ ax
        traj_y = M_t @ ay
        traj_heading = torch.cat(
            [
                torch.atan2(
                    traj_y[:, :1, 0] - y0.unsqueeze(-1), traj_x[:, :1, 0] - x0.unsqueeze(-1)
                ),
                torch.atan2(
                    traj_y[:, 1:, 0] - traj_y[:, :-1, 0], traj_x[:, 1:, 0] - traj_x[:, :-1, 0]
                ),
            ],
            dim=1,
        )

        interpolated = torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1)

        if keep_remaining and ego_future.shape[1] > P:
            return torch.concatenate([interpolated, ego_future[:, P:, :]], axis=1)
        else:
            return interpolated


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
        data_dict["neighbor_agents_future"] = aug_neighbors_future.squeeze(0).detach().cpu().numpy()
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
