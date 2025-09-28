from typing import List, Optional, Union

import numpy as np
import torch

NUM_REFINE = 20
REFINE_HORIZON = 2.0
TIME_INTERVAL = 0.1


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
    nexpand = heading.ndim - 1
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
    Data augmentation that perturbs the current ego position and generates a feasible trajectory that
    satisfies polynomial constraints.
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        normalize=True,
        wheel_base: float = 2.75,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        """
        Initialize the augmentor,
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        """
        self._augment_prob = augment_prob
        self._normalize = normalize
        self._device = torch.device(device)
        lo: List[float] = ([0.0, -0.75, -0.2, -1, -0.5, -0.2, -0.1, 0.0, 0.0],)
        hi: List[float] = ([0.0, +0.75, +0.2, +1, +0.5, +0.2, +0.1, 0.0, 0.0],)
        self._low = torch.tensor(lo).to(self._device)
        self._high = torch.tensor(hi).to(self._device)
        self._wheel_base = wheel_base

        self.refine_horizon = REFINE_HORIZON
        self.num_refine = NUM_REFINE
        self.time_interval = TIME_INTERVAL

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
        aug_flag = (torch.rand(B) >= self._augment_prob).bool().to(self._device) & ~(
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

        return aug_flag, ego_current_state

    def normalize_angle(
        self, angle: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
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

    def interpolation_future_trajectory(self, aug_current_state, ego_future):
        """
        refine future trajectory with quintic spline interpolation

        Args:
            aug_current_state: (B, 16) current state of the ego vehicle after augmentation
            ego_future:        (B, 80, 3) future trajectory of the ego vehicle

        Returns:
            ego_future: refined future trajectory of the ego vehicle
        """

        P = self.num_refine
        dt = self.time_interval
        T = self.refine_horizon
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

        return torch.concatenate(
            [torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1), ego_future[:, P:, :]],
            axis=1,
        )
