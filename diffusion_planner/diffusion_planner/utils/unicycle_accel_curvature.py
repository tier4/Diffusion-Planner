# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union

import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch import nn

logger = logging.getLogger(__name__)


TensorOrNDArray = TypeVar("TensorOrNDArray", torch.Tensor, np.ndarray)


def so3_to_yaw_torch(rot_mat: torch.Tensor) -> torch.Tensor:
    """Computes the yaw angle given an so3 rotation matrix (assumes that rotation is described in
    xyz order)

    Args:
        rot_mat (torch.Tensor): [..., 3,3]

    Returns:
        torch.Tensor: [...]
    """
    # phi is rotation about z, theta is rotation about y
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)


def so3_to_yaw_np(rot_mat: np.ndarray) -> np.ndarray:
    """Computes the yaw angle given an so3 rotation matrix (assumes that rotation is described in
    xyz order)

    Args:
        rot_mat (np.ndarray): [..., 3,3]

    Returns:
        np.ndarray: [...]
    """
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return np.arctan2(cos_th_sin_phi, cos_th_cos_phi)


def euler_2_so3(euler_angles: np.ndarray, degrees: bool = True, seq: str = "xyz") -> np.ndarray:
    """Converts the euler angles representation to the so3 rotation matrix
    Args:
        euler_angles (np.array): euler angles [n,3]
        degrees bool: True if angle is given in degrees else False
        seq string: sequence in which the euler angles are given

    Out:
        (np array): rotations given so3 matrix representation [n,3,3]
    """
    return (
        R.from_euler(seq=seq, angles=euler_angles, degrees=degrees).as_matrix().astype(np.float32)
    )


def angle_wrap(
    radians: TensorOrNDArray,
) -> TensorOrNDArray:
    """This function wraps angles to lie within [-pi, pi).

    Args:
        radians (np.ndarray): The input array of angles (in radians).

    Returns:
        np.ndarray: Wrapped angles that lie within [-pi, pi).
    """
    return (radians + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(angle: Union[float, np.ndarray]) -> np.ndarray:
    """Creates one or many 2D rotation matrices.

    Args:
        angle (Union[float, np.ndarray]): The angle to rotate points by.
            if float, returns 2x2 matrix
            if np.ndarray, expects shape [...], and returns [...,2,2] array

    Returns:
        np.ndarray: The 2x2 rotation matri(x/ces).
    """
    batch_dims = 0
    if isinstance(angle, np.ndarray):
        batch_dims = angle.ndim

    rotmat: np.ndarray = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    return rotmat.transpose(*np.arange(2, batch_dims + 2), 0, 1)


def rotation_matrix_torch(angle: torch.Tensor) -> torch.Tensor:
    """Creates one or many 2D rotation matrices.

    Args:
        angle (torch.Tensor): The angle to rotate points by. Size: [...].

    Returns:
        torch.Tensor: The 2x2 rotation matri(x/ces). Size: [..., 2, 2].
    """
    rotmat: torch.Tensor = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )
    return rotmat


def transform_coords_2d_np(
    coords: np.ndarray,
    offset: Optional[np.ndarray] = None,
    angle: Optional[np.ndarray] = None,
    rot_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Args:
        coords (np.ndarray): [..., 2] coordinates
        offset (Optional[np.ndarray], optional): [..., 2] offset to translate. Defaults to None.
        angle (Optional[np.ndarray], optional): [...] angle to rotate by. Defaults to None.
        rot_mat (Optional[np.ndarray], optional): [..., 2,2] rotation matrix to apply. Defaults to None.
            If rot_mat is given, angle is ignored.

    Returns:
        np.ndarray: transformed coords
    """
    if rot_mat is None and angle is not None:
        rot_mat = rotation_matrix(angle)

    if rot_mat is not None:
        coords = np.einsum("...ij,...j->...i", rot_mat, coords)

    if offset is not None:
        coords += offset

    return coords


def stable_gramschmidt(M: torch.Tensor) -> torch.Tensor:
    """Orthonormalize two 3D vectors using a stable Gram-Schmidt step.

    Args:
        M: Tensor of shape (..., 3, 2) with vectors (x, y).

    Returns:
        Tensor of shape (..., 3, 3) with orthonormal (x, y, x×y).
    """
    EPS = 1e-7

    x = M[..., 0]
    y = M[..., 1]
    x = x / torch.clamp_min(torch.norm(x, dim=-1, keepdim=True), EPS)
    y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = y / torch.clamp_min(torch.norm(y, dim=-1, keepdim=True), EPS)
    z = torch.cross(x, y, dim=-1)
    R = torch.stack((x, y, z), dim=-1)
    return R


def rot_3d_to_2d(rot):
    """Converts a 3D rotation matrix to a 2D rotation matrix by taking the x and y axes of the 3D
    rotation matrix, projecting them to xy plan, and performing gram-schmidt orthogonalization.

    Args:
        rot (torch.Tensor): The 3D rotation matrix to convert.

    Returns:
        torch.Tensor: The 2D rotation matrix.
    """
    xu = rot[..., :2, 0]
    yu = rot[..., :2, 1]
    EPS = 1e-6
    # gram-schmidt
    xu = xu / (torch.norm(xu, dim=-1, keepdim=True) + EPS)
    yu = yu - torch.sum(xu * yu, dim=-1, keepdim=True) * xu
    yu = yu / (torch.norm(yu, dim=-1, keepdim=True) + EPS)
    return torch.stack((xu, yu), dim=-1)


def rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    """Converts a 2D rotation matrix to a 3D rotation matrix assuming flat xy plane.

    Args:
        rot (torch.Tensor): The 2D rotation matrix to convert.

    Returns:
        torch.Tensor: The 3D rotation matrix.
    """
    rot = torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device).repeat(rot.shape[:-2] + (1, 1)),
        ],
        dim=-2,
    )
    return rot


def traj4d_to_action(
    action_space: "UnicycleAccelCurvatureActionSpace",
    traj_history_4d: torch.Tensor,
    traj_future_4d: torch.Tensor,
    t0_states: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Convert 4D ego trajectories (x, y, cos, sin) into acceleration/steering actions.

    Args:
        action_space: Action space helper that performs the conversion.
        traj_history_4d: [B, Th, 4] past ego trajectory.
        traj_future_4d: [B, T, 4] future ego trajectory to convert.

    Returns:
        torch.Tensor: [B, T, 2] action sequence (acceleration, steering curvature).
    """
    zeros_hist = torch.zeros_like(traj_history_4d[..., :1])
    zeros_future = torch.zeros_like(traj_future_4d[..., :1])
    traj_history_xyz = torch.cat([traj_history_4d[..., :2], zeros_hist], dim=-1)
    traj_future_xyz = torch.cat([traj_future_4d[..., :2], zeros_future], dim=-1)

    cos_sin_history = traj_history_4d[..., 2:]
    heading_history = torch.atan2(cos_sin_history[..., 1], cos_sin_history[..., 0])
    traj_history_rot = rot_2d_to_3d(rotation_matrix_torch(heading_history))

    cos_sin_future = traj_future_4d[..., 2:]
    heading_future = torch.atan2(cos_sin_future[..., 1], cos_sin_future[..., 0])
    traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(heading_future))

    return action_space.traj_to_action(
        traj_history_xyz=traj_history_xyz,
        traj_history_rot=traj_history_rot,
        traj_future_xyz=traj_future_xyz,
        traj_future_rot=traj_future_rot,
        t0_states=t0_states,
    )


def action_to_traj4d(
    action_space: "UnicycleAccelCurvatureActionSpace",
    traj_history_4d: torch.Tensor,
    actions: torch.Tensor,
    t0_states: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Convert normalized actions back to 4D trajectory."""
    zeros_hist = torch.zeros_like(traj_history_4d[..., :1])
    traj_history_xyz = torch.cat([traj_history_4d[..., :2], zeros_hist], dim=-1)
    cos_sin_history = traj_history_4d[..., 2:]
    heading_history = torch.atan2(cos_sin_history[..., 1], cos_sin_history[..., 0])
    traj_history_rot = rot_2d_to_3d(rotation_matrix_torch(heading_history))

    traj_future_xyz, traj_future_rot = action_space.action_to_traj(
        actions, traj_history_xyz, traj_history_rot, t0_states=t0_states
    )
    heading_future = so3_to_yaw_torch(traj_future_rot)
    traj_future_4d = torch.stack(
        [
            traj_future_xyz[..., 0],
            traj_future_xyz[..., 1],
            torch.cos(heading_future),
            torch.sin(heading_future),
        ],
        dim=-1,
    )
    return traj_future_4d


def ratan2(s, c, eps=1e-4):
    """Robust arctan2 for pytorch
    torch.arctan2(0,0)=nan, this function avoids the nan situation and returns ratan2(0,0)=0
    """
    sign = (c >= 0).float() * 2 - 1
    eps = eps * (c.abs() < eps).type(c.dtype) * sign
    return torch.arctan2(s, c + eps)


def round_2pi(x: np.ndarray) -> np.ndarray:
    """Normalize angles to the range [-pi, pi].

    Args:
        x: Angle(s) in radians, can be numpy array or torch tensor

    Returns:
        Normalized angle(s) in the range [-pi, pi], same type as input
    """
    return np.atan2(np.sin(x), np.cos(x))


def round_2pi_torch(x: torch.Tensor) -> torch.Tensor:
    """Normalize angles to the range [-pi, pi] in torch.

    Args:
        x: Angle(s) in radians, can be numpy array or torch tensor

    Returns:
        Normalized angle(s) in the range [-pi, pi], same type as input
    """
    return torch.atan2(torch.sin(x), torch.cos(x))


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """Unwrap the last dimension of the tensor to make sure the diff is in (-pi, pi]."""
    d = torch.diff(phi, dim=-1)
    d = round_2pi_torch(d)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def first_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the banded matrix for the first-order smoothing term."""
    D = torch.zeros(*lead_shape, N - 1, N, dtype=dtype, device=device)
    rows = torch.arange(N - 1, device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 1.0
    return D


def second_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the banded matrix for the second-order smoothing term."""
    D = torch.zeros(*lead_shape, max(N - 2, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 2, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 2.0
    D[..., rows, rows + 2] = -1.0
    return D


def third_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the banded matrix for the third-order smoothing term."""
    D = torch.zeros(*lead_shape, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def construct_DTD(
    N: int,
    lead: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    """Construct the dense matrix D^T s D for multiple orders of smoothing.

    Explanation of the smoothing lambda term:

    For 1st/2nd/3rd smoothing term, we are multiplying the DTD by 1/dt**2, 1/dt**4 and 1/dt**6
    respectively. The reason is that, for example for the 2nd order smoothing term, we are
    minimizing the following term:
        sum_i={0, ..., N-1} lambda * w_smooth2_i * (d^2 x_i / dt^2)**2
    After taking the derivative, we will get the following on the LHS in the normal equation:
        lambda / dt**4 * w_smooth2_i * DTD_2nd.
    Similar explanation applies to the 1st and 3rd order smoothing terms.

    Args:
        N: int, the length of the solving variables.
        lead: tuple, the shape of the leading dimensions of the output matrix.
        device: torch.device, the device of the output matrix.
        dtype: torch.dtype, the dtype of the output matrix.
        w_smooth1: float | torch.Tensor | None, the weight for the first-order smoothing term.
        w_smooth2: float | torch.Tensor | None, the weight for the second-order smoothing term.
        w_smooth3: float | torch.Tensor | None, the weight for the third-order smoothing term.
        lam: float, the weight for the smoothing term.
        dt: float, the time step.

    Returns:
        DTD: torch.Tensor, the dense matrix D^T s D for multiple orders of smoothing.
    """
    DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)
    if w_smooth1 is not None:
        lam_1 = lam / dt**2
        if isinstance(w_smooth1, float):
            w_smooth1_tensor = torch.full(
                (*lead, max(N - 1, 0)), w_smooth1, dtype=dtype, device=device
            )
        else:
            w_smooth1_tensor = w_smooth1
        D1 = first_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_1 * einops.einsum(
            D1 * w_smooth1_tensor.unsqueeze(-1), D1, "... i j, ... i k -> ... j k"
        )

    if w_smooth2 is not None:
        lam_2 = lam / dt**4
        if isinstance(w_smooth2, float):
            w_smooth2_tensor = torch.full(
                (*lead, max(N - 2, 0)), w_smooth2, dtype=dtype, device=device
            )
        else:
            w_smooth2_tensor = w_smooth2
        D2 = second_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_2 * einops.einsum(
            D2 * w_smooth2_tensor.unsqueeze(-1), D2, "... i j, ... i k -> ... j k"
        )

    if w_smooth3 is not None:
        lam_3 = lam / dt**6
        if isinstance(w_smooth3, float):
            w_smooth3_tensor = torch.full(
                (*lead, max(N - 3, 0)), w_smooth3, dtype=dtype, device=device
            )
        else:
            w_smooth3_tensor = w_smooth3
        D3 = third_order_D(N, lead, device=device, dtype=dtype)

        DTD += lam_3 * einops.einsum(
            D3 * w_smooth3_tensor.unsqueeze(-1), D3, "... i j, ... i k -> ... j k"
        )

    return DTD


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Solve a single-point constrained sequence with multiple orders of smoothing.

    This function solves the following problem:
        min_x={x_1, ..., x_N} sum_i={0, ..., N-1} w_data_i (x_i - x_target_i)**2 + smooth_terms
        subject to:
            x_0 = x_init

    Args:
        x_init: the initial value.
        x_target: the target value.
        w_data: the weight for the data term.
        w_smooth1: the weight for the first-order smoothing term.
        w_smooth2: the weight for the second-order smoothing term.
        w_smooth3: the weight for the third-order smoothing term.
        lam: the weight for the smoothing term.
        ridge: the ridge for regularization.
        dt: the time step.

    Returns:
        x: the solved value.
    """
    device, dtype = x_target.device, x_target.dtype
    *lead, N = x_target.shape
    if N <= 0:
        raise ValueError("x_mid must have a positive last-dimension length N.")
    if w_data is None:
        w_data = torch.ones_like(x_target)
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)

    # Solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, x_target, "... i j, ... i -> ... j")

    # The dim is N + 1 because we have x_init as the first element
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * x_init.unsqueeze(-1)

    ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    # strip off the x_init term
    lhs = ATA + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N)

    x = torch.cat([x_init.unsqueeze(-1), x], dim=-1)  # (..., N+1)
    return x


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_xs_eq_y(
    s: torch.Tensor,
    y: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Solve the following problem:

    min_x={x_0, ..., x_N-1} sum_i={0, ..., N-1} w_data_i (x_i * s_i - y_i)**2 + smooth_terms

    Args:
        s (..., N): the slope
        y (..., N): the y-value
        w_data (..., N): the weight for the data term
        w_smooth1: the weight for the first-order smoothing term
        w_smooth2: the weight for the second-order smoothing term
        w_smooth3: the weight for the third-order smoothing term
        lam: the weight for smoothness term
        ridge: the ridge for regularization
        dt: the time step

    Returns:
        x: the solved value.
    """
    device, dtype = y.device, y.dtype
    *lead, N = y.shape
    if w_data is None:
        w_data = torch.ones_like(y)
    if w_data.shape != y.shape:
        raise ValueError("w_data must have the same shape as y")

    # Solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.diag_embed(s)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, y, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )

    # NOTE: Since there is no terminal constraint, we need to handle the singularity case by
    # increasing the ridge term.
    L = None
    while L is None:
        try:
            ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
            lhs = ATA + DTD + ridge_term
            # Ensure dtype consistency for torch.compile fake tensor meta pass
            if rhs.dtype != lhs.dtype:
                rhs = rhs.to(lhs.dtype)
            L = torch.linalg.cholesky(lhs)
        except RuntimeError as e:
            logger.error(f"Error in cholesky decomposition: {e}", exc_info=True)
            ridge *= 10
            logger.warning(f"Resolving singularity using ridge {ridge}")

    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v_without_v0(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    """Given the dxy and theta, compute the velocity.
    The velocity is defined by the trapezoidal integration:

    define:
        u_t = [cos theta_t, sin theta_t]
        Δp_t = p_t+1 - p_t
    We have:
        Δp_t = dt / 2 * (v_t u_t + v_t+1 u_t+1)
        v_t u_t + v_t+1 u_t+1 = 2 * Δp_t / dt
    => Solve v_t, t=0, ..., N by least squares

    This function is shared by the accel and jerk curvature action spaces.

    Args:
        dxy: (..., N, 2) p_t+1 - p_t for t=0, ..., N
        theta: (..., N+1)
        dt: float, the time step
        v_lambda: float, the lambda for the velocity smoothing term
        v_ridge: float, the ridge for the velocity regularization term

    Returns:
        v: (..., N+1) the estimated velocity from 0 to N
    """
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy  # (..., N, 2)

    w = torch.ones_like(dxy[..., 0])

    # solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, b_data, "... i j, ... i -> ... j")

    # The dim is N + 1 because we have x_init as the first element
    # NOTE: for Tikhonov regularization
    # 1st order means we want small acceleration
    # 2nd order means we want small jerk
    # 3rd order means we want small difference between jerk
    # We use 3rd order here as we do not want to penalize the jerk itself directly but only
    # smoothness of the jerk.
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )

    ridge_term = v_ridge * torch.eye(N + 1, dtype=dtype, device=device).expand(*lead, N + 1, N + 1)
    # strip off the x_init term
    lhs = ATA + DTD + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N+1)

    return y  # (..., N+1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    """Given the dxy and theta, compute the velocity.
    The velocity is defined by the trapezoidal integration:

    define:
        u_t = [cos theta_t, sin theta_t]
        Δp_t = p_t+1 - p_t
    We have:
        Δp_t = dt / 2 * (v_t u_t + v_t+1 u_t+1)
        v_t u_t + v_t+1 u_t+1 = 2 * Δp_t / dt
    => Solve v_t, t=1, ..., N by least squares
    Args:
        dxy: (..., N, 2) p_t+1 - p_t for t=0, ..., N
        theta: (..., N+1)
        v0: (...,)

    Returns:
        v: (..., N+1) the estimated velocity from 0 to N
    """
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy  # (..., N, 2)

    w = torch.ones_like(dxy[..., 0])

    # solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        # rhs is A^T w_data b, but we need to include the x_init terms into the rhs as it is a
        # constant.
        rhs = einops.einsum(Aw_data[..., :, 1:], b_data, "... i j, ... i -> ... j")
    rhs -= ATA[..., 1:, 0] * v0.unsqueeze(-1)

    # The dim is N + 1 because we have x_init as the first element
    # NOTE: for Tikhonov regularization
    # 1st order means we want small acceleration
    # 2nd order means we want small jerk
    # 3rd order means we want small difference between jerk
    # We use 3rd order here as we do not want to penalize the jerk itself directly but only
    # smoothness of the jerk.
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * v0.unsqueeze(-1)

    ridge_term = v_ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    # strip off the x_init term
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N)

    return torch.cat([v0.unsqueeze(-1), y], dim=-1)  # (..., N+1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def theta_smooth(
    traj_future_rot: torch.Tensor,
    dt: float = 1.0,
    theta_lambda: float = 1e-4,
    theta_ridge: float = 1e-4,
) -> torch.Tensor:
    """Smooth the heading of the trajectory.

    Args:
        traj_future_rot: (..., T, 3, 3)
    """
    theta = so3_to_yaw_torch(traj_future_rot)
    theta = unwrap_angle(theta)
    theta_init = torch.zeros_like(theta[..., 0])
    return solve_single_constraint(
        x_init=theta_init,
        x_target=theta,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )


class ActionSpace(ABC, nn.Module):
    """Action space base class for the trajectory generation."""

    @abstractmethod
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Transform the future trajectory to the action space.

        Args:
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
            *args: other data for the action space
            **kwargs: other data for the action space

        Returns:
            action: (..., *action_space_dims)
        """

    @abstractmethod
    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the action space to the trajectory.

        Args:
            action: (..., *action_space_dims)
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            *args: other data for the action space
            **kwargs: other data for the action space

        Returns:
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
        """

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]:
        """Get the dimensions of the action space.

        Returns:
            action_space_dims: the action space dimensions
        """

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Check if the action is within the bounds.

        By default, we assume the action is within bounds (dummy implementation).

        Args:
            action: (..., *action_space_dims)

        Returns:
            is_within_bounds: (...,)
        """
        num_action_dims = len(self.get_action_space_dims())
        batch_shape = action.shape[:-num_action_dims] if num_action_dims > 0 else action.shape
        return torch.ones(batch_shape, dtype=torch.bool, device=action.device)


class UnicycleAccelCurvatureActionSpace(ActionSpace):
    """Unicycle Kinematic Model with acceleration and curvature as control inputs."""

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),  # min and max bounds for accel
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),  # min and max bounds for curvature
        dt: float = 0.1,
        n_waypoints: int = 80,
        theta_lambda: float = 1e-2,
        theta_ridge: float = 1e-4,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ):
        """Initialize the UnicycleAccelCurvatureActionSpace.

        Args:
            accel_mean: Mean for normalizing acceleration.
            accel_std: Std for normalizing acceleration.
            curvature_mean: Mean for normalizing curvature.
            curvature_std: Std for normalizing curvature.
            accel_bounds: Acceleration bounds (min, max).
                This value is used to check if the acceleration is within bounds.
            curvature_bounds: Curvature bounds (min, max).
                This value is used to check if the curvature is within bounds.
            dt: Time step interval.
            n_waypoints: Number of waypoints in the trajectory.
            theta_lambda: Lambda parameter for theta smoothing.
            theta_ridge: Ridge parameter for theta smoothing.
            v_lambda: Lambda parameter for velocity smoothing.
            v_ridge: Ridge parameter for velocity smoothing.
            a_lambda: Lambda parameter for acceleration smoothing.
            a_ridge: Ridge parameter for acceleration smoothing.
            kappa_lambda: Lambda parameter for curvature smoothing.
            kappa_ridge: Ridge parameter for curvature smoothing.
        """
        super().__init__()
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))
        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge
        self.a_lambda = a_lambda
        self.a_ridge = a_ridge
        self.kappa_lambda = kappa_lambda
        self.kappa_ridge = kappa_ridge

    def get_action_space_dims(self) -> tuple[int, int]:
        """Get the dimensions of the action space."""
        return (self.n_waypoints, 2)

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Check if a normalized action is within bounds.

        Args:
            action: (..., N, 2)

        Returns:
            is_within_bounds: (...,)
        """
        accel = action[..., 0]
        kappa = action[..., 1]
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean
        is_accel_within_bounds = (accel >= self.accel_bounds[0]) & (accel <= self.accel_bounds[1])
        is_kappa_within_bounds = (kappa >= self.curvature_bounds[0]) & (
            kappa <= self.curvature_bounds[1]
        )
        return torch.all(is_accel_within_bounds & is_kappa_within_bounds, dim=-1)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the acceleration from the velocity.

        Define:
            Δv_t = v_t+1 - v_t

        According to the kinematic model
            Δv_t = dt * a_t

        => solve it by single-constrained solver

        Args:
            v: (..., N+1)

        Returns:
            a: (..., N)
        """
        dv = (v[..., 1:] - v[..., :-1]) / self.dt  # (..., N)
        # NOTE: for Tikhonov regularization
        # 1st order means we want small jerk
        # 2nd order means we want small difference between jerk
        # We use 2nd order here as we do not want to penalize the jerk itself directly but only
        # smoothness of the jerk.
        a = solve_xs_eq_y(
            s=torch.ones_like(dv),
            y=dv,
            dt=self.dt,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
        )
        return a

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the curvature from the theta, v, a, jerk.

        The kappa is computed by

            s = dt * v + dt^2 * a / 2
            kappa = dtheta / s

        where dtheta is the unwrapped heading difference.

        Args:
            theta: (..., N+1) unwrapped heading
            v: (..., N+1) velocity
            a: (..., N) acceleration

        Returns:
            kappa: (..., N)
        """
        dtheta = theta[..., 1:] - theta[..., :-1]  # (..., N)
        dt = self.dt
        s = dt * v[..., :-1] + (dt**2) / 2.0 * a  # (..., N)

        w = torch.ones_like(dtheta)
        # NOTE: for Tikhonov regularization
        # 1st order means we want small kappa 1st order difference
        # 2nd order means we want small kappa 2nd order difference
        return solve_xs_eq_y(
            s=s,
            y=dtheta,
            w_data=w,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self, traj_history_xyz: torch.Tensor, traj_history_rot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Estimate the t0 states from the trajectory history."""
        full_xy = traj_history_xyz[..., :2]  # (..., N_hist, 2)
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]  # (..., N_hist-1, 2)
        theta = so3_to_yaw_torch(traj_history_rot)
        theta = unwrap_angle(theta)

        v = dxy_theta_to_v_without_v0(
            dxy=dxy, theta=theta, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge
        )  # (..., N+1)
        v_t0 = v[..., -1]
        return {"v": v_t0}

    @torch.no_grad()
    @torch._dynamo.disable()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
        output_all_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Transform the future trajectory to the action space.

        Here we assume the traj_history_xyz[..., -1, :] is the current position and is all zeros.

        Args:
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
            t0_states: initial state estimate
            output_all_states: whether to output all the states

        Returns:
            action: (..., T, 2)
        """
        # Validate inputs
        if traj_future_xyz.shape[-2] != self.n_waypoints:
            raise ValueError(
                f"future trajectory must have length {self.n_waypoints} "
                f"but got {traj_future_xyz.shape[-2]}"
            )

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        # Concatenate last history and future
        # NOTE: we assume the traj_history_xyz[..., -1, :] is the current position and it is all
        # zero.
        full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[
            ..., :2
        ]  # (..., N+1, 2)

        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]  # (..., N, 2)
        theta = theta_smooth(
            traj_future_rot=traj_future_rot,
            dt=self.dt,
            theta_lambda=self.theta_lambda,
            theta_ridge=self.theta_ridge,
        )

        v0 = t0_states["v"]
        v = dxy_theta_to_v(
            dxy=dxy, theta=theta, v0=v0, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge
        )  # (..., N+1)

        accel = self._v_to_a(v)  # (..., N+1), (..., N)

        kappa = self._theta_v_a_to_kappa(theta, v, accel)  # (..., N)

        # normalize acceleration and kappa
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = (accel - accel_mean) / accel_std
        kappa = (kappa - kappa_mean) / kappa_std

        if not output_all_states:
            return torch.stack([accel, kappa], dim=-1)  # (..., N, 2)
        else:
            return torch.stack([accel, kappa], dim=-1), torch.stack(
                [v[:, :-1], accel, theta[:, :-1]], dim=-1
            )

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the action space to the trajectory.

        Args:
            action: (..., T, 2)
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            t0_states: initial state estimate

        Returns:
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
        """
        accel, kappa = action[..., 0], action[..., 1]

        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        v0 = t0_states["v"]
        dt = self.dt

        dt_2_term = 0.5 * (self.dt**2)
        velocity = torch.cat(
            [
                v0.unsqueeze(-1),
                (v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)),
            ],
            dim=-1,
        )  # (..., N+1)
        initial_yaw = torch.zeros_like(v0)
        theta = torch.cat(
            [
                initial_yaw.unsqueeze(-1),
                (
                    initial_yaw.unsqueeze(-1)
                    + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                    + torch.cumsum(kappa * accel * dt_2_term, dim=-1)
                ),
            ],
            dim=-1,
        )  # (..., N+1)
        half_dt_term = 0.5 * dt
        initial_x = torch.zeros_like(v0)
        initial_y = torch.zeros_like(v0)
        x = (
            initial_x.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt_term, dim=-1)
        )  # (..., N)
        y = (
            initial_y.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt_term, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt_term, dim=-1)
        )  # (..., N)
        batch_dim = traj_history_xyz.shape[:-2]
        traj_future_xyz = torch.zeros(
            *batch_dim,
            self.n_waypoints,
            3,
            device=traj_history_xyz.device,
            dtype=traj_history_xyz.dtype,
        )
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        # Handle only_xy case for output
        traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]

        traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))

        return traj_future_xyz, traj_future_rot


def smoothing_future_trajectory(
    ego_agent_past: torch.Tensor, ego_current_state: torch.Tensor, ego_future: torch.Tensor
) -> torch.Tensor:
    action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=80)
    t0_states = {"v": ego_current_state[:, 4]}
    ego_actions = traj4d_to_action(
        action_space,
        ego_agent_past,
        ego_future,
        t0_states=t0_states,
    )
    smoothed_ego_future = action_to_traj4d(
        action_space,
        ego_agent_past,
        ego_actions,
        t0_states=t0_states,
    )
    return smoothed_ego_future
