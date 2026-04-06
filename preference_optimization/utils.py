"""Utility functions for preference optimization."""

from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from diffusion_planner.train_epoch import heading_to_cos_sin


def load_npz_data(npz_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load NPZ file into tensors on the specified device.

    Args:
        npz_path: Path to the NPZ file
        device: Device to load tensors onto

    Returns:
        Dictionary of tensors with observation data
    """
    with np.load(str(npz_path)) as loaded:
        data: dict[str, torch.Tensor] = {}

        for key, value in loaded.items():
            if key in {"map_name", "token", "delay"}:
                continue
            data[key] = torch.tensor(np.expand_dims(value, axis=0)).to(device)

    if "goal_pose" in data:
        data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
    if "ego_agent_past" in data:
        data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])

    if "ego_shape" not in data:
        wheel_base = 2.79
        ego_length = 4.34
        ego_width = 1.70
        data["ego_shape"] = torch.tensor(
            [[wheel_base, ego_length, ego_width]], dtype=torch.float32, device=device
        )

    # v4 decoder requires delay (always 0 at inference, training uses random delay)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)

    return data


def calculate_path_length(trajectory: np.ndarray) -> float:
    """Calculate negative path length (longer paths = smaller values for preference ranking).

    Args:
        trajectory: Trajectory array [T, 4] (x, y, heading, velocity)

    Returns:
        Negative sum of distances between consecutive points
    """
    xy = trajectory[:, :2]
    diffs = np.diff(xy, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(-np.sum(dists))


def calculate_fde(trajectory_1: np.ndarray, trajectory_2: np.ndarray) -> float:
    """Calculate Final Displacement Error between two trajectory endpoints.

    Args:
        trajectory_1: First trajectory [T, 4] (x, y, heading, velocity)
        trajectory_2: Second trajectory [T, 4]

    Returns:
        Euclidean distance between final positions (in meters)
    """
    final_pos_1 = trajectory_1[-1, :2]
    final_pos_2 = trajectory_2[-1, :2]
    fde = np.linalg.norm(final_pos_1 - final_pos_2)
    return float(fde)


def calculate_ade(trajectory_1: np.ndarray, trajectory_2: np.ndarray) -> float:
    """Calculate Average Displacement Error between two trajectories.

    ADE is the mean Euclidean distance across all timesteps.

    Args:
        trajectory_1: First trajectory [T, 4] (x, y, cos, sin)
        trajectory_2: Second trajectory [T, 4] or [T, 3] (x, y, heading)

    Returns:
        Mean Euclidean distance across all timesteps (in meters)
    """
    positions_1 = trajectory_1[:, :2]
    positions_2 = trajectory_2[:, :2]
    displacements = np.sqrt(np.sum((positions_1 - positions_2) ** 2, axis=1))
    return float(np.mean(displacements))


def calculate_initial_position_displacement(
    trajectory_1: np.ndarray, trajectory_2: np.ndarray
) -> float:
    """Calculate Euclidean distance between the first poses of two trajectories.

    Args:
        trajectory_1: First trajectory [T, 4] (x, y, cos(yaw), sin(yaw))
        trajectory_2: Second trajectory [T, 4] (x, y, cos(yaw), sin(yaw))

    Returns:
        Euclidean distance between the initial (x, y) positions (in meters)
    """
    return float(np.linalg.norm(trajectory_1[0, :2] - trajectory_2[0, :2]))


def calculate_initial_yaw_difference(
    trajectory_1: np.ndarray, trajectory_2: np.ndarray
) -> float:
    """Calculate the absolute yaw difference at the first pose of two trajectories.

    Yaw is extracted from cos/sin encoding (columns 2 and 3) using atan2.
    The difference is wrapped to [-pi, pi] before taking the absolute value,
    so the result is always in [0, 180] degrees.

    Args:
        trajectory_1: First trajectory [T, 4] (x, y, cos(yaw), sin(yaw))
        trajectory_2: Second trajectory [T, 4] (x, y, cos(yaw), sin(yaw))

    Returns:
        Absolute yaw difference in degrees, in range [0, 180]
    """
    yaw_1 = np.arctan2(trajectory_1[0, 3], trajectory_1[0, 2])
    yaw_2 = np.arctan2(trajectory_2[0, 3], trajectory_2[0, 2])
    diff = (yaw_2 - yaw_1 + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.abs(diff)))


def should_prune_by_initial_pose(
    trajectory_1: np.ndarray,
    trajectory_2: np.ndarray,
    pos_threshold_m: float,
    yaw_threshold_deg: float,
) -> tuple[bool, float, float]:
    """Determine whether a trajectory pair should be pruned based on initial pose alignment.

    Pruning is triggered when the initial position displacement OR the initial yaw
    difference between the two trajectories exceeds its respective threshold.

    Args:
        trajectory_1: Deterministic trajectory [T, 4] (x, y, cos(yaw), sin(yaw))
        trajectory_2: Stochastic trajectory [T, 4] (x, y, cos(yaw), sin(yaw))
        pos_threshold_m: Maximum allowed initial position displacement (meters)
        yaw_threshold_deg: Maximum allowed initial yaw difference (degrees)

    Returns:
        Tuple of (should_prune, displacement_m, yaw_diff_deg)
        - should_prune: True if displacement > pos_threshold_m OR yaw_diff > yaw_threshold_deg
        - displacement_m: Euclidean distance between initial positions
        - yaw_diff_deg: Absolute yaw difference in degrees
    """
    displacement = calculate_initial_position_displacement(trajectory_1, trajectory_2)
    yaw_diff = calculate_initial_yaw_difference(trajectory_1, trajectory_2)
    should_prune = displacement > pos_threshold_m or yaw_diff > yaw_threshold_deg
    return should_prune, displacement, yaw_diff


@torch.no_grad()
def generate_trajectory_pair(
    policy_model,
    model_args,
    data: dict[str, torch.Tensor],
    noise_scale: float = 2.5,
    fde_threshold: float = 2.0,
    ade_threshold: float = 1.0,
    max_retries: int = 50,
    device: torch.device | None = None,
    gt_similarity_mode: bool = True,
    gt_trajectory: np.ndarray | None = None,
    enable_initial_pruning: bool = True,
    initial_pos_threshold: float = 0.055,
    initial_yaw_threshold_deg: float = 0.55,
    # New unified guidance parameter:
    guidance: GuidanceSetConfig | None = None,
    # Deprecated: kept for backward compatibility, ignored when guidance is not None.
    enable_guidance: bool = False,
    use_collision: bool = True,
    use_route_following: bool = False,
    use_lane_keeping: bool = False,
    use_centerline_following: bool = False,
    guidance_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, torch.Tensor, float, float, bool]:
    """Generate two diverse trajectories with threshold-based retry logic.

    Generates pairs of trajectories:
    - First trajectory: deterministic (temperature=0)
    - Second trajectory: stochastic (with noise)

    Two FDE/ADE modes available:
    - Diversity mode: Retry until FDE between trajectories >= fde_threshold
    - GT-similarity mode (default): Retry until ADE between stochastic and GT <= ade_threshold

    Additionally, initial-pose pruning can be enabled to reject stochastic candidates
    whose first pose is too misaligned with the deterministic trajectory. When
    enable_initial_pruning=False, pruning metrics are still computed and returned so
    the caller can display a visual indicator without affecting generation.

    Args:
        policy_model: The diffusion planner model
        model_args: Model configuration arguments
        data: Input observation data
        noise_scale: Noise scale for second trajectory (default: 2.5)
        fde_threshold: FDE threshold - minimum between trajectories (diversity mode)
        ade_threshold: ADE threshold - maximum to GT (GT-similarity mode)
        max_retries: Maximum number of generation attempts (default: 50)
        device: Computation device (default: model's device)
        gt_similarity_mode: If True (default), find stochastic trajectory close to GT using ADE
        gt_trajectory: Ground truth trajectory [T, 3] (x, y, heading) for GT-similarity mode
        enable_initial_pruning: If True, skip candidates whose initial pose is misaligned
        initial_pos_threshold: Maximum initial position displacement to accept (meters)
        initial_yaw_threshold_deg: Maximum initial yaw difference to accept (degrees)
        guidance: GuidanceSetConfig describing which guidance functions to apply and
            their scales. When provided, the deprecated boolean flags below are ignored.
        guidance_scale: Deprecated. Use guidance=GuidanceSetConfig(global_scale=...).
            Ignored (with a DeprecationWarning) when guidance= is provided.
            When guidance= is None and no legacy flags are set, temporarily overrides
            the decoder's guidance scale.

    Returns:
        Tuple of (trajectory_1, trajectory_2, final_metric, attempts_used, ego_shape,
                  initial_displacement, initial_yaw_diff, is_pruned)
        - trajectory_1: Deterministic trajectory [T, 4]
        - trajectory_2: Stochastic trajectory [T, 4]
        - final_metric: ADE to GT (GT mode) or FDE between trajectories (diversity mode)
        - attempts_used: Number of attempts used
        - ego_shape: Vehicle shape parameters
        - initial_displacement: Position displacement at first pose (meters)
        - initial_yaw_diff: Yaw difference at first pose (degrees)
        - is_pruned: True if the returned trajectory_2 fails the initial pose check
    """
    device = device or next(policy_model.parameters()).device
    data = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data = model_args.observation_normalizer(data)

    ego_shape = data["ego_shape"]

    # Temporarily configure guidance on the decoder.
    # The guidance function and scale are restored after generation regardless
    # of which return path is taken.
    _original_guidance_fn = policy_model.decoder._guidance_fn
    _original_guidance_scale = policy_model.decoder._guidance_scale

    import warnings

    # Build GuidanceSetConfig from legacy boolean params if new-style config not provided.
    if guidance is None and enable_guidance and (
        use_collision or use_route_following or use_lane_keeping or use_centerline_following
    ):
        warnings.warn(
            "Boolean guidance flags are deprecated. Use guidance=GuidanceSetConfig(...).",
            DeprecationWarning,
            stacklevel=2,
        )
        guidance = GuidanceSetConfig(
            global_scale=guidance_scale if guidance_scale is not None else 0.5,
            functions=[
                GuidanceConfig("collision",            enabled=use_collision),
                GuidanceConfig("route_following",      enabled=use_route_following),
                GuidanceConfig("lane_keeping",         enabled=use_lane_keeping),
                GuidanceConfig("centerline_following", enabled=use_centerline_following),
            ],
        )

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Generate deterministic trajectory WITHOUT guidance — always the unconstrained
    # baseline regardless of guidance settings. Guidance only shapes traj_2.
    policy_model.decoder._guidance_fn = None
    data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4).to(device)
    _, outputs = policy_model(data)
    traj_1 = outputs["prediction"][0, 0].cpu().numpy()

    # Configure guidance for the stochastic trajectory generation.
    if guidance is not None and guidance.active_functions():
        if guidance_scale is not None:
            warnings.warn(
                "guidance_scale is ignored when guidance= is provided; "
                "set the scale via GuidanceSetConfig(global_scale=...).",
                DeprecationWarning,
                stacklevel=2,
            )
        from diffusion_planner.model.guidance.composer import GuidanceComposer
        policy_model.decoder._guidance_fn = GuidanceComposer(guidance)
        policy_model.decoder._guidance_scale = guidance.global_scale
    else:
        policy_model.decoder._guidance_fn = None
        if guidance_scale is not None:
            policy_model.decoder._guidance_scale = guidance_scale

    # Initialize best tracking based on FDE/ADE mode
    if gt_similarity_mode and gt_trajectory is not None:
        best_metric = float("inf")
    else:
        best_metric = 0.0
    best_traj_2 = None
    best_disp = 0.0
    best_yaw_diff = 0.0

    # Last generated trajectory - used as fallback when all candidates are pruned.
    # Initialised to traj_1 so the return is always a valid array even if the
    # very first candidate triggers an exception before last_traj_2 is assigned.
    last_traj_2 = traj_1
    last_disp = 0.0
    last_yaw_diff = 0.0

    # When guidance is active, use zeros as the starting point so the guidance
    # signal is the sole source of diversity rather than mixing it with noise.
    # Because the input is deterministic (zeros), the output is also deterministic,
    # so retrying would produce the same trajectory every time. Generate once and
    # skip the retry loop entirely.
    guidance_active = policy_model.decoder._guidance_fn is not None

    if guidance_active:
        data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4).to(device)
        _, outputs = policy_model(data)
        traj_2 = outputs["prediction"][0, 0].cpu().numpy()
        is_pruned_candidate, disp, yaw_diff = should_prune_by_initial_pose(
            traj_1, traj_2, initial_pos_threshold, initial_yaw_threshold_deg
        )
        policy_model.decoder._guidance_fn = _original_guidance_fn
        policy_model.decoder._guidance_scale = _original_guidance_scale
        return traj_1, traj_2, 0.0, 1, ego_shape, disp, yaw_diff, is_pruned_candidate

    for attempt in range(max_retries):
        # Generate stochastic trajectory with scaled random noise.
        noise = noise_scale * torch.randn(B, P, future_len + 1, 4).to(device)
        data["sampled_trajectories"] = noise
        _, outputs = policy_model(data)
        traj_2 = outputs["prediction"][0, 0].cpu().numpy()

        # Always compute initial pose metrics for every candidate
        is_pruned_candidate, disp, yaw_diff = should_prune_by_initial_pose(
            traj_1, traj_2, initial_pos_threshold, initial_yaw_threshold_deg
        )
        last_traj_2, last_disp, last_yaw_diff = traj_2, disp, yaw_diff

        # When pruning is active, skip candidates that fail the initial pose check
        if enable_initial_pruning and is_pruned_candidate:
            continue

        if gt_similarity_mode and gt_trajectory is not None:
            ade = calculate_ade(traj_2, gt_trajectory)

            if ade < best_metric:
                best_metric, best_traj_2, best_disp, best_yaw_diff = ade, traj_2, disp, yaw_diff

            if ade <= ade_threshold:
                policy_model.decoder._guidance_fn = _original_guidance_fn
                policy_model.decoder._guidance_scale = _original_guidance_scale
                return traj_1, traj_2, ade, attempt + 1, ego_shape, disp, yaw_diff, is_pruned_candidate
        else:
            fde = calculate_fde(traj_1, traj_2)

            if fde > best_metric:
                best_metric, best_traj_2, best_disp, best_yaw_diff = fde, traj_2, disp, yaw_diff

            if fde >= fde_threshold:
                policy_model.decoder._guidance_fn = _original_guidance_fn
                policy_model.decoder._guidance_scale = _original_guidance_scale
                return traj_1, traj_2, fde, attempt + 1, ego_shape, disp, yaw_diff, is_pruned_candidate

    # Max retries reached — restore guidance configuration before returning.
    policy_model.decoder._guidance_fn = _original_guidance_fn
    policy_model.decoder._guidance_scale = _original_guidance_scale

    if best_traj_2 is not None:
        # At least one candidate passed the initial pose check (if pruning was enabled)
        # or best by FDE/ADE metric (if pruning was disabled)
        is_pruned_final = best_disp > initial_pos_threshold or best_yaw_diff > initial_yaw_threshold_deg
        return traj_1, best_traj_2, best_metric, max_retries, ego_shape, best_disp, best_yaw_diff, is_pruned_final
    else:
        # All max_retries candidates were pruned (only reachable when enable_initial_pruning=True)
        return traj_1, last_traj_2, 0.0, max_retries, ego_shape, last_disp, last_yaw_diff, True


@torch.no_grad()
def generate_deterministic_trajectory(
    policy_model,
    model_args,
    data: dict,
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate a deterministic (zero-noise) ego trajectory for an observation.

    Applies observation normalization internally; the input data must be raw (not
    yet normalized).  This mirrors the traj_1 branch in generate_trajectory_pair.

    Args:
        policy_model: The diffusion planner model.
        model_args: Model configuration arguments.
        data: Raw observation dict (as returned by load_npz_data).
        device: Computation device (defaults to model device).

    Returns:
        Trajectory [T, 4] (x, y, cos(yaw), sin(yaw)) in physical ego-centric metres.
    """
    device = device or next(policy_model.parameters()).device
    data = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data = model_args.observation_normalizer(data)

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    _original_guidance_fn = policy_model.decoder._guidance_fn
    policy_model.decoder._guidance_fn = None
    data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4, device=device)
    _, outputs = policy_model(data)
    policy_model.decoder._guidance_fn = _original_guidance_fn

    return outputs["prediction"][0, 0].cpu().numpy()
