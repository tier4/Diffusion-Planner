"""Raw subscore entry point.

``compute_subscores_batch`` runs the input marshalling (ego_shape / neighbors /
goal) and every per-subscore computation, but stops before any reward *shaping*
(weights / gates / survival aggregation). It returns the raw per-trajectory
subscores + diagnostics as a dict; ``rlvr.reward.compute_reward_batch`` feeds
this straight into ``_shape_reward`` (single source of truth — no duplicated
marshalling), and the validation loop logs the continuous subscores.

These are reward-shaping subscores (custom thresholds, goal-based ego-progress,
penalty signs), i.e. EPDMS-INSPIRED, NOT a faithful EPDMS port. Single scene /
N trajectories (map + neighbor terms use one scene's tensors). Use
``compute_subscores_scene_batch`` when scoring a batch of independent scenes.
"""

from __future__ import annotations

import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import (
    ROAD_BORDER_NO_DATA_DISTANCE_M,
    compute_centerline_score_batch,
    compute_feasibility_score_batch,
    compute_kinematic_gate,
    compute_lane_departure_penalty,
    compute_progress_score_batch,
    compute_red_light_score_batch,
    compute_road_border_penalty,
    compute_safety_score_batch,
    compute_smoothness_score_batch,
    compute_static_collision_penalty,
    compute_ttc_score_batch,
)

__all__ = ["compute_subscores_batch", "compute_subscores_scene_batch"]

_NEIGHBOR_COORD_EPS_M = 1e-6
_NEIGHBOR_SHAPE_EPS_M = 1e-3
_UNKNOWN_NEIGHBOR_SHAPE_M = 2.0
_DEFAULT_NEIGHBOR_WIDTH_M = 2.0
_DEFAULT_NEIGHBOR_LENGTH_M = 4.5


def _slice_scene_data(
    data: dict[str, torch.Tensor],
    scene_idx: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Return one scene from a scene-batched data dict, preserving batch dims."""
    out: dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
            out[key] = value[scene_idx : scene_idx + 1]
        else:
            out[key] = value
    return out


def _stack_subscore_outputs(
    per_scene: list[dict[str, torch.Tensor | list[int | None]]],
) -> dict[str, torch.Tensor | list[list[int | None]]]:
    """Stack per-scene outputs into B-major tensors/lists."""
    if not per_scene:
        return {}

    stacked: dict[str, torch.Tensor | list[list[int | None]]] = {}
    for key in per_scene[0]:
        values = [scene_out[key] for scene_out in per_scene]
        first = values[0]
        if torch.is_tensor(first):
            if not all(torch.is_tensor(v) for v in values):
                raise TypeError(f"mixed subscore output types for {key}")
            stacked[key] = torch.stack(values, dim=0)
        elif isinstance(first, list):
            if not all(isinstance(v, list) for v in values):
                raise TypeError(f"mixed subscore output types for {key}")
            stacked[key] = [list(v) for v in values]
        else:
            raise TypeError(f"unsupported subscore output type for {key}: {type(first)!r}")
    return stacked


@torch.no_grad()
def compute_subscores_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> dict[str, torch.Tensor | list[int | None]]:
    """Raw per-trajectory subscores + diagnostics for ``ego_trajs`` ``(N, T, 4)``.

    Marshals ``data`` (ego_shape / neighbors / goal) and calls every subscore —
    no weighting, gating, or aggregation. Returns a dict whose continuous
    subscores and penalties are ``(N,)`` tensors; ``*_crossing_steps`` and
    ``collision_step`` are length-N lists; ``rb_min_dist`` / ``sc_min_dist`` are
    the per-trajectory minima over t>=1 (the t=0 step is not model-controllable).
    ``data`` must carry ``ego_shape``; map / neighbor terms degrade to their
    neutral values when the corresponding keys are absent.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    # --- Ego shape --- (no silent fallback; mirrors compute_reward_batch)
    if "ego_shape" not in data:
        raise ValueError(
            "compute_subscores_batch: data is missing 'ego_shape' (wheel_base, "
            "length, width). Populate ego_shape upstream and re-run."
        )
    es = data["ego_shape"]
    if es.dim() == 2:
        es = es[0]
    if es.numel() < 3:
        raise ValueError(
            f"compute_subscores_batch: ego_shape has shape {tuple(es.shape)}; "
            "expected at least 3 elements (wheel_base, length, width)."
        )
    ego_shape = es[:3].to(device)

    # --- Neighbor data for collision ---
    neighbor_futures = torch.zeros(0, T, 4, device=device)
    neighbor_shapes = torch.zeros(0, 2, device=device)
    neighbor_valid = torch.zeros(0, T, dtype=torch.bool, device=device)

    if "neighbor_agents_future" in data:
        nf = data["neighbor_agents_future"]
        if nf.dim() == 4:
            nf = nf[0]
        if nf.shape[1] >= T and nf.shape[2] >= 4:
            nf_data = nf[:, :T, :4]  # (N_nb, T, 4) = x, y, cos, sin
        elif nf.shape[0] > 0 and nf.shape[1] >= T and nf.shape[2] == 3:
            raise ValueError(
                "neighbor_agents_future has 3 columns (x, y, heading_rad) but "
                "4 columns (x, y, cos, sin) are required. Re-generate the NPZ "
                "with the updated tensor_converter / _backfill_neighbor_futures."
            )
        if nf.shape[1] >= T and nf.shape[2] >= 4:
            slot_valid = nf_data[:, :, :2].abs().sum(dim=(1, 2)) > _NEIGHBOR_COORD_EPS_M
            if slot_valid.any():
                neighbor_futures = nf_data[slot_valid]
                neighbor_valid = (
                    neighbor_futures[:, :, :2].abs().sum(dim=-1) > _NEIGHBOR_COORD_EPS_M
                )

                if "neighbor_agents_past" in data:
                    nap = data["neighbor_agents_past"]
                    if nap.dim() == 4:
                        nap = nap[0]
                    ns = nap[slot_valid, -1, :]
                    if ns.shape[-1] >= 8:
                        neighbor_shapes = ns[:, [6, 7]]  # width, length
                    else:
                        neighbor_shapes = torch.full(
                            (neighbor_futures.shape[0], 2),
                            _UNKNOWN_NEIGHBOR_SHAPE_M,
                            device=device,
                        )
                else:
                    neighbor_shapes = torch.full(
                        (neighbor_futures.shape[0], 2),
                        _UNKNOWN_NEIGHBOR_SHAPE_M,
                        device=device,
                    )

    zero_shapes = neighbor_shapes.abs().sum(dim=-1) < _NEIGHBOR_SHAPE_EPS_M
    if zero_shapes.any():
        neighbor_shapes[zero_shapes] = torch.tensor(
            [_DEFAULT_NEIGHBOR_WIDTH_M, _DEFAULT_NEIGHBOR_LENGTH_M],
            device=device,
        )

    # --- Goal pose ---
    goal_pose = torch.zeros(4, device=device)
    if "goal_pose" in data:
        gp = data["goal_pose"]
        if gp.dim() == 2:
            gp = gp[0]
        if gp.numel() >= 4:
            goal_pose = gp[:4].to(device)

    # --- Batched subscore computation (no shaping) ---
    safety_scores, collision_steps = compute_safety_score_batch(
        ego_trajs, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid, config
    )
    progress_scores = compute_progress_score_batch(ego_trajs, goal_pose, data)
    smoothness_scores = compute_smoothness_score_batch(ego_trajs, config)
    feasibility_scores, off_road_fractions = compute_feasibility_score_batch(
        ego_trajs, ego_shape, data, config
    )
    centerline_scores = compute_centerline_score_batch(
        ego_trajs,
        ego_shape,
        data,
        usage_mode=config.centerline_usage_mode,
        time_weight_min=config.centerline_time_weight_min,
    )
    red_light_scores = compute_red_light_score_batch(ego_trajs, data, config)
    ttc_result = compute_ttc_score_batch(
        ego_trajs, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid
    )
    ttc_scores = ttc_result["score"]
    (
        rb_crossing_gate,
        rb_near_pen,
        rb_wide_pen,
        rb_crossing_steps,
        rb_cont_penalty,
        rb_per_ts_min,
    ) = compute_road_border_penalty(ego_trajs, ego_shape, data, config=config)

    if config.enable_lane_departure:
        (
            lane_crossing_gate,
            lane_near_frac,
            lane_wide_frac,
            lane_crossing_steps,
            lane_cont_penalty,
        ) = compute_lane_departure_penalty(ego_trajs, ego_shape, data, config=config)
    else:
        lane_crossing_gate = torch.ones(N, device=device)
        lane_near_frac = torch.zeros(N, device=device)
        lane_wide_frac = torch.zeros(N, device=device)
        lane_crossing_steps: list[int | None] = [None] * N
        lane_cont_penalty = torch.zeros(N, device=device)

    if config.static_collision_enabled:
        sc_result = compute_static_collision_penalty(
            ego_trajs, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid, config
        )
        sc_crossing_gate = sc_result["crossing_gate"]
        sc_near_pen = sc_result["near_penalty"]
        sc_wide_pen = sc_result["wide_penalty"]
        sc_cont_pen = sc_result["cont_penalty"]
        sc_crossing_steps = sc_result["first_crossing_steps"]
        sc_per_ts_min = sc_result["per_timestep_min"]
        sc_n_stopped_scene = int(sc_result["stopped_mask"].sum().item())
    else:
        sc_crossing_gate = torch.ones(N, device=device)
        sc_near_pen = torch.zeros(N, device=device)
        sc_wide_pen = torch.zeros(N, device=device)
        sc_cont_pen = torch.zeros(N, device=device)
        sc_crossing_steps: list[int | None] = [None] * N
        sc_per_ts_min = torch.full((N, T), 99.0, device=device)
        sc_n_stopped_scene = 0

    kinematic_gate = compute_kinematic_gate(ego_trajs, config, ego_shape)

    # Min clearances over t>=1 only (t=0 is not model-controllable).
    if T > 1:
        rb_min_dist = rb_per_ts_min[:, 1:].min(dim=1).values
        sc_min_dist = sc_per_ts_min[:, 1:].min(dim=1).values
    else:
        rb_min_dist = torch.full((N,), ROAD_BORDER_NO_DATA_DISTANCE_M, device=device)
        sc_min_dist = torch.full((N,), 99.0, device=device)

    return {
        # raw subscores (signs as produced by the subscore functions)
        "safety": safety_scores,
        "ttc": ttc_scores,
        "ttc_unsafe_at_t": ttc_result["unsafe_at_t"],
        "ttc_first_unsafe_steps": ttc_result["first_unsafe_steps"],
        "ttc_first_collision_steps": ttc_result["first_collision_steps"],
        "ttc_min_clearance": ttc_result["min_clearance"],
        "progress": progress_scores,
        "comfort": smoothness_scores,
        "feasibility": feasibility_scores,
        "centerline": centerline_scores,
        "red_light": red_light_scores,
        "kinematic_gate": kinematic_gate,  # 1.0 feasible, 0.0 violated
        "off_road_fraction": off_road_fractions,
        # road-border (DAC) diagnostics
        "rb_crossing_gate": rb_crossing_gate,  # 1.0 ok, 0.0 crossing
        "rb_near_penalty": rb_near_pen,
        "rb_wide_penalty": rb_wide_pen,
        "rb_cont_penalty": rb_cont_penalty,
        "rb_min_dist": rb_min_dist,
        "rb_crossing_steps": rb_crossing_steps,
        # lane-departure diagnostics
        "lane_crossing_gate": lane_crossing_gate,
        "lane_near_frac": lane_near_frac,
        "lane_wide_frac": lane_wide_frac,
        "lane_cont_penalty": lane_cont_penalty,
        "lane_crossing_steps": lane_crossing_steps,
        # static-collision diagnostics
        "sc_crossing_gate": sc_crossing_gate,
        "sc_near_penalty": sc_near_pen,
        "sc_wide_penalty": sc_wide_pen,
        "sc_cont_penalty": sc_cont_pen,
        "sc_min_dist": sc_min_dist,
        "sc_n_stopped": torch.full((N,), float(sc_n_stopped_scene), device=device),
        "sc_crossing_steps": sc_crossing_steps,
        "collision_step": collision_steps,
    }


@torch.no_grad()
def compute_subscores_scene_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> dict[str, torch.Tensor | list[list[int | None]]]:
    """Raw subscores for ``B`` scenes, each with ``N`` candidate trajectories.

    ``compute_subscores_batch`` remains the single-scene primitive. This wrapper
    gives callers a B-major API so scene classification can handle multiple NPZs
    per batch while preserving all per-scene map and neighbor semantics.

    Args:
        ego_trajs: ``(B, N, T, 4)`` x/y/cos/sin trajectories.
        data: Scene-batched observation dict. Tensor values whose leading dim is
            ``B`` are sliced per scene; scalar/shared values are forwarded.
        config: RewardConfig with subscore thresholds.

    Returns:
        Dict matching ``compute_subscores_batch`` but with tensor outputs
        stacked as ``(B, N, ...)`` and step diagnostics as B-major nested lists.
    """
    if ego_trajs.dim() != 4:
        raise ValueError(
            "compute_subscores_scene_batch expects ego_trajs shaped (B, N, T, 4); "
            f"got {tuple(ego_trajs.shape)}"
        )
    B = ego_trajs.shape[0]
    if B == 0:
        return {}

    per_scene = [
        compute_subscores_batch(ego_trajs[b], _slice_scene_data(data, b, B), config)
        for b in range(B)
    ]
    return _stack_subscore_outputs(per_scene)
