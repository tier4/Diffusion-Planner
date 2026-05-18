"""Closed-loop simulation using SceneContext and Diffusion-Planner.

At each timestep:
1. Convert SceneContext to model tensors (ego as ego)
2. Run model inference -> ego + neighbor trajectories (80 steps each)
3. Advance all agents by 1 step using the predicted positions
4. Update SceneContext, save visualization

Usage:
    python -m scenario_generation.simulate \
        --model_path /path/to/best_model.pth \
        --npz /path/to/scene.npz \
        --output_dir /path/to/output \
        --steps 80
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from scenario_generation.gt_route_extractor import assign_gt_goals_and_routes
from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import AgentType, SceneContext
from scenario_generation.tensor_converter import MapTensorCache, dump_step_npz, to_model_tensors
from scenario_generation.visualize import draw_scene, draw_trajectory


def load_model(model_path: str | Path, device: str = "cuda"):
    """Load Diffusion-Planner model and args."""
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    from diffusion_planner.utils.config import Config

    args_file = str(Path(model_path).parent / "args.json")
    args = Config(args_file)
    model = Diffusion_Planner(args)
    ckpt = torch.load(str(model_path), map_location=device)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, args


def _ego_to_world(pred_xy: np.ndarray, pred_cos_sin: np.ndarray,
                  ego_x: float, ego_y: float, ego_heading: float) -> tuple[np.ndarray, np.ndarray]:
    """Transform model predictions from ego-centric back to world frame.

    Args:
        pred_xy: (..., 2) positions in ego frame.
        pred_cos_sin: (..., 2) [cos_h, sin_h] in ego frame.
        ego_x, ego_y, ego_heading: Ego pose in world frame.

    Returns:
        (world_xy, world_headings) where world_headings are in radians.
    """
    c, s = math.cos(ego_heading), math.sin(ego_heading)
    # Inverse rotation: R^T = [[cos, -sin], [sin, cos]]
    wx = ego_x + pred_xy[..., 0] * c - pred_xy[..., 1] * s
    wy = ego_y + pred_xy[..., 0] * s + pred_xy[..., 1] * c
    world_xy = np.stack([wx, wy], axis=-1).astype(np.float32)

    # Transform heading back
    pred_h = np.arctan2(pred_cos_sin[..., 1], pred_cos_sin[..., 0])
    world_h = (pred_h + ego_heading).astype(np.float32)

    return world_xy, world_h


def _advance_agent(
    agent,
    new_world_pos: np.ndarray,
    dt: float = 0.1,
    new_speed: float | None = None,
    new_accel: float | None = None,
    new_yaw_rate: float | None = None,
    new_steering: float | None = None,
):
    """Advance a single agent in-place given its new world position.

    Uses in-place shift + overwrite instead of concatenation to avoid
    allocations.

    The optional ``new_*`` kwargs carry the tracker's own physically-
    correct telemetry (see ``MPCTracker.last_*`` / ``PerfectTracker.
    last_*``). When given, they replace the corresponding 5-step MA
    derivatives — those MAs were meant to denoise the jittery pred[0]
    teleport output, but in MPC / perfect mode the tracker's bicycle
    model already produces physically smooth signals, so the MA just
    adds ~0.4 s of spurious lag. The lagged values end up in
    ``ego_current_state`` and get fed back to the model, creating a
    closed-loop feedback lag where "slow down" plans never settle.
    Teleport-mode callers pass None for all of them and get the MA
    fallback that's still justified there.
    """
    agent.past_trajectory[:-1] = agent.past_trajectory[1:]
    agent.past_trajectory[-1] = new_world_pos

    traj = agent.past_trajectory
    T = traj.shape[0]
    W = min(5, T - 1) if T >= 2 else 0

    if new_speed is not None:
        new_yaw = float(new_world_pos[2])
        smoothed_vel = np.array(
            [new_speed * math.cos(new_yaw), new_speed * math.sin(new_yaw)],
            dtype=np.float32,
        )
    elif W >= 2:
        diffs = np.diff(traj[T - 1 - W: T, :2], axis=0) / dt
        smoothed_vel = diffs.mean(axis=0).astype(np.float32)
    else:
        smoothed_vel = ((new_world_pos[:2] - traj[-2, :2]) / dt).astype(np.float32) \
            if T >= 2 else np.zeros(2, dtype=np.float32)

    if agent.past_velocities is not None:
        old_smoothed_vel = agent.past_velocities[-1].copy()
        agent.past_velocities[:-1] = agent.past_velocities[1:]
        agent.past_velocities[-1] = smoothed_vel
    else:
        old_smoothed_vel = smoothed_vel

    # Acceleration: use the tracker's commanded accel when given (MPC /
    # perfect); otherwise fall back to the 5-step MA of velocity diffs,
    # which is what teleport mode needs to denoise the jittery pred[0]
    # finite differences.
    if new_accel is not None:
        new_yaw = float(new_world_pos[2])
        agent.acceleration = np.array(
            [new_accel * math.cos(new_yaw), new_accel * math.sin(new_yaw)],
            dtype=np.float32,
        )
    elif agent.past_velocities is not None and agent.past_velocities.shape[0] >= W + 1:
        vel_window = agent.past_velocities[-W - 1:]
        if vel_window.shape[0] >= 2:
            accel_diffs = np.diff(vel_window, axis=0) / dt
            agent.acceleration = accel_diffs.mean(axis=0).astype(np.float32)
        else:
            agent.acceleration = ((smoothed_vel - old_smoothed_vel) / dt).astype(np.float32)
    else:
        agent.acceleration = ((smoothed_vel - old_smoothed_vel) / dt).astype(np.float32)

    # Yaw rate: tracker's kinematic value when given, else MA over heading diffs.
    if new_yaw_rate is not None:
        agent.yaw_rate = float(new_yaw_rate)
    elif T >= W + 1:
        head_window = traj[T - 1 - W: T, 2]
        dh_window = np.diff(head_window)
        dh_window = np.arctan2(np.sin(dh_window), np.cos(dh_window))
        agent.yaw_rate = float(dh_window.mean() / dt)
    else:
        dh = float(new_world_pos[2] - traj[-2, 2]) if T >= 2 else 0.0
        dh = (dh + math.pi) % (2 * math.pi) - math.pi
        agent.yaw_rate = dh / dt

    # Steering angle: MPC supplies the actual commanded δ; perfect tracker
    # has no δ control (heading snaps to reference) so we derive it via
    # the bicycle model from (yaw_rate, speed, wheelbase), same fallback
    # teleport mode uses.
    speed = float(np.hypot(smoothed_vel[0], smoothed_vel[1]))
    if new_steering is not None and abs(new_steering) > 1e-9:
        agent.steering_angle = float(new_steering)
    elif speed > 0.2:
        agent.steering_angle = float(math.atan2(agent.wheelbase * agent.yaw_rate, speed))
    else:
        agent.steering_angle = 0.0

    # Roll the per-step turn-indicator history forward by one slot. The
    # newest slot (index -1) is filled in by the replay loop after
    # reading the model's turn_indicator_logit output for this agent —
    # we just leave index -1 at 0 (NONE) here so callers that don't have
    # the model output still get a valid-shaped buffer.
    if agent.turn_indicators is not None:
        agent.turn_indicators[:-1] = agent.turn_indicators[1:]
        agent.turn_indicators[-1] = 0


def advance_scene(scene: SceneContext, agent_predictions: dict[str, np.ndarray],
                  dt: float = 0.1) -> None:
    """Advance the scene by one timestep using per-agent predictions (in-place).

    Args:
        scene: SceneContext to modify in-place.
        agent_predictions: Maps agent_id -> (80, 4) ego-centric prediction
            [x, y, cos_h, sin_h] from running that agent as ego.
        dt: Timestep duration.
    """
    for agent in scene.agents:
        if agent.id not in agent_predictions:
            continue

        pred = agent_predictions[agent.id]  # (80, 4) in that agent's ego frame
        step = pred[0]  # First step

        # Convert from agent's ego frame back to world
        ax, ay = agent.current_position
        ah = agent.current_heading
        new_xy, new_h = _ego_to_world(
            step[:2].reshape(1, 2), step[2:4].reshape(1, 2),
            float(ax), float(ay), ah,
        )
        new_pos = np.array([new_xy[0, 0], new_xy[0, 1], new_h[0]], dtype=np.float32)
        _advance_agent(agent, new_pos, dt)


def advance_scene_mpc(
    scene: SceneContext,
    agent_predictions: dict[str, np.ndarray],
    trackers: dict,
    dt: float = 0.1,
    apply_postprocessing: bool = True,
    tracker_type: str = "mpc",
    mpc_horizon_steps: int = 20,
    mpc_n_knots: int = 5,
    ego_max_steer: float = 0.6,
) -> None:
    """Advance the scene using trajectory tracking (in-place).

    Instead of teleporting each agent to ``pred[0]``, transforms the
    full 80-step reference trajectory to world frame and feeds it to a
    per-agent tracker which produces physically plausible steps.

    Args:
        scene: SceneContext to modify in-place.
        agent_predictions: Maps agent_id -> (80, 4) ego-centric prediction
            [x, y, cos_h, sin_h].
        trackers: Maps agent_id -> tracker instance.  Missing entries are
            created lazily.
        dt: Timestep duration.
        apply_postprocessing: When True, apply velocity smoothing and
            force-stop to the reference before tracking (see
            ``mpc_tracker.postprocess_reference``).
        tracker_type: ``"mpc"`` for bicycle-model MPC, ``"perfect"``
            for Euler velocity-limited follower.
    """
    from scenario_generation.mpc_tracker import (
        MPCTracker,
        PerfectTracker,
        postprocess_reference,
    )

    for agent in scene.agents:
        if agent.id not in agent_predictions:
            continue

        pred = agent_predictions[agent.id]  # (80, 4) ego-centric

        # Transform full trajectory to world frame
        ax, ay = agent.current_position
        ah = agent.current_heading
        world_xy, world_h = _ego_to_world(
            pred[:, :2], pred[:, 2:4], float(ax), float(ay), ah,
        )

        # Build world-frame reference (N, 3) [x, y, yaw]
        ref_world = np.column_stack([world_xy, world_h])

        if apply_postprocessing:
            ref_world = postprocess_reference(world_xy, world_h, dt=dt)

        # Lazy-init tracker for this agent
        if agent.id not in trackers:
            if tracker_type == "mpc":
                max_steer = ego_max_steer if agent.id == scene.ego_agent_id else 0.6
                trackers[agent.id] = MPCTracker(
                    wheelbase=agent.wheelbase, dt=dt,
                    horizon_steps=mpc_horizon_steps, n_knots=mpc_n_knots,
                    max_steer=max_steer,
                )
            elif tracker_type == "perfect":
                trackers[agent.id] = PerfectTracker(dt=dt)
            else:
                raise ValueError(f"Unknown tracker_type {tracker_type!r}")

        tracker = trackers[agent.id]

        # Current state: [x, y, yaw, speed]
        vel = agent.current_velocity
        speed = float(vel[0] * math.cos(ah) + vel[1] * math.sin(ah))
        speed = max(speed, 0.0)
        x0 = np.array([float(ax), float(ay), ah, speed], dtype=np.float64)

        new_pos, new_speed = tracker.track(x0, ref_world)
        # Use None — not 0.0 — when a tracker lacks the telemetry
        # attribute. _advance_agent treats None as "fall back to MA
        # estimate"; 0.0 would be treated as a valid zero command and
        # silently force accel/yaw_rate/steering to zero every step,
        # defeating the whole point of the tracker-telemetry path.
        last_accel = getattr(tracker, "last_accel", None)
        last_yaw_rate = getattr(tracker, "last_yaw_rate", None)
        last_steering = getattr(tracker, "last_steering", None)
        _advance_agent(
            agent, new_pos, dt,
            new_speed=float(new_speed),
            new_accel=None if last_accel is None else float(last_accel),
            new_yaw_rate=None if last_yaw_rate is None else float(last_yaw_rate),
            new_steering=None if last_steering is None else float(last_steering),
        )


def _build_color_map(scene: SceneContext) -> dict[str, str]:
    """Assign a stable color to each agent, matching the overview rendering."""
    from scenario_generation.visualize import _EGO_COLOR, _agent_color
    colors: dict[str, str] = {}
    nb_idx = 0
    for agent in scene.agents:
        if agent.id == scene.ego_agent_id:
            colors[agent.id] = _EGO_COLOR
        else:
            colors[agent.id] = _agent_color(agent.agent_type, nb_idx)
            nb_idx += 1
    return colors


def _draw_agent_view(
    scene: SceneContext,
    agent_id: str,
    agent_predictions: dict[str, np.ndarray],
    world_history: list[np.ndarray],
    step: int,
    n_steps: int,
    output_path: Path,
    color_map: dict[str, str] | None = None,
):
    """Render a zoomed view centered on a single agent."""
    from scenario_generation.visualize import (
        _EGO_COLOR,
        _agent_color,
        draw_agent_box,
        draw_lanes,
        draw_road_borders,
        draw_route,
        draw_stop_lines,
        draw_trajectory,
    )

    agent = scene.get_agent(agent_id)
    pos = agent.current_position
    heading = agent.current_heading
    color = (color_map or {}).get(agent_id, _EGO_COLOR)

    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Map elements
    draw_lanes(ax, scene.map_data)
    draw_road_borders(ax, scene.map_data)
    draw_stop_lines(ax, scene.map_data)

    # Other agents as bounding boxes in their assigned color
    for other in scene.agents:
        if other.id == agent_id:
            continue
        opos = other.current_position
        oh = other.current_heading
        ocolor = (color_map or {}).get(other.id, "#aaaaaa")
        draw_agent_box(ax, opos[0], opos[1], oh, other.length, other.width,
                       ocolor, alpha=0.35, lw=0.8, zorder=5)

    # This agent's bounding box
    draw_agent_box(ax, pos[0], pos[1], heading, agent.length, agent.width,
                   color, alpha=0.9, lw=2.0, zorder=20)

    # Heading arrow
    arrow_len = max(agent.length, 2.0)
    dx = arrow_len * math.cos(heading)
    dy = arrow_len * math.sin(heading)
    ax.annotate("", xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=21)

    # Planned trajectory
    if agent_id in agent_predictions:
        pred = agent_predictions[agent_id]
        plan_xy, plan_h = _ego_to_world(
            pred[:, :2], pred[:, 2:4], float(pos[0]), float(pos[1]), heading,
        )
        plan_traj = np.concatenate([plan_xy, plan_h[:, np.newaxis]], axis=-1)
        draw_trajectory(ax, plan_traj, "#3366cc", label="Plan", lw=2, zorder=25,
                        show_footprints=True, length=agent.length, width=agent.width)

    # Route
    if agent.route_lanes is not None:
        draw_route(ax, agent.route_lanes, color=color, alpha=0.3)

    # World path history
    if len(world_history) > 1:
        h = np.array(world_history)
        ax.plot(h[:, 0], h[:, 1], "-", color=color, lw=2, alpha=0.7, zorder=22, label="Path")

    # Goal
    if agent.goal_pose is not None:
        gx, gy, gh = agent.goal_pose
        ax.plot(gx, gy, "*", color="#ff3333", ms=18, markeredgecolor="black",
                markeredgewidth=1.0, zorder=30, label="Goal")
        draw_agent_box(ax, gx, gy, gh, agent.length, agent.width,
                       "#ff3333", alpha=0.2, lw=1.0, zorder=29)

    # GT future
    if agent.future_trajectory is not None:
        gt = agent.future_trajectory
        ax.plot(gt[:, 0], gt[:, 1], "--", color="#22bb22", lw=1.5, alpha=0.5, zorder=15, label="GT")

    # Zoom: center on agent, show ~30m radius
    zoom_pts = [pos.reshape(1, 2)]
    if agent_id in agent_predictions:
        zoom_pts.append(plan_xy[:20])
    if agent.goal_pose is not None:
        zoom_pts.append(agent.goal_pose[:2].reshape(1, 2))
    if len(world_history) > 1:
        zoom_pts.append(np.array(world_history[-min(20, len(world_history)):]))
    all_pts = np.vstack(zoom_pts)
    cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
    half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
    half = max(half, 15)  # minimum 15m radius
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{agent_id}  step {step:03d}  t={step * 0.1:.1f}s", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    fig.clf()


def _save_and_close(fig, path: Path, dpi: int = 100) -> None:
    """Save a matplotlib figure and release resources.

    Avoids plt.close() which touches the global pyplot figure manager
    and is not thread-safe under concurrent saves.
    """
    fig.savefig(path, dpi=dpi)
    fig.clf()


def _cat_tensor_dicts(
    dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate single-sample tensor dicts along batch dim 0."""
    keys = dicts[0].keys()
    return {k: torch.cat([d[k] for d in dicts], dim=0) for k in keys}


@torch.no_grad()
def _predict_as_ego(model, model_args, scene: SceneContext,
                    agent_id: str, device: str) -> np.ndarray:
    """Run a single agent as ego and return its ego prediction (80, 4)."""
    data = to_model_tensors(scene, agent_id, model_args, device)
    model.decoder._guidance_fn = None
    _, outputs = model(data)
    return outputs["prediction"][0, 0].cpu().numpy()


@torch.no_grad()
def _predict_batch(
    model, model_args, scene: SceneContext,
    agent_ids: list[str], device: str,
    map_cache: MapTensorCache | None = None,
    return_turn_indicators: bool = False,
    inference_delay: int = 0,
    turn_indicator_keep_bias: float = 0.25,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, int]]:
    """Run batched inference for multiple agents-as-ego.

    Builds per-agent tensor dicts, concatenates along batch dim,
    runs one forward pass, splits results back per agent.

    Args:
        map_cache: Optional pre-built MapTensorCache. When the scene's
            map_data is static across steps, pass a single cache built
            once to avoid rebuilding every call. Built internally when
            not provided.
        return_turn_indicators: When True, also returns a per-agent
            ``{id: class_idx}`` dict with the argmax of each agent's
            ``turn_indicator_logit`` output (class index in
            {0=NONE, 1=DISABLE, 2=LEFT, 3=RIGHT, 4=KEEP}). Used by the
            closed-loop replay to feed model-predicted turn signals back
            into the next frame's ``turn_indicators`` history, matching
            the C++ ``TurnIndicatorManager`` control flow.
        turn_indicator_keep_bias: Subtracted from the KEEP (class 4)
            logit before argmax to imitate the C++ planner. A 0.25 bias
            means KEEP has to beat the next-best class by 0.25 before we
            pick it — avoids spurious KEEP locks on near-ties. Only
            applied when ``return_turn_indicators=True``. Default 0.25;
            set to 0.0 to reproduce the raw argmax.

    Returns ``{agent_id: (80, 4) ego-centric prediction}``, or a tuple
    ``(preds, turn_idx)`` when ``return_turn_indicators=True``.
    """
    if not agent_ids:
        return ({}, {}) if return_turn_indicators else {}

    if map_cache is None:
        map_cache = MapTensorCache(scene.map_data)
    tensor_dicts = [
        to_model_tensors(scene, aid, model_args, device, map_cache=map_cache,
                         inference_delay=inference_delay)
        for aid in agent_ids
    ]

    if len(agent_ids) == 1:
        _, outputs = model(tensor_dicts[0])
        preds = {agent_ids[0]: outputs["prediction"][0, 0].cpu().numpy()}
        if not return_turn_indicators:
            return preds
        ti_logit = outputs.get("turn_indicator_logit")
        ti = {}
        if ti_logit is not None:
            biased = ti_logit[0].clone()
            if turn_indicator_keep_bias != 0.0 and biased.shape[-1] > 4:
                biased[..., 4] -= turn_indicator_keep_bias
            ti[agent_ids[0]] = int(biased.argmax(dim=-1).cpu().item())
        return preds, ti

    batched = _cat_tensor_dicts(tensor_dicts)
    _, outputs = model(batched)
    preds_arr = outputs["prediction"][:, 0].cpu().numpy()
    preds = {aid: preds_arr[i] for i, aid in enumerate(agent_ids)}
    if not return_turn_indicators:
        return preds
    ti_logit = outputs.get("turn_indicator_logit")
    ti: dict[str, int] = {}
    if ti_logit is not None:
        biased = ti_logit.clone()
        if turn_indicator_keep_bias != 0.0 and biased.shape[-1] > 4:
            biased[..., 4] -= turn_indicator_keep_bias
        ti_cls = biased.argmax(dim=-1).cpu().numpy()
        for i, aid in enumerate(agent_ids):
            ti[aid] = int(ti_cls[i])
    return preds, ti


@torch.no_grad()
def _refresh_line_strings(
    scene: SceneContext, builder,
    query_world_xy: np.ndarray,
    scene_origin_world: np.ndarray,
) -> None:
    """Rebuild scene.map_data.line_strings from the lanelet2 builder.

    Queries line_strings near ``query_world_xy`` (current ego world pos),
    then transforms from world frame to scene frame using
    ``scene_origin_world`` (the ego's world pose at sim start = scene origin).

    Args:
        query_world_xy: (2,) current ego world position (for spatial query).
        scene_origin_world: (3,) [x, y, yaw] of the scene frame origin in
            world frame (= ego world pose at step 0).
    """
    from scenario_generation.transforms import _rotation_matrix, transform_positions

    ls_world = builder.build_line_strings_tensor(query_world_xy.astype(np.float32))

    init_xy = scene_origin_world[:2].astype(np.float64)
    R_init = _rotation_matrix(float(scene_origin_world[2]))

    ls_scene = ls_world.copy()
    for li in range(ls_scene.shape[0]):
        pts = ls_scene[li, :, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.any():
            ls_scene[li, valid, :2] = transform_positions(
                pts[valid].astype(np.float64), R_init, init_xy,
            ).astype(np.float32)
    scene.map_data.line_strings = ls_scene



def run_simulation(model, model_args, scene: SceneContext, n_steps: int,
                   output_dir: Path, device: str = "cuda", per_agent: bool = False,
                   mode: str = "closed_loop",
                   builder=None,
                   ego_world_pose: np.ndarray | None = None,
                   map_refresh_steps: int = 5,
                   skip_viz: bool = False,
                   static_agent_ids: set[str] | None = None,
                   dump_npz: bool = False,
                   progress_fn=None,
                   zero_neighbors: bool = False):
    """Run simulation with configurable ego behavior.

    Modes:
        closed_loop: All agents (ego + neighbors) re-planned every step.
        semi_closed_loop: Ego follows its initial full trajectory prediction.
            Only neighbors get per-step re-planning.

    Args:
        builder: Optional LaneletSceneBuilder for map refresh (lanes,
            line_strings with road borders, polygons). When provided with
            ``ego_world_pose``, the scene is converted to world frame so
            map refresh works correctly (same as replay.py).
        ego_world_pose: (3,) [x, y, yaw] of the ego in map world frame at
            the start of the simulation. Required when ``builder`` is set.
        map_refresh_steps: Rebuild map_data every N steps (default 5).
        skip_viz: Skip PNG generation (faster for GUI use).
        static_agent_ids: Agent IDs to exclude from prediction (stationary).
        dump_npz: Save per-step NPZ files alongside PNGs.
        progress_fn: Optional callable(fraction, desc) for progress reporting.
        zero_neighbors: Zero neighbor_agents_past in model input so the
            model only sees placed obstacles, not original NPZ neighbors.
    """
    if mode not in ("closed_loop", "semi_closed_loop"):
        raise ValueError(f"Unknown mode {mode!r}. Use 'closed_loop' or 'semi_closed_loop'.")
    if builder is not None and ego_world_pose is None:
        import warnings
        warnings.warn("builder provided without ego_world_pose — map refresh will be disabled",
                      stacklevel=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = deepcopy(scene)
    ego_id = scene.ego_agent_id

    scene.agents = [a for a in scene.agents if a.agent_type == AgentType.VEHICLE]
    n_agents = len(scene.agents)

    world_histories: dict[str, list[np.ndarray]] = {}
    for agent in scene.agents:
        world_histories[agent.id] = [agent.current_position.copy()]

    static_ids: set[str] = set(static_agent_ids or [])
    for agent in scene.agents:
        if agent.id == ego_id or agent.id in static_ids:
            continue
        speed = np.linalg.norm(agent.current_velocity)
        goal_dist = np.linalg.norm(agent.goal_pose[:2] - agent.current_position) if agent.goal_pose is not None else float("inf")
        if speed < 0.5 and goal_dist < 1.0:
            static_ids.add(agent.id)
    if static_ids:
        print(f"Static agents: {static_ids}")

    simulated_ids = {a.id for a in scene.agents} - static_ids

    # In semi-closed-loop, ego prediction is computed once at step 0
    ego_initial_plan = None
    if mode == "semi_closed_loop":
        ego_agent = scene.get_agent(ego_id)
        ego_pred = _predict_as_ego(model, model_args, scene, ego_id, device)
        ax0, ay0 = ego_agent.current_position
        ah0 = ego_agent.current_heading
        plan_xy, plan_h = _ego_to_world(
            ego_pred[:, :2], ego_pred[:, 2:4], float(ax0), float(ay0), ah0,
        )
        ego_initial_plan = np.concatenate([plan_xy, plan_h[:, np.newaxis]], axis=-1)
        simulated_ids.discard(ego_id)
        print(f"Semi-closed-loop: ego follows initial {len(ego_initial_plan)}-step plan")

    n_simulated = len(simulated_ids)
    color_map = _build_color_map(scene)

    # Pre-create per-agent output dirs
    if per_agent:
        for agent in scene.agents:
            (output_dir / agent.id).mkdir(parents=True, exist_ok=True)

    # Optionally rebuild line_strings from builder to get road border flags
    # that psim NPZs lack. Only line_strings is replaced — lanes, polygons
    # from the NPZ are correct and untouched.
    _can_refresh_ls = builder is not None and ego_world_pose is not None
    _scene_origin = np.array(ego_world_pose, dtype=np.float64) if ego_world_pose is not None else None
    _init_yaw = float(_scene_origin[2]) if _scene_origin is not None else 0.0
    if _can_refresh_ls:
        _refresh_line_strings(scene, builder, _scene_origin[:2], _scene_origin)
    map_cache = MapTensorCache(scene.map_data)

    ids_to_predict = [a.id for a in scene.agents if a.id in simulated_ids]

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="save") as save_pool:
        for step in range(n_steps):
            if _can_refresh_ls and step > 0 and map_refresh_steps > 0 and step % map_refresh_steps == 0:
                ep = scene.get_agent(ego_id).current_position
                eh = scene.get_agent(ego_id).current_heading
                ci, si = math.cos(_init_yaw), math.sin(_init_yaw)
                cur_wx = _scene_origin[0] + ci * ep[0] - si * ep[1]
                cur_wy = _scene_origin[1] + si * ep[0] + ci * ep[1]
                _refresh_line_strings(
                    scene, builder,
                    np.array([cur_wx, cur_wy], dtype=np.float64),
                    _scene_origin,
                )
                map_cache = MapTensorCache(scene.map_data)

            if zero_neighbors:
                _placed_ids = set(static_agent_ids or [])
                _saved_agents = scene.agents[:]
                scene.agents = [a for a in scene.agents
                                if a.id == ego_id or a.id in _placed_ids]
                agent_predictions = _predict_batch(
                    model, model_args, scene, [ego_id], device,
                    map_cache=map_cache,
                )
                scene.agents = _saved_agents
            else:
                agent_predictions = _predict_batch(
                    model, model_args, scene, ids_to_predict, device,
                    map_cache=map_cache,
                )

            mode_label = "CL" if mode == "closed_loop" else "semi-CL"
            if not skip_viz:
                print(f"[{mode_label}] Step {step:03d}/{n_steps}  "
                      f"({n_simulated} re-planned, {n_agents} total)")
            if progress_fn is not None:
                progress_fn((step + 1) / n_steps, f"Simulating {step+1}/{n_steps}")

            # --- Dump NPZ ---
            if dump_npz:
                npz_data = dump_step_npz(scene, map_cache, future_len=model_args.future_len)
                if ego_id in agent_predictions:
                    ego_pred = agent_predictions[ego_id]
                    heading = np.arctan2(ego_pred[:, 3], ego_pred[:, 2])
                    npz_data["ego_agent_future"] = np.column_stack(
                        [ego_pred[:, :2], heading]).astype(np.float32)
                # Save sidecar JSON with ego world pose
                if _scene_origin is not None:
                    import json as _json
                    ep = scene.get_agent(ego_id).current_position
                    eh = scene.get_agent(ego_id).current_heading
                    ci, si = math.cos(_init_yaw), math.sin(_init_yaw)
                    wx = _scene_origin[0] + ci * ep[0] - si * ep[1]
                    wy = _scene_origin[1] + si * ep[0] + ci * ep[1]
                    wyaw = _init_yaw + eh
                    sidecar = {"x": float(wx), "y": float(wy),
                               "qz": math.sin(wyaw / 2), "qw": math.cos(wyaw / 2),
                               "qx": 0.0, "qy": 0.0}
                    (output_dir / f"replay_step_{step:04d}.json").write_text(
                        _json.dumps(sidecar))
                np.savez(output_dir / f"replay_step_{step:04d}.npz", **npz_data)

            # --- Visualize ---
            if not skip_viz:
                from matplotlib.figure import Figure
                fig = Figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 1, 1)
                draw_scene(ax, scene, ego_id)

                from scenario_generation.visualize import _agent_color
                nb_idx = 0
                for agent in scene.agents:
                    if agent.id not in agent_predictions:
                        continue
                    pred = agent_predictions[agent.id]
                    ax_pos, ay_pos = agent.current_position
                    ah = agent.current_heading
                    plan_xy, plan_h = _ego_to_world(
                        pred[:, :2], pred[:, 2:4], float(ax_pos), float(ay_pos), ah,
                    )
                    plan_traj = np.concatenate([plan_xy, plan_h[:, np.newaxis]], axis=-1)

                    if agent.id == ego_id:
                        color = "#3366cc"
                        label = "Ego plan"
                        zorder = 25
                    else:
                        color = _agent_color(agent.agent_type, nb_idx)
                        label = None
                        zorder = 18
                        nb_idx += 1

                    draw_trajectory(ax, plan_traj, color, label=label, lw=1.0, zorder=zorder,
                                    show_footprints=(agent.id == ego_id),
                                    length=agent.length, width=agent.width)

                if ego_initial_plan is not None:
                    remaining = ego_initial_plan[step:]
                    if len(remaining) > 1:
                        draw_trajectory(ax, remaining, "#3366cc", label="Ego plan (fixed)",
                                        lw=1.5, zorder=25, show_footprints=True,
                                        length=scene.get_agent(ego_id).length,
                                        width=scene.get_agent(ego_id).width)

                hist = world_histories.get(ego_id, [])
                if len(hist) > 1:
                    h = np.array(hist)
                    ax.plot(h[:, 0], h[:, 1], "-", color="#3366cc", lw=2, alpha=0.8, zorder=22)

                ax.set_title(f"[{mode_label}] Step {step:03d}/{n_steps}  t={step*0.1:.1f}s  "
                             f"({n_agents} agents)", fontsize=11)
                fig.tight_layout()

                step_futures = [
                    save_pool.submit(_save_and_close, fig, output_dir / f"step_{step:03d}.png"),
                ]
                if per_agent:
                    for agent in scene.agents:
                        step_futures.append(
                            save_pool.submit(
                                _draw_agent_view,
                                scene, agent.id, agent_predictions,
                                world_histories.get(agent.id, []),
                                step, n_steps, output_dir / agent.id / f"step_{step:03d}.png",
                                color_map,
                            )
                        )
                for f in step_futures:
                    f.result()

            # Advance all agents
            if mode == "semi_closed_loop" and ego_initial_plan is not None and step < len(ego_initial_plan):
                wp = ego_initial_plan[step]
                _advance_agent(scene.get_agent(ego_id),
                               np.array([wp[0], wp[1], wp[2]], dtype=np.float32))
            advance_scene(scene, agent_predictions)
            for agent in scene.agents:
                world_histories[agent.id].append(agent.current_position.copy())

    print(f"Done. {n_steps} frames saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop simulation with Diffusion-Planner")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to best_model.pth")
    parser.add_argument("--npz", type=Path, required=True, help="NPZ scene file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for images")
    parser.add_argument("--steps", type=int, default=80, help="Number of simulation steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--use_gt_goals", action="store_true",
                        help="Set neighbor goals and routes from their GT future trajectories")
    parser.add_argument("--per_agent", action="store_true",
                        help="Save per-agent zoomed images in addition to the overview")
    parser.add_argument("--mode", choices=["closed_loop", "semi_closed_loop"],
                        default="closed_loop",
                        help="closed_loop: all agents re-planned each step. "
                             "semi_closed_loop: ego follows initial trajectory, "
                             "neighbors re-planned.")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"Loading model from {args.model_path}")
    model, model_args = load_model(args.model_path, device)

    print(f"Loading scene from {args.npz}")
    scene = from_npz(args.npz)

    if args.use_gt_goals:
        n = assign_gt_goals_and_routes(scene)
        print(f"Assigned GT goals and routes to {n} agents")

    run_simulation(model, model_args, scene, args.steps, args.output_dir, device,
                   per_agent=args.per_agent, mode=args.mode)


if __name__ == "__main__":
    main()
