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
from scenario_generation.tensor_converter import MapTensorCache, to_model_tensors
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


def _advance_agent(agent, new_world_pos: np.ndarray, dt: float = 0.1):
    """Advance a single agent in-place given its new world position.

    Uses in-place shift + overwrite instead of concatenation to avoid
    allocations. Derivatives (velocity, acceleration, yaw_rate) are
    computed from *smoothed* windows of the shifted past buffers instead
    of raw one-step finite differences — at 10 Hz the one-step numerical
    diff amplifies high-freq jitter from the diffusion sampler by ~10×
    on velocity and ~100× on acceleration. A short 5-step central
    average over the most recent window (past 0.4 s) suppresses that
    without lag. Steering angle is derived from the bicycle model:
    ``δ = atan2(wheelbase * yaw_rate, speed)``.
    """
    agent.past_trajectory[:-1] = agent.past_trajectory[1:]
    agent.past_trajectory[-1] = new_world_pos

    # Velocity window: use the last 5 position differences to average out
    # per-step noise. 5 steps = 0.4 s, short enough to react, long enough
    # to denoise.
    traj = agent.past_trajectory
    T = traj.shape[0]
    W = min(5, T - 1) if T >= 2 else 0
    if W >= 2:
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

    # Acceleration: finite diff of the smoothed velocity series — this is
    # far cleaner than `(new_vel_raw - old_vel_raw) / dt` which compounded
    # jitter from both steps.
    if agent.past_velocities is not None and agent.past_velocities.shape[0] >= W + 1:
        vel_window = agent.past_velocities[-W - 1:]
        if vel_window.shape[0] >= 2:
            accel_diffs = np.diff(vel_window, axis=0) / dt
            agent.acceleration = accel_diffs.mean(axis=0).astype(np.float32)
        else:
            agent.acceleration = ((smoothed_vel - old_smoothed_vel) / dt).astype(np.float32)
    else:
        agent.acceleration = ((smoothed_vel - old_smoothed_vel) / dt).astype(np.float32)

    # Heading rate via circular difference over the same window.
    if T >= W + 1:
        head_window = traj[T - 1 - W: T, 2]
        dh_window = np.diff(head_window)
        dh_window = np.arctan2(np.sin(dh_window), np.cos(dh_window))
        agent.yaw_rate = float(dh_window.mean() / dt)
    else:
        dh = float(new_world_pos[2] - traj[-2, 2]) if T >= 2 else 0.0
        dh = (dh + math.pi) % (2 * math.pi) - math.pi
        agent.yaw_rate = dh / dt

    # Steering angle from the bicycle model.
    speed = float(np.hypot(smoothed_vel[0], smoothed_vel[1]))
    if speed > 0.2:
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

        new_pos, _new_speed = tracker.track(x0, ref_world)
        _advance_agent(agent, new_pos, dt)


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

    model.decoder._guidance_fn = None
    if len(agent_ids) == 1:
        _, outputs = model(tensor_dicts[0])
        preds = {agent_ids[0]: outputs["prediction"][0, 0].cpu().numpy()}
        if not return_turn_indicators:
            return preds
        ti_logit = outputs.get("turn_indicator_logit")
        ti = {}
        if ti_logit is not None:
            ti[agent_ids[0]] = int(ti_logit[0].argmax(dim=-1).cpu().item())
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
        ti_cls = ti_logit.argmax(dim=-1).cpu().numpy()
        for i, aid in enumerate(agent_ids):
            ti[aid] = int(ti_cls[i])
    return preds, ti


@torch.no_grad()
def run_simulation(model, model_args, scene: SceneContext, n_steps: int,
                   output_dir: Path, device: str = "cuda", per_agent: bool = False,
                   mode: str = "closed_loop"):
    """Run simulation with configurable ego behavior.

    Modes:
        closed_loop: All agents (ego + neighbors) re-planned every step.
        semi_closed_loop: Ego follows its initial full trajectory prediction.
            Only neighbors get per-step re-planning.
    """
    if mode not in ("closed_loop", "semi_closed_loop"):
        raise ValueError(f"Unknown mode {mode!r}. Use 'closed_loop' or 'semi_closed_loop'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = deepcopy(scene)
    ego_id = scene.ego_agent_id

    scene.agents = [a for a in scene.agents if a.agent_type == AgentType.VEHICLE]
    n_agents = len(scene.agents)

    world_histories: dict[str, list[np.ndarray]] = {}
    for agent in scene.agents:
        world_histories[agent.id] = [agent.current_position.copy()]

    static_ids: set[str] = set()
    for agent in scene.agents:
        if agent.id == ego_id:
            continue
        speed = np.linalg.norm(agent.current_velocity)
        goal_dist = np.linalg.norm(agent.goal_pose[:2] - agent.current_position) if agent.goal_pose is not None else float("inf")
        if speed < 0.5 and goal_dist < 1.0:
            static_ids.add(agent.id)
    if static_ids:
        print(f"Static agents (speed<0.5, goal<1m): {static_ids}")

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

    # Build map cache once; map_data is static across simulation steps
    map_cache = MapTensorCache(scene.map_data)

    # Precompute ordered list of agents to predict; simulated_ids is constant
    ids_to_predict = [a.id for a in scene.agents if a.id in simulated_ids]

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="save") as save_pool:
        for step in range(n_steps):
            agent_predictions = _predict_batch(
                model, model_args, scene, ids_to_predict, device,
                map_cache=map_cache,
            )

            mode_label = "CL" if mode == "closed_loop" else "semi-CL"
            print(f"[{mode_label}] Step {step:03d}/{n_steps}  "
                  f"({n_simulated} re-planned, {n_agents} total)")

            # --- Visualize ---
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

            # Draw ego initial plan in semi-closed-loop
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

            # Submit overview save to thread pool
            step_futures = [
                save_pool.submit(_save_and_close, fig, output_dir / f"step_{step:03d}.png"),
            ]

            # Submit per-agent views to thread pool
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

            # Drain all saves for this step before advancing scene state
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
