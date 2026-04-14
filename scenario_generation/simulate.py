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
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from scenario_generation.gt_route_extractor import assign_gt_goals_and_routes
from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import AgentType, SceneContext
from scenario_generation.tensor_converter import to_model_tensors
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

    Args:
        agent: Agent to update.
        new_world_pos: (3,) [x, y, heading_rad] in world frame.
        dt: Timestep duration.
    """
    old_vel = agent.current_velocity
    new_vel = ((new_world_pos[:2] - agent.current_position) / dt).astype(np.float32)
    new_accel = ((new_vel - old_vel) / dt).astype(np.float32)
    dh = float(new_world_pos[2] - agent.current_heading)
    dh = (dh + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    new_yaw_rate = dh / dt

    agent.past_trajectory = np.concatenate([
        agent.past_trajectory[1:], new_world_pos.reshape(1, 3)
    ], axis=0)
    if agent.past_velocities is not None:
        agent.past_velocities = np.concatenate([
            agent.past_velocities[1:], new_vel.reshape(1, 2)
        ], axis=0)
    agent.acceleration = new_accel
    agent.yaw_rate = new_yaw_rate


def advance_scene(scene: SceneContext, agent_predictions: dict[str, np.ndarray],
                  dt: float = 0.1) -> SceneContext:
    """Advance the scene by one timestep using per-agent predictions.

    Args:
        scene: Current scene state.
        agent_predictions: Maps agent_id -> (80, 4) ego-centric prediction
            [x, y, cos_h, sin_h] from running that agent as ego.
        dt: Timestep duration.

    Returns:
        New SceneContext with all agents advanced by one step.
    """
    new_scene = deepcopy(scene)

    for agent in new_scene.agents:
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

    return new_scene


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
        draw_agent_box, draw_lanes, draw_road_borders, draw_route,
        draw_stop_lines, draw_trajectory, _EGO_COLOR, _agent_color,
    )

    agent = scene.get_agent(agent_id)
    pos = agent.current_position
    heading = agent.current_heading
    color = (color_map or {}).get(agent_id, _EGO_COLOR)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

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
        draw_trajectory(ax, plan_traj, "#ff4444", label="Plan", lw=2, zorder=25,
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
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _predict_as_ego(model, model_args, scene: SceneContext,
                    agent_id: str, device: str) -> np.ndarray:
    """Run a single agent as ego and return its ego prediction (80, 4)."""
    data = to_model_tensors(scene, agent_id, model_args, device)
    model.decoder._guidance_fn = None
    _, outputs = model(data)
    return outputs["prediction"][0, 0].cpu().numpy()


@torch.no_grad()
def run_simulation(model, model_args, scene: SceneContext, n_steps: int,
                   output_dir: Path, device: str = "cuda", per_agent: bool = False):
    """Run closed-loop simulation with every agent planned as ego.

    At each step, every agent gets its own forward pass as ego. Each agent
    is advanced using step 0 of its own ego prediction.

    Args:
        model: Loaded Diffusion-Planner model.
        model_args: Model configuration.
        scene: Initial scene state.
        n_steps: Number of simulation steps.
        output_dir: Directory to save images.
        device: Torch device.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ego_id = scene.ego_agent_id
    n_agents = len(scene.agents)

    # Track world trajectories for all agents
    world_histories: dict[str, list[np.ndarray]] = {}
    for agent in scene.agents:
        world_histories[agent.id] = [agent.current_position.copy()]

    # Remove non-vehicle agents entirely
    scene.agents = [a for a in scene.agents if a.agent_type == AgentType.VEHICLE]
    n_agents = len(scene.agents)

    # Identify static agents: low speed and goal within 1m
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
    n_simulated = len(simulated_ids)

    # Stable color assignment for all agents
    color_map = _build_color_map(scene)

    for step in range(n_steps):
        # Run model for every simulated agent as ego
        agent_predictions: dict[str, np.ndarray] = {}
        for agent in scene.agents:
            if agent.id not in simulated_ids:
                continue
            pred = _predict_as_ego(model, model_args, scene, agent.id, device)
            agent_predictions[agent.id] = pred

        print(f"Step {step:03d}/{n_steps}  ({n_simulated} vehicles simulated, {n_agents} total)")

        # --- Visualize ---
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        draw_scene(ax, scene, ego_id)

        # Draw planned trajectories for simulated agents (in world frame)
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
                color = "#ff4444"
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

        # Draw world path history for ego
        hist = world_histories[ego_id]
        if len(hist) > 1:
            h = np.array(hist)
            ax.plot(h[:, 0], h[:, 1], "-", color="#3366cc", lw=2, alpha=0.8, zorder=22)

        ax.set_title(f"Step {step:03d} / {n_steps}  t={step * 0.1:.1f}s  ({n_agents} agents)", fontsize=11)

        fig.tight_layout()
        fig.savefig(output_dir / f"step_{step:03d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Per-agent zoomed views
        if per_agent:
            for agent in scene.agents:
                agent_dir = output_dir / agent.id
                agent_dir.mkdir(parents=True, exist_ok=True)
                _draw_agent_view(
                    scene, agent.id, agent_predictions,
                    world_histories.get(agent.id, []),
                    step, n_steps, agent_dir / f"step_{step:03d}.png",
                    color_map=color_map,
                )

        # Advance all agents
        scene = advance_scene(scene, agent_predictions)
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
                   per_agent=args.per_agent)


if __name__ == "__main__":
    main()
