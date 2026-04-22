"""Closed-loop replay of a saved :class:`scenario_generation.route.Route`.

High-level flow
---------------

1. Load the Route and rebuild a ``LaneletSceneBuilder`` from ``route.map_path``.
2. Build an initial ``SceneContext`` containing just the ego (no neighbors),
   placed at ``route.start_pose`` with its 31-step history synthesised
   backwards along the lanelet centerlines (see
   ``LaneletSceneBuilder.generate_history``).
3. Enter the closed-loop: each simulation tick we

   * Run batched inference on every currently-alive agent as ego (reusing
     :func:`scenario_generation.simulate._predict_batch` — a single forward
     pass with all agents concatenated along the batch dim, no sequential
     per-agent calls).
   * Advance every agent one physical step via
     :func:`scenario_generation.simulate.advance_scene`.
   * Periodically run the NPC spawn manager:

     - **Despawn** any neighbor farther than ``despawn_distance`` m from ego.
     - **Spawn** up to ``max_active_npcs`` (hard cap) neighbors near the ego
       with a small per-tick probability, using a realistic synthesised
       history and a route that is sometimes biased to overlap the ego's
       own route (see ``SpawnConfig.ego_overlap_ratio``).
   * Save an overview PNG per tick.

4. Terminate when the ego arrives within ``goal_tolerance_m`` of
   ``route.goal_pose`` or after ``n_steps`` ticks.

Traffic lights are managed by :class:`TrafficLightController`
(``scenario_generation.traffic_light``), which discovers TL regulatory
elements from the lanelet2 map, builds cycle groups, and writes the 5-dim
one-hot into ``scene.map_data.lanes[:, :, 8:13]`` every map refresh.

Batched inference note
----------------------

Each alive agent is a separate scene dict (different ego-centric coordinate
frames) but ``_predict_batch`` concatenates them along ``batch_dim=0`` for a
single ``model(data)`` call. This is the point the user emphasised: we never
loop inference sequentially; even spawned neighbors enter the same batch on
the very next tick.

The ``MapTensorCache`` is rebuilt whenever a new lanelet is added to
``scene.map_data`` during NPC spawning (there is no ``invalidate()`` method —
see CLAUDE.md and the cache definition in ``tensor_converter.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from scenario_generation.gui.lanelet_scene_builder import (
    LaneletSceneBuilder,
    _obb_collides,
    _obb_corners,
)
from scenario_generation.route import Route
from scenario_generation.scene_context import Agent, AgentType, SceneContext
from scenario_generation.simulate import (
    _ego_to_world,
    _predict_batch,
    _save_and_close,
    advance_scene,
    advance_scene_mpc,
    load_model,
)
from scenario_generation.tensor_converter import MapTensorCache, dump_step_npz
from scenario_generation.traffic_light import TrafficLightController

# Reuse the Savitzky-Golay smoother from the RL pipeline. Used there by
# ranked-SFT to smooth diffusion-planner outputs before the SFT loss; same
# defaults (window=11, order=3) and cos/sin-renormalisation logic apply at
# replay time to suppress diffusion-sampler jitter. Importing rather than
# duplicating so the two pipelines stay in sync.
from rlvr.grpo_sft_trainer import _smooth_trajectory as _sg_smooth_trajectory

# Live per-step lane / border / centerline scoring (only loaded when
# dump_npz_dir + reward_config_path are set). Matches the exact same
# primitives ranked-SFT uses for its reward, so the metrics log here and
# the training run speak the same thresholds.
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import (
    RewardConfig,
    compute_centerline_score_batch,
    compute_reward_batch,
)
from scenario_generation.visualize import (
    _agent_color,
    draw_agent_box,
    draw_trajectory,
)


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class SpawnConfig:
    """Controls the NPC spawn/despawn manager.

    All distances in metres, all times in simulation steps (each step = 0.1 s).

    Attributes:
        spawn_period_steps: Run the spawn/despawn tick every N steps.
            ``10`` ≈ once per simulated second.
        max_active_npcs: Hard upper bound on the number of concurrent
            neighbors. Neighbor count is not kept constant — it drifts in
            ``[0, max_active_npcs]`` driven by ``spawn_probability`` and
            despawn distance.
        spawn_probability: Per-tick chance of attempting a spawn when the
            active count is below the cap.
        min_spawn_distance: A spawn candidate must sit at least this far
            from ego.
        max_spawn_distance: A spawn candidate must sit no further than this
            from ego.
        despawn_distance: Any neighbor farther than this from ego is dropped.
            Default 120 m matches the user's spec.
        forward_bias: Probability that a spawn candidate is restricted to
            lanelets in front of the ego (vs. free directional choice).
        min_npc_separation: Minimum centre-to-centre distance between a new
            spawn and any existing agent's OBB. Matches the upstream
            npc_manager constant.
        goal_tolerance_m: Ego-to-goal distance that triggers a
            "goal reached" termination.
        max_steps: Maximum simulation ticks before forced termination.
            6000 ticks = 600 s = 10 minutes of simulated time.
        seed: RNG seed used for spawn candidate selection, vehicle
            dimensions, speeds, route choices. ``None`` for non-deterministic.
        ego_overlap_ratio: Fraction of spawned NPCs that are routed to
            overlap with the ego's route_lanelet_ids. Per user: 0.30 (30%
            overlap, 70% random forward routes).
        npc_min_speed: Floor for a spawned NPC's initial speed (m/s).
        npc_max_speed: Ceiling for a spawned NPC's initial speed (m/s).
        npc_route_length_m: Minimum arc-length fed to ``find_route`` for each
            new NPC — drives how many lanelets of route_lanes it gets.
        curvature_threshold: Max allowable |Δheading| between consecutive
            centerline segments for a lanelet to be spawnable. Matches the
            default used by ``is_lanelet_straight``.
        map_refresh_steps: Rebuild ``scene.map_data`` with the closest
            lanelets to the ego every this many steps. Matches the training
            distribution where the ``(140, 20, 33)`` lane tensor is packed
            with the closest lanelets to ego — not a fixed pre-baked subset.
        max_map_lanelets: Upper bound on the number of lanelets packed into
            ``map_data.lanes``. Must match / not exceed
            ``tensor_converter._NUM_LANES`` (currently 140).
        map_mask_range_m: Half-side of the AABB lane-filter around ego.
            Matches the Diffusion-Planner ROS node's ``judge_inside``
            (``mask_range = 100.0``). A lanelet passes the filter when its
            center, first, or last centerline point is within this square.
    """

    spawn_period_steps: int = 10
    max_active_npcs: int = 8
    spawn_probability: float = 0.3
    min_spawn_distance: float = 15.0
    max_spawn_distance: float = 60.0
    despawn_distance: float = 120.0
    forward_bias: float = 0.8
    min_npc_separation: float = 8.0
    goal_tolerance_m: float = 2.0
    max_steps: int = 6000
    seed: int | None = None
    ego_overlap_ratio: float = 0.3
    npc_min_speed: float = 3.0
    npc_max_speed: float = 12.0
    npc_route_length_m: float = 120.0
    curvature_threshold: float = 0.3
    # Closest-approach window: when the ego has been within this radius of
    # the goal AND now has the goal *behind* it (negative dot product
    # against ego-forward), terminate as "goal_passed". The diffusion
    # planner doesn't stop at the goal, so without this it can pass within
    # 5-15 m of the goal then drive off into the horizon.
    goal_pass_window_m: float = 25.0
    map_refresh_steps: int = 5
    max_map_lanelets: int = 140
    # ROS node uses 100 m; empirical survey of our training NPZs shows
    # ~23 (min) – 89 (max) non-zero lanes per scene, median 61. 100 m on the
    # Shinagawa map tops out at ~22 lanelets — at the bottom of training
    # distribution. 200 m yields ~62, matching the median.
    map_mask_range_m: float = 200.0
    # Savitzky-Golay smoothing applied to each agent's predicted
    # trajectory before ``advance_scene`` uses its first step. Matches the
    # defaults from ``rlvr.grpo_sft_trainer._smooth_trajectory`` (ranked
    # SFT uses the same smoother on generated trajectories before the SFT
    # loss). Set ``sg_smooth_enabled=False`` to disable (e.g. for A/B
    # comparison).
    sg_smooth_enabled: bool = True
    sg_filter_window: int = 11
    sg_filter_order: int = 3
    # Advance mode: how the vehicle moves each step.
    #   "teleport"  — original behaviour, snap to pred[0] (default)
    #   "mpc"       — bicycle-model MPC tracking with 2 s lookahead
    #   "perfect"   — Euler integration with velocity from reference
    #                  (matches Autoware autoware_perfect_tracker)
    advance_mode: str = "teleport"
    mpc_horizon_steps: int = 20
    mpc_n_knots: int = 5
    # Ego vehicle dimensions. Override for non-default vehicles (e.g. larger
    # buses with longer wheelbase and wider footprint).
    ego_length: float = 4.5
    ego_width: float = 1.9
    ego_wheelbase: float = 2.925  # 4.5 * 0.65
    ego_max_steer: float = 0.6
    # Model inference delay: number of initial timesteps kept fixed as prefix.
    # Matches the "delay" input tensor to the diffusion decoder.
    inference_delay: int = 0
    # Skip traffic-light state propagation entirely. Useful for MPC-gen
    # data runs where TL-driven speed drops would bias the replay ego
    # toward stop-and-go behaviour we don't want in training.
    enable_traffic_lights: bool = True
    # Overlay the live metric values + closest road-border line on each
    # per-step PNG. Requires dump_npz_dir + reward_config_path (so the
    # metrics have actually been computed). Adds ~1 ms per frame.
    overlay_metrics_on_png: bool = False
    # When set, per-step observations are dumped as training-style NPZs to
    # this directory. GT future is zeroed out (ranked-SFT generates its own).
    dump_npz_dir: str | None = None
    # Path to a training-style (GRPO) config JSON. Required when dump_npz_dir
    # is set: per-step lane / border / centerline metrics are logged using
    # the same thresholds the training run will use, so downstream scene
    # selection can re-threshold without re-running the sim.
    reward_config_path: str | None = None
    # Initial ego speed for history synthesis (m/s). Default uses midpoint
    # of NPC speed band (7.5), but real data may be much slower (e.g. 1.75
    # on low-speed exit curves). Set to match the scenario being replayed.
    ego_init_speed: float | None = None
    # Run model inference one agent at a time instead of batching all
    # agents into a single forward pass. Slower but useful for diagnosing
    # whether batched inference affects trajectory quality.
    sequential_inference: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Re-check field invariants.

        Call after CLI overrides or any direct field mutation so late-bound
        changes (e.g. ``cfg.max_steps = args.steps``) still fail fast instead
        of silently carrying bad values into ``run_route_replay``.
        """
        if self.ego_length <= 0 or self.ego_width <= 0 or self.ego_wheelbase <= 0:
            raise ValueError(
                f"ego dimensions must be positive "
                f"(length={self.ego_length}, width={self.ego_width}, "
                f"wheelbase={self.ego_wheelbase})"
            )
        if self.ego_wheelbase > self.ego_length:
            raise ValueError(
                f"ego_wheelbase must be <= ego_length "
                f"(wheelbase={self.ego_wheelbase}, length={self.ego_length})"
            )
        if not 0 < self.ego_max_steer < math.pi / 2:
            raise ValueError(
                f"ego_max_steer must be in (0, pi/2); got {self.ego_max_steer}"
            )
        if self.inference_delay < 0:
            raise ValueError(
                f"inference_delay must be non-negative; got {self.inference_delay}"
            )
        if self.max_steps < 1:
            raise ValueError(
                f"max_steps must be >= 1; got {self.max_steps}"
            )
        if self.ego_init_speed is not None and self.ego_init_speed < 0:
            raise ValueError(
                f"ego_init_speed must be >= 0 when set; got {self.ego_init_speed}"
            )
        if self.dump_npz_dir and not self.reward_config_path:
            raise ValueError(
                "reward_config_path is required when dump_npz_dir is set; "
                "per-step metrics need thresholds that match the training "
                "reward function (no silent defaults)."
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "SpawnConfig":
        """Load a SpawnConfig from a JSON file.

        Unknown keys (including ``_comment_*`` keys used as inline JSON
        comments) are silently dropped so configs can carry documentation
        without tripping the dataclass constructor.
        """
        with open(path) as f:
            data = json.load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ── NPC spawn manager ────────────────────────────────────────────────────────


class SceneNPCManager:
    """Spawns and despawns ``Agent`` objects inside a running ``SceneContext``.

    This class intentionally owns no state that outlives a single replay —
    construct a fresh one per ``run_route_replay`` invocation.

    The manager does **not** call the model. Inference for every agent
    (including freshly spawned neighbors) happens in the main replay loop via
    the shared ``_predict_batch`` one-forward-pass.
    """

    def __init__(
        self,
        builder: LaneletSceneBuilder,
        ego_route_ll_ids: list[int],
        spawn_config: SpawnConfig,
        tl_controller: TrafficLightController | None = None,
    ) -> None:
        self.builder = builder
        self.ego_route_ll_ids = ego_route_ll_ids
        self.cfg = spawn_config
        self.tl_controller = tl_controller
        self._sim_time: float = 0.0
        self._rng = random.Random(spawn_config.seed)
        self._np_rng = np.random.default_rng(spawn_config.seed)
        self._next_id = 0
        # Set of lanelet ids currently present in ``scene.map_data``. Tracked
        # incrementally so we only rebuild the MapTensorCache when new
        # lanelets actually appear.
        self._known_lanelet_ids: set[int] = set()
        # Ego route lanelets the ego has already driven through.
        # Updated every tick so NPC goals avoid untransited ego lanelets.
        self._ego_transited: set[int] = set()
        self._ego_untransited: set[int] = set(ego_route_ll_ids)

    def register_known_lanelets(self, lanelet_ids: list[int]) -> None:
        self._known_lanelet_ids.update(lanelet_ids)

    def update_ego_progress(self, ego_xy: np.ndarray) -> None:
        """Mark the ego's current lanelet (and all prior) as transited."""
        ll = self.builder.snap_to_nearest_ll(ego_xy)
        if ll is None or ll not in self._ego_untransited:
            return
        # Mark everything up to and including this lanelet as transited.
        for rid in self.ego_route_ll_ids:
            self._ego_transited.add(rid)
            self._ego_untransited.discard(rid)
            if rid == ll:
                break

    def tick(self, scene: SceneContext) -> None:
        """Run one spawn/despawn cycle.

        The map_data rebuild is owned by the main replay loop (it refreshes
        periodically based on ego position + closest lanelets). This method
        only mutates ``scene.agents``.
        """
        ego_agent = scene.ego_agent
        if ego_agent is None:
            return
        ego_pos = ego_agent.current_position

        # --- Despawn pass ---
        kept_agents: list[Agent] = []
        removed = 0
        for agent in scene.agents:
            if agent.id == scene.ego_agent_id:
                kept_agents.append(agent)
                continue
            d = float(np.linalg.norm(agent.current_position - ego_pos))
            if d > self.cfg.despawn_distance:
                removed += 1
                continue
            kept_agents.append(agent)
        scene.agents = kept_agents
        if removed > 0:
            print(f"  [NPCManager] despawned {removed} (beyond {self.cfg.despawn_distance:.0f} m)")

        # --- Spawn pass ---
        active_nb = sum(1 for a in scene.agents if a.id != scene.ego_agent_id)
        if active_nb >= self.cfg.max_active_npcs:
            return
        if self._rng.random() >= self.cfg.spawn_probability:
            return

        new_agent, _added_ll_ids = self._try_spawn_one(scene)
        if new_agent is None:
            return
        scene.agents.append(new_agent)
        print(f"  [NPCManager] spawned {new_agent.id}")

    # -- internals --------------------------------------------------------

    def _try_spawn_one(self, scene: SceneContext) -> tuple[Agent | None, list[int]]:
        """Attempt to synthesise one valid NPC near the ego. Returns
        ``(agent_or_None, list_of_lanelet_ids_the_new_agent_touches)``."""
        ego = scene.ego_agent
        ego_pos = ego.current_position
        ego_heading = ego.current_heading
        ego_forward = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float32)

        candidate_ids = self.builder.lanelets_near_point(
            ego_pos, self.cfg.max_spawn_distance,
        )
        # Drop lanelets that are too curved for history synthesis.
        candidate_ids = [
            ll_id for ll_id in candidate_ids
            if self.builder.is_lanelet_straight(ll_id, self.cfg.curvature_threshold)
        ]
        if not candidate_ids:
            return None, []

        forward_only = self._rng.random() < self.cfg.forward_bias
        if forward_only:
            filtered = []
            for ll_id in candidate_ids:
                cl = self.builder._cache[ll_id].raw_centerline
                mid = cl[len(cl) // 2]
                to_mid = mid - ego_pos
                if np.dot(to_mid, ego_forward) > 0:
                    filtered.append(ll_id)
            candidate_ids = filtered or candidate_ids  # fall back if forward filter empties

        # Existing agents' OBBs to collision-test against.
        existing_corners = [
            _obb_corners(
                a.current_position[0], a.current_position[1],
                a.current_heading, a.length, a.width,
            )
            for a in scene.agents
        ]

        for _ in range(20):  # up to 20 attempts
            ll_id = self._rng.choice(candidate_ids)
            # Skip lanelets that ARE traffic-light-controlled (inside the
            # intersection). Spawning is allowed on lanelets *before* the
            # intersection; speed is adjusted below based on TL state.
            if self.tl_controller is not None and \
               self.tl_controller.get_group_for_lanelet(ll_id) is not None:
                continue
            c = self.builder._cache[ll_id]
            cl = c.raw_centerline
            # Pick an arc-length away from the lanelet endpoints.
            total = float(c.arc_length)
            if total < 1.0:
                continue
            margin = min(3.0, total * 0.1)
            target_arc = self._rng.uniform(margin, total - margin)
            arc_lengths = c.cum_arc_lengths
            seg_idx = int(np.searchsorted(arc_lengths, target_arc)) - 1
            seg_idx = max(0, min(seg_idx, len(cl) - 2))
            seg_len = arc_lengths[seg_idx + 1] - arc_lengths[seg_idx]
            if seg_len < 1e-6:
                continue
            t = (target_arc - arc_lengths[seg_idx]) / max(seg_len, 1e-6)
            pos = cl[seg_idx] + t * (cl[seg_idx + 1] - cl[seg_idx])
            pos = pos.astype(np.float32)

            # Range check against ego. Dynamic minimum: at higher ego
            # speeds, push the min distance out to ego_speed * 3 s to
            # prevent NPCs from popping in dangerously close.
            d_ego = float(np.linalg.norm(pos - ego_pos))
            ego_speed = float(np.linalg.norm(ego.current_velocity))
            dynamic_min = max(self.cfg.min_spawn_distance, ego_speed * 3.0)
            if d_ego < dynamic_min or d_ego > self.cfg.max_spawn_distance:
                continue

            # Lane heading at this point.
            if seg_idx < len(cl) - 1:
                dxdy = cl[seg_idx + 1] - cl[seg_idx]
            else:
                dxdy = cl[seg_idx] - cl[seg_idx - 1]
            heading = float(math.atan2(dxdy[1], dxdy[0]))

            length = float(self._rng.uniform(4.0, 5.0))
            width = float(self._rng.uniform(1.7, 2.0))
            wheelbase = length * 0.65

            corners = _obb_corners(pos[0], pos[1], heading, length, width)
            collides = False
            for ec in existing_corners:
                ec_center = ec.mean(axis=0)
                if np.linalg.norm(pos - ec_center) < self.cfg.min_npc_separation:
                    collides = True
                    break
                if _obb_collides(corners, ec):
                    collides = True
                    break
            if collides:
                continue

            # Speed from lane speed limit (±20%), falling back to config range.
            cache_entry = self.builder._cache[ll_id]
            if cache_entry.has_speed_limit and cache_entry.speed_limit_mps > 0:
                sl = cache_entry.speed_limit_mps
                speed = float(self._rng.uniform(sl * 0.8, sl * 1.2))
                speed = max(speed, 0.5)  # floor to avoid near-zero spawns
            else:
                speed = float(self._rng.uniform(self.cfg.npc_min_speed, self.cfg.npc_max_speed))

            # Route for this neighbor.
            route_ll_ids = self._pick_route(ll_id)
            goal = self.builder._route_goal(route_ll_ids)
            route_lanes, route_sl, route_hsl = self.builder._route_to_33dim(route_ll_ids)
            if self.tl_controller is not None:
                self.tl_controller.write_to_route_lanes(
                    route_lanes, route_ll_ids, self._sim_time,
                )

            # Reject spawn if too close to a red light on its route.
            if self._too_close_to_red_tl(pos, route_ll_ids):
                continue

            history, history_ll_ids = self.builder.generate_history(
                pos, heading, speed, ll_id,
            )
            velocities = np.zeros((history.shape[0], 2), dtype=np.float32)
            for k in range(1, history.shape[0]):
                velocities[k] = (history[k, :2] - history[k - 1, :2]) / 0.1
            velocities[0] = velocities[1]

            # Realistic kinematic initialisation from the synthesised
            # history — lanelet centerline tracing already gives curvature,
            # so yaw_rate / steering / acceleration are computable rather
            # than the zero placeholders we had before.
            dh_spawn = np.arctan2(
                np.sin(np.diff(history[-5:, 2])),
                np.cos(np.diff(history[-5:, 2])),
            )
            yaw_rate_spawn = float(dh_spawn.mean() / 0.1) if len(dh_spawn) > 0 else 0.0
            speed_spawn = float(np.linalg.norm(velocities[-1]))
            accel_spawn = (velocities[-1] - velocities[-3]) / (2 * 0.1) \
                if len(velocities) >= 3 else np.zeros(2, dtype=np.float32)
            steering_spawn = float(math.atan2(wheelbase * yaw_rate_spawn, max(speed_spawn, 0.2))) \
                if speed_spawn > 0.2 else 0.0

            agent_id = f"npc_{self._next_id}"
            self._next_id += 1
            agent = Agent(
                id=agent_id,
                agent_type=AgentType.VEHICLE,
                length=length,
                width=width,
                wheelbase=wheelbase,
                past_trajectory=history,
                past_velocities=velocities,
                acceleration=accel_spawn.astype(np.float32),
                steering_angle=steering_spawn,
                yaw_rate=yaw_rate_spawn,
                goal_pose=goal,
                route_lanes=route_lanes,
                route_speed_limit=route_sl,
                route_has_speed_limit=route_hsl,
                turn_indicators=np.zeros(history.shape[0], dtype=np.int32),
                age_steps=0,
                route_lanelet_ids=route_ll_ids,
            )
            touched = list(set(route_ll_ids) | set(history_ll_ids) | {ll_id})
            return agent, touched

        return None, []

    def _too_close_to_red_tl(
        self,
        spawn_pos: np.ndarray,
        route_ll_ids: list[int],
        min_dist: float = 30.0,
    ) -> bool:
        """Return True if the spawn is within ``min_dist`` of a RED/YELLOW TL
        on the NPC's route. Such spawns are rejected outright — a vehicle
        appearing 10 m before a red light with full speed is unrealistic
        and confuses the ego.
        """
        from scenario_generation.traffic_light import TL_RED, TL_YELLOW
        tl = self.tl_controller
        if tl is None:
            return False

        for rid in route_ll_ids:
            gid = tl.get_group_for_lanelet(rid)
            if gid is None:
                continue
            color = tl.color_for_group(gid, self._sim_time)
            if color not in (TL_RED, TL_YELLOW):
                continue  # green TL, check remaining
            # Red/yellow TL — check distance to its start.
            if rid not in self.builder._cache:
                return True
            tl_start = self.builder._cache[rid].raw_centerline[0]
            dist = float(np.linalg.norm(spawn_pos - tl_start))
            if dist < min_dist:
                return True

        return False

    def _trim_route_off_ego(self, route: list[int]) -> list[int]:
        """Trim route so its goal lanelet is not on an untransited ego lanelet.

        Walks backward from the end of the route, dropping lanelets that
        belong to the ego's future path, until a safe goal is found.
        Returns at least the first lanelet (the spawn lanelet).
        """
        if not self._ego_untransited:
            return route
        end = len(route)
        while end > 1 and route[end - 1] in self._ego_untransited:
            end -= 1
        return route[:end]

    def _pick_route(self, start_ll_id: int) -> list[int]:
        """Select a forward route for a freshly-spawned NPC.

        With probability ``ego_overlap_ratio`` we retry ``find_route`` a few
        times searching for a candidate that shares at least one lanelet with
        the ego's route — more interactions with ego, less natural traffic
        diversity. Otherwise a single random forward route.

        The route is trimmed so its goal does not land on an ego route
        lanelet that the ego has not yet transited.
        """
        want_overlap = self._rng.random() < self.cfg.ego_overlap_ratio
        ego_set = set(self.ego_route_ll_ids)
        best = self.builder.find_route(start_ll_id, self.cfg.npc_route_length_m)
        if want_overlap and not (set(best) & ego_set):
            for _ in range(5):
                candidate = self.builder.find_route(start_ll_id, self.cfg.npc_route_length_m)
                if set(candidate) & ego_set:
                    best = candidate
                    break
        return self._trim_route_off_ego(best)


# ── Replay loop ──────────────────────────────────────────────────────────────


_LANE_COLOR = "#bbbbbb"
_LANE_BORDER_COLOR = "#888888"
_ROAD_BORDER_COLOR = "#dd2222"
_EGO_COLOR = "#3366cc"
_ROUTE_COLOR = "#3366cc"
_VIEW_HALF_M = 50.0  # ±50 m window around ego keeps lane detail legible


def _draw_lane_network(ax, map_data, alpha: float = 0.7) -> None:
    """Draw lane centerlines **and** left/right borders from the 33-dim tensor.

    Borders are reconstructed via ``centerline + lane[:, 4:6]`` (left) and
    ``centerline + lane[:, 6:8]`` (right). ``map_data.line_strings`` stays
    zero-filled in replay scenes so we can't rely on it; the borders we draw
    here come from the lane tensor which is always populated.
    """
    from matplotlib.collections import LineCollection

    lanes = map_data.lanes
    centerlines, lefts, rights = [], [], []
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        centerlines.append(pts[valid])
        if lane.shape[1] > 7:
            lefts.append((pts + lane[:, 4:6])[valid])
            rights.append((pts + lane[:, 6:8])[valid])

    if centerlines:
        ax.add_collection(LineCollection(
            centerlines, colors=_LANE_COLOR, linewidths=0.6,
            alpha=alpha * 0.4, zorder=1,
        ))
    if lefts:
        ax.add_collection(LineCollection(
            lefts, colors=_LANE_BORDER_COLOR, linewidths=1.1,
            alpha=alpha, zorder=2,
        ))
    if rights:
        ax.add_collection(LineCollection(
            rights, colors=_LANE_BORDER_COLOR, linewidths=1.1,
            alpha=alpha, zorder=2,
        ))


def _draw_road_borders(ax, road_border_polylines, view_center=None, view_half_m=None) -> None:
    """Draw actual road-border polylines (curbs/walls) in red. Separate from
    lane markings; these come from the builder's line_strings_cache (type
    ``road_border``), not from the lane tensor.
    """
    if not road_border_polylines:
        return
    from matplotlib.collections import LineCollection
    # AABB filter to avoid drawing the whole map each tick
    if view_center is not None and view_half_m is not None:
        cx, cy = view_center
        half = view_half_m * 1.5  # keep a bit of margin
        filtered = []
        for pl in road_border_polylines:
            if pl.shape[0] < 2:
                continue
            in_view = (
                (pl[:, 0] >= cx - half) & (pl[:, 0] <= cx + half)
                & (pl[:, 1] >= cy - half) & (pl[:, 1] <= cy + half)
            )
            if in_view.any():
                filtered.append(pl)
        polylines = filtered
    else:
        polylines = [pl for pl in road_border_polylines if pl.shape[0] >= 2]
    if not polylines:
        return
    ax.add_collection(LineCollection(
        polylines, colors=_ROAD_BORDER_COLOR, linewidths=2.0,
        alpha=0.9, zorder=5,  # above lanes, below agents
    ))


def _nearest_border_point(
    probe_xy: np.ndarray,
    border_polylines: list[np.ndarray] | None,
) -> np.ndarray | None:
    """Nearest point on any road-border polyline to ``probe_xy`` (world frame).

    Pure geometry used only to position the viz pointer — the authoritative
    body-to-border distance comes from ``rlvr.reward.compute_road_border_penalty``
    via the per-step metrics log. Do not read the returned point's distance
    as a metric.
    """
    if not border_polylines:
        return None
    best_pt: np.ndarray | None = None
    best_d = float("inf")
    px, py = float(probe_xy[0]), float(probe_xy[1])
    for pl in border_polylines:
        if pl is None or pl.shape[0] < 2:
            continue
        p1 = pl[:-1].astype(np.float64)
        p2 = pl[1:].astype(np.float64)
        seg = p2 - p1
        seg_len2 = (seg * seg).sum(axis=1)
        seg_len2[seg_len2 < 1e-9] = 1e-9
        t = (((px - p1[:, 0]) * seg[:, 0] + (py - p1[:, 1]) * seg[:, 1])
             / seg_len2)
        t = np.clip(t, 0.0, 1.0)
        closest = p1 + t[:, None] * seg
        dists = np.hypot(closest[:, 0] - px, closest[:, 1] - py)
        idx = int(dists.argmin())
        if dists[idx] < best_d:
            best_d = float(dists[idx])
            best_pt = closest[idx]
    return best_pt


def _ego_obb_corners(
    ex: float, ey: float, heading: float, length: float, width: float,
) -> np.ndarray:
    """Four OBB corners of the ego footprint in world frame (pure geometry).

    Matches the rear-axle convention used by ``scenario_generation.visualize
    .draw_agent_box``: baselink (ego x, y) sits rear_overhang behind the
    back of the box.
    """
    rear_overhang = (length - length * 0.65) / 2
    x0, x1 = -rear_overhang, length - rear_overhang
    y0, y1 = -width / 2, width / 2
    local = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.float64)
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (R @ local.T).T + np.array([ex, ey], dtype=np.float64)


def _save_step_figure(
    scene: SceneContext,
    agent_predictions: dict,
    output_path: Path,
    step: int,
    n_steps: int,
    route_polylines: list[np.ndarray] | None = None,
    view_half_m: float = _VIEW_HALF_M,
    tl_controller: TrafficLightController | None = None,
    route_lanelet_ids: list[int] | None = None,
    sim_time: float = 0.0,
    road_border_polylines: list[np.ndarray] | None = None,
    metrics: dict | None = None,
) -> None:
    """Render + save the overview PNG for a single replay step.

    Viewport is fixed to ``±view_half_m`` metres around the ego, so lane
    borders stay visible and NPC detail remains readable at every step.
    """
    from matplotlib.figure import Figure

    ego = scene.ego_agent
    if ego is None:
        return
    ex, ey = ego.current_position

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("#f8f8f8")

    # 1) Lane network (centerlines + left / right lane markings, gray).
    _draw_lane_network(ax, scene.map_data)

    # 1b) Road borders (curbs/walls) from the lanelet map, drawn in red.
    _draw_road_borders(ax, road_border_polylines, view_center=(ex, ey),
                       view_half_m=view_half_m)

    # 2) Ego route polyline (drawn below agents but above lanes).
    if route_polylines:
        for pl in route_polylines:
            if pl.shape[0] >= 2:
                ax.plot(
                    pl[:, 0], pl[:, 1], "-", color=_ROUTE_COLOR,
                    lw=2.5, alpha=0.6, zorder=3,
                )

    # 2b) Traffic-light coloured overlay on ALL lanes in map_data that have
    #     active TL state (route, parallel, and perpendicular). Read
    #     centerline XY directly from the lane tensor [0:2].
    if tl_controller is not None:
        from matplotlib.collections import LineCollection
        tl_segments: dict[str, list[np.ndarray]] = {}  # hex → list of polylines
        lanes = scene.map_data.lanes
        # Use the map_data_ll_ids that were stored when building map_data.
        # They are passed via route_lanelet_ids for the route overlay, but
        # for ALL lanes we need the builder's _last_map_data_ids — which
        # we can't access here. Instead, read the TL one-hot directly from
        # the lane tensor channels [8:13].
        for i in range(lanes.shape[0]):
            lane = lanes[i]
            pts = lane[:, :2]
            if np.abs(pts).sum() < 1e-6:
                continue
            tl_onehot = lane[0, 8:13]
            if tl_onehot.sum() < 0.5:
                continue
            ch = int(np.argmax(tl_onehot))
            from scenario_generation.traffic_light import TL_HEX, TL_NONE
            if ch == TL_NONE:
                continue
            hex_color = TL_HEX.get(ch)
            if hex_color is None:
                continue
            valid = np.abs(pts).sum(axis=1) > 0.1
            if valid.sum() < 2:
                continue
            tl_segments.setdefault(hex_color, []).append(pts[valid])

        for hex_color, segs in tl_segments.items():
            ax.add_collection(LineCollection(
                segs, colors=hex_color, linewidths=2.5,
                alpha=0.85, zorder=4,
            ))

    # 3) Agents + per-agent predicted trajectories.
    # Assign colors by hashing the agent id (or extracting a stable numeric
    # suffix from ``npc_N``) so a given neighbor keeps the same color even
    # when other NPCs spawn / despawn and reshuffle the iteration order.
    # Previously we indexed the palette by iteration rank, which made
    # colors jump on every spawn / despawn tick.
    def _stable_color(agent) -> str:
        if agent.id == scene.ego_agent_id:
            return _EGO_COLOR
        # npc_5 → 5; falls back to Python hash otherwise.
        sid = agent.id
        idx = None
        if "_" in sid:
            suffix = sid.rsplit("_", 1)[-1]
            if suffix.isdigit():
                idx = int(suffix)
        if idx is None:
            idx = abs(hash(sid))
        return _agent_color(agent.agent_type, idx)

    for agent in scene.agents:
        is_ego = agent.id == scene.ego_agent_id
        color = _stable_color(agent)

        pos = agent.current_position
        heading = agent.current_heading

        # Past trail (dashed, light).
        past = agent.past_trajectory
        valid = np.abs(past[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 1:
            ax.plot(
                past[valid, 0], past[valid, 1], "--", color=color,
                lw=0.9, alpha=0.5, zorder=7,
            )

        # Bounding box + heading arrow.
        draw_agent_box(
            ax, pos[0], pos[1], heading, agent.length, agent.width,
            color, alpha=0.85 if is_ego else 0.55, lw=2 if is_ego else 1,
            zorder=20 if is_ego else 15,
        )
        arrow_len = max(agent.length, 2.5)
        ax.annotate(
            "",
            xy=(pos[0] + arrow_len * math.cos(heading),
                pos[1] + arrow_len * math.sin(heading)),
            xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12),
            zorder=21 if is_ego else 16,
        )
        ax.annotate(
            agent.id, (pos[0], pos[1]), fontsize=7, color=color,
            ha="center", va="bottom", xytext=(0, 6), textcoords="offset points",
            zorder=22,
        )

        # Predicted trajectory from the model (in that agent's ego frame).
        if agent.id in agent_predictions:
            pred = agent_predictions[agent.id]
            plan_xy, plan_h = _ego_to_world(
                pred[:, :2], pred[:, 2:4],
                float(pos[0]), float(pos[1]), heading,
            )
            plan_traj = np.concatenate([plan_xy, plan_h[:, np.newaxis]], axis=-1)
            draw_trajectory(
                ax, plan_traj, color,
                lw=1.8 if is_ego else 1.0,
                zorder=25 if is_ego else 18,
                show_footprints=is_ego,
                length=agent.length, width=agent.width,
            )

    # 4) Ego goal marker (if within viewport).
    if ego.goal_pose is not None:
        gx, gy = float(ego.goal_pose[0]), float(ego.goal_pose[1])
        if abs(gx - ex) <= view_half_m and abs(gy - ey) <= view_half_m:
            ax.plot(gx, gy, "*", color="#d62728", ms=18, zorder=30,
                    markeredgecolor="black", markeredgewidth=0.8)

    # 5) Viewport: fixed square around ego.
    ax.set_xlim(ex - view_half_m, ex + view_half_m)
    ax.set_ylim(ey - view_half_m, ey + view_half_m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    goal_d = float(np.linalg.norm(ego.current_position - ego.goal_pose[:2])) \
        if ego.goal_pose is not None else float("nan")

    # Ego state readout: speed, steering, current turn-signal class.
    ego_speed = float(np.hypot(ego.current_velocity[0], ego.current_velocity[1]))
    ego_speed_kph = ego_speed * 3.6
    _TI_NAMES = {0: "NONE", 1: "DISABLE", 2: "LEFT", 3: "RIGHT", 4: "KEEP"}
    ti_cls = (
        int(ego.turn_indicators[-1]) if ego.turn_indicators is not None else 0
    )
    ti_label = _TI_NAMES.get(ti_cls, f"?{ti_cls}")
    steer_deg = math.degrees(ego.steering_angle)
    title = (
        f"Step {step:04d}/{n_steps}  t={step * 0.1:.1f}s  agents={len(scene.agents)}"
        f"\nego  v={ego_speed:.1f} m/s ({ego_speed_kph:.0f} km/h)  "
        f"steer={steer_deg:+.0f}°  turn={ti_label}  goal_d={goal_d:.1f} m"
    )

    if metrics is not None:
        gate_s = "IN" if metrics.get("lane_gate", 1.0) >= 0.5 else "CROSS"
        title += (
            f"\nrb_min={metrics.get('rb_min_dist', float('nan')):.2f} m  "
            f"cl={metrics.get('cl_score', float('nan')):+.3f}  "
            f"lane={gate_s}  "
            f"lane_near={metrics.get('lane_near_frac', 0.0):.2f}"
        )
        # Position the viz pointer using the nearest border point to the
        # ego rear axle, then anchor the line on the nearest OBB corner
        # (body edge, not baselink) so the visual length roughly tracks
        # the body-to-border distance shown in the label.
        border_pt = _nearest_border_point(ego.current_position, road_border_polylines)
        if border_pt is not None:
            corners = _ego_obb_corners(
                ex, ey, ego.current_heading,
                float(ego.length), float(ego.width),
            )
            d_corner = np.hypot(corners[:, 0] - border_pt[0],
                                corners[:, 1] - border_pt[1])
            start = corners[int(d_corner.argmin())]
            body_d = metrics.get("rb_min_dist", float("nan"))
            ax.plot([start[0], border_pt[0]], [start[1], border_pt[1]],
                    "k--", linewidth=1.3, alpha=0.7, zorder=29)
            ax.plot(border_pt[0], border_pt[1], "ko", markersize=6, zorder=30,
                    markeredgecolor="white", markeredgewidth=0.8)
            mx, my = (start[0] + border_pt[0]) / 2, (start[1] + border_pt[1]) / 2
            ax.annotate(f"{body_d:.2f} m",
                        xy=(mx, my), fontsize=8, color="black",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="black",
                                  alpha=0.7),
                        zorder=31)

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _save_and_close(fig, output_path)


@torch.no_grad()
@torch.no_grad()
def _score_step(
    npz_data: dict[str, np.ndarray],
    step: int,
    device: str,
    reward_cfg: RewardConfig,
    spawn_config: SpawnConfig,
) -> dict:
    """Score the current ego pose against the dumped map tensors.

    The dumped NPZ has ``ego_agent_future`` zeroed, so the "trajectory" here
    is a 1-step origin placeholder (t=0 + t=1 both at origin). The penalty
    primitives skip t=0 for near/wide fractions; the duplicate t=1 slot
    gives them one timestep to evaluate, representing the current pose.

    Everything dispatches to ``compute_reward_batch`` so that adding a new
    field to ``rlvr.reward.RewardBreakdown`` automatically flows into the
    log — no per-field mapping here to maintain. The one extra is an
    explicit baselink-mode centerline score (the ``RewardBreakdown``
    already holds a centerline term, but whether it used body or baselink
    depends on the training config; the heatmap wants the rear-axle
    version).
    """
    def _to_t(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(arr)).float().to(device)
        return t.unsqueeze(0) if t.dim() == 3 else t

    d: dict[str, torch.Tensor] = {}
    for k in ("lanes", "route_lanes", "line_strings", "ego_shape",
              "neighbor_agents_future", "neighbor_agents_past", "goal_pose"):
        if k in npz_data:
            d[k] = _to_t(npz_data[k])

    ego_shape_cl = torch.tensor(
        [spawn_config.ego_wheelbase, spawn_config.ego_length, spawn_config.ego_width],
        device=device, dtype=torch.float32,
    )

    traj = torch.zeros(1, 2, 4, device=device)
    traj[0, :, 2] = 1.0

    breakdowns = compute_reward_batch(traj, d, reward_cfg)
    br = breakdowns[0]

    cl_baselink = compute_centerline_score_batch(
        traj, ego_shape_cl, d,
        usage_cap=reward_cfg.centerline_usage_cap,
        usage_mode="baselink",
    )

    # Dump every RewardBreakdown field by iterating the dataclass, so
    # adding a new component only requires touching rlvr.reward — this
    # function stays untouched.
    out: dict = {"step": step}
    for k, v in asdict(br).items():
        if isinstance(v, (bool, int, float)) or v is None:
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = float(v.item())
        # Anything else (should not happen for RewardBreakdown) gets dropped
        # rather than breaking JSON serialization.

    # Derived convenience fields not in RewardBreakdown:
    #   collision: bool from collision_step
    #   lane_gate: 0/1 alias of (not lane_crossing) — selector reads it
    #   cl_score:  raw rear-axle centerline magnitude
    out["collision"] = out.get("collision_step") is not None
    out["lane_gate"] = 0.0 if out.get("lane_crossing") else 1.0
    out["cl_score"] = float(cl_baselink[0].item())
    return out


def run_route_replay(
    model,
    model_args,
    builder: LaneletSceneBuilder,
    route: Route,
    output_dir: Path,
    spawn_config: SpawnConfig | None = None,
    device: str = "cuda",
) -> dict:
    """Run closed-loop replay of ``route`` with dynamic NPC spawning.

    Args:
        model: Loaded Diffusion-Planner (``eval()`` already called).
        model_args: ``Config`` instance returned alongside ``model`` by
            :func:`scenario_generation.simulate.load_model`.
        builder: Lanelet scene builder for the map ``route.map_path`` points
            at (rebuild one fresh per call — cheap vs. GPU inference).
        route: Authored route spec. Must be resolved (``route_lanelet_ids``
            non-empty); falls back to greedy ``find_route`` with a warning
            when unresolved.
        output_dir: Directory for per-step PNGs. Created if missing.
        spawn_config: NPC manager tuning. Defaults to :class:`SpawnConfig()`.
        device: Torch device.

    Returns:
        Dict with ``final_step``, ``goal_reached`` (bool), ``reason`` (str),
        and ``n_npc_spawned`` for downstream scripting.
    """
    if spawn_config is None:
        spawn_config = SpawnConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed ALL random sources for full reproducibility across runs.
    if spawn_config.seed is not None:
        torch.manual_seed(spawn_config.seed)
        torch.cuda.manual_seed_all(spawn_config.seed)
        np.random.seed(spawn_config.seed)
        random.seed(spawn_config.seed)

    # --- Step 0: build the initial scene from the Route. ---
    ego_route_ids = route.route_lanelet_ids
    if not ego_route_ids:
        if route.start_lanelet_id is None:
            raise ValueError("Route has no start_lanelet_id and no resolved path")
        print("  [WARN] Route.route_lanelet_ids is empty; falling back to find_route")
        ego_route_ids = builder.find_route(
            route.start_lanelet_id, spawn_config.npc_route_length_m,
        )

    # Snap the ego to a lanelet on the saved route (prevents the initial lane
    # from drifting to a parallel lane that isn't part of the route).
    start_pose = route.start_pose
    start_ll_id = builder.snap_to_nearest_ll(
        start_pose[:2], candidate_ids=ego_route_ids,
    ) or route.start_lanelet_id
    if start_ll_id is None:
        raise ValueError("Could not determine start lanelet for the ego")

    # Snap the ego's x,y onto the chosen lanelet's centerline for stability.
    cl = builder._cache[start_ll_id].raw_centerline
    dists = np.linalg.norm(cl - start_pose[:2], axis=1)
    closest = int(np.argmin(dists))
    snapped_xy = cl[closest].astype(np.float32)
    heading = float(start_pose[2])
    # Initial ego speed for history synthesis.
    if spawn_config.ego_init_speed is not None:
        init_speed = spawn_config.ego_init_speed
    else:
        init_speed = 0.5 * (spawn_config.npc_min_speed + spawn_config.npc_max_speed)

    # Synthesise realistic history along the route's predecessor lanelets.
    history, history_ll_ids = builder.generate_history(
        snapped_xy, heading, init_speed, start_ll_id,
    )
    # Override history[-1] with the user-specified heading so the ego faces
    # the right way at t=0 even when the lane heading differs slightly.
    history[-1, 2] = heading
    velocities = np.zeros((history.shape[0], 2), dtype=np.float32)
    for k in range(1, history.shape[0]):
        velocities[k] = (history[k, :2] - history[k - 1, :2]) / 0.1
    velocities[0] = velocities[1]

    # Build map_data to mirror the Diffusion-Planner ROS node:
    # - closest lanelets to ego via a ±100 m AABB pre-filter + distance sort
    # - ego route + history pinned (route context never drops, even when the
    #   ego approaches a dense junction where the closest-N would saturate)
    # - each alive NPC's current lanelet pinned so neighbors always have
    #   lane context even when outside the ego bbox (NPCs can live up to
    #   despawn_distance = 120 m > bbox = 100 m from ego)
    # - final hard cap at ``max_map_lanelets`` (140 = tensor_converter._NUM_LANES).
    def _compute_map_lanelet_ids(
        ego_xy: np.ndarray,
        neighbor_positions: list[np.ndarray],
    ) -> list[int]:
        closest = builder.closest_lanelets(
            ego_xy, spawn_config.max_map_lanelets,
            mask_range=spawn_config.map_mask_range_m,
        )
        pinned: list[int] = list(ego_route_ids) + list(history_ll_ids)
        for nb_xy in neighbor_positions:
            ll = builder.snap_to_nearest_ll(nb_xy)
            if ll is not None:
                pinned.append(ll)

        # Deduplicate: pinned IDs first (always included), then closest.
        seen: set[int] = set()
        ordered: list[int] = []
        for ll_id in pinned + list(closest):
            if ll_id in seen:
                continue
            seen.add(ll_id)
            ordered.append(ll_id)
            if len(ordered) >= spawn_config.max_map_lanelets:
                break
        return ordered

    all_lanelet_ids = _compute_map_lanelet_ids(snapped_xy, [])
    map_data = builder._build_map_data(all_lanelet_ids, center_xy=snapped_xy)

    # Initial route_lanes uses the C++-style forward window (not the full
    # saved route — training data has median 4 non-zero route slots, so
    # packing all 25 over-provides context). Refreshed every
    # ``map_refresh_steps`` in the main loop as the ego advances.
    initial_route_window = builder.select_route_segment_indices(
        ego_route_ids, snapped_xy, max_segments=25,
    ) or ego_route_ids[:25]
    route_lanes, route_sl, route_hsl = builder._route_to_33dim(initial_route_window)
    # Initial kinematic derivatives from the synthesized history so the
    # first inference call sees realistic non-zero yaw_rate + steering +
    # acceleration (otherwise the model's first ~1-2 steps behave as if
    # the ego just teleported in with zero state).
    dh_init = np.arctan2(
        np.sin(np.diff(history[-5:, 2])),
        np.cos(np.diff(history[-5:, 2])),
    )
    yaw_rate_init = float(dh_init.mean() / 0.1) if len(dh_init) > 0 else 0.0
    speed_init = float(np.linalg.norm(velocities[-1]))
    accel_init = (velocities[-1] - velocities[-3]) / (2 * 0.1) if len(velocities) >= 3 else np.zeros(2, dtype=np.float32)
    steering_init = float(math.atan2(spawn_config.ego_wheelbase * yaw_rate_init, max(speed_init, 0.2))) if speed_init > 0.2 else 0.0

    ego = Agent(
        id="ego",
        agent_type=AgentType.VEHICLE,
        length=spawn_config.ego_length, width=spawn_config.ego_width, wheelbase=spawn_config.ego_wheelbase,
        past_trajectory=history,
        past_velocities=velocities,
        acceleration=accel_init.astype(np.float32),
        steering_angle=steering_init,
        yaw_rate=yaw_rate_init,
        goal_pose=route.goal_pose.astype(np.float32),
        route_lanes=route_lanes,
        route_speed_limit=route_sl,
        route_has_speed_limit=route_hsl,
        turn_indicators=np.zeros(history.shape[0], dtype=np.int32),
        route_lanelet_ids=list(ego_route_ids),
    )
    scene = SceneContext(agents=[ego], map_data=map_data, ego_agent_id="ego", dt=0.1)

    # Road-border polylines (world frame) via the public accessor; this
    # filters to only road_border entries (stop_line skipped).
    road_border_polylines = builder.road_border_polylines()

    # Route polyline (world frame) for per-step visualisation. Keep the
    # lanelet ID list in sync so the TL overlay can colour each segment.
    _route_vis_ll_ids: list[int] = [
        ll_id for ll_id in ego_route_ids if ll_id in builder._cache
    ]
    route_polylines = [
        builder._cache[ll_id].raw_centerline[:, :2]
        for ll_id in _route_vis_ll_ids
    ]

    # --- Traffic light controller. ---
    tl_controller: TrafficLightController | None = None
    if spawn_config.enable_traffic_lights:
        tl_controller = TrafficLightController(
            builder, ego_route_ids, seed=spawn_config.seed,
        )
        # Apply initial TL state to the freshly-built map_data AND ego route_lanes.
        tl_controller.tick(scene, 0.0, builder._last_map_data_ids, ego_xy=snapped_xy)
        tl_controller.write_to_route_lanes(
            scene.ego_agent.route_lanes, initial_route_window, 0.0,
        )

    # --- NPC manager. ---
    npc_manager = SceneNPCManager(builder, ego_route_ids, spawn_config, tl_controller)
    npc_manager.register_known_lanelets(all_lanelet_ids)

    map_cache = MapTensorCache(scene.map_data)
    n_npc_spawned = 0
    goal_reached = False
    reason = "max_steps"
    min_goal_d = float("inf")  # closest approach to goal seen so far

    # Per-step trajectory log for post-hoc evaluation.
    trajectory_log: list[dict] = []

    # Live lane / border / centerline scoring, logged per step when NPZ dump
    # + reward_config_path are both set. The downstream scene selector reads
    # this log; no offline NPZ re-scoring needed.
    metrics_log: list[dict] = []
    reward_cfg: RewardConfig | None = None
    if spawn_config.dump_npz_dir and spawn_config.reward_config_path:
        reward_cfg = load_reward_config(spawn_config.reward_config_path)

    # Tracker state (lazy-init per agent inside advance_scene_mpc).
    _use_tracker = spawn_config.advance_mode in ("mpc", "perfect")
    mpc_trackers: dict = {}
    if _use_tracker:
        print(f"  Advance mode: {spawn_config.advance_mode}"
              + (f" (horizon={spawn_config.mpc_horizon_steps}, "
                 f"knots={spawn_config.mpc_n_knots})"
                 if spawn_config.advance_mode == "mpc" else ""))

    # --- Main loop. ---
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="save") as save_pool:
        pending_saves: list = []
        for step in range(spawn_config.max_steps):
            # Keep NPC manager's sim time in sync for TL writes on spawn.
            npc_manager._sim_time = step * 0.1

            # Track which ego route lanelets have been transited so NPC
            # goals avoid landing on the ego's future path.
            npc_manager.update_ego_progress(scene.ego_agent.current_position)

            # Run the NPC manager (after step 0 so the ego has a meaningful
            # ``current_position`` — always true by construction here).
            if step > 0 and step % spawn_config.spawn_period_steps == 0:
                before = sum(1 for a in scene.agents if a.id != scene.ego_agent_id)
                npc_manager.tick(scene)
                after = sum(1 for a in scene.agents if a.id != scene.ego_agent_id)
                if after > before:
                    n_npc_spawned += (after - before)
                # Prune trackers for despawned agents.
                if _use_tracker:
                    alive_ids = {a.id for a in scene.agents}
                    for stale_id in list(mpc_trackers.keys() - alive_ids):
                        del mpc_trackers[stale_id]

            # Refresh map_data + ego.route_lanes. Mirrors the C++ Autoware
            # planner, which rebuilds these every inference frame. We
            # throttle to ``map_refresh_steps`` (default 5 = 0.5 s at
            # dt=0.1). The refresh now also populates polygons +
            # line_strings (intersection areas + road borders + stop lines)
            # via center_xy, and slides the route window forward via
            # select_route_segment_indices.
            ego_xy = scene.ego_agent.current_position

            # Rebuild map_data periodically (expensive: closest-lanelet
            # query + polygon/line_string tensors).
            if step == 0 or step % spawn_config.map_refresh_steps == 0:
                neighbor_xys = [
                    a.current_position for a in scene.agents
                    if a.id != scene.ego_agent_id
                ]
                new_ids = _compute_map_lanelet_ids(ego_xy, neighbor_xys)
                scene.map_data = builder._build_map_data(new_ids, center_xy=ego_xy)
                npc_manager._known_lanelet_ids = set(new_ids)
                map_cache = MapTensorCache(scene.map_data)

            # TL state + route_lanes refresh EVERY step so the model
            # always sees the current signal phase. The TL one-hot is
            # written directly into scene.map_data.lanes which the
            # map_cache references (shared underlying array).
            if tl_controller is not None:
                tl_controller.tick(
                    scene, step * 0.1, builder._last_map_data_ids,
                    ego_xy=ego_xy,
                )

            # Refresh route_lanes for ALL agents (ego + NPCs) so the
            # sliding window stays centered on each agent's current
            # position. Without this, NPC route context goes stale and
            # the model plans trajectories in the wrong direction.
            for a in scene.agents:
                if a.route_lanelet_ids is None:
                    continue
                a_xy = a.current_position
                fwd = builder.select_route_segment_indices(
                    a.route_lanelet_ids, a_xy, max_segments=25,
                )
                if fwd:
                    rl, rsl, rhsl = builder._route_to_33dim(fwd)
                    if tl_controller is not None:
                        tl_controller.write_to_route_lanes(rl, fwd, step * 0.1)
                    a.route_lanes = rl
                    a.route_speed_limit = rsl
                    a.route_has_speed_limit = rhsl

            # Inference for every alive agent, in one batched forward pass.
            # Also returns per-agent argmax of the model's turn-indicator
            # logit head so we can feed it back into each agent's
            # turn_indicators history on the next frame — mirrors the C++
            # ``TurnIndicatorManager`` control loop.
            ids_to_predict = [a.id for a in scene.agents if a.agent_type == AgentType.VEHICLE]
            if spawn_config.sequential_inference:
                # One forward pass per agent (batch_size=1 each).
                agent_predictions: dict[str, np.ndarray] = {}
                agent_turn_indicators: dict[str, int] = {}
                for aid in ids_to_predict:
                    p, ti = _predict_batch(
                        model, model_args, scene, [aid], device,
                        map_cache=map_cache, return_turn_indicators=True,
                        inference_delay=spawn_config.inference_delay,
                    )
                    agent_predictions.update(p)
                    agent_turn_indicators.update(ti)
            else:
                agent_predictions, agent_turn_indicators = _predict_batch(
                    model, model_args, scene, ids_to_predict, device,
                    map_cache=map_cache, return_turn_indicators=True,
                    inference_delay=spawn_config.inference_delay,
                )

            # Optional Savitzky-Golay smoothing on each agent's predicted
            # trajectory. Reuses the same smoother the RL ranked-SFT
            # pipeline applies before its SFT loss (rlvr/grpo_sft_trainer.
            # _smooth_trajectory). Helps when the diffusion sampler emits
            # jitter at the first step; benign at worst when the output is
            # already clean.
            if spawn_config.sg_smooth_enabled:
                for aid, traj in agent_predictions.items():
                    agent_predictions[aid] = _sg_smooth_trajectory(
                        traj,
                        spawn_config.sg_filter_window,
                        spawn_config.sg_filter_order,
                    )

            # Optional: dump per-step observation NPZ (training-scene format).
            # Captures the scene as the model sees it just before this step's
            # prediction. Future trajectories are filled with zeros (ranked-
            # SFT generates its own, doesn't use GT).
            if getattr(spawn_config, "dump_npz_dir", None):
                npz_dir = Path(spawn_config.dump_npz_dir)
                npz_dir.mkdir(parents=True, exist_ok=True)
                # NPZ neighbor count is locked by the past array's fixed shape
                # (_MAX_NUM_NEIGHBORS=32), not by model_args.predicted_neighbor_num
                # (which counts predicted future trajectories, not past slots).
                data = dump_step_npz(
                    scene,
                    map_cache,
                    future_len=getattr(model_args, "future_len", 80),
                )
                np.savez(npz_dir / f"replay_step_{step:04d}.npz", **data)

                if reward_cfg is not None:
                    metrics_log.append(_score_step(
                        data, step, device, reward_cfg, spawn_config,
                    ))

            # Save PNG (concurrent with next step's compute).
            out_path = output_dir / f"step_{step:04d}.png"
            overlay_metrics = (
                metrics_log[-1]
                if spawn_config.overlay_metrics_on_png and metrics_log
                else None
            )
            pending_saves.append(save_pool.submit(
                _save_step_figure,
                deepcopy(scene), agent_predictions, out_path,
                step, spawn_config.max_steps, route_polylines,
                _VIEW_HALF_M, tl_controller, _route_vis_ll_ids,
                step * 0.1, road_border_polylines,
                overlay_metrics,
            ))

            # Drain finished saves so memory doesn't balloon. Call
            # .result() to surface any exceptions from background saves.
            if step % 50 == 0:
                still_pending = []
                for f in pending_saves:
                    if f.done():
                        f.result()  # raises if save thread failed
                    else:
                        still_pending.append(f)
                pending_saves = still_pending

            # Goal check (BEFORE advancing — measures arrival accurately).
            # Two termination conditions:
            #   (a) within ``goal_tolerance_m`` of the goal — clean arrival
            #   (b) ego *passed* the goal: goal is behind ego AND was within
            #       ``goal_pass_window_m`` recently. The diffusion planner
            #       isn't a goal-stop controller, so a perfect-arrival check
            #       at small radius (e.g. 2 m) often misses by a few metres
            #       and the ego then drives away. (b) catches that case.
            ego_pos = scene.ego_agent.current_position
            ego_heading = scene.ego_agent.current_heading
            goal_xy = route.goal_pose[:2]
            d_goal = float(np.linalg.norm(ego_pos - goal_xy))
            min_goal_d = min(min_goal_d, d_goal)

            # Log ego state before termination checks so the terminal frame
            # (at goal_reached/goal_passed) is preserved for post-hoc metrics.
            ego_agent = scene.ego_agent
            ego_speed = float(np.linalg.norm(ego_agent.past_velocities[-1])) \
                if ego_agent.past_velocities is not None else 0.0
            trajectory_log.append({
                "step": step,
                "x": float(ego_pos[0]),
                "y": float(ego_pos[1]),
                "heading": float(ego_heading),
                "speed": ego_speed,
                "goal_d": d_goal,
            })

            if d_goal <= spawn_config.goal_tolerance_m:
                goal_reached = True
                reason = "goal_reached"
                print(f"  Goal reached at step {step} (distance {d_goal:.2f} m)")
                break
            # Has the ego passed the goal? Vector ego→goal vs ego forward.
            ego_forward = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float32)
            to_goal = goal_xy - ego_pos
            dot = float(np.dot(to_goal, ego_forward))
            if dot < 0 and min_goal_d <= spawn_config.goal_pass_window_m:
                goal_reached = True
                reason = "goal_passed"
                print(
                    f"  Ego passed the goal at step {step} (current d={d_goal:.1f} m, "
                    f"closest approach was {min_goal_d:.1f} m within "
                    f"{spawn_config.goal_pass_window_m:.0f} m of goal)"
                )
                break

            if spawn_config.advance_mode in ("mpc", "perfect"):
                advance_scene_mpc(
                    scene, agent_predictions, mpc_trackers,
                    tracker_type=spawn_config.advance_mode,
                    mpc_horizon_steps=spawn_config.mpc_horizon_steps,
                    mpc_n_knots=spawn_config.mpc_n_knots,
                    ego_max_steer=spawn_config.ego_max_steer,
                )
            else:
                advance_scene(scene, agent_predictions)

            # Increment age for all agents so the tensor converter knows
            # how many history frames are "real" vs pre-spawn fabrication.
            for a in scene.agents:
                a.age_steps += 1

            # Closed-loop turn-indicator feedback: overwrite the freshly-
            # rolled last slot of each agent's turn_indicators history
            # with the argmax of the model's turn_indicator_logit. Matches
            # the C++ TurnIndicatorManager (minus the hold-duration
            # filter, which we skip for now — can be added later).
            for a in scene.agents:
                if a.turn_indicators is None:
                    continue
                ti_cls = agent_turn_indicators.get(a.id)
                if ti_cls is not None:
                    a.turn_indicators[-1] = int(ti_cls)

            if step % 50 == 0:
                print(
                    f"  step {step:04d}/{spawn_config.max_steps}  "
                    f"agents={len(scene.agents)}  "
                    f"ego=({ego_pos[0]:.1f}, {ego_pos[1]:.1f})  "
                    f"goal_d={np.linalg.norm(ego_pos - goal_xy):.1f} m"
                )

        for f in pending_saves:
            f.result()

    final_step = step
    print(f"Done. {final_step + 1} frames saved to {output_dir}; reason={reason}")

    # Save trajectory log for post-hoc evaluation.
    traj_log_path = output_dir / "trajectory_log.json"
    with open(traj_log_path, "w") as f:
        json.dump(trajectory_log, f)

    metrics_log_path: Path | None = None
    if metrics_log:
        metrics_log_path = output_dir / "metrics_log.json"
        # Record the config source so the downstream selector knows which
        # thresholds the stored lane_near_frac etc. were computed against.
        payload = {
            "reward_config_path": spawn_config.reward_config_path,
            "dump_npz_dir": spawn_config.dump_npz_dir,
            "ego_shape": [
                spawn_config.ego_wheelbase,
                spawn_config.ego_length,
                spawn_config.ego_width,
            ],
            "steps": metrics_log,
        }
        with open(metrics_log_path, "w") as f:
            json.dump(payload, f)

    out = {
        "final_step": final_step,
        "goal_reached": goal_reached,
        "reason": reason,
        "n_npc_spawned": n_npc_spawned,
        "trajectory_log_path": str(traj_log_path),
    }
    if metrics_log_path is not None:
        out["metrics_log_path"] = str(metrics_log_path)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop replay of a saved scenario_generation Route "
                    "with dynamic NPC spawning.",
    )
    parser.add_argument("--map_path", type=str, default=None,
                        help="Override the map path stored in the Route pickle")
    parser.add_argument("--route", type=Path, required=True, help="Path to route.pkl")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to best_model.pth")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory for per-step PNGs")
    parser.add_argument("--steps", type=int, default=None,
                        help="Max simulation steps (overrides config.max_steps; "
                             "default from config = 6000 = 10 min at dt=0.1)")
    parser.add_argument("--max_npcs", type=int, default=None,
                        help="Override hard cap on concurrent neighbors")
    parser.add_argument("--spawn_probability", type=float, default=None,
                        help="Override spawn probability per spawn tick")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to a SpawnConfig JSON. Required — replay "
                             "refuses to run with dataclass defaults because "
                             "they don't match any production recipe. Author "
                             "a config per run.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    route = Route.load(args.route)
    map_path = args.map_path or route.map_path
    print(f"Loading builder from {map_path}")
    builder = LaneletSceneBuilder(map_path)

    cfg = SpawnConfig.from_json(args.config)
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.max_npcs is not None:
        cfg.max_active_npcs = args.max_npcs
    if args.spawn_probability is not None:
        cfg.spawn_probability = args.spawn_probability
    if args.seed is not None:
        cfg.seed = args.seed
    # Re-run validation after CLI overrides so bad values (e.g. --steps 0)
    # are rejected at startup instead of much later.
    cfg.validate()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Loading model from {args.model_path}")
    model, model_args = load_model(args.model_path, device)

    run_route_replay(
        model=model, model_args=model_args, builder=builder, route=route,
        output_dir=args.output_dir, spawn_config=cfg, device=device,
    )


if __name__ == "__main__":
    main()
