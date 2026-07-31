from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from planner_metrics.subscores import (
    compute_road_border_penalty,
)
from rlvr.autoresearch.tools.classify_scene_failures import (
    _apply_scene_thresholds,
    _ego_shape_from_data,
    _moving_collision_all_rear_end,
    _moving_collision_step_gated,
    classify_loaded_scenes_batch,
    current_ego_neighbor_clearance,
)
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from scenario_generation.conflict_detector import detect_expert_disagreement_projected
from scenario_generation.reproducer_rollout import DT, _route_key
from scenario_generation.tools._heatmap_common import project_points_to_polyline

_SUPPORTED_REALIZED_EVENT_LABELS = frozenset(
    {"moving_collision", "static_collision", "road_border_crossing", "expert_disagreement"}
)

# Realized-lag (closed-loop) fail-to-resume detection: the paper-faithful Conflict above
# compares the model's OPEN-LOOP proposal against the expert future, which is blind to a
# dithering model whose plans keep promising a departure the closed loop never executes,
# and whose lag flags at chunk START fall before the rollout so their credit windows can
# never be saved. The realized branch instead compares where the ego ACTUALLY is against
# where the expert clock is, on the same route polyline. The gap must be sustained this
# many consecutive sim steps (1 s at 10 Hz) before flagging, so a transient gap while the
# recorded expert decelerates into a stop the model already reached does not fire.
REALIZED_LAG_SUSTAIN_STEPS = 10


def load_credit_windows(path: Path | None) -> dict[str, dict[str, int | float]] | None:
    if path is None:
        return None
    with open(path) as f:
        raw = json.load(f)
    frame_hz = raw.get("_frame_hz")
    defaults = raw.get("_defaults")
    if not isinstance(frame_hz, int | float) or float(frame_hz) <= 0:
        raise ValueError(f"{path}: _frame_hz must be a positive number")
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: _defaults must define width_s and gap_s")

    def _seconds_to_frames(label: str, field: str, value: object) -> tuple[float, int]:
        if not isinstance(value, int | float):
            raise ValueError(f"{path}: {label}.{field} must be a number of seconds")
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"{path}: {label}.{field} must be >= 0, got {seconds}")
        frames = int(round(seconds * float(frame_hz)))
        return seconds, frames

    out: dict[str, dict[str, int | float]] = {}
    for key, value in raw.items():
        if str(key).startswith("_"):
            continue
        label = str(key)
        if not isinstance(value, dict):
            raise ValueError(
                f"{path}: {label} must be an object with width_s/gap_s; scalar "
                "frame counts are not supported"
            )
        width_s, width_frames = _seconds_to_frames(
            label,
            "width_s",
            value.get("width_s", defaults.get("width_s")),
        )
        gap_s, gap_frames = _seconds_to_frames(
            label,
            "gap_s",
            value.get("gap_s", defaults.get("gap_s")),
        )
        if width_frames < 1:
            raise ValueError(f"{path}: {label}.width_s must round to at least 1 frame")
        out[label] = {
            "width_s": width_s,
            "gap_s": gap_s,
            "width_frames": width_frames,
            "gap_frames": gap_frames,
        }
    return out


def _np_dict_to_scoring_tensors(
    np_dict: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Normalize a scene dict (numpy arrays OR torch tensors) to scoring tensors.

    Values that are already tensors on the right device/dtype pass through
    without a copy (``Tensor.to`` is a no-op then) — the rollout hands the
    scorer GPU-resident slices of the batched model input, so the per-segment
    host->device re-upload this function used to force is gone.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in np_dict.items():
        if key in {"lanes_has_speed_limit", "route_lanes_has_speed_limit"}:
            dtype = torch.bool
        elif key in {"turn_indicators", "delay"}:
            dtype = torch.long
        else:
            dtype = torch.float32
        if torch.is_tensor(value):
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
    if "delay" not in out:
        out["delay"] = torch.zeros((1,), dtype=torch.long, device=device)
    return out


def build_reproducer_danger_scorer(
    *,
    reward_config: Path,
    threshold_config: Path,
    device: str,
    enable_conflict_detector: bool = False,
    allowed_labels: set[str] | None = None,
    count_rear_end_collisions: bool | None = None,
):
    reward_cfg = load_reward_config(reward_config)
    if count_rear_end_collisions is not None:
        # Keep mining rear-end consistent with the repair side and the RAW-col
        # event trigger: --count_rear_end_collisions -> do NOT ignore them.
        reward_cfg.ignore_rear_end_collisions = not count_rear_end_collisions
    scorer_args = SimpleNamespace(
        threshold_config=threshold_config,
        moving_near_thresh=None,
        static_near_thresh=None,
        rb_near_thresh=None,
        sc_cross_thresh=None,
        rb_cross_thresh=None,
        enable_conflict_detector=bool(enable_conflict_detector),
    )
    thresholds = _apply_scene_thresholds(reward_cfg, scorer_args)
    torch_device = torch.device(device)
    allowed = set(allowed_labels) if allowed_labels else None

    def _scorer(built, preds, data, _device) -> list[dict[str, Any]]:
        datas = [
            _np_dict_to_scoring_tensors(np_dict, device=torch_device)
            for _s, np_dict, *_rest in built
        ]
        scene_paths = [f"{_route_key(s.tl)}_{idx:08d}" for s, _np, _nb, idx, *_ in built]
        ego = torch.as_tensor(preds[:, None, :, :4], dtype=torch.float32, device=torch_device)
        rows = classify_loaded_scenes_batch(
            scene_paths,
            ego,
            datas,
            reward_cfg,
            moving_collision_thresh=float(thresholds["moving_collision_thresh"]),
            moving_near_thresh=float(thresholds["moving_near_thresh"]),
            static_near_thresh=float(thresholds["static_near_thresh"]),
            rb_near_thresh=float(thresholds["rb_near_thresh"]),
            device=torch_device,
            args=scorer_args,
        )
        for row in rows:
            row["trajectory_source"] = "reproducer_det"
            if allowed is not None:
                labels = [label for label in row.get("labels", []) if label in allowed]
                row["labels"] = labels or ["clean"]
        return rows

    return _scorer


def build_realized_event_scorer(
    *,
    reward_config: Path,
    threshold_config: Path,
    device: str,
    allowed_labels: set[str] | None = None,
    count_rear_end_collisions: bool | None = None,
):
    allowed = set(allowed_labels) if allowed_labels else set(_SUPPORTED_REALIZED_EVENT_LABELS)
    unsupported = sorted(allowed - _SUPPORTED_REALIZED_EVENT_LABELS)
    if unsupported:
        raise ValueError(
            "realized-event verification currently supports only "
            f"{sorted(_SUPPORTED_REALIZED_EVENT_LABELS)}; got unsupported labels {unsupported}"
        )

    reward_cfg = load_reward_config(reward_config)
    if count_rear_end_collisions is not None:
        # See build_reproducer_danger_scorer: keep realized-event moving-collision
        # detection consistent with the repair side / RAW-col trigger.
        reward_cfg.ignore_rear_end_collisions = not count_rear_end_collisions
    scorer_args = SimpleNamespace(
        threshold_config=threshold_config,
        moving_near_thresh=None,
        static_near_thresh=None,
        rb_near_thresh=None,
        sc_cross_thresh=None,
        rb_cross_thresh=None,
        enable_conflict_detector=False,
        expert_disagreement_wait_speed_mps=None,
        expert_disagreement_wait_progress_m=None,
        expert_disagreement_forward_progress_gap_m=None,
        expert_disagreement_lag_progress_gap_m=None,
        expert_disagreement_moving_speed_mps=None,
    )
    thresholds = _apply_scene_thresholds(reward_cfg, scorer_args)
    torch_device = torch.device(device)
    rb_cross_thresh = float(thresholds["rb_cross_thresh"])
    static_collision_thresh = float(thresholds["sc_cross_thresh"])
    moving_collision_thresh = float(thresholds["moving_collision_thresh"])

    def _scorer(
        np_dict: dict[str, Any],
        *,
        collided: bool,
        step: int | None = None,
        model_pred_world: np.ndarray | None = None,
        expert_future_world: np.ndarray | None = None,
        expert_future_speed: np.ndarray | None = None,
        ref_polyline_world: np.ndarray | None = None,
        realized_lag_streak: int = 0,
        realized_lag_gap_m: float | None = None,
    ) -> dict[str, Any]:
        labels: list[str] = []
        row: dict[str, Any] = {
            "trajectory_source": "reproducer_realized",
            "realized_collision": bool(collided),
        }
        tensors = _np_dict_to_scoring_tensors(np_dict, device=torch_device)

        if "moving_collision" in allowed:
            moving_clearance = current_ego_neighbor_clearance(
                tensors,
                reward_cfg,
                device=torch_device,
                neighbor_kind="moving",
            )
            row["moving_collision_step"] = None
            row["moving_min_dist"] = float("inf")
            row["rear_end_collision"] = False
            distances = moving_clearance["distances"]
            if distances.numel():
                row["moving_min_dist"] = float(moving_clearance["min_clearance"])
                # Unified gated collision definition (overlap + rear-end
                # suppression per reward_cfg), NOT a raw proximity threshold.
                row["moving_collision_step"] = _moving_collision_step_gated(
                    moving_clearance["ego_now"],
                    moving_clearance["ego_shape"],
                    moving_clearance["neighbors"],
                    moving_clearance["neighbor_shapes"],
                    moving_clearance["neighbor_valid"],
                    reward_cfg,
                    moving_collision_thresh,
                )
                if row["moving_collision_step"] is not None:
                    labels.append("moving_collision")
                    # Keep rear-ends but TAG them, so downstream can drop them if
                    # desired instead of losing the collision at detection time.
                    row["rear_end_collision"] = _moving_collision_all_rear_end(
                        moving_clearance["ego_now"],
                        moving_clearance["ego_shape"],
                        moving_clearance["neighbors"],
                        moving_clearance["neighbor_shapes"],
                        moving_clearance["neighbor_valid"],
                        moving_collision_thresh,
                    )

        if "static_collision" in allowed:
            static_clearance = current_ego_neighbor_clearance(
                tensors,
                reward_cfg,
                device=torch_device,
                neighbor_kind="static",
            )
            row["static_collision_step"] = None
            row["static_min_dist"] = 99.0
            row["stopped_neighbor_count"] = int(static_clearance["stopped_mask"].sum().item())
            if static_clearance["distances"].numel():
                row["static_min_dist"] = float(static_clearance["min_clearance"])
                if row["static_min_dist"] <= static_collision_thresh:
                    row["static_collision_step"] = 0
                    labels.append("static_collision")

        if "road_border_crossing" in allowed:
            ego_shape = tensors["ego_shape"].reshape(-1)[:3]
            ego_now = torch.tensor(
                [[[0.0, 0.0, 1.0, 0.0]]],
                dtype=torch.float32,
                device=torch_device,
            )
            _gate, _near, _wide, _steps, _cont, per_timestep_min = compute_road_border_penalty(
                ego_now,
                ego_shape,
                tensors,
                reward_cfg,
            )
            rb_min_dist = float(per_timestep_min[0, 0].item())
            row["rb_min_dist"] = rb_min_dist
            if rb_min_dist < rb_cross_thresh:
                labels.append("road_border_crossing")

        if "expert_disagreement" in allowed:
            row["expert_disagreement"] = False
            row["expert_disagreement_step"] = None
            row["expert_disagreement_max_dev"] = 0.0
            row["expert_disagreement_reason"] = ""
            row["expert_disagreement_model_end_progress"] = 0.0
            row["expert_disagreement_expert_end_progress"] = 0.0
            row["expert_disagreement_model_end_speed"] = 0.0
            row["expert_disagreement_expert_end_speed"] = 0.0
            # Paper-faithful Conflict: compare the model's OPEN-LOOP proposal against the
            # logged expert future, both re-expressed as route ARC-LENGTH progress, expert
            # shifted +1 step (see detect_expert_disagreement_projected). No silent
            # fallback: the mining path must always supply these inputs.
            if (
                model_pred_world is None
                or expert_future_world is None
                or expert_future_speed is None
                or ref_polyline_world is None
            ):
                raise ValueError(
                    "expert_disagreement is enabled but the open-loop proposal / expert "
                    "future / route inputs were not provided (model_pred_world, "
                    "expert_future_world, expert_future_speed, ref_polyline_world). The "
                    "mining path must pass them."
                )
            model_xy = np.asarray(model_pred_world, dtype=np.float64).reshape(-1, 2)
            expert_xy = np.asarray(expert_future_world, dtype=np.float64).reshape(-1, 2)
            expert_speed = np.asarray(expert_future_speed, dtype=np.float32).reshape(-1)
            ref_xy = np.asarray(ref_polyline_world, dtype=np.float64).reshape(-1, 2)
            if ref_xy.shape[0] < 2:
                raise ValueError("ref_polyline_world must have >= 2 points for projection")
            # Cumulative arc length of the reference (recorded expert) polyline — same
            # formula _heatmap_common.build_route_polyline uses.
            seg = np.diff(ref_xy, axis=0)
            seg_len = np.sqrt((seg * seg).sum(axis=1))
            ref_s = np.concatenate([[0.0], np.cumsum(seg_len)])
            # Project model + expert points onto the route -> arc-length progress.
            model_progress = project_points_to_polyline(model_xy, ref_xy, ref_s)[:, 0].astype(
                np.float32
            )
            expert_progress = project_points_to_polyline(expert_xy, ref_xy, ref_s)[:, 0].astype(
                np.float32
            )
            # Diffusion model has no speed channel -> finite-diff its world xy. Leading
            # zero keeps speed[t] aligned with progress[t] (interval ENDING at t), matching
            # the _progress_and_speed convention.
            model_step = np.linalg.norm(np.diff(model_xy, axis=0), axis=1) / DT
            model_speed = np.concatenate([[0.0], model_step]).astype(np.float32)
            result = detect_expert_disagreement_projected(
                model_progress,
                model_speed,
                expert_progress,
                expert_speed,
                wait_speed_mps=float(thresholds["expert_disagreement_wait_speed_mps"]),
                wait_progress_m=float(thresholds["expert_disagreement_wait_progress_m"]),
                forward_progress_gap_m=float(
                    thresholds["expert_disagreement_forward_progress_gap_m"]
                ),
                lag_progress_gap_m=float(thresholds["expert_disagreement_lag_progress_gap_m"]),
                moving_speed_mps=float(thresholds["expert_disagreement_moving_speed_mps"]),
            )
            row["expert_disagreement_max_dev"] = float(result.max_deviation)
            row["expert_disagreement_reason"] = result.reason
            row["expert_disagreement_model_end_progress"] = result.model_end_progress
            row["expert_disagreement_expert_end_progress"] = result.expert_end_progress
            row["expert_disagreement_model_end_speed"] = result.model_end_speed
            row["expert_disagreement_expert_end_speed"] = result.expert_end_speed
            if result.expert_disagreement:
                row["expert_disagreement"] = True
                row["expert_disagreement_step"] = step
                labels.append("expert_disagreement")

            # Realized-lag branch (see REALIZED_LAG_SUSTAIN_STEPS). The rollout loop
            # tracks the streak per segment and passes it in; the thresholds live on
            # this scorer (expert_lag_thresholds attribute) so both sides use one
            # source. Reason string matches the frozen 3-branch port so the repair
            # side's depart-morph gate fires without changes.
            row["expert_disagreement_realized_lag"] = False
            if realized_lag_gap_m is not None:
                row["expert_disagreement_realized_gap_m"] = float(realized_lag_gap_m)
            if not row["expert_disagreement"] and realized_lag_streak >= REALIZED_LAG_SUSTAIN_STEPS:
                row["expert_disagreement"] = True
                row["expert_disagreement_step"] = step
                row["expert_disagreement_reason"] = "model_lagging_expert"
                row["expert_disagreement_realized_lag"] = True
                labels.append("expert_disagreement")

        row["labels"] = labels or ["clean"]
        row["label"] = labels[0] if labels else "clean"
        return row

    # Single source for the realized-lag thresholds: the rollout loop reads these to
    # maintain the per-segment streak it passes back into the scorer above. None when
    # expert_disagreement is not an allowed label (the loop then skips the tracking).
    _scorer.expert_lag_thresholds = (
        {
            "lag_progress_gap_m": float(thresholds["expert_disagreement_lag_progress_gap_m"]),
            "moving_speed_mps": float(thresholds["expert_disagreement_moving_speed_mps"]),
        }
        if "expert_disagreement" in allowed
        else None
    )
    return _scorer


def build_realized_reward_scorer(
    *,
    reward_config: Path,
    device: str,
    horizon: int,
    sample_step: int = 10,
):
    """Realized closed-loop reward, computed natively during a mining rollout.

    Returned as ``(hook, finalize)``:
      - ``hook(built, preds, data, _device)`` is installed as the rollout's per-step
        ``danger_scorer`` (free in the mining path). It only RECORDS — each segment's
        realized world pose every step, and (every ``sample_step`` steps) the scene
        scoring tensors. It returns all-clean rows so it never perturbs event mining.
      - ``finalize()`` scores the REALIZED trajectory for the segments buffered SINCE
        the last call: at each sampled pose ``t`` it expresses the realized future
        ``poses[t+1 : t+1+H]`` in the ego frame at ``t`` and rewards it against the
        context captured at ``t``; it then ACCUMULATES into a persistent running total
        and CLEARS the buffers. Returns the running ``(mean_reward, n_poses)``.

    ``run_segments_batched`` is invoked once per mining batch, so the caller must call
    ``finalize()`` after each batch: this bounds memory to one batch of contexts and —
    because ``_SegState`` objects (and their ``id``) are freed and reused between
    batches — prevents cross-batch ``id(s)`` aliasing from merging poses of different
    chunks. Windows that cross an unstick teleport (detected via ``s.snap_count``) are
    skipped so the discontinuous jump is not baked into a "realized future".

    Reward is per-pose (the reward function shapes an H-step trajectory), so the
    realized DRIVE is rewarded — not the model's per-step plan. No disk save/reload,
    no second simulation.
    """
    from rlvr.reward import compute_reward_batch  # local: avoid module-load cycle
    from scenario_generation.transforms import _rotation_matrix

    reward_cfg = load_reward_config(reward_config)
    tdev = torch.device(device)
    # Per-BATCH buffers (cleared by finalize); id(s) is unique within a single
    # run_segments_batched call, so it is a safe within-batch key.
    poses: dict[int, dict[int, np.ndarray]] = {}
    contexts: dict[int, dict[int, dict]] = {}
    # Shown neighbor world poses per step (uuid -> (wx, wy, wh)); captured EVERY step so a
    # sampled pose t can assemble the realized neighbor FUTURE over t+1..t+H. Slot order at
    # a sampled step mirrors sim_nb's nearest-first ordering, so the reconstructed future
    # aligns slot-for-slot with the captured neighbor_agents_past.
    nbr_world: dict[int, dict[int, dict]] = {}
    slot_uuids_at: dict[int, dict[int, list]] = {}
    teleport_ks: dict[int, set] = {}  # seg -> step indices where an unstick teleport fired
    last_snaps: dict[int, int] = {}
    acc = {"sum": 0.0, "n": 0}  # persistent running total across batches

    def hook(built, preds, data, _device):
        rows = []
        for item in built:
            s, np_dict = item[0], item[1]
            rest = item[2:]
            suuid = rest[2] if len(rest) > 2 else None  # slot -> uuid (sim mode)
            wbu = rest[3] if len(rest) > 3 else None  # uuid -> current shown world pose
            k = int(s.k)
            sid = id(s)
            poses.setdefault(sid, {})[k] = np.asarray(s.live_pose, dtype=np.float64).copy()
            if wbu:
                nbr_world.setdefault(sid, {})[k] = {
                    u: np.asarray(p, dtype=np.float64) for u, p in wbu.items()
                }
            snaps = int(getattr(s, "snap_count", 0))
            if snaps > last_snaps.get(sid, 0):
                teleport_ks.setdefault(sid, set()).add(k)  # a teleport landed at this step
            last_snaps[sid] = snaps
            if k % sample_step == 0:
                # keep scoring tensors on CPU; moved to GPU in one batch at finalize
                contexts.setdefault(sid, {})[k] = _np_dict_to_scoring_tensors(np_dict, device="cpu")
                if suuid is not None:
                    slot_uuids_at.setdefault(sid, {})[k] = list(suuid)
        return [{"labels": ["clean"], "label": "clean"} for _ in built]

    def _ego_future(world: np.ndarray, t: int) -> np.ndarray:
        x0, y0, yaw0 = world[t]
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        fut = world[t + 1 : t + 1 + horizon]
        dx = fut[:, 0] - x0
        dy = fut[:, 1] - y0
        xe = dx * c - dy * s
        ye = dx * s + dy * c
        dyaw = fut[:, 2] - yaw0
        return np.column_stack([xe, ye, np.cos(dyaw), np.sin(dyaw)]).astype(np.float32)

    def _neighbor_future(sid: int, k: int, ego_pose: np.ndarray, n_slots: int) -> np.ndarray | None:
        """Realized neighbor future in the ego frame at step k: for each slot's uuid, its
        SHOWN world pose at k+1..k+H (from nbr_world), transformed like sim_nb's past block
        so slots align with neighbor_agents_past. Returns (n_slots, H, 4) or None if no
        neighbor telemetry was captured (non-sim mode)."""
        slots = slot_uuids_at.get(sid, {}).get(k)
        wmap = nbr_world.get(sid)
        if slots is None or wmap is None:
            return None
        x0, y0, yaw0 = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
        rot = _rotation_matrix(yaw0)  # world delta -> ego frame (matches SimNeighborTracker.build)
        nf = np.zeros((n_slots, horizon, 4), dtype=np.float32)
        for i, u in enumerate(slots[:n_slots]):
            for j in range(1, horizon + 1):
                wp = wmap.get(k + j, {}).get(u)
                if wp is None:
                    continue  # neighbor absent that step -> zero (slot/step-invalid downstream)
                dxy = (wp[:2] - np.array([x0, y0])) @ rot.T
                lh = wp[2] - yaw0
                nf[i, j - 1] = (dxy[0], dxy[1], np.cos(lh), np.sin(lh))
        return nf

    def finalize() -> tuple[float, int]:
        for sid, kmap in poses.items():
            ks = sorted(kmap)
            kpos = {k: i for i, k in enumerate(ks)}  # step index -> contiguous row
            world = np.stack([kmap[k] for k in ks])
            tps = teleport_ks.get(sid, set())
            for k, sd in sorted(contexts.get(sid, {}).items()):
                if k not in kpos:
                    continue
                # need the H future steps present AND no teleport within (k, k+H]
                if any((k + j) not in kpos for j in range(1, horizon + 1)):
                    continue
                if any((k + j) in tps for j in range(1, horizon + 1)):
                    continue
                fut = _ego_future(world, kpos[k])
                if fut.shape[0] < horizon:
                    continue
                sd_gpu = {kk: (vv.to(tdev) if torch.is_tensor(vv) else vv) for kk, vv in sd.items()}
                # Realized neighbor future: score collision/TTC against where neighbors
                # ACTUALLY were over k+1..k+H (the shown motion), not an empty set.
                nap = sd_gpu.get("neighbor_agents_past")
                n_slots = int(nap.shape[-3]) if nap is not None and nap.dim() >= 3 else 0
                nf = _neighbor_future(sid, k, world[kpos[k]], n_slots) if n_slots else None
                if nf is not None:
                    sd_gpu["neighbor_agents_future"] = torch.from_numpy(nf[None]).to(tdev)
                ego = torch.from_numpy(fut[None]).to(tdev)
                rb = compute_reward_batch(ego, sd_gpu, reward_cfg)
                acc["sum"] += float(rb[0].total)
                acc["n"] += 1
        poses.clear()
        contexts.clear()
        nbr_world.clear()
        slot_uuids_at.clear()
        teleport_ks.clear()
        last_snaps.clear()
        return (acc["sum"] / acc["n"] if acc["n"] else float("nan")), acc["n"]

    return hook, finalize
