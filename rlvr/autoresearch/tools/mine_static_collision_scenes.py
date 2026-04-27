#!/usr/bin/env python3
"""Mine real-data NPZs for scenes where the model's prediction collides with
or approaches stationary neighbours.

For each candidate scene:

1. Load the NPZ (has real ``neighbor_agents_future`` from GT log — no
   synthesis needed, unlike the sim-replay case).
2. Quick reject if no neighbour in the past has near-zero velocity + low
   total past displacement (no stopped cars → nothing to collide with).
3. Run the model for 1 step → 80-step ego prediction.
4. Score with ``rlvr.reward.compute_static_collision_penalty`` against
   the real (not synthesised) neighbour futures. Stopped-neighbour mask
   inside the primitive uses the real v0 / displacement signal.
5. Record the predicted ``sc_min_dist`` + closest-pair points.

Scenes are ranked and bucketed into the four clearance zones:

    cross   d <  sc_cross_thresh         (default 0.2 m — "visually a collision")
    near    sc_cross_thresh <= d < sc_near_thresh   (default < 0.4 m)
    wide    sc_near_thresh  <= d < sc_wide_thresh   (default < 0.7 m)
    cont    sc_wide_thresh  <= d < sc_cont_thresh   (default < 1.0 m)

The tool picks ``--target_per_zone`` scenes from each (``10`` default ×
4 zones → 40 scenes, plus the original count fill). Total ≈ 50.

For each picked scene it emits a 2-panel audit PNG (spatial view with
ego footprint rendered AT the first-crossing timestep + offender NPC +
closest-pair line, bottom = clearance-vs-prediction-time with zone
shading) — identical layout to
``rlvr.autoresearch.tools.audit_static_collision`` (viz helper is shared).

Usage:

    source /opt/ros/humble/setup.bash && source .venv/bin/activate
    python -m rlvr.autoresearch.tools.mine_static_collision_scenes \\
        --scenes /media/.../j6_train_all.json \\
        --model_path /media/.../x2_model_base/best_model.pth \\
        --config /path/to/reward_config_static_on.json \\
        --output_dir /media/.../j6_static_collision_mining \\
        --target_per_zone 12 \\
        --max_scan 5000

The reward config JSON MUST set ``static_collision_enabled=true`` — the
tool refuses to run otherwise (no silent fallback).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.autoresearch.tools.audit_static_collision import (
    _ZONE_COLORS,
    _draw_audit_figure,
    _zone_for,
)
from rlvr.reward import (
    _build_ego_bbox_corners,
    compute_static_collision_penalty,
)


def _has_stopped_neighbor_past(
    nb_past: np.ndarray,
    vel_thresh: float,
    disp_thresh: float,
) -> bool:
    """Cheap pre-filter: does the scene contain at least one neighbour that
    was essentially stationary across its past?

    ``nb_past`` layout: (N_nb, T_past, 11) with
    [x, y, cos_h, sin_h, vx, vy, width, length, type(3)].
    """
    valid = np.abs(nb_past[:, -1, :2]).sum(axis=-1) > 1e-6
    if not valid.any():
        return False
    speeds = np.linalg.norm(nb_past[valid, -1, 4:6], axis=-1)
    # Total past displacement: first-valid slot minus last-valid slot,
    # modulo the valid mask per neighbour (we check if any neighbour was
    # within disp_thresh across its valid past).
    disp = np.linalg.norm(nb_past[valid, -1, :2] - nb_past[valid, 0, :2], axis=-1)
    stopped = (speeds < vel_thresh) & (disp < disp_thresh)
    return bool(stopped.any())


@torch.no_grad()
def _score_scene(
    data_np: dict[str, np.ndarray],
    ego_pred: np.ndarray,
    reward_cfg,
    device: str,
) -> dict | None:
    """Score one scene's ego prediction against real stopped neighbours.

    Mirrors ``audit_static_collision._score_prediction`` but uses the
    NPZ's REAL ``neighbor_agents_future`` (GT logged motion) — no
    broadcast. Moving neighbours are automatically filtered out by the
    primitive's stopped mask (|v0|<thresh AND disp<thresh).

    Returns None when no stopped neighbour survives the primitive's mask
    (e.g. the candidate neighbours were actually creeping / parked cars
    that in fact move during the 8 s future).
    """
    ego_traj = torch.from_numpy(ego_pred.astype(np.float32)).to(device).unsqueeze(0)
    T = ego_traj.shape[1]

    es = np.asarray(data_np["ego_shape"])
    if es.ndim == 2:
        es = es[0]
    ego_shape = torch.from_numpy(es[:3].astype(np.float32)).to(device)

    # Real neighbour future from the NPZ. (N_nb, 80, 3) [x, y, yaw].
    nb_fut_raw = np.asarray(data_np["neighbor_agents_future"])
    if nb_fut_raw.ndim == 4:
        nb_fut_raw = nb_fut_raw[0]
    if nb_fut_raw.shape[1] < T:
        return None
    nb_fut_raw = nb_fut_raw[:, :T, :]
    # Convert to (N_nb, T, 4) [x, y, cos, sin] for the primitive.
    yaw = nb_fut_raw[..., 2:3]
    nb_fut_4 = np.concatenate(
        [nb_fut_raw[..., :2], np.cos(yaw), np.sin(yaw)], axis=-1
    ).astype(np.float32)

    neighbor_futures = torch.from_numpy(nb_fut_4).to(device)
    slot_valid = neighbor_futures[:, :, :2].abs().sum(dim=(1, 2)) > 1e-6
    if not slot_valid.any():
        return None
    neighbor_futures = neighbor_futures[slot_valid]
    neighbor_valid = neighbor_futures[:, :, :2].abs().sum(dim=-1) > 1e-6

    nb_past = np.asarray(data_np["neighbor_agents_past"])
    if nb_past.ndim == 4:
        nb_past = nb_past[0]
    nb_past_valid = nb_past[slot_valid.cpu().numpy()]
    shapes_np = nb_past_valid[:, -1, [6, 7]].astype(np.float32)
    neighbor_shapes = torch.from_numpy(shapes_np).to(device)
    zero = neighbor_shapes.abs().sum(dim=-1) < 1e-3
    if zero.any():
        neighbor_shapes[zero] = torch.tensor([2.0, 4.5], device=device)

    sc = compute_static_collision_penalty(
        ego_traj, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid,
        reward_cfg,
    )

    if int(sc["stopped_mask"].sum().item()) == 0:
        return None

    per_ts_min = sc["per_timestep_min"][0].cpu().numpy()
    ego_pts = sc["ego_closest_pt"][0].cpu().numpy()
    npc_pts = sc["npc_closest_pt"][0].cpu().numpy()
    argmin_nb = sc["argmin_neighbor"][0].cpu().numpy()
    first_cross = sc["first_crossing_steps"][0]

    if T > 1:
        argmin_t = int(np.argmin(per_ts_min[1:])) + 1
        sc_min_dist = float(per_ts_min[argmin_t])
    else:
        argmin_t = 0
        sc_min_dist = float(per_ts_min[0])

    if first_cross is not None:
        viz_t = int(first_cross)
        viz_d = float(per_ts_min[viz_t])
    else:
        viz_t = argmin_t
        viz_d = sc_min_dist

    return {
        "sc_min_dist": sc_min_dist,
        "static_crossing": bool(sc["crossing_gate"][0].item() < 0.5),
        "sc_n_stopped": int(sc["stopped_mask"].sum().item()),
        "first_crossing_step": first_cross,
        "argmin_t": argmin_t,
        "viz_t": viz_t,
        "viz_d": viz_d,
        "viz_neighbor_idx": int(argmin_nb[viz_t]),
        "ego_closest_pt": ego_pts[viz_t].tolist(),
        "npc_closest_pt": npc_pts[viz_t].tolist(),
        "per_ts_min": per_ts_min.tolist(),
        "_arrays": {
            "ego_pts": ego_pts,
            "npc_pts": npc_pts,
            "argmin_nb": argmin_nb,
            "per_ts_min": per_ts_min,
            "ego_traj": ego_pred,
            "ego_corners": _build_ego_bbox_corners(
                ego_traj, ego_shape,
            )[0].cpu().numpy(),
            "nb_fut_4": nb_fut_4,
            "neighbor_shapes_all": nb_past[:, -1, [6, 7]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True,
                        help="JSON list of NPZ paths.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Reward config JSON with "
                             "static_collision_enabled=true.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_per_zone", type=int, default=12,
                        help="Aim for N scenes per zone (cross/near/wide/"
                             "cont). Default 12 → ~48 total.")
    parser.add_argument("--max_scan", type=int, default=5000,
                        help="Stop scanning after this many scenes even "
                             "if zone targets aren't met. Default 5000.")
    parser.add_argument("--shuffle_seed", type=int, default=42,
                        help="Shuffle the scene list (with this seed) "
                             "before scanning so we don't bias toward "
                             "the head of the list.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vel_thresh", type=float, default=0.1,
                        help="Past-velocity threshold for the cheap "
                             "stopped-neighbour pre-filter. Keep aligned "
                             "with sc_neighbor_vel_thresh in the reward "
                             "config.")
    parser.add_argument("--disp_thresh", type=float, default=0.5,
                        help="Past-displacement threshold for the "
                             "cheap pre-filter.")
    parser.add_argument("--ego_min_speed", type=float, default=0.1,
                        help="Reject scenes where the ego's current "
                             "speed is below this (m/s). Default 0.1 "
                             "— a stationary ego at a red light or in "
                             "a jam is not an interesting collision "
                             "candidate because there's no actual "
                             "motion risk.")
    args = parser.parse_args()

    reward_cfg = load_reward_config(args.config)
    if not getattr(reward_cfg, "static_collision_enabled", False):
        raise SystemExit(
            f"{args.config} must set static_collision_enabled=true"
        )

    cross_t = float(reward_cfg.sc_cross_thresh)
    near_t = float(reward_cfg.sc_near_thresh)
    wide_t = float(reward_cfg.sc_wide_thresh)
    cont_t = float(reward_cfg.sc_cont_thresh)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scenes = list(json.load(f))
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(scenes)
    scenes = scenes[: args.max_scan]

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # Lazy model load — avoids pulling ROS-touching modules until needed.
    from scenario_generation.npz_loader import from_npz
    from scenario_generation.simulate import _predict_batch, load_model

    print(f"Loading model {args.model_path}")
    model, model_args = load_model(str(args.model_path), device)

    zone_hits: dict[str, list[dict]] = {"cross": [], "near": [], "wide": [], "cont": []}
    n_scanned = 0
    n_ego_stopped_rejected = 0
    n_with_stopped = 0
    n_inferred = 0

    for path_str in scenes:
        path = Path(path_str)
        n_scanned += 1

        try:
            with np.load(path, allow_pickle=True) as raw:
                if "neighbor_agents_past" not in raw.files:
                    continue
                nb_past = np.asarray(raw["neighbor_agents_past"])
                if nb_past.ndim == 4:
                    nb_past = nb_past[0]
                # Reject scenes where the ego itself is stationary — a
                # stopped ego at a red light or in a jam isn't an
                # interesting collision candidate because the ego isn't
                # actually going to hit anything (the "prediction hits
                # car ahead" is academic). Use ego_current_state[4] (=
                # speed magnitude in ego frame, canonicalised, from
                # tensor_converter line 115).
                ego_speed = 0.0
                if "ego_current_state" in raw.files:
                    ecs = np.asarray(raw["ego_current_state"])
                    if ecs.ndim == 2:
                        ecs = ecs[0]
                    if ecs.size >= 5:
                        ego_speed = float(abs(ecs[4]))
                if ego_speed < args.ego_min_speed:
                    n_ego_stopped_rejected += 1
                    continue
        except Exception:
            continue

        # Cheap pre-filter: at least one stopped neighbor.
        if not _has_stopped_neighbor_past(nb_past, args.vel_thresh, args.disp_thresh):
            continue
        n_with_stopped += 1

        try:
            with np.load(path, allow_pickle=True) as raw:
                data_np = {k: raw[k] for k in raw.files if k != "version"}
        except Exception:
            continue

        try:
            scene = from_npz(str(path))
            preds = _predict_batch(
                model, model_args, scene, [scene.ego_agent_id], device,
            )
            ego_pred = preds.get(scene.ego_agent_id)
        except Exception:
            continue
        if ego_pred is None:
            continue
        n_inferred += 1

        result = _score_scene(data_np, ego_pred, reward_cfg, device)
        if result is None:
            continue

        d = result["sc_min_dist"]
        if d < cross_t:
            zone = "cross"
        elif d < near_t:
            zone = "near"
        elif d < wide_t:
            zone = "wide"
        elif d < cont_t:
            zone = "cont"
        else:
            continue  # safe — not interesting

        if len(zone_hits[zone]) >= args.target_per_zone:
            # Already enough of this zone; only accept if it's more severe
            # than the current worst in that bucket.
            worst_in_zone = max(r["sc_min_dist"] for r in zone_hits[zone])
            if d >= worst_in_zone:
                continue
            # Replace the least-severe entry.
            worst_idx = max(range(len(zone_hits[zone])),
                            key=lambda i: zone_hits[zone][i]["sc_min_dist"])
            del zone_hits[zone][worst_idx]

        result["npz_path"] = str(path)
        result["zone"] = zone
        zone_hits[zone].append(result)

        if n_inferred % 50 == 0:
            counts = {z: len(zone_hits[z]) for z in zone_hits}
            print(f"  scan {n_scanned}/{len(scenes)}  "
                  f"ego_stopped_rej={n_ego_stopped_rejected}  "
                  f"stopped={n_with_stopped}  "
                  f"inferred={n_inferred}  zones={counts}")

        # Stop early if all zones are filled.
        if all(len(zone_hits[z]) >= args.target_per_zone for z in zone_hits):
            print(f"  All zones filled at scan={n_scanned}")
            break

    print()
    print("Summary:")
    for z in ("cross", "near", "wide", "cont"):
        print(f"  {z}: {len(zone_hits[z])} scenes")

    # Draw + dump summary.
    summary: list[dict] = []
    for zone in ("cross", "near", "wide", "cont"):
        entries = sorted(zone_hits[zone], key=lambda r: r["sc_min_dist"])
        for rank, r in enumerate(entries):
            sc_min = r["sc_min_dist"]
            viz_t = r["viz_t"]
            viz_d = r["viz_d"]
            nb_idx = r["viz_neighbor_idx"]
            path = Path(r["npz_path"])
            tag = f"{zone}_{rank:02d}_{path.stem}"
            png_path = args.output_dir / f"{tag}.png"

            # Re-load the NPZ for draw (we kept _arrays but need line_strings
            # + ego_shape from the NPZ dict; the _arrays dict intentionally
            # doesn't include heavy map tensors).
            with np.load(path, allow_pickle=True) as raw:
                data_np = {k: raw[k] for k in raw.files if k != "version"}

            _draw_audit_figure(
                data_np, step=rank, arrays=r["_arrays"],
                sc_min_dist=viz_d, argmin_t=viz_t,
                argmin_nb_global=nb_idx,
                output_path=png_path,
                viz_threshold_m=2.0,
            )

            row = {k: v for k, v in r.items() if not k.startswith("_")}
            row["png"] = str(png_path)
            row["tag"] = tag
            summary.append(row)

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "scenes_list": str(args.scenes),
            "model_path": str(args.model_path),
            "reward_config_path": str(args.config),
            "zone_thresholds": {
                "cross": cross_t, "near": near_t,
                "wide": wide_t, "cont": cont_t,
            },
            "n_scanned": n_scanned,
            "n_ego_stopped_rejected": n_ego_stopped_rejected,
            "n_with_stopped": n_with_stopped,
            "n_inferred": n_inferred,
            "ego_min_speed": float(args.ego_min_speed),
            "n_selected": len(summary),
            "zone_counts": {z: len(zone_hits[z]) for z in zone_hits},
            "scenes": summary,
        }, f)
    print(f"Wrote {summary_path} ({len(summary)} selected scenes, "
          f"{n_inferred} inferences across {n_scanned} scans)")
    # Scene list for downstream re-use.
    list_path = args.output_dir / "scene_list.json"
    with open(list_path, "w") as f:
        json.dump([r["npz_path"] for r in summary], f)
    print(f"Wrote {list_path}")


if __name__ == "__main__":
    main()
