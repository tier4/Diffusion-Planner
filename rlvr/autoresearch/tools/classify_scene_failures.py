#!/usr/bin/env python3
"""Classify NPZ scenes by reward/safety failure modes and write training lists.

This is a geometry/reward based classifier, not the explorer-policy classifier.
It scores either the GT future already stored in each NPZ or a deterministic
model prediction, then emits:

* ``classified_scenes.jsonl``: one rich diagnostics row per scene.
* ``summary.json``: label counts and thresholds used.
* ``lists/*.json``: plain NPZ path lists per label, plus ``all_flagged.json``
  and ``clean.json``. These lists are directly usable by RSFT / IL loaders.

Usage:
    python -m rlvr.autoresearch.tools.classify_scene_failures \\
        --scenes scenes.json --config reward_config.json \\
        --output_dir /tmp/scene_flags --trajectory gt

    python -m rlvr.autoresearch.tools.classify_scene_failures \\
        --scenes scenes.json --config reward_config.json \\
        --output_dir /tmp/scene_flags_det --trajectory det \\
        --model_path /path/to/best_model.pth --batch_size 32
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from planner_metrics.subscores import (
    compute_ego_neighbor_signed_clearance,
    compute_safety_score_batch,
)
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import RewardConfig, compute_subscores_batch

_ALWAYS_WRITE_LISTS = (
    "all_flagged",
    "clean",
    "road_border_crossing",
    "road_border_near",
    "lane_crossing",
    "static_collision",
    "static_near_miss",
    "moving_collision",
    "moving_near_miss",
    "moving_ttc",
)


def _load_scene_paths(path: str | Path) -> list[str]:
    with open(path) as f:
        paths = json.load(f)
    if not isinstance(paths, list):
        raise ValueError(f"{path} must contain a JSON list of NPZ paths")
    return [str(p) for p in paths]


def _future_heading_to_cos_sin(fut: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., T, 3)`` x/y/yaw futures to ``(..., T, 4)`` x/y/cos/sin."""
    if fut.shape[-1] >= 4:
        return fut
    if fut.shape[-1] != 3:
        raise ValueError(f"future tensor last dim must be 3 or >=4, got {tuple(fut.shape)}")
    return torch.cat([fut[..., :2], torch.cos(fut[..., 2:3]), torch.sin(fut[..., 2:3])], dim=-1)


def _prepare_scoring_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone data and normalize future tensors for reward/scoring code.

    Scene-editor exports may store futures as x/y/yaw. The reward geometry
    expects x/y/cos/sin, so convert in memory. This leaves source NPZs intact.
    """
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    if "ego_agent_future" in out:
        out["ego_agent_future"] = _future_heading_to_cos_sin(out["ego_agent_future"])
    if "neighbor_agents_future" in out:
        out["neighbor_agents_future"] = _future_heading_to_cos_sin(out["neighbor_agents_future"])
    return out


def _ego_shape_from_data(data: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if "ego_shape" not in data:
        raise ValueError("scene is missing required ego_shape")
    ego_shape = data["ego_shape"]
    if ego_shape.dim() == 2:
        ego_shape = ego_shape[0]
    if ego_shape.numel() < 3:
        raise ValueError(f"ego_shape has shape {tuple(ego_shape.shape)}, expected >=3 values")
    return ego_shape[:3].to(device)


def _gt_trajectory(data: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if "ego_agent_future" not in data:
        raise ValueError("scene is missing ego_agent_future; cannot use --trajectory gt")
    traj = data["ego_agent_future"]
    if traj.dim() == 3:
        traj = traj[0]
    traj = _future_heading_to_cos_sin(traj)
    return traj[:, :4].to(device).unsqueeze(0)


def _neighbor_inputs(
    data: dict[str, torch.Tensor],
    T: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neighbor_futures = torch.zeros(0, T, 4, device=device)
    neighbor_shapes = torch.zeros(0, 2, device=device)
    neighbor_valid = torch.zeros(0, T, dtype=torch.bool, device=device)

    if "neighbor_agents_future" not in data:
        return neighbor_futures, neighbor_shapes, neighbor_valid

    nf = data["neighbor_agents_future"]
    if nf.dim() == 4:
        nf = nf[0]
    if nf.shape[0] == 0:
        return neighbor_futures, neighbor_shapes, neighbor_valid
    if nf.shape[1] < T or (nf.shape[2] != 3 and nf.shape[2] < 4):
        raise ValueError(
            "neighbor_agents_future must be shaped (N,T,3|4) with x,y,yaw or x,y,cos,sin; "
            f"got {tuple(nf.shape)} for trajectory length {T}"
        )

    nf_data = _future_heading_to_cos_sin(nf[:, :T])
    slot_valid = nf_data[:, :, :2].abs().sum(dim=(1, 2)) > 1e-6
    if not slot_valid.any():
        return neighbor_futures, neighbor_shapes, neighbor_valid

    neighbor_futures = nf_data[slot_valid].to(device)
    neighbor_valid = neighbor_futures[:, :, :2].abs().sum(dim=-1) > 1e-6

    if "neighbor_agents_past" in data:
        nap = data["neighbor_agents_past"]
        if nap.dim() == 4:
            nap = nap[0]
        ns = nap[slot_valid, -1, :]
        if ns.shape[-1] >= 8:
            neighbor_shapes = ns[:, [6, 7]].to(device)  # width, length
        else:
            neighbor_shapes = torch.full((neighbor_futures.shape[0], 2), 2.0, device=device)
    else:
        neighbor_shapes = torch.full((neighbor_futures.shape[0], 2), 2.0, device=device)

    zero_shapes = neighbor_shapes.abs().sum(dim=-1) < 1e-3
    if zero_shapes.any():
        neighbor_shapes = neighbor_shapes.clone()
        neighbor_shapes[zero_shapes] = torch.tensor([2.0, 4.5], device=device)

    return neighbor_futures, neighbor_shapes, neighbor_valid


def _stopped_neighbor_mask(
    neighbor_futures: torch.Tensor,
    neighbor_valid: torch.Tensor,
    config: RewardConfig,
) -> torch.Tensor:
    N_nb = neighbor_futures.shape[0]
    device = neighbor_futures.device
    if N_nb == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    nb_xy = neighbor_futures[:, :, :2]
    if nb_xy.shape[1] < 2:
        return torch.zeros(N_nb, dtype=torch.bool, device=device)

    both_valid_01 = neighbor_valid[:, 0] & neighbor_valid[:, 1]
    v0 = torch.zeros(N_nb, device=device)
    if both_valid_01.any():
        v0[both_valid_01] = (nb_xy[both_valid_01, 1] - nb_xy[both_valid_01, 0]).norm(
            dim=-1
        ) / config.dt

    disp_all = (nb_xy - nb_xy[:, 0:1]).norm(dim=-1)
    max_disp = disp_all.masked_fill(~neighbor_valid, 0.0).max(dim=1).values
    has_any_valid = neighbor_valid.any(dim=1)

    return (
        has_any_valid
        & both_valid_01
        & (v0 < config.sc_neighbor_vel_thresh)
        & (max_disp < config.sc_neighbor_disp_thresh)
    )


def _first_step(steps: list[int | None]) -> int | None:
    values = [s for s in steps if s is not None]
    return min(values) if values else None


def _moving_diagnostics(
    ego_traj: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
    moving_near_thresh: float,
    device: torch.device,
) -> dict[str, Any]:
    T = ego_traj.shape[1]
    ego_shape = _ego_shape_from_data(data, device)
    neighbor_futures, neighbor_shapes, neighbor_valid = _neighbor_inputs(data, T, device)
    stopped_mask = _stopped_neighbor_mask(neighbor_futures, neighbor_valid, config)
    moving_mask = ~stopped_mask
    moving_count = int(moving_mask.sum().item())

    empty = {
        "moving_neighbor_count": moving_count,
        "stopped_neighbor_count": int(stopped_mask.sum().item()),
        "moving_min_dist": 99.0,
        "moving_argmin_neighbor": None,
        "moving_argmin_t": None,
        "moving_collision_step": None,
        "moving_near_miss": False,
    }
    if moving_count == 0:
        return empty

    mf = neighbor_futures[moving_mask]
    ms = neighbor_shapes[moving_mask]
    mv = neighbor_valid[moving_mask]
    moving_global_idx = moving_mask.nonzero(as_tuple=True)[0]

    distances = compute_ego_neighbor_signed_clearance(ego_traj, ego_shape, mf, ms, mv)
    flat_idx = int(distances.reshape(-1).argmin().item())
    _, M, T = distances.shape
    argmin_m = (flat_idx // T) % M
    argmin_t = flat_idx % T
    min_dist = float(distances.reshape(-1)[flat_idx].item())

    _, moving_collision_steps = compute_safety_score_batch(
        ego_traj,
        ego_shape,
        mf,
        ms,
        mv,
        config,
    )
    moving_collision_step = _first_step(moving_collision_steps)

    return {
        "moving_neighbor_count": moving_count,
        "stopped_neighbor_count": int(stopped_mask.sum().item()),
        "moving_min_dist": min_dist,
        "moving_argmin_neighbor": int(moving_global_idx[argmin_m].item()),
        "moving_argmin_t": int(argmin_t),
        "moving_collision_step": moving_collision_step,
        "moving_near_miss": moving_collision_step is None and min_dist < moving_near_thresh,
    }


def classify_loaded_scene(
    scene_path: str,
    ego_traj: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
    *,
    moving_near_thresh: float,
    static_near_thresh: float,
    rb_near_thresh: float,
    device: torch.device,
) -> dict[str, Any]:
    data = _prepare_scoring_data(data)
    subs = compute_subscores_batch(ego_traj, data, config)
    moving = _moving_diagnostics(ego_traj, data, config, moving_near_thresh, device)

    labels: list[str] = []
    rb_crossing = bool(subs["rb_crossing_gate"][0].item() < 0.5)
    rb_min_dist = float(subs["rb_min_dist"][0].item())
    if rb_crossing:
        labels.append("road_border_crossing")
    elif rb_min_dist < rb_near_thresh:
        labels.append("road_border_near")

    lane_crossing = bool(subs["lane_crossing_gate"][0].item() < 0.5)
    if lane_crossing:
        labels.append("lane_crossing")

    sc_crossing = bool(subs["sc_crossing_gate"][0].item() < 0.5)
    sc_min_dist = float(subs["sc_min_dist"][0].item())
    sc_n_stopped = int(subs["sc_n_stopped"][0].item())
    if sc_crossing:
        labels.append("static_collision")
    elif sc_n_stopped > 0 and sc_min_dist < static_near_thresh:
        labels.append("static_near_miss")

    if moving["moving_collision_step"] is not None:
        labels.append("moving_collision")
    elif moving["moving_near_miss"]:
        labels.append("moving_near_miss")

    ttc_first_unsafe = subs["ttc_first_unsafe_steps"][0]
    if ttc_first_unsafe is not None and moving["moving_collision_step"] is None:
        labels.append("moving_ttc")

    if not labels:
        labels.append("clean")

    return {
        "scene_path": scene_path,
        "labels": labels,
        "trajectory_source": None,  # filled by caller
        "rb_min_dist": rb_min_dist,
        "rb_crossing": rb_crossing,
        "rb_crossing_step": subs["rb_crossing_steps"][0],
        "lane_crossing": lane_crossing,
        "lane_crossing_step": subs["lane_crossing_steps"][0],
        "static_crossing": sc_crossing,
        "sc_min_dist": sc_min_dist,
        "sc_crossing_step": subs["sc_crossing_steps"][0],
        "sc_n_stopped": sc_n_stopped,
        "collision_step": subs["collision_step"][0],
        "ttc_score": float(subs["ttc"][0].item()),
        "ttc_first_unsafe_step": ttc_first_unsafe,
        "ttc_first_collision_step": subs["ttc_first_collision_steps"][0],
        **moving,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_outputs(
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    output_dir: Path,
    thresholds: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "classified_scenes.jsonl"
    with open(jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    by_label: dict[str, list[str]] = defaultdict(list)
    all_flagged: list[str] = []
    clean: list[str] = []
    for row in rows:
        path = row["scene_path"]
        labels = row["labels"]
        for label in labels:
            by_label[label].append(path)
        if labels == ["clean"]:
            clean.append(path)
        else:
            all_flagged.append(path)

    by_label["all_flagged"] = all_flagged
    by_label["clean"] = clean
    lists_dir = output_dir / "lists"
    for label in sorted(set(_ALWAYS_WRITE_LISTS) | set(by_label)):
        _write_json(lists_dir / f"{label}.json", by_label.get(label, []))

    counts = Counter(label for row in rows for label in row["labels"])
    summary = {
        "n_input": len(rows) + len(errors),
        "n_classified": len(rows),
        "n_errors": len(errors),
        "label_counts": dict(sorted(counts.items())),
        "thresholds": thresholds,
        "outputs": {
            "classified_scenes_jsonl": str(jsonl),
            "lists_dir": str(lists_dir),
        },
        "errors": errors,
    }
    _write_json(output_dir / "summary.json", summary)


def _classify_gt(
    scene_paths: list[str],
    config: RewardConfig,
    args,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for idx, scene_path in enumerate(scene_paths):
        try:
            data = _prepare_scoring_data(load_npz_data(scene_path, device))
            ego_traj = _gt_trajectory(data, device)
            row = classify_loaded_scene(
                scene_path,
                ego_traj,
                data,
                config,
                moving_near_thresh=args.moving_near_thresh,
                static_near_thresh=args.static_near_thresh,
                rb_near_thresh=args.rb_near_thresh,
                device=device,
            )
            row["trajectory_source"] = "gt"
            rows.append(row)
            print(f"  [{idx:4d}] {Path(scene_path).name}: {','.join(row['labels'])}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"scene_path": scene_path, "error": str(exc)})
            print(f"  [{idx:4d}] ERROR {Path(scene_path).name}: {exc}")
    return rows, errors


@torch.no_grad()
def _classify_det(
    scene_paths: list[str],
    config: RewardConfig,
    args,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not args.model_path:
        raise ValueError("--trajectory det requires --model_path")
    model, model_args = load_model(args.model_path, device)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for start in range(0, len(scene_paths), args.batch_size):
        batch_paths = scene_paths[start : start + args.batch_size]
        datas: list[dict[str, torch.Tensor]] = []
        valid_paths: list[str] = []
        for scene_path in batch_paths:
            try:
                datas.append(load_npz_data(scene_path, device))
                valid_paths.append(scene_path)
            except Exception as exc:  # noqa: BLE001
                errors.append({"scene_path": scene_path, "error": str(exc)})
                print(f"  [err ] {Path(scene_path).name}: {exc}")
        if not datas:
            continue
        det_trajs = det_inference_batched(model, model_args, datas, device)
        for bi, scene_path in enumerate(valid_paths):
            try:
                row = classify_loaded_scene(
                    scene_path,
                    det_trajs[bi : bi + 1],
                    _prepare_scoring_data(datas[bi]),
                    config,
                    moving_near_thresh=args.moving_near_thresh,
                    static_near_thresh=args.static_near_thresh,
                    rb_near_thresh=args.rb_near_thresh,
                    device=device,
                )
                row["trajectory_source"] = "det"
                rows.append(row)
                print(f"  [{start + bi:4d}] {Path(scene_path).name}: {','.join(row['labels'])}")
            except Exception as exc:  # noqa: BLE001
                errors.append({"scene_path": scene_path, "error": str(exc)})
                print(f"  [{start + bi:4d}] ERROR {Path(scene_path).name}: {exc}")
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ scene paths")
    parser.add_argument("--config", required=True, help="Reward config JSON")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trajectory", choices=("gt", "det"), default="gt")
    parser.add_argument("--model_path", default=None, help="Required when --trajectory det")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--moving_near_thresh", type=float, default=1.0)
    parser.add_argument("--static_near_thresh", type=float, default=None)
    parser.add_argument("--rb_near_thresh", type=float, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_reward_config(args.config)
    if args.static_near_thresh is None:
        args.static_near_thresh = float(config.sc_near_thresh)
    if args.rb_near_thresh is None:
        args.rb_near_thresh = float(config.rb_near_thresh)

    scene_paths = _load_scene_paths(args.scenes)
    if args.max_scenes is not None:
        scene_paths = scene_paths[: args.max_scenes]

    if args.trajectory == "gt":
        rows, errors = _classify_gt(scene_paths, config, args, device)
    else:
        rows, errors = _classify_det(scene_paths, config, args, device)

    thresholds = {
        "moving_near_thresh": float(args.moving_near_thresh),
        "static_near_thresh": float(args.static_near_thresh),
        "rb_near_thresh": float(args.rb_near_thresh),
        "sc_cross_thresh": float(config.sc_cross_thresh),
        "rb_cross_thresh": float(config.rb_cross_thresh),
    }
    _write_outputs(rows, errors, Path(args.output_dir), thresholds)
    print(f"\nClassified {len(rows)}/{len(scene_paths)} scenes; errors={len(errors)}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
