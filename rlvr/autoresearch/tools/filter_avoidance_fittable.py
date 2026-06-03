#!/usr/bin/env python3
"""Filter candidate avoidance scenes by K=N fittability under a model.

For each candidate scene, generates K trajectories under the given model
(using a collision-aware generation variant), scores each with
``compute_reward_batch``, and keeps the scene only if at least one
generation passes ALL gates AND achieves >= the clearance threshold.

Class A (stopped): sc_min_dist >= threshold, static_crossing=False
Class B (moving): collision_step is None AND dynamic clearance >= threshold

Emits ``kept_scenes.json`` (with best-of-K trajectory saved per scene),
``dropped_scenes.json``, and a summary report.

Usage:
    python -m rlvr.autoresearch.tools.filter_avoidance_fittable \
        --candidates <candidate_scenes_all.json> \
        --model_path <champion_model.pth> \
        --config <reward_sc.json> \
        --output_dir <dir> \
        --clearance_thresh 0.5 \
        --variant rsft_v2_col4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import copy

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
)
from rlvr.reward import compute_reward_batch, compute_static_collision_penalty


def _compute_moving_clearance(
    ego_traj: torch.Tensor,  # (1, T, 4)
    data_np: dict,
    neighbor_slot: int,
    rcfg,
    device: torch.device,
) -> float:
    """Compute min OBB clearance between ego trajectory and a specific moving neighbor.

    Reuses compute_static_collision_penalty with raised velocity thresholds so
    the moving neighbor is classified as 'stopped' for OBB distance purposes.
    """
    es = data_np["ego_shape"]
    if isinstance(es, torch.Tensor):
        es = es.cpu().numpy()
    es = np.asarray(es).reshape(-1)[:3]
    ego_shape = torch.from_numpy(es.astype(np.float32)).to(device)

    nb_fut = data_np["neighbor_agents_future"]
    if isinstance(nb_fut, torch.Tensor):
        nb_fut = nb_fut.cpu().numpy()
    nb_fut = np.asarray(nb_fut)
    if nb_fut.ndim == 4:
        nb_fut = nb_fut[0]

    nb_past = data_np["neighbor_agents_past"]
    if isinstance(nb_past, torch.Tensor):
        nb_past = nb_past.cpu().numpy()
    nb_past = np.asarray(nb_past)
    if nb_past.ndim == 4:
        nb_past = nb_past[0]

    T = ego_traj.shape[1]
    slot_fut = nb_fut[neighbor_slot, :T, :]  # (T, 4) x,y,cos,sin or x,y,heading
    if slot_fut.shape[-1] == 3:
        yaw = slot_fut[..., 2:3]
        slot_fut = np.concatenate([slot_fut[..., :2], np.cos(yaw), np.sin(yaw)], axis=-1)
    slot_fut = slot_fut.astype(np.float32)

    neighbor_futures = torch.from_numpy(slot_fut).unsqueeze(0).to(device)  # (1, T, 4)
    neighbor_valid = neighbor_futures[:, :, :2].abs().sum(dim=-1) > 1e-6  # (1, T)

    w = float(nb_past[neighbor_slot, -1, 6])
    l = float(nb_past[neighbor_slot, -1, 7])
    if w < 0.1 or l < 0.1:
        # No silent fallback: fittability cannot be assessed without the obstacle's
        # true size. Fail loudly so the caller skips this scene (logged) rather than
        # scoring against an invented box.
        raise ValueError(
            f"neighbor slot {neighbor_slot} has degenerate shape (w={w:.3f}, l={l:.3f}); "
            f"skipping — cannot assess avoidance fittability without true obstacle size")
    neighbor_shapes = torch.tensor([[w, l]], device=device)

    # Override velocity thresholds so the moving neighbor is treated as 'stopped'
    rcfg_mov = copy.copy(rcfg)
    rcfg_mov.sc_neighbor_vel_thresh = 1000.0
    rcfg_mov.sc_neighbor_disp_thresh = 1000.0

    sc = compute_static_collision_penalty(
        ego_traj, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid, rcfg_mov,
    )
    per_ts = sc["per_timestep_min"][0].cpu().numpy()  # (T,)
    if T > 1:
        return float(per_ts[1:].min())
    return float(per_ts[0])


def _passes_common_gates(r) -> bool:
    return (
        not r.rb_crossing
        and not r.lane_crossing
        and not r.kinematic_violated
        and r.total > -50.0  # not stopped / off-route
    )


def _passes_stopped_filter(r, thresh: float) -> bool:
    return (
        _passes_common_gates(r)
        and not r.static_crossing
        and r.sc_min_dist >= thresh
    )


def _passes_moving_common(r) -> bool:
    return _passes_common_gates(r) and r.collision_step is None


@torch.no_grad()
def filter_scenes(
    model,
    model_args,
    candidates: list[dict],
    rcfg,
    device: torch.device,
    clearance_thresh: float,
    variant: str,
    K: int,
    noise_range: tuple[float, float],
    scene_batch_size: int = 8,
    save_curated_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run K=N generation + fittability filter on candidate scenes.

    Returns (kept, dropped) lists with per-scene details.
    When save_curated_dir is set, writes a copy of each kept NPZ with
    ego_agent_future replaced by the best-of-K trajectory (for curated RSFT).
    """
    kept, dropped = [], []

    for start in range(0, len(candidates), scene_batch_size):
        batch_cands = candidates[start : start + scene_batch_size]
        datas = []
        valid_cands = []

        for cand in batch_cands:
            p = cand["npz_path"]
            try:
                d = load_npz_data(p, device)
                datas.append(d)
                valid_cands.append(cand)
            except Exception as e:
                print(f"  [skip] {Path(p).name}: {e}")
                dropped.append({**cand, "reason": f"load_error: {e}"})

        if not datas:
            continue

        batch = _stack_scene_data(datas, device)
        norm_batch = _normalize_batch(batch, model_args)

        all_trajs = generate_all_scenes_batched(
            model, model_args, norm_batch, K,
            noise_range=noise_range, device=device,
            generation_variant=variant,
            use_route_cl_guidance=True,
        )  # (N, K, T, 4)

        for bi, cand in enumerate(valid_cands):
            scene_trajs = all_trajs[bi]  # (K, T, 4)
            scene_class = cand["class"]
            best_reward = -1e9
            best_k = -1
            best_clearance = 0.0
            gate_failures = {"rb": 0, "lane": 0, "kin": 0, "sc": 0,
                             "collision": 0, "stopped": 0, "clearance": 0}
            n_passing = 0

            for k in range(K):
                traj_1T4 = scene_trajs[k : k + 1]
                rewards = compute_reward_batch(traj_1T4, datas[bi], rcfg)
                r = rewards[0]

                if not _passes_common_gates(r):
                    if r.rb_crossing:
                        gate_failures["rb"] += 1
                    if r.lane_crossing:
                        gate_failures["lane"] += 1
                    if r.kinematic_violated:
                        gate_failures["kin"] += 1
                    if r.total <= -50.0:
                        gate_failures["stopped"] += 1
                    continue

                if scene_class == "stopped":
                    if not _passes_stopped_filter(r, clearance_thresh):
                        if r.static_crossing:
                            gate_failures["sc"] += 1
                        if r.sc_min_dist < clearance_thresh:
                            gate_failures["clearance"] += 1
                        continue
                    clearance = r.sc_min_dist
                else:  # moving
                    if not _passes_moving_common(r):
                        if r.collision_step is not None:
                            gate_failures["collision"] += 1
                        continue
                    mov_slot = cand.get("neighbor_slot")
                    if mov_slot is None:
                        gate_failures["clearance"] += 1
                        continue
                    mov_clr = _compute_moving_clearance(
                        scene_trajs[k:k+1], datas[bi], mov_slot, rcfg, device,
                    )
                    if mov_clr < clearance_thresh:
                        gate_failures["clearance"] += 1
                        continue
                    clearance = mov_clr

                n_passing += 1
                if r.total > best_reward:
                    best_reward = float(r.total)
                    best_k = k
                    best_clearance = float(clearance)

            scene_name = Path(cand["npz_path"]).name
            result = {
                **cand,
                "best_k": best_k,
                "best_reward": round(best_reward, 4) if best_k >= 0 else None,
                "best_clearance": round(best_clearance, 4) if best_k >= 0 else None,
                "n_passing": n_passing,
                "gate_failures": gate_failures,
            }

            if best_k >= 0:
                kept.append(result)
                print(
                    f"  [KEPT] {scene_name} class={scene_class} "
                    f"k={best_k} clr={best_clearance:.3f}m reward={best_reward:.2f}"
                )
                if save_curated_dir is not None:
                    best_traj = scene_trajs[best_k].cpu().numpy()  # (T, 4)
                    src_data = dict(np.load(cand["npz_path"]))
                    src_data["ego_agent_future"] = best_traj.astype(np.float32)
                    out_npz = save_curated_dir / scene_name
                    np.savez_compressed(str(out_npz), **src_data)
                    result["curated_npz"] = str(out_npz)
            else:
                result["reason"] = "no_passing_generation"
                dropped.append(result)
                print(
                    f"  [DROP] {scene_name} class={scene_class} "
                    f"gates={gate_failures}"
                )

    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description="Filter avoidance scenes by K=N fittability")
    parser.add_argument("--candidates", required=True, help="Path to candidate_scenes_all.json")
    parser.add_argument("--model_path", required=True, help="Champion model path")
    parser.add_argument("--config", required=True, help="Reward config JSON (reward_sc.json)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--clearance_thresh", type=float, required=True,
                        help="Minimum clearance in metres (e.g. 0.5)")
    parser.add_argument("--variant", default="rsft_v2_col4",
                        help="Generation variant (default: rsft_v2_col4)")
    parser.add_argument("--K", type=int, default=16, help="Number of generations per scene")
    parser.add_argument("--noise_range", default="0.5,2.0",
                        help="Noise scale range (default: 0.5,2.0)")
    parser.add_argument("--scene_batch_size", type=int, default=8,
                        help="Scenes per GPU batch")
    parser.add_argument("--save_curated_dir", type=str, default=None,
                        help="If set, save kept NPZs with best-of-K as ego_agent_future")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_range = tuple(float(x) for x in args.noise_range.split(","))

    with open(args.candidates) as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidate scenes from {args.candidates}")

    model, model_args = load_model(args.model_path, device)
    rcfg = load_reward_config(args.config)
    print(f"Model loaded from {args.model_path}")
    print(f"Reward config: {args.config}")
    print(f"Clearance threshold: {args.clearance_thresh} m")
    print(f"Variant: {args.variant}, K={args.K}, noise={noise_range}")

    curated_dir = None
    if args.save_curated_dir:
        curated_dir = Path(args.save_curated_dir)
        curated_dir.mkdir(parents=True, exist_ok=True)

    kept, dropped = filter_scenes(
        model, model_args, candidates, rcfg, device,
        clearance_thresh=args.clearance_thresh,
        variant=args.variant,
        K=args.K,
        noise_range=noise_range,
        scene_batch_size=args.scene_batch_size,
        save_curated_dir=curated_dir,
    )

    # Save results
    with open(out_dir / "kept_scenes.json", "w") as f:
        json.dump(kept, f, indent=2)
    with open(out_dir / "dropped_scenes.json", "w") as f:
        json.dump(dropped, f, indent=2)

    n_stopped_kept = sum(1 for s in kept if s["class"] == "stopped")
    n_moving_kept = sum(1 for s in kept if s["class"] == "moving")
    n_stopped_drop = sum(1 for s in dropped if s["class"] == "stopped")
    n_moving_drop = sum(1 for s in dropped if s["class"] == "moving")

    summary = {
        "total_candidates": len(candidates),
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_stopped": n_stopped_kept,
        "kept_moving": n_moving_kept,
        "dropped_stopped": n_stopped_drop,
        "dropped_moving": n_moving_drop,
        "clearance_thresh": args.clearance_thresh,
        "variant": args.variant,
        "K": args.K,
    }
    if kept:
        clrs = [s["best_clearance"] for s in kept]
        summary["kept_clearance_stats"] = {
            "min": min(clrs), "max": max(clrs),
            "mean": round(sum(clrs) / len(clrs), 4),
        }

    with open(out_dir / "filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== FILTER SUMMARY ===")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Kept: {len(kept)} ({n_stopped_kept} stopped, {n_moving_kept} moving)")
    print(f"  Dropped: {len(dropped)} ({n_stopped_drop} stopped, {n_moving_drop} moving)")
    if kept:
        print(f"  Clearance range: [{min(clrs):.3f}, {max(clrs):.3f}] m")


if __name__ == "__main__":
    main()
