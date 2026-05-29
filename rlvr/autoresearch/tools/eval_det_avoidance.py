#!/usr/bin/env python3
"""Deterministic-only avoidance eval: score det trajectory on static-collision scenes.

Single forward pass per scene (noise=0, no guidance). Reports per-scene
metrics and aggregate distribution stats (mean, p5, p25, p50, p75, p95,
min, max) for sc_min_dist, rb_min_dist, and safety flags.

Usage:
    python -m rlvr.autoresearch.tools.eval_det_avoidance \
        --model_path <model.pth> \
        --scenes <scenes.json> \
        --config <reward_config.json> \
        --ego_shape WB,L,W \
        --output_dir <dir>

Outputs:
    <output_dir>/det_avoidance_summary.json   per-scene + aggregate stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import compute_reward_batch


def load_model(model_path: str, device: torch.device):
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    from diffusion_planner.utils.config import Config

    model_dir = Path(model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    model_args = Config(str(args_path))
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, model_args


@torch.no_grad()
def det_inference_batched(
    model, model_args, datas: list[dict], device: torch.device,
    norm_batch: dict | None = None,
) -> torch.Tensor:
    """Run deterministic inference (noise=0) on a batch of scenes.

    Args:
        model: Diffusion_Planner model.
        model_args: Model config.
        datas: List of per-scene dicts from load_npz_data.
        device: Torch device.
        norm_batch: Pre-computed normalized batch (skip stacking if provided).

    Returns (B, T, 4) tensor of ego trajectories.
    """
    from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data

    if norm_batch is None:
        batch = _stack_scene_data(datas, device)
        norm_batch = _normalize_batch(batch, model_args)
    else:
        B_nb = norm_batch["ego_current_state"].shape[0]
        if B_nb != len(datas):
            raise ValueError(
                f"norm_batch has batch size {B_nb} but len(datas)={len(datas)}. "
                f"They must match when norm_batch is pre-provided."
            )

    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    saved_fn = decoder._guidance_fn
    decoder._guidance_fn = None
    try:
        P = 1 + model_args.predicted_neighbor_num
        future_len = model_args.future_len
        norm_batch_d = {k: v for k, v in norm_batch.items()}
        norm_batch_d["sampled_trajectories"] = torch.zeros(
            len(datas), P, future_len + 1, 4, device=device,
        )
        _, det_out = model(norm_batch_d)
        return det_out["prediction"][:, 0].detach()  # (B, T, 4)
    finally:
        decoder._guidance_fn = saved_fn


def reward_breakdown_to_det_dict(r) -> dict:
    """Extract deterministic scoring fields from a RewardBreakdown."""
    return {
        "det_cl": float(r.centerline),
        "det_total": float(r.total),
        "det_sc_min_dist": float(getattr(r, "sc_min_dist", 99.0)),
        "det_rb_min_dist": float(getattr(r, "rb_min_dist", 99.0)),
        "det_rb_cross": bool(getattr(r, "rb_crossing", False)),
        "det_lane_cross": bool(getattr(r, "lane_crossing", False)),
        "det_kin_violated": bool(getattr(r, "kinematic_violated", False)),
        "det_static_crossing": bool(getattr(r, "static_crossing", False)),
        "det_sc_n_stopped": int(getattr(r, "sc_n_stopped", 0)),
    }


def score_det_scenes(
    model, model_args, scene_paths: list[str], rcfg, ego_shape: np.ndarray,
    device: torch.device, batch_size: int = 32,
) -> list[dict]:
    """Run det inference + reward scoring on all scenes. Returns per-scene dicts."""
    results = []

    for start in range(0, len(scene_paths), batch_size):
        batch_paths = scene_paths[start : start + batch_size]
        datas = []
        valid_paths = []

        for p in batch_paths:
            try:
                raw_keys = set(np.load(p, allow_pickle=True).keys())
                if "ego_shape" not in raw_keys:
                    print(f"  [skip] {Path(p).name}: missing ego_shape")
                    continue
                d = load_npz_data(p, device)
                es = d["ego_shape"].cpu().numpy().reshape(-1)[:3]
                if not np.allclose(es, ego_shape, atol=1e-2):
                    print(
                        f"  [skip] {Path(p).name}: ego_shape={es.tolist()} "
                        f"vs --ego_shape={ego_shape.tolist()}"
                    )
                    continue
                datas.append(d)
                valid_paths.append(p)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {Path(p).name}: {e}")

        if not datas:
            continue

        det_trajs = det_inference_batched(model, model_args, datas, device)

        for bi, p in enumerate(valid_paths):
            traj_1T4 = det_trajs[bi : bi + 1]
            r = compute_reward_batch(traj_1T4, datas[bi], rcfg)[0]
            results.append({
                "scene": Path(p).name,
                "scene_path": str(p),
                "sc_min_dist": float(getattr(r, "sc_min_dist", 99.0)),
                "rb_min_dist": float(getattr(r, "rb_min_dist", 99.0)),
                "cl": float(r.centerline),
                "total": float(r.total),
                "static_crossing": bool(r.static_crossing),
                "rb_cross": bool(r.rb_crossing),
                "lane_cross": bool(r.lane_crossing),
                "kin_violated": bool(r.kinematic_violated),
                "sc_n_stopped": int(getattr(r, "sc_n_stopped", 0)),
            })
            flag = "COL" if r.static_crossing else "   "
            print(
                f"  [{start + bi:3d}] {flag}  "
                f"sc={float(getattr(r, 'sc_min_dist', 99.0)):+.3f}m  "
                f"rb={float(getattr(r, 'rb_min_dist', 99.0)):.3f}m  "
                f"{Path(p).name}"
            )

    return results


def aggregate_stats(results: list[dict]) -> dict:
    """Compute distribution stats over per-scene metrics."""
    n = len(results)
    if n == 0:
        return {}

    def _dist(vals):
        a = np.array(vals)
        return {
            "mean": float(a.mean()),
            "p5": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    return {
        "n_scenes": n,
        "static_crossings": sum(r["static_crossing"] for r in results),
        "rb_crossings": sum(r["rb_cross"] for r in results),
        "lane_crossings": sum(r["lane_cross"] for r in results),
        "kin_violated": sum(r["kin_violated"] for r in results),
        "sc_min_dist": _dist([r["sc_min_dist"] for r in results]),
        "rb_min_dist": _dist([r["rb_min_dist"] for r in results]),
        "cl": _dist([r["cl"] for r in results]),
        "total": _dist([r["total"] for r in results]),
    }


def print_avoidance_summary(agg: dict) -> None:
    """Print avoidance-focused summary (sc_min_dist only)."""
    n = agg["n_scenes"]
    d = agg["sc_min_dist"]
    print(f"\n{'=' * 65}")
    print(f"  Avoidance Eval — {n} scenes")
    print(f"{'=' * 65}")
    print(f"  Static crossings:  {agg['static_crossings']}/{n}")
    print()
    print(
        f"  sc_min_dist  "
        f"mean={d['mean']:+.3f}  "
        f"p5={d['p5']:+.3f}  "
        f"p25={d['p25']:+.3f}  "
        f"p50={d['p50']:+.3f}  "
        f"p75={d['p75']:+.3f}  "
        f"p95={d['p95']:+.3f}  "
        f"min={d['min']:+.3f}  "
        f"max={d['max']:+.3f}"
    )
    print(f"{'=' * 65}\n")


def print_summary(agg: dict) -> None:
    """Print a human-readable summary table."""
    n = agg["n_scenes"]
    print(f"\n{'=' * 65}")
    print(f"  Deterministic Avoidance Eval — {n} scenes")
    print(f"{'=' * 65}")
    print(f"  Static crossings:  {agg['static_crossings']}/{n}")
    print(f"  RB crossings:      {agg['rb_crossings']}/{n}")
    print(f"  Lane crossings:    {agg['lane_crossings']}/{n}")
    print(f"  Kin violated:      {agg['kin_violated']}/{n}")
    print()
    for key in ("sc_min_dist", "rb_min_dist", "cl"):
        d = agg[key]
        print(
            f"  {key:14s}  "
            f"mean={d['mean']:+.3f}  "
            f"p5={d['p5']:+.3f}  "
            f"p25={d['p25']:+.3f}  "
            f"p50={d['p50']:+.3f}  "
            f"p75={d['p75']:+.3f}  "
            f"p95={d['p95']:+.3f}  "
            f"min={d['min']:+.3f}  "
            f"max={d['max']:+.3f}"
        )
    print(f"{'=' * 65}\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ego_shape", required=True,
                        help="WB,L,W e.g. 4.76,7.24,2.29")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    rcfg = load_reward_config(args.config)

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    print(f"[eval_det_avoidance] {len(scene_paths)} scenes, model={args.model_path}")

    model, model_args = load_model(args.model_path, device)

    results = score_det_scenes(
        model, model_args, scene_paths, rcfg, ego_shape,
        device, batch_size=args.batch_size,
    )

    agg = aggregate_stats(results)
    print_summary(agg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "det_avoidance_summary.json"
    with open(out_path, "w") as f:
        json.dump({"aggregate": agg, "scenes": results}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
