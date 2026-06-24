"""Unified deterministic metrics evaluation.

One deterministic forward pass per scene, scored against ALL reward metrics in a single
table: avoidance (static-collision clearance + crossings), road border, lane departure,
centerline, path length, plus collision / kinematic flags. Optionally applies a LoRA.

This is the "evaluate the model on everything" tool — it reuses the same inference
(`det_inference_batched`) and scoring (`compute_reward_batch`) the training pipeline uses,
so the numbers match. For L2 (ego/neighbor displacement vs GT) use `valid_predictor`; for
guidance-policy evaluation use `eval_policy_avoidance`.

Usage:
    python -m rlvr.autoresearch.tools.eval_full_metrics \
        --model_path <base.pth> [--lora_path <lora_dir>] \
        --scenes <scenes.json> --config <reward.json> --ego_shape WB,L,W \
        --output_dir <dir> [--batch_size 32]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import compute_reward_batch


def _path_len(traj_t4: np.ndarray) -> float:
    """Travelled arc length (m) of an [T,4] (x,y,cos,sin) trajectory."""
    xy = traj_t4[:, :2]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _dist(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return {}
    return {
        "mean": float(a.mean()),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def score_scenes(model, model_args, scene_paths, rcfg, ego_shape, device, batch_size=32):
    """Det inference + full reward scoring. Returns (rows, trajs_by_path).

    trajs_by_path maps scene_path -> [T,4] numpy trajectory (for optional rendering).
    """
    rows: list[dict] = []
    trajs_by_path: dict[str, np.ndarray] = {}
    for start in range(0, len(scene_paths), batch_size):
        batch = scene_paths[start : start + batch_size]
        datas, valid = [], []
        for p in batch:
            try:
                with np.load(p, allow_pickle=True) as npz:
                    if "ego_shape" not in set(npz.keys()):
                        print(f"  [skip] {Path(p).name}: missing ego_shape")
                        continue
                d = load_npz_data(p, device)
                es = d["ego_shape"].cpu().numpy().reshape(-1)[:3]
                if not np.allclose(es, ego_shape, atol=1e-2):
                    print(
                        f"  [skip] {Path(p).name}: ego_shape {es.tolist()} != {ego_shape.tolist()}"
                    )
                    continue
                datas.append(d)
                valid.append(p)
            except Exception as e:  # noqa: BLE001 - log + skip unreadable scenes
                print(f"  [skip] {Path(p).name}: {e}")
        if not datas:
            continue
        trajs = det_inference_batched(model, model_args, datas, device)
        for bi, p in enumerate(valid):
            t = trajs[bi : bi + 1]
            r = compute_reward_batch(t, datas[bi], rcfg)[0]
            traj_np = t[0].cpu().numpy()
            trajs_by_path[str(p)] = traj_np
            rows.append(
                {
                    "scene": Path(p).name,
                    "scene_path": str(p),
                    "sc_min_dist": float(getattr(r, "sc_min_dist", 99.0)),
                    "sc_n_stopped": int(getattr(r, "sc_n_stopped", 0)),
                    "static_crossing": bool(r.static_crossing),
                    "rb_min_dist": float(getattr(r, "rb_min_dist", 99.0)),
                    "rb_crossing": bool(r.rb_crossing),
                    "rb_near_penalty": float(r.rb_near_penalty),
                    "rb_wide_penalty": float(r.rb_wide_penalty),
                    "lane_crossing": bool(r.lane_crossing),
                    "lane_near_frac": float(r.lane_near_frac),
                    "lane_wide_frac": float(r.lane_wide_frac),
                    "centerline": float(r.centerline),
                    "off_road_fraction": float(r.off_road_fraction),
                    "collision": r.collision_step is not None,
                    "kin_violated": bool(r.kinematic_violated),
                    "path_len": _path_len(traj_np),
                    "total": float(r.total),
                }
            )
    return rows, trajs_by_path


def _load(model_path, lora_path, device):
    model, model_args = load_model(model_path, device)
    if lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint

        model = load_lora_checkpoint(model, lora_path)
        model.to(device).eval()
        print(f"[eval_full_metrics] applied LoRA: {lora_path}")
    return model, model_args


def render_scenes(rows_a, trajs_a, label_a, rows_b, trajs_b, label_b, out_dir):
    """Per-scene PNGs of the det trajectory(ies) on each scene, using the canonical renderers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rlvr.autoresearch.tools.viz_cl_recovery import draw_scene_base, draw_traj

    viz_dir = Path(out_dir) / "scenes"
    viz_dir.mkdir(parents=True, exist_ok=True)
    rows_b_by_path = {r["scene_path"]: r for r in (rows_b or [])}
    for ra in rows_a:
        sp = ra["scene_path"]
        name = Path(sp).stem
        ta = trajs_a.get(sp)
        if ta is None:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))
        draw_scene_base(ax, sp)
        draw_traj(ax, ta, f"{label_a} (sc={ra['sc_min_dist']:.2f}m)", "#1f77b4", sp)
        pts = [ta[:, :2], np.zeros((1, 2))]
        if trajs_b and sp in trajs_b:
            tb = trajs_b[sp]
            rb = rows_b_by_path.get(sp, {})
            draw_traj(ax, tb, f"{label_b} (sc={rb.get('sc_min_dist', 99):.2f}m)", "#d62728", sp)
            pts.append(tb[:, :2])
        allp = np.vstack(pts)
        cx, cy = float(allp[:, 0].mean()), float(allp[:, 1].mean())
        half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.6 + 6
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(name, fontsize=10)
        fig.savefig(viz_dir / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"  rendered {len(rows_a)} scene PNGs -> {viz_dir}")


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_scenes": 0}

    def cnt(key):
        return int(sum(1 for r in rows if r[key]))

    n_stopped_scenes = sum(1 for r in rows if r["sc_n_stopped"] > 0)
    return {
        "n_scenes": n,
        "n_scenes_with_stopped_npc": n_stopped_scenes,
        "static_crossings": cnt("static_crossing"),
        "rb_crossings": cnt("rb_crossing"),
        "lane_crossings": cnt("lane_crossing"),
        "collisions": cnt("collision"),
        "kin_violations": cnt("kin_violated"),
        "collision_rate": cnt("collision") / n,
        "static_crossing_rate": cnt("static_crossing") / n,
        "sc_min_dist": _dist([r["sc_min_dist"] for r in rows]),
        "rb_min_dist": _dist([r["rb_min_dist"] for r in rows]),
        "centerline": _dist([r["centerline"] for r in rows]),
        "path_len": _dist([r["path_len"] for r in rows]),
        "lane_near_frac_mean": float(np.mean([r["lane_near_frac"] for r in rows])),
        "lane_wide_frac_mean": float(np.mean([r["lane_wide_frac"] for r in rows])),
    }


def print_summary(agg: dict) -> None:
    n = agg["n_scenes"]
    print("\n" + "=" * 70)
    print(f"  Full deterministic metrics — {n} scenes")
    print("=" * 70)
    print(
        f"  Collisions:        {agg['collisions']}/{n}   "
        f"Static crossings: {agg['static_crossings']}/{n} "
        f"(scenes w/ stopped NPC: {agg['n_scenes_with_stopped_npc']})"
    )
    print(
        f"  RB crossings:      {agg['rb_crossings']}/{n}   Lane crossings: {agg['lane_crossings']}/{n}"
    )
    print(f"  Kin violations:    {agg['kin_violations']}/{n}")
    for key, label in (
        ("sc_min_dist", "sc_min_dist  "),
        ("rb_min_dist", "rb_min_dist  "),
        ("centerline", "centerline   "),
        ("path_len", "path_len (m) "),
    ):
        d = agg[key]
        if d:
            print(
                f"  {label} mean={d['mean']:+.3f}  p5={d['p5']:+.3f}  p25={d['p25']:+.3f}  "
                f"p50={d['p50']:+.3f}  p75={d['p75']:+.3f}  p95={d['p95']:+.3f}  "
                f"min={d['min']:+.3f}  max={d['max']:+.3f}"
            )
    if agg["sc_min_dist"].get("mean") == 99.0:
        print(
            "  NOTE: sc_min_dist is 99 everywhere — the reward config has no stopped-NPC "
            "scoring (use an SC-enabled config, e.g. one with static_collision_enabled=true)."
        )


def print_h2h(agg_a, agg_b, label_a, label_b):
    """Compact A-vs-B comparison of the headline metrics."""
    print("\n" + "=" * 70)
    print(f"  Head-to-head:  A={label_a}   B={label_b}")
    print("=" * 70)
    n = agg_a["n_scenes"]

    def row(name, a, b, fmt="{:+.3f}"):
        print(f"  {name:<22} A={fmt.format(a):>10}   B={fmt.format(b):>10}")

    row("collisions", agg_a["collisions"], agg_b["collisions"], "{:d}/" + str(n))
    row("static_crossings", agg_a["static_crossings"], agg_b["static_crossings"], "{:d}/" + str(n))
    row("rb_crossings", agg_a["rb_crossings"], agg_b["rb_crossings"], "{:d}/" + str(n))
    row("lane_crossings", agg_a["lane_crossings"], agg_b["lane_crossings"], "{:d}/" + str(n))
    for key in ("sc_min_dist", "rb_min_dist", "centerline", "path_len"):
        a, b = agg_a[key].get("mean"), agg_b[key].get("mean")
        if a is not None and b is not None:
            row(f"{key} mean", a, b)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None, help="Optional LoRA adapter dir to apply")
    ap.add_argument("--model_b", default=None, help="Optional 2nd model for head-to-head")
    ap.add_argument("--lora_b", default=None, help="Optional LoRA for model B")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W e.g. 4.76,7.24,2.29")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--render", action="store_true", help="Render per-scene PNGs to <out>/scenes")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    rcfg = load_reward_config(args.config)

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    from diffusion_planner.utils.scene_skip import filter_scene_list

    scene_paths = filter_scene_list(scene_paths, label="eval_full_metrics")
    print(f"[eval_full_metrics] {len(scene_paths)} scenes, model A={args.model_path}")

    model_a, margs_a = _load(args.model_path, args.lora_path, device)
    rows_a, trajs_a = score_scenes(
        model_a, margs_a, scene_paths, rcfg, ego_shape, device, args.batch_size
    )
    if not rows_a:
        raise SystemExit(
            f"All {len(scene_paths)} scenes were skipped (ego_shape mismatch or missing). "
            f"Check --ego_shape={args.ego_shape} matches the NPZs."
        )
    agg_a = aggregate(rows_a)
    print(f"\n--- {args.label_a} ---")
    print_summary(agg_a)

    rows_b, trajs_b, agg_b = None, None, None
    if args.model_b:
        print(f"\n[eval_full_metrics] model B={args.model_b}")
        model_b, margs_b = _load(args.model_b, args.lora_b, device)
        rows_b, trajs_b = score_scenes(
            model_b, margs_b, scene_paths, rcfg, ego_shape, device, args.batch_size
        )
        agg_b = aggregate(rows_b)
        print(f"\n--- {args.label_b} ---")
        print_summary(agg_b)
        print_h2h(agg_a, agg_b, args.label_a, args.label_b)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render:
        render_scenes(rows_a, trajs_a, args.label_a, rows_b, trajs_b, args.label_b, out_dir)

    summary = {
        "model_a": args.model_path,
        "lora_a": args.lora_path,
        "label_a": args.label_a,
        "aggregate": agg_a,
        "scenes": rows_a,
    }
    if args.model_b:
        summary.update(
            {
                "model_b": args.model_b,
                "lora_b": args.lora_b,
                "label_b": args.label_b,
                "aggregate_b": agg_b,
                "scenes_b": rows_b,
            }
        )
    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
