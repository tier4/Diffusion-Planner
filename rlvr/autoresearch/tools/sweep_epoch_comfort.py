#!/usr/bin/env python3
"""Sweep the per-epoch LoRA checkpoints of a ranked-SFT run and report
deterministic avoidance metrics per epoch (static/rb/lane crossings + sc_min_dist
distribution). Useful for picking the candidate epoch(s) of any LoRA training run
before the expensive L2 / psim verdict.

For each requested epoch it:
  1. merges <run_dir>/lora_epoch_NNN into --base_model via
     `preference_optimization.merge_lora` (the canonical merge interface),
  2. scores --scenes with `eval_det_avoidance.score_det_scenes` (reuses the
     reward.py OBB scoring path — no reimplemented geometry),
  3. records the aggregate from `eval_det_avoidance.aggregate_stats`.

Merged checkpoints are written under <output_dir>/merged_epNNN and deleted after
scoring unless --keep_merged is given (each is ~230MB).

Usage:
    python -m rlvr.autoresearch.tools.sweep_epoch_comfort \
        --run_dir <run with lora_epoch_NNN> \
        --base_model <base.pth> \
        --scenes <scenes.json> \
        --config <reward_config.json> \
        --ego_shape WB,L,W \
        --output_dir <dir> \
        --epochs 1-24            # or "2,4,6" or "1-24:2" (stride) or "all"

Outputs:
    <output_dir>/sweep_summary.json   per-epoch aggregate table
    <output_dir>/avoid_epNNN/det_avoidance_summary.json   per-epoch full detail
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data as _load_npz_comfort
from rlvr.autoresearch.tools.eval_det_avoidance import (
    aggregate_stats,
    load_model,
    score_det_scenes,
)
from rlvr.autoresearch.tools.eval_driving_metrics import (
    generate_trajectory,
    lat_accel_smoothed,
)
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import _build_sg_diff_kernel


def _jerk_mag(xy: np.ndarray, w: int = 11) -> np.ndarray:
    """|d3(x,y)/dt3| from positions (T,2) via SG deriv=3 (reuses reward kernel)."""
    if xy.shape[0] < w:
        return np.zeros(1)
    k = _build_sg_diff_kernel(window=w, poly=3, deriv=3, delta=0.1)
    pad = w // 2
    t = torch.from_numpy(xy).float().permute(1, 0).unsqueeze(0)
    t = torch.nn.functional.pad(t, (pad, pad), mode="replicate")
    j = torch.nn.functional.conv1d(t, k.view(1, 1, -1).expand(2, 1, -1), groups=2)[0].numpy()
    return np.sqrt(j[0] ** 2 + j[1] ** 2)


def comfort_for_epoch(model, model_args, scene_paths, device) -> dict:
    """Open-loop PLAN comfort over scenes: |lat_accel| + jerk of the det trajectory.

    Reuses eval_driving_metrics.{generate_trajectory, lat_accel_smoothed}. Reports
    full sub-metrics (mean/p50/p95/max) so inter-epoch comfort change is visible.
    """
    la_all, jk_all = [], []
    for p in scene_paths:
        try:
            d = _load_npz_comfort(p, device)
            traj = generate_trajectory(model, model_args, d, device)
        except Exception:  # one bad scene must not abort a multi-epoch sweep
            continue
        traj = traj.detach().cpu().numpy() if hasattr(traj, "detach") else np.asarray(traj)
        xy = traj[:, :2].astype(np.float32)
        la_all.append(np.abs(lat_accel_smoothed(xy)))
        jk_all.append(_jerk_mag(xy))
    if not la_all:
        return {}
    la = np.concatenate(la_all)
    jk = np.concatenate(jk_all)
    pct = lambda a, q: float(np.percentile(a, q))
    return {
        "lat_mean": float(la.mean()),
        "lat_p50": pct(la, 50),
        "lat_p95": pct(la, 95),
        "lat_max": float(la.max()),
        "jerk_mean": float(jk.mean()),
        "jerk_p95": pct(jk, 95),
    }


def parse_epochs(spec: str, run_dir: Path) -> list[int]:
    """Parse "all" | "1-24" | "1-24:2" | "2,4,6" into a sorted epoch list."""
    available = sorted(
        int(m.group(1))
        for p in run_dir.glob("lora_epoch_*")
        if (m := re.fullmatch(r"lora_epoch_(\d+)", p.name))
    )
    if not available:
        raise FileNotFoundError(f"No lora_epoch_NNN dirs found in {run_dir}")
    if spec == "all":
        return available
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            rng, _, stride = part.partition(":")
            lo, hi = (int(x) for x in rng.split("-"))
            step = int(stride) if stride else 1
            out.update(range(lo, hi + 1, step))
        else:
            out.add(int(part))
    sel = sorted(e for e in out if e in available)
    missing = sorted(e for e in out if e not in available)
    if missing:
        print(f"[sweep] requested epochs not present, skipping: {missing}")
    if not sel:
        raise ValueError(f"None of the requested epochs exist in {run_dir}")
    return sel


def merge_epoch(base_model: Path, lora_dir: Path, out_pth: Path) -> None:
    """Invoke the canonical merge_lora CLI for one epoch."""
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "preference_optimization.merge_lora",
        "--model_path",
        str(base_model),
        "--lora_dir",
        str(lora_dir),
        "--output",
        str(out_pth),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_pth.exists():
        raise RuntimeError(
            f"merge_lora failed for {lora_dir}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
        )
    # merge_lora reads args.json from the BASE model dir; copy it next to the merged
    # checkpoint so downstream load_model finds it.
    base_args = base_model.parent / "args.json"
    if base_args.exists():
        shutil.copy(base_args, out_pth.parent / "args.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir", required=True, help="Training run dir containing lora_epoch_NNN/"
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base .pth the LoRA was trained on (its dir must hold args.json)",
    )
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--config", required=True, help="Reward config JSON")
    parser.add_argument("--ego_shape", required=True, help="WB,L,W e.g. 4.76,7.24,2.29")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", required=True, help='"all" | "1-24" | "1-24:2" | "2,4,6"')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--keep_merged",
        action="store_true",
        help="Keep merged .pth per epoch (default: delete to save disk)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_model = Path(args.base_model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    rcfg = load_reward_config(args.config)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    epochs = parse_epochs(args.epochs, run_dir)
    print(f"[sweep] {len(epochs)} epochs: {epochs}")
    print(f"[sweep] {len(scene_paths)} scenes, reward={args.config}")

    rows: list[dict] = []
    for e in epochs:
        ep = f"{e:03d}"
        lora_dir = run_dir / f"lora_epoch_{ep}"
        merged = out_dir / f"merged_ep{ep}" / "best_model.pth"
        print(f"\n[sweep] ===== epoch {ep} =====")
        merge_epoch(base_model, lora_dir, merged)
        model, model_args = load_model(str(merged), device)
        results = score_det_scenes(
            model,
            model_args,
            scene_paths,
            rcfg,
            ego_shape,
            device,
            batch_size=args.batch_size,
        )
        comfort = comfort_for_epoch(model, model_args, scene_paths, device)
        del model
        torch.cuda.empty_cache()
        if not args.keep_merged:
            shutil.rmtree(merged.parent, ignore_errors=True)
        if not results:
            print(f"[sweep] ep{ep}: all scenes skipped")
            continue
        agg = aggregate_stats(results)
        # write per-epoch detail
        ep_out = out_dir / f"avoid_ep{ep}"
        ep_out.mkdir(parents=True, exist_ok=True)
        with open(ep_out / "det_avoidance_summary.json", "w") as f:
            json.dump({"aggregate": agg, "scenes": results}, f, indent=2)
        scm = agg["sc_min_dist"]
        row = {
            "epoch": e,
            "n": agg["n_scenes"],
            "static_crossings": agg["static_crossings"],
            "rb_crossings": agg["rb_crossings"],
            "lane_crossings": agg["lane_crossings"],
            "sc_min_mean": scm["mean"],
            "sc_min_p5": scm["p5"],
            "sc_min_min": scm["min"],
            **{f"comfort_{k}": v for k, v in comfort.items()},
        }
        rows.append(row)
        cstr = (
            f"  lat[mean={comfort['lat_mean']:.3f} p95={comfort['lat_p95']:.3f} "
            f"max={comfort['lat_max']:.3f}] jerk[mean={comfort['jerk_mean']:.2f}]"
            if comfort
            else "  comfort=NA"
        )
        print(
            f"[sweep] ep{ep}: static={row['static_crossings']}/{row['n']}  "
            f"rb={row['rb_crossings']}  lane={row['lane_crossings']}  "
            f"sc_min mean={scm['mean']:+.3f} p5={scm['p5']:+.3f} min={scm['min']:+.3f}" + cstr
        )

    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"  Sweep summary ({len(rows)} epochs)")
    print(f"{'=' * 72}")
    print(
        f"  {'ep':>3}  {'static':>7}  {'rb':>4}  {'lane':>4}  "
        f"{'sc_mean':>8}  {'sc_p5':>7}  {'sc_min':>7}  {'lat_mean':>8}  {'lat_p95':>7}  {'lat_max':>7}  {'jerk_m':>6}"
    )
    for r in rows:
        print(
            f"  {r['epoch']:>3}  {r['static_crossings']:>3}/{r['n']:<3}  "
            f"{r['rb_crossings']:>4}  {r['lane_crossings']:>4}  "
            f"{r['sc_min_mean']:>+8.3f}  {r['sc_min_p5']:>+7.3f}  {r['sc_min_min']:>+7.3f}  "
            f"{r.get('comfort_lat_mean', float('nan')):>8.3f}  {r.get('comfort_lat_p95', float('nan')):>7.3f}  "
            f"{r.get('comfort_lat_max', float('nan')):>7.3f}  {r.get('comfort_jerk_mean', float('nan')):>6.2f}"
        )
    print(f"{'=' * 72}")
    print(f"Wrote {out_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
