#!/usr/bin/env python3
"""Per-epoch FULL-metric sweep of a LoRA training run.

Extends sweep_lora_epochs (avoidance-only) to the complete per-epoch metric
array needed for honest epoch picks:
  - deterministic avoidance on MULTIPLE scene sets (e.g. a raw base set AND a
    perturbed hard-val set): crossings + full sc_min_dist distribution
    (mean/p5/p25/p50/p75/min) via eval_det_avoidance (reward.py OBB path),
  - open-loop PLAN comfort (lat-accel p95 / jerk p95 / curve speed) via
    eval_plan_comfort — catches plan-sharpness comfort regressions that the
    geometric metrics miss,
  - ego/neighbor L2 on a validation list via valid_320 (DDP subprocess).

For each epoch: merge lora_epoch_NNN onto --base_model with the canonical
merge_lora CLI, score everything, append a row to <output_dir>/full_sweep.json
and print a markdown table row. Merged checkpoints are kept under
<output_dir>/merged_epNNN unless --no_keep_merged.

Usage:
    python -m rlvr.autoresearch.tools.sweep_full_metrics \
        --run_dir <run with lora_epoch_NNN> \
        --base_model <warmstart.pth (dir must hold args.json)> \
        --epochs all|1-12|2,4,6|1-24:2 \
        --avoid_sets base=<scenes.json> hardval=<scenes.json> \
        --config <reward_config.json> --ego_shape WB,L,W \
        --comfort_scenes <scenes.json> \
        --l2_val <val_list.json> --args_json <args.json> \
        --output_dir <dir> [--l2_port 29571] [--batch_size 32]

Any of the metric groups can be skipped by omitting its flag (--avoid_sets /
--comfort_scenes / --l2_val), so the tool also serves as a quick single-axis
sweeper. No silent defaults: reward config and ego_shape are required whenever
--avoid_sets is given.
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

from rlvr.autoresearch.tools.eval_det_avoidance import (
    aggregate_stats,
    load_model,
    score_det_scenes,
)
from rlvr.autoresearch.tools.eval_plan_comfort import _dist, eval_plan_comfort
from rlvr.autoresearch.tools.sweep_lora_epochs import merge_epoch, parse_epochs


def run_l2(
    merged: Path, args_json: Path, val_list: Path, port: int, batch_size: int
) -> tuple[float, float]:
    """Run valid_320 (DDP, 1 GPU) on a merged checkpoint; return (ego, neighbor)."""
    repo = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--standalone",
        f"--master_port={port}",
        str(repo / "scripts" / "valid_320.py"),
        "--resume_model_path",
        str(merged),
        "--args_json_path",
        str(args_json),
        "--valid_set_list",
        str(val_list),
        "--batch_size",
        str(batch_size),
        "--ddp",
        "true",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
    m = re.search(r"avg_loss_ego=([\d.]+) avg_loss_neighbor=([\d.]+)", res.stdout)
    if not m:
        raise RuntimeError(
            f"valid_320 produced no avg_loss line for {merged}:\n"
            f"{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
        )
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run_dir", required=True)
    p.add_argument(
        "--base_model",
        required=True,
        help="Warmstart .pth the LoRA trains on (merge base; dir holds args.json)",
    )
    p.add_argument("--epochs", required=True, help='"all" | "1-24" | "2,4,6" | "1-24:2"')
    p.add_argument(
        "--avoid_sets",
        nargs="*",
        default=[],
        help="NAME=scenes.json pairs for deterministic avoidance eval",
    )
    p.add_argument("--config", help="Reward config JSON (required with --avoid_sets)")
    p.add_argument("--ego_shape", help="WB,L,W (required with --avoid_sets)")
    p.add_argument("--comfort_scenes", help="Scene list for open-loop plan comfort")
    p.add_argument("--l2_val", help="Validation list for ego/neighbor L2")
    p.add_argument("--args_json", help="Model args.json (required with --l2_val)")
    p.add_argument("--l2_port", type=int, default=29571)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--no_keep_merged", action="store_true", help="Delete each merged checkpoint after scoring"
    )
    args = p.parse_args()

    if args.avoid_sets and (not args.config or not args.ego_shape):
        p.error("--avoid_sets requires --config and --ego_shape")
    if args.l2_val and not args.args_json:
        p.error("--l2_val requires --args_json")

    run_dir, out_dir = Path(args.run_dir), Path(args.output_dir)
    base_model = Path(args.base_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avoid_sets: dict[str, list[str]] = {}
    for spec in args.avoid_sets:
        name, _, path = spec.partition("=")
        if not path:
            p.error(f"--avoid_sets entry must be NAME=path, got {spec!r}")
        with open(path) as f:
            avoid_sets[name] = json.load(f)
    if avoid_sets:
        from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config

        rcfg = load_reward_config(args.config)
        ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    comfort_scenes = json.load(open(args.comfort_scenes)) if args.comfort_scenes else None

    epochs = parse_epochs(args.epochs, run_dir)
    print(
        f"[full-sweep] epochs {epochs}; avoid sets "
        f"{ {k: len(v) for k, v in avoid_sets.items()} }; "
        f"comfort={len(comfort_scenes) if comfort_scenes else 0}; l2={args.l2_val}"
    )

    summary_path = out_dir / "full_sweep.json"
    rows: list[dict] = json.loads(summary_path.read_text()) if summary_path.exists() else []
    done = {r["epoch"] for r in rows}

    for e in epochs:
        if e in done:
            print(f"[full-sweep] ep{e:03d} already in {summary_path.name}, skipping")
            continue
        ep = f"{e:03d}"
        merged = out_dir / f"merged_ep{ep}" / "best_model.pth"
        print(f"\n[full-sweep] ===== epoch {ep} =====")
        merge_epoch(base_model, run_dir / f"lora_epoch_{ep}", merged)
        row: dict = {"epoch": e}

        model, model_args = load_model(str(merged), device)
        for name, paths in avoid_sets.items():
            results = score_det_scenes(
                model, model_args, paths, rcfg, ego_shape, device, batch_size=args.batch_size
            )
            agg = aggregate_stats(results)
            ep_out = out_dir / f"avoid_{name}_ep{ep}"
            ep_out.mkdir(parents=True, exist_ok=True)
            (ep_out / "det_avoidance_summary.json").write_text(
                json.dumps({"aggregate": agg, "scenes": results}, indent=2)
            )
            scm = agg["sc_min_dist"]
            row[name] = {
                "n": agg["n_scenes"],
                "static_crossings": agg["static_crossings"],
                "rb_crossings": agg["rb_crossings"],
                "lane_crossings": agg["lane_crossings"],
                "sc_min": {k: scm[k] for k in ("mean", "p5", "p25", "p50", "p75", "min")},
            }
            print(
                f"  [{name}] static={agg['static_crossings']}/{agg['n_scenes']} "
                f"rb={agg['rb_crossings']} lane={agg['lane_crossings']} "
                f"sc_min mean={scm['mean']:+.3f} p5={scm['p5']:+.3f} min={scm['min']:+.3f}"
            )
        if comfort_scenes:
            la95, jk95, cspd = eval_plan_comfort(model, model_args, comfort_scenes, device)
            row["plan_comfort"] = {
                "lat_accel_p95": _dist(la95),
                "jerk_p95": _dist(jk95),
                "curve_speed": _dist(cspd),
            }
            la, cs = row["plan_comfort"]["lat_accel_p95"], row["plan_comfort"]["curve_speed"]
            cs_mean = "—" if cs["mean"] is None else f"{cs['mean']:.2f}"
            print(
                f"  [comfort] plan lat_accel p95 mean={la['mean']:.2f} "
                f"p95={la['p95']:.2f} | curve_speed mean={cs_mean}"
            )
        del model
        torch.cuda.empty_cache()

        if args.l2_val:
            ego, nbr = run_l2(
                merged, Path(args.args_json), Path(args.l2_val), args.l2_port, args.batch_size
            )
            row["l2"] = {"ego": ego, "neighbor": nbr}
            print(f"  [l2] ego={ego:.4f} neighbor={nbr:.4f}")

        if args.no_keep_merged:
            shutil.rmtree(merged.parent, ignore_errors=True)
        rows.append(row)
        rows.sort(key=lambda r: r["epoch"])
        summary_path.write_text(json.dumps(rows, indent=2))

    # markdown summary table
    lines = [
        "| ep | "
        + " | ".join(f"{n} static/rb/lane | {n} sc_min mean/p5/min" for n in avoid_sets)
        + (" | plan latA p95 | curve spd |" if comfort_scenes else "")
        + (" ego L2 | nbr L2 |" if args.l2_val else "")
    ]
    lines.append(
        "|"
        + "---|"
        * (1 + 2 * len(avoid_sets) + (2 if comfort_scenes else 0) + (2 if args.l2_val else 0))
    )
    for r in rows:
        cells = [str(r["epoch"])]
        for n in avoid_sets:
            if n in r:
                a = r[n]
                cells.append(
                    f"{a['static_crossings']}/{a['n']} · {a['rb_crossings']} · "
                    f"{a['lane_crossings']}"
                )
                s = a["sc_min"]
                cells.append(f"{s['mean']:+.3f} / {s['p5']:+.3f} / {s['min']:+.3f}")
            else:
                cells += ["—", "—"]
        if comfort_scenes:
            pc = r.get("plan_comfort")
            if pc:
                cs_mean = pc["curve_speed"]["mean"]
                cells.append(f"{pc['lat_accel_p95']['mean']:.2f}")
                cells.append("—" if cs_mean is None else f"{cs_mean:.2f}")
            else:
                cells += ["—", "—"]
        if args.l2_val:
            l2 = r.get("l2")
            cells += [f"{l2['ego']:.4f}", f"{l2['neighbor']:.4f}"] if l2 else ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    (out_dir / "full_sweep.md").write_text(md + "\n")
    print(f"\n{md}\nWrote {summary_path} + full_sweep.md")


if __name__ == "__main__":
    main()
