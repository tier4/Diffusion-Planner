#!/usr/bin/env python3
"""Run L2 validation on one or more merged model .pth files.

Usage:
    python -m rlvr.autoresearch.tools.run_l2_eval \
        --args_json <args.json> --val_set <path_list.json> \
        <merged1.pth> [merged2.pth ...]

For each model:
  1. Runs valid_predictor.py via torchrun (DDP mode, batch_size=32)
  2. Saves predictions to <model_dir>/l2_<stem>/
  3. Saves full output to <model_dir>/l2_<stem>.log
  4. Prints summary: ego L2, neighbor L2, % change from baseline
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

BASELINE_EGO = 5.388
BASELINE_NEIGH = 4.393


def run_eval(model_pth: Path, args_json: Path, val_set: Path, port: int) -> dict | None:
    stem = model_pth.stem
    outdir = model_pth.parent / f"l2_{stem}"
    logfile = model_pth.parent / f"l2_{stem}.log"

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--standalone",
        f"--master_port={port}",
        "diffusion_planner/valid_predictor.py",
        "--resume_model_path",
        str(model_pth),
        "--args_json_path",
        str(args_json),
        "--valid_set_list",
        str(val_set),
        "--batch_size",
        "32",
        "--ddp",
        "true",
        "--save_predictions_dir",
        str(outdir),
    ]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    with open(logfile, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}, see {logfile}")
        return None

    return parse_log(logfile)


def parse_log(logfile: Path) -> dict | None:
    text = logfile.read_text()
    ego_m = re.search(r"avg_loss_ego=([0-9.]+)", text)
    neigh_m = re.search(r"avg_loss_neighbor=([0-9.]+)", text)
    if not ego_m or not neigh_m:
        return None
    return {"ego": float(ego_m.group(1)), "neigh": float(neigh_m.group(1))}


def fmt_pct(val: float, baseline: float) -> str:
    return f"{(val / baseline - 1) * 100:+.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Run L2 validation on merged checkpoints")
    parser.add_argument("models", nargs="+", type=Path, help="Merged .pth files")
    parser.add_argument("--args_json", type=Path, required=True, help="Base model args.json")
    parser.add_argument("--val_set", type=Path, required=True, help="Validation path_list.json")
    parser.add_argument("--port", type=int, default=29505, help="torchrun master port")
    parser.add_argument("--baseline_ego", type=float, default=BASELINE_EGO)
    parser.add_argument("--baseline_neigh", type=float, default=BASELINE_NEIGH)
    args = parser.parse_args()

    print("=" * 64)
    print(f"L2 Evaluation")
    print(f"  args_json: {args.args_json}")
    print(f"  val_set:   {args.val_set}")
    print("=" * 64)

    for i, model_pth in enumerate(args.models):
        stem = model_pth.stem
        print(f"\n--- {stem} ---")
        result = run_eval(model_pth, args.args_json, args.val_set, args.port + i)
        if result:
            ego_pct = fmt_pct(result["ego"], args.baseline_ego)
            neigh_pct = fmt_pct(result["neigh"], args.baseline_neigh)
            print(
                f"  ego={result['ego']:.4f} ({ego_pct})  neigh={result['neigh']:.4f} ({neigh_pct})"
            )
        else:
            logfile = model_pth.parent / f"l2_{stem}.log"
            print(f"  ERROR: could not parse results from {logfile}")

    print(f"\n{'=' * 64}")
    print("Done. Logs saved next to each .pth file.")
    print("=" * 64)


if __name__ == "__main__":
    main()
