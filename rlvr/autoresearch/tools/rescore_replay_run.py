#!/usr/bin/env python3
"""Recompute ``metrics_log.json`` for a ``scenario_generation.replay`` run.

Motivation: when a reward component is added or its thresholds change, the
training-time pipeline picks the new behaviour up immediately, but any
already-dumped replay run keeps its old metrics log. Re-running the full
MPC replay just to regenerate scores is wasteful — the NPZs the metrics
depend on are already on disk. This tool reuses
``scenario_generation.replay._score_step`` to rescore every NPZ in place
and overwrite ``metrics_log.json``.

Usage:
    python -m rlvr.autoresearch.tools.rescore_replay_run \\
        --run_dir /path/to/mpc_gen_run/ \\
        --config  /path/to/grpo_config.json \\
        [--output /path/to/new_metrics_log.json]  # default: run_dir/metrics_log.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from scenario_generation.replay import SpawnConfig, _score_step


_NPZ_RE = re.compile(r"replay_step_(\d+)\.npz$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Replay output directory (must contain npz/).")
    parser.add_argument("--config", type=Path, required=True,
                        help="Reward config JSON (no silent defaults).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output metrics log path "
                             "(default: <run_dir>/metrics_log.json).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_dir = args.run_dir
    npz_dir = run_dir / "npz"
    if not npz_dir.is_dir():
        raise SystemExit(f"{npz_dir} missing")

    spawn_cfg_path = run_dir / "spawn_config.json"
    if not spawn_cfg_path.exists():
        raise SystemExit(f"{spawn_cfg_path} missing — needed for ego shape")
    spawn_cfg = SpawnConfig.from_json(spawn_cfg_path)

    reward_cfg = load_reward_config(args.config)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    npz_paths = sorted(npz_dir.glob("replay_step_*.npz"))
    if not npz_paths:
        raise SystemExit(f"no replay_step_*.npz in {npz_dir}")

    steps: list[dict] = []
    for i, path in enumerate(npz_paths):
        m = _NPZ_RE.search(path.name)
        if not m:
            continue
        step = int(m.group(1))
        with np.load(path, allow_pickle=True) as raw:
            data = {k: raw[k] for k in raw.files if k != "version"}
        steps.append(_score_step(data, step, device, reward_cfg, spawn_cfg))
        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(npz_paths)}")

    out_path = args.output or (run_dir / "metrics_log.json")
    payload = {
        "reward_config_path": str(args.config),
        "dump_npz_dir": str(npz_dir),
        "ego_shape": [
            spawn_cfg.ego_wheelbase, spawn_cfg.ego_length, spawn_cfg.ego_width,
        ],
        "steps": steps,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved {len(steps)} step records to {out_path}")


if __name__ == "__main__":
    main()
