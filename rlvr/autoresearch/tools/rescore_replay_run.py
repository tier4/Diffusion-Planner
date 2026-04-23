#!/usr/bin/env python3
"""Recompute ``metrics_log.json`` for a ``scenario_generation.replay`` run.

Two modes, picked explicitly on the CLI — no fallbacks, no silent
defaults:

``--mode instant``
    Scores each dumped NPZ against the reward primitives using only the
    current ego pose (the same data the live sim already produces when
    ``_score_step`` is called with ``prediction=None``). Fast, no model
    needed. Use case: a reward component was added or a threshold
    changed, and you want to regenerate the log without re-running the
    MPC replay.

``--mode full``
    ``instant`` + the ``pred_*`` block produced by running model
    inference on each NPZ and scoring the resulting 80-step prediction.
    Requires ``--model_path``. Use case: evaluate how a DIFFERENT model
    would have planned given the observations the baseline run dumped.
    The instantaneous block is model-independent, so it stays identical
    across model comparisons — only ``pred_*`` differs. Load two metrics
    logs into scene_search (or diff them offline) to compare plan
    quality at matching world positions.

Important: the dumped NPZs are the closed-loop trajectory the *baseline*
model actually drove. Post-hoc rescoring with a different model
produces the single-step open-loop prediction that model would have
emitted from the same observation — it does NOT simulate what the
alternative model would have done if rolled out closed-loop from the
same seed (that still requires ``scenario_generation.replay``).

Usage:
    python -m rlvr.autoresearch.tools.rescore_replay_run \\
        --run_dir /path/to/mpc_gen_run/ \\
        --config  /path/to/grpo_config.json \\
        --mode    instant \\
        [--output /path/to/new_metrics_log.json]

    python -m rlvr.autoresearch.tools.rescore_replay_run \\
        --run_dir /path/to/mpc_gen_run/ \\
        --config  /path/to/grpo_config.json \\
        --mode    full \\
        --model_path /path/to/model.pth \\
        [--output /path/to/new_metrics_log.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from scenario_generation.replay import SpawnConfig, _score_step


_NPZ_RE = re.compile(r"replay_step_(\d+)\.npz$")


def _score_instant(npz_paths, device, reward_cfg, spawn_cfg) -> list[dict]:
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
            print(f"  Scored {i+1}/{len(npz_paths)} (instant)")
    return steps


def _score_full(npz_paths, device, reward_cfg, spawn_cfg, model_path) -> list[dict]:
    # Lazy imports: ROS / scene_context are only needed in full mode.
    from scenario_generation.npz_loader import from_npz
    from scenario_generation.simulate import _predict_batch, load_model

    print(f"  Loading model {model_path}")
    model, model_args = load_model(str(model_path), device)

    steps: list[dict] = []
    for i, path in enumerate(npz_paths):
        m = _NPZ_RE.search(path.name)
        if not m:
            continue
        step = int(m.group(1))
        with np.load(path, allow_pickle=True) as raw:
            data = {k: raw[k] for k in raw.files if k != "version"}

        scene = from_npz(str(path))
        preds = _predict_batch(
            model, model_args, scene, [scene.ego_agent_id], device,
            inference_delay=spawn_cfg.inference_delay,
        )
        ego_pred = preds.get(scene.ego_agent_id)
        if ego_pred is None:
            raise SystemExit(
                f"Inference returned no prediction for ego at {path} — refusing "
                f"to emit a partial metrics_log. Check that the NPZ has a valid "
                f"ego_agent_past."
            )
        steps.append(_score_step(
            data, step, device, reward_cfg, spawn_cfg, prediction=ego_pred,
        ))
        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(npz_paths)} (full)")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Replay output directory (must contain npz/).")
    parser.add_argument("--config", type=Path, required=True,
                        help="Reward config JSON (no silent defaults).")
    parser.add_argument("--mode", choices=["instant", "full"], required=True,
                        help="'instant' = current-pose metrics only (no model "
                             "needed). 'full' = instantaneous + pred_* from a "
                             "specified model (inference per NPZ).")
    parser.add_argument("--model_path", type=Path, default=None,
                        help="Model checkpoint. REQUIRED when --mode full; "
                             "MUST NOT be supplied when --mode instant.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output metrics log path "
                             "(default: <run_dir>/metrics_log.json).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch / numpy / random before loading the "
                             "model (full mode only). Match the sim's seed to "
                             "make post-hoc predictions as close as possible "
                             "to the live run — still not bit-identical "
                             "because the live process advanced RNG state "
                             "through map builds / MPC solves / etc. between "
                             "inferences, which this tool doesn't replay.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Mode / model_path consistency — explicit, no fallbacks.
    if args.mode == "full" and args.model_path is None:
        parser.error("--mode full requires --model_path")
    if args.mode == "instant" and args.model_path is not None:
        parser.error(
            "--mode instant refuses --model_path (prediction scoring requires "
            "--mode full; passing a model here would silently be ignored). "
            "Drop --model_path or switch to --mode full."
        )

    run_dir = args.run_dir
    npz_dir = run_dir / "npz"
    if not npz_dir.is_dir():
        raise SystemExit(f"{npz_dir} missing")

    spawn_cfg_path = run_dir / "spawn_config.json"
    if not spawn_cfg_path.exists():
        raise SystemExit(
            f"{spawn_cfg_path} missing — the tool needs the spawn config to "
            f"read ego dimensions and inference_delay. Copy the one from the "
            f"original run."
        )
    spawn_cfg = SpawnConfig.from_json(spawn_cfg_path)

    reward_cfg = load_reward_config(args.config)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    npz_paths = sorted(npz_dir.glob("replay_step_*.npz"))
    if not npz_paths:
        raise SystemExit(f"no replay_step_*.npz in {npz_dir}")

    if args.mode == "instant":
        steps = _score_instant(npz_paths, device, reward_cfg, spawn_cfg)
    else:
        steps = _score_full(npz_paths, device, reward_cfg, spawn_cfg, args.model_path)

    out_path = args.output or (run_dir / "metrics_log.json")
    payload = {
        "reward_config_path": str(args.config),
        "dump_npz_dir": str(npz_dir),
        "ego_shape": [
            spawn_cfg.ego_wheelbase, spawn_cfg.ego_length, spawn_cfg.ego_width,
        ],
        "mode": args.mode,
        "model_path": str(args.model_path) if args.model_path else None,
        "steps": steps,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved {len(steps)} step records to {out_path}")


if __name__ == "__main__":
    main()
