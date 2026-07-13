#!/usr/bin/env python3
"""Patience-onset evaluation: does the det plan stop when/where the GT stops?

Runs deterministic inference on a patience benchmark (see
``build_patience_benchmark``) for one or two models and reports, per scene and
aggregated, over the full 8 s horizon:

  * fail_to_stop  — GT reaches < ``stop_speed`` m/s but the det plan never does
  * stop_onset_delay_s — det stop-onset time minus GT stop-onset time
  * over_distance_m — det 8 s path length minus GT's

This is the OPEN-LOOP guard of the patience benchmark; pair it with a
closed-loop probe (``valid_predictor_closed_loop`` on the same scenes) — a
model can pass open-loop (conditioned on GT history) yet still creep in closed
loop where per-step biases compound.

Usage:
    python -m rlvr.autoresearch.tools.eval_patience_onset \
        --model <a.pth> [--model_b <b.pth>] --scenes <benchmark.json> --out <report.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model

_DT = 0.1


def _stop_metrics(xy: np.ndarray, stop_speed: float) -> dict:
    v = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / _DT
    stopped = v < stop_speed
    return {
        "stops": bool(stopped.any()),
        "stop_onset_s": float(np.argmax(stopped) * _DT) if stopped.any() else None,
        "dist_m": float(v.sum() * _DT),
        "min_v": float(v.min()),
    }


def _eval_model(model_path: str, datas: list[dict], device: torch.device, batch: int):
    model, margs = load_model(model_path, device)
    outs = []
    for i in range(0, len(datas), batch):
        ego = det_inference_batched(model, margs, datas[i : i + batch], device)
        outs.append(ego[..., :2].cpu().numpy())
    del model
    return np.concatenate(outs)


def evaluate(
    model_paths: dict[str, str],
    scene_paths: list[str],
    *,
    stop_speed: float,
    device: torch.device,
    batch: int,
) -> dict:
    datas = [load_npz_data(p, device) for p in scene_paths]
    gt = {
        p: d["ego_agent_future"][0, :, :2].detach().cpu().numpy()
        for p, d in zip(scene_paths, datas)
    }
    preds = {label: _eval_model(mp, datas, device, batch) for label, mp in model_paths.items()}

    rows = []
    for i, p in enumerate(scene_paths):
        g = _stop_metrics(gt[p], stop_speed)
        row: dict = {"scene": p, "gt": g}
        for label in model_paths:
            m = _stop_metrics(preds[label][i], stop_speed)
            m["fail_to_stop"] = bool(g["stops"] and not m["stops"])
            m["stop_onset_delay_s"] = (
                round(m["stop_onset_s"] - g["stop_onset_s"], 2)
                if g["stops"] and m["stops"] and g["stop_onset_s"] is not None
                else None
            )
            m["over_distance_m"] = round(m["dist_m"] - g["dist_m"], 2)
            row[label] = m
        rows.append(row)

    summary: dict = {"n_scenes": len(rows), "stop_speed_mps": stop_speed}
    for label in model_paths:
        fails = sum(1 for r in rows if r[label]["fail_to_stop"])
        delays = [
            r[label]["stop_onset_delay_s"]
            for r in rows
            if r[label]["stop_onset_delay_s"] is not None
        ]
        overs = [r[label]["over_distance_m"] for r in rows]
        summary[label] = {
            "fail_to_stop": fails,
            "stop_onset_delay_mean_s": round(float(np.mean(delays)), 3) if delays else None,
            "stop_onset_delay_p95_s": round(float(np.percentile(delays, 95)), 3)
            if delays
            else None,
            "over_distance_mean_m": round(float(np.mean(overs)), 3),
            "over_distance_p95_m": round(float(np.percentile(overs, 95)), 3),
        }
    return {"summary": summary, "scenes": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint to evaluate")
    ap.add_argument("--model_b", help="optional second checkpoint (e.g. the baseline)")
    ap.add_argument("--scenes", required=True, help="patience benchmark JSON")
    ap.add_argument("--out", required=True, help="report JSON")
    ap.add_argument("--stop_speed", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    scene_paths = [str(p) for p in json.load(open(args.scenes))]
    if not scene_paths:
        raise ValueError(f"--scenes list is empty: {args.scenes}")
    model_paths = {"model": args.model}
    if args.model_b:
        model_paths["model_b"] = args.model_b

    report = evaluate(
        model_paths,
        scene_paths,
        stop_speed=args.stop_speed,
        device=torch.device(args.device),
        batch=args.batch_size,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
