#!/usr/bin/env python3
"""Classify closed-loop replay steps into training bands by reward.py CL score.

Reads the ``metrics_log.json`` emitted by :mod:`scenario_generation.replay`
(which already logs ``cl_score`` per step — the value
``compute_centerline_score_batch`` produces on the CURRENT ego pose, 1:1
with the training reward signal), bands each step into
``clean / warm / hot / very_hot / lane_change``, and writes per-band
JSON scene-lists pointing at the corresponding ``replay_step_NNNN.npz``
files.

Design notes:

* Banding uses reward.py's own ``cl_score`` so nothing is lost in
  translation between training signal and data mining.
* Stopped scenes are filtered out — ranked SFT on a stopped ego reward-
  hacks trivially.
* Lane-change scenes are heuristically flagged: any run of N
  consecutive steps where ``cl_score`` is saturated (≤ -cap * 0.95)
  for at least ``lane_change_min_steps`` is bucketed separately so it
  doesn't inflate the lane-keeping training set. The same steps are
  ALSO removed from the banded lists.
* Declustering drops scenes that fall within ``decluster_window`` of a
  higher-band scene — so one long drift event contributes a few
  representative NPZs, not 30 near-duplicates.

Usage:
    python -m scenario_generation.tools.classify_replay_steps \\
        --run /path/to/perfect_d0 \\
        --bands_config scenario_generation/configs/classify_bands_default.json \\
        --output_dir /path/to/scene_lists/

The bands_config JSON is REQUIRED — we refuse to mine training data
with silent defaults because band cut-offs directly decide what the
downstream RSFT run learns from. Edit (or copy and edit) the default
config file to change thresholds, don't override on the CLI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_metrics(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics_log.json"
    if not p.exists():
        raise SystemExit(f"metrics_log.json missing at {p}")
    with open(p) as f:
        return json.load(f)["steps"]


def _assign_band(cl: float, bands: dict) -> str | None:
    for name, (lo, hi) in bands.items():
        if lo <= cl < hi:
            return name
    return None


def _ego_speed(run_dir: Path, step: int) -> float:
    """Read the authoritative speed from ``ego_current_state[4]``.

    tensor_converter writes full world-frame speed magnitude into the
    NPZ's ``ego_current_state`` slot 4 (see
    scenario_generation/tensor_converter.py::_build_ego_current_state).
    Don't re-derive from position diffs — that's brittle if the dump
    frame conventions ever change.
    """
    fp = run_dir / "npz" / f"replay_step_{step:04d}.npz"
    with np.load(fp, allow_pickle=True) as d:
        state = d["ego_current_state"]
    return float(state[4])


def _detect_lane_change_runs(cl_scores: np.ndarray,
                             saturated_thresh: float,
                             min_steps: int) -> set[int]:
    """Return the set of step indices that fall inside a saturated-CL run
    of at least ``min_steps`` consecutive samples."""
    sat = cl_scores <= saturated_thresh
    out: set[int] = set()
    i = 0
    n = len(sat)
    while i < n:
        if sat[i]:
            j = i
            while j < n and sat[j]:
                j += 1
            if j - i >= min_steps:
                out.update(range(i, j))
            i = j
        else:
            i += 1
    return out


def _decluster(idx: list[int], window: int) -> list[int]:
    """Keep the first of any cluster of indices closer than ``window`` apart."""
    if not idx:
        return []
    kept = [idx[0]]
    for k in idx[1:]:
        if k - kept[-1] >= window:
            kept.append(k)
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True,
                   help="Replay output dir (contains metrics_log.json + npz/)")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--bands_config", type=Path, required=True,
                   help="Path to JSON with bands + stopped_filter + "
                        "lane_change_detection + decluster_window. See "
                        "scenario_generation/configs/classify_bands_default.json "
                        "— this is REQUIRED (no silent defaults).")
    args = p.parse_args()

    with open(args.bands_config) as f:
        cfg = json.load(f)
    for key in ("bands", "stopped_filter", "lane_change_detection",
                "decluster_window", "gate_filters"):
        if key not in cfg:
            raise SystemExit(
                f"bands_config missing required key '{key}'. "
                f"See scenario_generation/configs/classify_bands_default.json for the full schema."
            )
    bands = {k: tuple(v) for k, v in cfg["bands"].items()}
    min_speed = float(cfg["stopped_filter"]["min_speed_m_s"])
    lc_sat = float(cfg["lane_change_detection"]["saturated_thresh"])
    lc_min = int(cfg["lane_change_detection"]["min_consecutive_steps"])
    decluster_window = int(cfg["decluster_window"])
    gates = cfg["gate_filters"]
    rb_x_excl = bool(gates.get("rb_crossing_excludes", True))
    rb_min_thresh = float(gates.get("rb_min_dist_threshold_m", 0.0))
    lane_x_excl = bool(gates.get("lane_crossing_excludes", False))
    pred_coll_first_n = int(gates.get("pred_collision_first_n_steps", 0))
    print(f"Config: {args.bands_config}")
    print(f"  bands: {bands}")
    print(f"  min_speed={min_speed} m/s  decluster={decluster_window}  "
          f"lane_change_sat={lc_sat}  lane_change_min_steps={lc_min}")
    print(f"  gates: rb_cross_excl={rb_x_excl} rb_min>={rb_min_thresh}  "
          f"lane_cross_excl={lane_x_excl}  pred_coll_first_n={pred_coll_first_n}")

    steps = _load_metrics(args.run)
    cl_scores = np.array([float(s.get("cl_score", 0.0)) for s in steps])
    n = len(steps)
    print(f"Loaded {n} steps from {args.run}")
    print(f"  cl_score: min={cl_scores.min():.3f} p5={np.percentile(cl_scores, 5):.3f} "
          f"p50={np.percentile(cl_scores, 50):.3f} p95={np.percentile(cl_scores, 95):.3f} max={cl_scores.max():.3f}")

    # 1. Lane-change detection before banding so lane-change steps don't
    #    pollute very_hot.
    lc_steps = _detect_lane_change_runs(cl_scores, lc_sat, lc_min)
    print(f"Flagged {len(lc_steps)} steps inside lane-change-like saturation runs")

    # 2. Speed filter: read ego speed from NPZs. For TL runs there are
    #    many stopped frames; for no-TL this is usually a no-op.
    print(f"Reading NPZs for stopped-filter (min_speed={min_speed} m/s)...")
    stopped_steps: set[int] = set()
    for i in range(n):
        v = _ego_speed(args.run, i)
        if v < min_speed:
            stopped_steps.add(i)
    print(f"  stopped: {len(stopped_steps)} / {n} ({100*len(stopped_steps)/n:.1f}%)")

    # 3. Gate-violation filters (reward.py fields in metrics_log, no re-derive).
    #    Drop scenes where the CURRENT ego pose is already over a gate, or
    #    where the model's 80-step prediction crashes in the first N steps.
    gated_steps: set[int] = set()
    for i, s in enumerate(steps):
        if rb_x_excl and s.get("rb_crossing"):
            gated_steps.add(i); continue
        if rb_min_thresh > 0 and float(s.get("rb_min_dist", 1.0)) < rb_min_thresh:
            gated_steps.add(i); continue
        if lane_x_excl and s.get("lane_crossing"):
            gated_steps.add(i); continue
        if pred_coll_first_n > 0:
            pcs = s.get("pred_collision_step")
            if pcs is not None and int(pcs) < pred_coll_first_n:
                gated_steps.add(i); continue
    print(f"  gate-filtered: {len(gated_steps)} / {n} "
          f"({100*len(gated_steps)/n:.1f}%)")

    # 4. Band the remaining steps (drop stopped, gated, and lane-change
    #    from banded training sets; lane-change keeps its own bucket).
    per_band: dict[str, list[int]] = {name: [] for name in bands}
    per_band["lane_change"] = sorted(lc_steps - stopped_steps - gated_steps)
    for i, cl in enumerate(cl_scores):
        if i in stopped_steps or i in gated_steps or i in lc_steps:
            continue
        band = _assign_band(cl, bands)
        if band is None or band == "clean":
            continue
        per_band[band].append(i)

    # 4. Decluster each band.
    print(f"\nPre-decluster counts:")
    for name, ids in per_band.items():
        print(f"  {name}: {len(ids)}")
    for name in list(per_band.keys()):
        if name == "lane_change":
            # keep lane-change dense so we can see where the lane change
            # starts / ends in a diagnostic later.
            continue
        per_band[name] = _decluster(per_band[name], decluster_window)
    print(f"\nPost-decluster counts (window={decluster_window}):")
    for name, ids in per_band.items():
        print(f"  {name}: {len(ids)}")

    # 5. Emit scene-list JSONs and a summary.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = args.run / "npz"
    summary = {
        "run_dir": str(args.run),
        "bands": {k: list(v) for k, v in bands.items()},
        "decluster_window": decluster_window,
        "bands_config": str(args.bands_config),
        "min_speed": min_speed,
        "lane_change_sat_thresh": lc_sat,
        "lane_change_min_steps": lc_min,
        "n_total_steps": n,
        "n_stopped": len(stopped_steps),
        "n_gate_filtered": len(gated_steps),
        "counts": {name: len(ids) for name, ids in per_band.items()},
    }
    for name, ids in per_band.items():
        out = args.output_dir / f"{name}.json"
        paths = [str((npz_dir / f"replay_step_{i:04d}.npz").resolve()) for i in ids]
        with open(out, "w") as f:
            json.dump(paths, f, indent=2)
        print(f"  wrote {len(paths)} scenes → {out}")

    # Composite sets matching the three training candidates
    composites = {
        "candidate_A_warm": ["warm"],
        "candidate_B_warm_hot": ["warm", "hot"],
        "candidate_C_warm_hot_vhot": ["warm", "hot", "very_hot"],
    }
    for cname, bn_list in composites.items():
        merged = sorted({i for bn in bn_list for i in per_band[bn]})
        paths = [str((npz_dir / f"replay_step_{i:04d}.npz").resolve()) for i in merged]
        out = args.output_dir / f"{cname}.json"
        with open(out, "w") as f:
            json.dump(paths, f, indent=2)
        print(f"  wrote {len(paths)} scenes → {out}")
        summary["counts"][cname] = len(paths)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
