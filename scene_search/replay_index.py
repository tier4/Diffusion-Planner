"""Build a scene_search spatial index from a `scenario_generation.replay` run.

A replay run does NOT emit per-NPZ JSON sidecars (``dump_step_npz`` writes
observation tensors only); instead it writes ``trajectory_log.json`` and —
when live metric logging is enabled — ``metrics_log.json`` alongside the
``npz/`` dump. This module joins the two logs into the same index-entry
format the rest of scene_search expects, plus an extra ``metrics`` field
carrying the per-step drift scores that power the heatmap overlay and the
reward-threshold constraint.

Each entry mirrors ``search_scenes.read_sidecar``:
    {
        "npz_path": ...,
        "x": ..., "y": ..., "heading_deg": ...,
        "timestamp": ...,
        "metrics": {lane_gate, lane_near_frac, rb_min_dist, cl_score, ...},
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_replay_run(run_dir: str | Path) -> list[dict]:
    """Load all per-step entries from one replay run directory.

    Args:
        run_dir: Output directory of ``scenario_generation.replay``. Must
            contain ``trajectory_log.json``; ``metrics_log.json`` and
            ``npz/replay_step_NNNN.npz`` are optional (steps with missing
            NPZs are dropped, missing metrics default to None).

    Returns:
        List of per-step index entries.
    """
    run = Path(run_dir)
    traj_path = run / "trajectory_log.json"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"{traj_path} missing; --replay_runs expects a run dir emitted "
            f"by scenario_generation.replay."
        )
    trajectory = _load_json(traj_path)

    metrics_by_step: dict[int, dict] = {}
    metrics_path = run / "metrics_log.json"
    if metrics_path.exists():
        payload = _load_json(metrics_path)
        # Newer replay writes {"steps": [...], ...}; older raw-list format
        # is tolerated so old dumps still load.
        steps = payload["steps"] if isinstance(payload, dict) else payload
        for rec in steps:
            s = int(rec["step"])
            # Copy every metric field the replay emits. Missing fields in
            # older logs are simply absent from the dict; downstream code
            # reads with .get() and treats None/missing as "no data".
            metrics_by_step[s] = {k: v for k, v in rec.items() if k != "step"}

    npz_dir = run / "npz"
    entries: list[dict] = []
    for rec in trajectory:
        step = int(rec["step"])
        npz_path = npz_dir / f"replay_step_{step:04d}.npz"
        if not npz_path.exists():
            continue
        entries.append({
            "npz_path": str(npz_path),
            "x": float(rec["x"]),
            "y": float(rec["y"]),
            "heading_deg": math.degrees(float(rec["heading"])),
            "timestamp": float(step) * 0.1,  # dt=0.1 s per step
            "metrics": metrics_by_step.get(step, {}),
            "replay_run": str(run),
            "replay_step": step,
        })
    return entries


def load_replay_runs(run_dirs: list[str | Path]) -> list[dict]:
    """Concatenate entries from multiple replay runs. Run dirs are disjoint
    NPZ namespaces, so no de-duplication is needed."""
    out: list[dict] = []
    for rd in run_dirs:
        out.extend(load_replay_run(rd))
    return out
