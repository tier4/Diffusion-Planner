"""Extract post-event RELEASE BANDS for every mined R2LPL event.

The mining/repair pipeline carves credit windows that END at (or before) the
offense, so every repaired target teaches the entry into a braking manoeuvre
and never its resolution — the recorded driver's release/re-acceleration lies
just past the window edge in the source logs. A corpus built only from those
windows is systematically deceleration-biased, and post-training on it erodes
take-off behaviour a little more every epoch. Pairing each event's credit
window with frames from the SAME event's post-offense band restores the other
half of the manoeuvre, using recorded futures only — nothing is generated,
relabeled, or filtered out of the repaired set itself.

For each event row in a mining run's credit_windows.jsonl, this tool locates
the source sequence via the chunk manifest, then walks recorded frames from
the offense forward through ``--post_window_s`` seconds at ``--stride_s``
spacing. Traffic-light alignment decides what each frame may teach:

  * route lane RED (and not green)  -> keep unconditionally (patience);
  * route lane GREEN or no route TL -> keep only when the recorded future
    travels at least ``--min_takeoff_travel_m`` (a genuine release — a creep
    would re-teach the bias this tool exists to cancel).

Every threshold is required; the tool fails loudly on unmatched chunk keys.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# route_lanes feature columns 8..12 carry the traffic-light one-hot
# [green, yellow, red, white, none] (diffusion_planner.dimensions).
_TL_SLICE = slice(8, 13)
_TL_GREEN = 0
_TL_RED = 2

_FRAME_RE = re.compile(r"_(\d+)\.npz$")


def _chunk_key_of_event(event_key: str) -> str:
    """``<chunk_key>_<idx>_<step>_danger_<label>`` -> ``<chunk_key>``.

    Chunk keys themselves contain underscores (log names do), so strip the
    known suffixes from the right instead of splitting from the left.
    """
    stem = event_key.rsplit("_danger_", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"unparseable event_key: {event_key!r}")
    return "_".join(parts[:-2])


def _sequence_index(scene_path: str) -> tuple[str, dict[int, str]]:
    d = os.path.dirname(scene_path)
    idx: dict[int, str] = {}
    for f in os.listdir(d):
        m = _FRAME_RE.search(f)
        if m and f.endswith(".npz"):
            idx[int(m.group(1))] = os.path.join(d, f)
    if not idx:
        raise ValueError(f"no frame-indexed .npz files next to {scene_path}")
    return d, idx


def _pathlen(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _route_tl_state(d: np.lib.npyio.NpzFile) -> tuple[bool, bool]:
    rl = d["route_lanes"]
    valid = np.abs(rl).sum(axis=(1, 2)) > 0
    if not valid.any():
        return False, False
    onehot = rl[valid][:, 0, _TL_SLICE]
    hot = onehot.max(axis=1) > 0
    cls = onehot.argmax(axis=1)
    green = bool((hot & (cls == _TL_GREEN)).any())
    red = bool((hot & (cls == _TL_RED)).any())
    return green, red


def _band_worker(
    args: tuple[str, list[int], int, int, float],
) -> list[str]:
    start_scene_path, offense_frames, post_frames, stride_frames, min_takeoff_m = args
    _, idx = _sequence_index(start_scene_path)
    last_frame = max(idx)
    out: list[str] = []
    for f0 in offense_frames:
        for f in range(f0, min(f0 + post_frames, last_frame) + 1, stride_frames):
            p = idx.get(f)
            if p is None:
                continue
            try:
                with np.load(p) as d:
                    green, red = _route_tl_state(d)
                    if red and not green:
                        out.append(p)
                        continue
                    fut = _pathlen(d["ego_agent_future"][:, :2])
            except Exception:
                continue
            if fut >= min_takeoff_m:
                out.append(p)
    return out


def _load_chunk_manifest(chunk_manifest: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    with open(chunk_manifest) as fh:
        for line in fh:
            row = json.loads(line)
            manifest[str(row["chunk_key"])] = str(row["start_scene_path"])
    if not manifest:
        raise ValueError(f"{chunk_manifest} is empty")
    return manifest


def _load_event_offenses(credit_windows: Path) -> dict[str, set[int]]:
    events: dict[str, set[int]] = {}
    with open(credit_windows) as fh:
        for line in fh:
            row = json.loads(line)
            key = _chunk_key_of_event(str(row["event_key"]))
            events.setdefault(key, set()).add(int(row["offense_frame_id"]))
    if not events:
        raise ValueError(f"{credit_windows} contains no event rows")
    return events


def build_release_bands(
    credit_windows: Path,
    chunk_manifest: Path,
    out: Path,
    *,
    post_window_s: float,
    stride_s: float,
    min_takeoff_travel_m: float,
    frame_hz: float,
    workers: int,
) -> list[str]:
    if post_window_s <= 0 or stride_s <= 0 or frame_hz <= 0:
        raise ValueError("post_window_s, stride_s and frame_hz must be > 0")
    if min_takeoff_travel_m <= 0:
        raise ValueError(f"min_takeoff_travel_m must be > 0: {min_takeoff_travel_m}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1: {workers}")

    manifest = _load_chunk_manifest(chunk_manifest)
    events = _load_event_offenses(credit_windows)
    unmatched = sorted(k for k in events if k not in manifest)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)}/{len(events)} event chunk keys are absent from "
            f"{chunk_manifest} (first: {unmatched[0]!r}) — wrong manifest for this run?"
        )

    post_frames = int(round(post_window_s * frame_hz))
    stride_frames = max(1, int(round(stride_s * frame_hz)))
    jobs = [
        (manifest[key], sorted(frames), post_frames, stride_frames, min_takeoff_travel_m)
        for key, frames in sorted(events.items())
    ]
    rows: set[str] = set()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for result in ex.map(_band_worker, jobs, chunksize=8):
            rows.update(result)
    kept = sorted(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(kept))
    tmp.replace(out)
    print(
        f"[release-bands] {len(kept)} band rows from {len(events)} event chunks "
        f"(window {post_window_s}s, stride {stride_s}s, takeoff >= {min_takeoff_travel_m} m)"
    )
    return kept


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credit_windows", type=Path, required=True)
    parser.add_argument("--chunk_manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--post_window_s", type=float, required=True)
    parser.add_argument("--stride_s", type=float, required=True)
    parser.add_argument("--min_takeoff_travel_m", type=float, required=True)
    parser.add_argument("--frame_hz", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)
    build_release_bands(
        args.credit_windows,
        args.chunk_manifest,
        args.out,
        post_window_s=args.post_window_s,
        stride_s=args.stride_s,
        min_takeoff_travel_m=args.min_takeoff_travel_m,
        frame_hz=args.frame_hz,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
