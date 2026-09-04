"""Build a RESUME-focused patience benchmark from the frozen waits pool.

Selects scenes where the recorded ego is quasi-stationary at t0 (speed <
stop_speed over the first hold_steps of the GT future) and RESUMES (speed >=
resume_speed sustained for sustain_steps) within the 80-step future — the
take-off decision the onset-preferred benchmark cannot see (its GT rarely
resumes in-horizon, hence resume_onset_delay = null for every model).
Event-dedup: one scene per (log dir, frame-window), preferring the offset
whose GT resume happens mid-horizon (max headroom on both sides).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

DT = 0.1


def gt_speed(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    fut = d["ego_agent_future"][:, :2]
    cur = d["ego_current_state"][:2]
    pts = np.vstack([cur, fut])
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop_speed", type=float, required=True)
    ap.add_argument("--resume_speed", type=float, required=True)
    ap.add_argument("--hold_steps", type=int, required=True)
    ap.add_argument("--sustain_steps", type=int, required=True)
    ap.add_argument("--max_scenes", type=int, required=True)
    ap.add_argument("--event_frame_gap", type=int, required=True)
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    cands = []
    bad = 0
    for p in pool:
        try:
            sp = gt_speed(p)
        except Exception:
            bad += 1
            continue
        if sp[: args.hold_steps].max() >= args.stop_speed:
            continue
        moving = sp >= args.resume_speed
        # first index where `sustain_steps` consecutive moving samples begin
        run = np.convolve(moving.astype(int), np.ones(args.sustain_steps, int), "valid")
        hits = np.where(run == args.sustain_steps)[0]
        if len(hits) == 0:
            continue
        onset = int(hits[0])
        # mid-horizon preference: distance from ideal center step 40
        cands.append((p, onset, abs(onset - 40)))

    # event dedup by log dir + frame index proximity
    def key(p):
        m = re.search(r"_(\d+)\.npz$", p)
        if m is None:
            # A silent frame-0 fallback collapses all unparseable scenes of a log
            # dir into one "event" and dedups unrelated scenes against each other.
            raise ValueError(f"cannot parse trailing frame index from {p}")
        return str(Path(p).parent), int(m.group(1))

    cands.sort(key=lambda c: (key(c[0])[0], key(c[0])[1]))
    by_dir = {}
    for p, onset, score in cands:
        dirn, frame = key(p)
        group = by_dir.setdefault(dirn, [])
        merged = False
        for g in group:
            if abs(frame - g["frame"]) < args.event_frame_gap:
                if score < g["score"]:
                    g.update(path=p, frame=frame, onset=onset, score=score)
                merged = True
                break
        if not merged:
            group.append({"path": p, "frame": frame, "onset": onset, "score": score})

    if pool and not cands:
        raise RuntimeError(
            f"no resume candidates in a pool of {len(pool)} scenes ({bad} unreadable) — "
            "wrong pool, unreadable NPZs, or thresholds no scene satisfies"
        )
    events = [g for group in by_dir.values() for g in group]
    events.sort(key=lambda g: g["score"])
    picked = [g["path"] for g in events[: args.max_scenes]]
    json.dump(picked, open(args.out, "w"), indent=1)
    print(
        f"pool={len(pool)} unreadable={bad} resume_candidates={len(cands)} "
        f"events={len(events)} picked={len(picked)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
