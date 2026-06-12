#!/usr/bin/env python3
"""Repair NPZs whose STOPPED neighbours have empty future tracks.

The reward/label pipeline classifies static obstacles through
``neighbor_agents_future`` validity; extraction sometimes leaves a parked
vehicle with an all-zero future track, making it INVISIBLE to sweeps,
margin labels and reward-based metrics (sc_n_stopped=0 -> "already_clean"
-> zero-target) while past-based eval extraction still sees it — i.e. the
policy is actively taught that the scene is empty road, then scored for
crashing into the car it was never told about.

Fix: for every neighbour that is (a) valid in the past, (b) stopped by past
displacement (< --stop_disp over the last second), and (c) has zero valid
future steps, fill the future track with its current pose repeated — the
physically exact future of a parked vehicle. All other fields copied
verbatim; output mirrors the input list with pool-prefixed filenames.

Usage:
    python -m rlvr.autoresearch.tools.pad_static_neighbor_futures \
        --scenes <list.json> --out_dir <dir> --out_list <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_list", required=True)
    parser.add_argument("--stop_disp", type=float, default=0.5,
                        help="max displacement (m) over the last 1 s of past "
                             "to classify a neighbour as stopped")
    args = parser.parse_args()

    with open(args.scenes) as f:
        paths = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, n_padded_total = [], 0
    for sp in paths:
        raw = dict(np.load(sp, allow_pickle=True))
        nb = raw["neighbor_agents_past"]
        nf = np.array(raw["neighbor_agents_future"], copy=True)
        n_padded = 0
        last = nb[:, -1, :]
        valid = np.abs(last[:, :2]).sum(axis=1) > 0.1
        for i in np.nonzero(valid)[0]:
            past_ok = np.abs(nb[i, -11, :2]).sum() > 0
            stopped = past_ok and float(
                np.linalg.norm(nb[i, -1, :2] - nb[i, -11, :2])) < args.stop_disp
            fut_valid = (np.abs(nf[i, :, :2]).sum(axis=1) > 0.1).sum()
            if stopped and fut_valid == 0:
                # Future of a parked car = current pose, repeated,
                # zero-velocity by identity. Past frames are
                # (x, y, cos, sin, ...); the future is either 4-col
                # (x, y, cos, sin) — copy directly — or 3-col
                # (x, y, heading_rad) — recover the angle, NEVER copy
                # the past's cos into the radians slot.
                T, F = nf.shape[1], nf.shape[2]
                if F == 4:
                    row = last[i, :4]
                elif F == 3:
                    row = np.array([last[i, 0], last[i, 1],
                                    np.arctan2(last[i, 3], last[i, 2])])
                else:
                    raise ValueError(
                        f"{sp}: unsupported neighbor_agents_future width {F} "
                        "(expected 3 or 4)")
                nf[i, :, :] = np.tile(row, (T, 1))
                n_padded += 1
        if n_padded:
            raw["neighbor_agents_future"] = nf.astype(
                np.array(raw["neighbor_agents_future"]).dtype)
        pool = Path(sp).parent.name
        out_path = out_dir / f"{pool}__{Path(sp).stem}.npz"
        np.savez(out_path, **raw)
        written.append(str(out_path))
        n_padded_total += n_padded
        if n_padded:
            print(f"  [pad ] {pool}/{Path(sp).name}: {n_padded} neighbour(s)")

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    print(f"\nWrote {len(written)} scenes ({n_padded_total} futures padded) "
          f"-> {args.out_list}")


if __name__ == "__main__":
    main()
