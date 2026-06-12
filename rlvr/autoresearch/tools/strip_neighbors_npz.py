#!/usr/bin/env python3
"""Counterfactual zero-target variants: copy scenes with ALL neighbors removed.

For guidance-explorer training: an avoidance scene with its neighbors
stripped has nothing to avoid, so the policy's correct output there is
exactly eta = 0. Pairing each solved avoidance scene with its neighbor-less
twin teaches the policy to key on the actual obstacles rather than the road
geometry (the observed failure mode: inertness leaks on unfamiliar normals,
e.g. firing on intersection turns with nothing nearby).

Zeroed fields: neighbor_agents_past, neighbor_agents_future (zero rows are
empty slots — the same convention as padding). Everything else is copied
verbatim.

Usage:
    python -m rlvr.autoresearch.tools.strip_neighbors_npz \
        --scenes <scenes.json> --out_dir <dir> --out_list <list.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_list", required=True)
    args = parser.parse_args()

    with open(args.scenes) as f:
        paths = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for sp in paths:
        raw = dict(np.load(sp, allow_pickle=True))
        if "neighbor_agents_past" not in raw:
            raise ValueError(f"{sp}: missing neighbor_agents_past")
        raw["neighbor_agents_past"] = np.zeros_like(raw["neighbor_agents_past"])
        if "neighbor_agents_future" in raw:
            raw["neighbor_agents_future"] = np.zeros_like(raw["neighbor_agents_future"])
        pool = Path(sp).parent.name
        out_path = out_dir / f"{pool}__{Path(sp).stem}_nonbr.npz"
        np.savez(out_path, **raw)
        written.append(str(out_path))

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    print(f"[strip] {len(written)} neighbor-less variants -> {args.out_list}")


if __name__ == "__main__":
    main()
