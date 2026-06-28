"""Pre-filter a scene list by the converter's per-frame skip flag (standalone pre-pass).

When a corpus is generated with the cpp converter's ``--write_skipped_npz=1`` (so the
closed-loop reproducer gets a gap-free timeline), every 10 Hz frame is written,
including the ones the production filter would normally drop (stopped at a red/yellow
light, no future progress, GT collision, off-lane, stale data). Each frame's JSON
sidecar carries ``is_skipped``. TRAINING must not learn from those frames.

The training/eval loaders (``DiffusionPlannerData``) no longer filter at load time, so
run this script ONCE up front to produce a pre-filtered scene list on disk, then point
training/eval at that filtered list.

It is a thin wrapper over the same shared helper used everywhere else
(``diffusion_planner.utils.scene_skip.filter_scene_list``), so the result is byte-for-byte
the same as the old in-loader filtering. Input and output are both a flat JSON list of
npz paths (the format ``DiffusionPlannerData`` consumes, e.g. ``path_list_valid.json``).

Backward-compatible: a frame with no resolvable sidecar (older corpora) is treated as NOT
skipped, so existing lists pass through unchanged.

Example::

    python ros_scripts/filter_scene_list.py \
        --scenes train_all.json --sidecar_root /path/to/npz_dir \
        --out train_all_noskip.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffusion_planner.utils.scene_skip import filter_scene_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenes", type=Path, required=True, help="input scene list (.json: flat list of npz paths)"
    )
    p.add_argument("--out", type=Path, required=True, help="output filtered scene list")
    p.add_argument(
        "--sidecar_root",
        type=Path,
        default=None,
        help="root of pose/skip JSON sidecars if not next to the NPZ (e.g. the "
        "pre-padding conversion tree when the padded NPZs dropped their sidecars)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenes_path = args.scenes
    out_path = args.out
    sidecar_root = args.sidecar_root

    scenes = json.loads(scenes_path.read_text())
    if not isinstance(scenes, list):
        raise ValueError(f"{scenes_path}: scene list must be a flat JSON list of npz paths")
    kept = filter_scene_list(scenes, sidecar_root=sidecar_root, label=str(scenes_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept))
    print(f"wrote {len(kept)} scenes -> {out_path}")


if __name__ == "__main__":
    main()
