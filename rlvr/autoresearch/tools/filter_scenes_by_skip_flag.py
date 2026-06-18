"""Filter a training scene list by the converter's per-frame skip flag.

When a corpus is generated with the cpp converter's ``--write_skipped_npz=1``
(so the closed-loop reproducer gets a gap-free timeline), every 10 Hz frame is
written, including the ones the production filter would normally drop (stopped at
a red light, no future progress, ...). Each frame's JSON sidecar carries
``is_skipped`` (and ``skipping_info.label``). TRAINING must not learn from those
frames, so this tool rewrites a scene list keeping only ``is_skipped == false``.

Backward-compatible: a sidecar without the field (older corpora) is treated as
NOT skipped, so existing lists pass through unchanged.

Example::

    python -m rlvr.autoresearch.tools.filter_scenes_by_skip_flag \
        --scenes train_all.json --sidecar_root /path/to/npz_dir \
        --out train_all_noskip.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sidecar_for(npz_path: Path, sidecar_root: Path | None) -> Path | None:
    sib = npz_path.with_suffix(".json")
    if sib.is_file():
        return sib
    if sidecar_root is not None:
        cand = sidecar_root / f"{npz_path.stem}.json"
        if cand.is_file():
            return cand
        matches = list(sidecar_root.rglob(f"{npz_path.stem}.json"))
        if matches:
            return matches[0]
    return None


def is_skipped(npz_path: str | Path, sidecar_root: Path | None = None) -> bool:
    """True if the frame's sidecar marks it skip_for_training (missing flag => False)."""
    sc = _sidecar_for(Path(npz_path), sidecar_root)
    if sc is None:
        return False
    try:
        d = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(d.get("is_skipped", False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenes", type=Path, required=True, help="input scene list (.json of npz paths)"
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
    scenes = json.loads(args.scenes.read_text())
    if not isinstance(scenes, list):
        raise ValueError(f"{args.scenes} is not a JSON list of npz paths")
    kept, dropped = [], 0
    for entry in scenes:
        path = entry if isinstance(entry, str) else (entry.get("path") or entry.get("npz"))
        if is_skipped(path, args.sidecar_root):
            dropped += 1
        else:
            kept.append(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kept))
    print(
        f"kept {len(kept)} / {len(scenes)} scenes (dropped {dropped} skip_for_training) -> {args.out}"
    )


if __name__ == "__main__":
    main()
