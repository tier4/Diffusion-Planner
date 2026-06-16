"""Select the OR-window NPZs from a per-session NPZ dir.

Reads parse_rosbag.py output NPZs (each with a sibling .json carrying ego
timestamp in nanoseconds), filters to frames inside [t_or - pre_s, t_or + post_s],
and writes a scenes list JSON in the format used by rlvr autoresearch tools.

Example:
    python -m rlvr.autoresearch.tools.cata_select_or_window \
        --npz_dir $ROUND/npz/session1 \
        --or_jst 2026-01-01T00:00:00+09:00 \
        --pre_s 3.0 --post_s 8.0 \
        --output $ROUND/npz/session1_or_window.json
"""

import argparse
import datetime as dt
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz_dir", type=Path, required=True)
    p.add_argument(
        "--or_jst",
        required=True,
        help="Override moment in ISO-8601 format (e.g. 2026-01-01T00:00:00+09:00).",
    )
    p.add_argument("--pre_s", type=float, default=3.0)
    p.add_argument("--post_s", type=float, default=8.0)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    or_t = dt.datetime.fromisoformat(args.or_jst)
    or_ns = int(or_t.timestamp() * 1e9)
    lo_ns = or_ns - int(args.pre_s * 1e9)
    hi_ns = or_ns + int(args.post_s * 1e9)

    npz_paths = sorted(args.npz_dir.glob("*.npz"))
    selected: list[tuple[int, str]] = []
    skipped = 0
    for npz in npz_paths:
        sidecar = npz.with_suffix(".json")
        if not sidecar.exists():
            skipped += 1
            continue
        try:
            ts = int(
                json.loads(sidecar.read_text())["timestamp"]
            )  # str or num; raises loudly on garbage
        except (KeyError, json.JSONDecodeError, ValueError, TypeError):
            skipped += 1
            continue
        if lo_ns <= ts <= hi_ns:
            selected.append((ts, str(npz)))

    selected.sort()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # --output is a PLAIN list of NPZ paths (the scenes-list format rlvr autoresearch
    # tools consume); window metadata goes to a .meta.json sidecar.
    paths = [p for _, p in selected]
    args.output.write_text(json.dumps(paths, indent=2))
    meta = {
        "or_jst": args.or_jst,
        "or_ns": or_ns,
        "window_pre_s": args.pre_s,
        "window_post_s": args.post_s,
        "n_total_npz": len(npz_paths),
        "n_in_window": len(selected),
        "n_skipped_no_sidecar": skipped,
        "timestamps_ns": [ts for ts, _ in selected],
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Wrote {args.output}: {len(selected)}/{len(npz_paths)} NPZs in window "
        f"[{args.or_jst} -{args.pre_s}s, +{args.post_s}s]. Skipped {skipped} without sidecar."
    )


if __name__ == "__main__":
    main()
