#!/usr/bin/env python3
"""Join OOD scores with EPDMS subscores into a single feature matrix.

Joins by bag name + timestamp (50ms tolerance). Labels frames as positive
if within [t-20s, t+10s] of an OR event. Computes ood_residual per
maneuver type.

Usage:
    uv run python scripts/build_feature_matrix.py \
      --ood_override data/latent_ood_scores_override.jsonl \
      --ood_normal data/latent_ood_scores_normal.jsonl \
      --epdms_csv /path/to/samples_epdms_all.csv \
      --or_transitions /path/to/override_transitions.json \
      --output data/feature_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CATEGORY_TO_MANEUVER = {
    "停止_車両": "stop",
    "停止_信号": "stop",
    "停止_自転車歩行者": "stop",
    "回避_車両": "avoid",
    "回避_路駐車": "avoid",
    "曲がり切れない": "turn",
    "曲がるタイミングが早い": "turn",
    "その他": "other",
}

OR_WINDOW_PRE = 20.0
OR_WINDOW_POST = 10.0
MATCH_TOLERANCE_SEC = 0.05


def load_ood_scores(jsonl_path: Path) -> list[dict]:
    """Load OOD scores with parsed bag name and timestamp."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            e = json.loads(line.strip())
            npz_path = e["npz_path"]

            # Extract bag name from npz path
            # Filename format: <bag_name>_<frame_id>.npz
            # The bag name contains the vehicle/date/chunk info
            stem = Path(npz_path).stem
            parts = stem.rsplit("_", 1)
            bag_name = parts[0] if len(parts) == 2 else stem

            # Get timestamp from companion JSON if available
            json_path = Path(npz_path).with_suffix(".json")
            if json_path.exists():
                with open(json_path) as jf:
                    meta = json.load(jf)
                ts_sec = meta["timestamp"] / 1e9
            else:
                ts_sec = None

            entries.append(
                {
                    "npz_path": npz_path,
                    "bag_name": bag_name,
                    "ts_sec": ts_sec,
                    "knn_mean": e["knn_mean"],
                }
            )
    return entries


def load_epdms(csv_path: Path) -> dict[str, list[dict]]:
    """Load EPDMS CSV keyed by bag name."""
    by_bag = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            by_bag[row["bag"]].append(row)
    return dict(by_bag)


def load_or_transitions(json_path: Path) -> dict[str, list[float]]:
    with open(json_path) as f:
        return json.load(f)


def is_in_or_window(ts: float, or_times: list[float]) -> bool:
    for ot in or_times:
        if (ot - OR_WINDOW_PRE) <= ts <= (ot + OR_WINDOW_POST):
            return True
    return False


def extract_category_from_path(npz_path: str) -> str:
    """Extract failure_mode category from npz directory path."""
    parts = Path(npz_path).parts
    for part in parts:
        if part in CATEGORY_TO_MANEUVER:
            return part
        # Strip common suffixes and retry
        for suffix in ("_test", "_v2", "_old"):
            stripped = part.removesuffix(suffix)
            if stripped in CATEGORY_TO_MANEUVER:
                return stripped
    return "unknown"


def extract_session_from_normal_path(npz_path: str) -> str:
    """Extract date_time session ID from normal npz path."""
    parts = Path(npz_path).parts
    # Find 'train' in the path, session is train/<date>/<time>
    for i, part in enumerate(parts):
        if part in ("train", "valid") and i + 2 < len(parts):
            return f"normal_{parts[i + 1]}_{parts[i + 2]}"
    # Fallback: use parent directory name
    return f"normal_{Path(npz_path).parent.name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_override", type=Path, required=True)
    parser.add_argument("--ood_normal", type=Path, required=True)
    parser.add_argument("--epdms_csv", type=Path, required=True)
    parser.add_argument("--or_transitions", type=Path, required=True)
    parser.add_argument(
        "--maneuver_npz_paths",
        type=Path,
        default=None,
        help="JSON with maneuver group -> npz path lists",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print("Loading OOD scores...")
    override_ood = load_ood_scores(args.ood_override)
    normal_ood = load_ood_scores(args.ood_normal)
    print(f"  Override: {len(override_ood)}, Normal: {len(normal_ood)}")

    print("Loading EPDMS...")
    epdms = load_epdms(args.epdms_csv)
    print(f"  {len(epdms)} bags, {sum(len(v) for v in epdms.values())} frames")

    print("Loading OR transitions...")
    or_trans = load_or_transitions(args.or_transitions)
    print(f"  {len(or_trans)} bags with OR events")

    # Assign maneuver types to normal scenes
    normal_maneuver = {}
    if args.maneuver_npz_paths and args.maneuver_npz_paths.exists():
        with open(args.maneuver_npz_paths) as f:
            groups = json.load(f)
        for group, paths in groups.items():
            maneuver = group.replace("normal_", "")
            for p in paths:
                normal_maneuver[p] = maneuver

    # Compute per-maneuver median OOD from normal scores
    maneuver_scores = defaultdict(list)
    for e in normal_ood:
        maneuver = normal_maneuver.get(e["npz_path"], "straight")
        maneuver_scores[maneuver].append(e["knn_mean"])

    maneuver_medians = {}
    for m, scores in maneuver_scores.items():
        maneuver_medians[m] = float(np.median(scores))
    print(f"  Maneuver medians: {maneuver_medians}")

    # Join override OOD with EPDMS
    EPDMS_COLS = ["nc", "dac", "ddc", "tlc", "ttc", "lk", "hc", "ec", "ep", "epdms"]
    OUTPUT_COLS = [
        "bag",
        "ts_sec",
        "category",
        "maneuver_type",
        "label",
        *EPDMS_COLS,
        "knn_mean",
        "ood_residual",
        "is_override",
        "ade_full",
        "fde_full",
    ]

    rows = []
    matched = 0
    unmatched = 0

    for ood_entry in override_ood:
        bag = ood_entry["bag_name"]
        ts = ood_entry["ts_sec"]
        if ts is None or bag not in epdms:
            unmatched += 1
            continue

        # Find nearest EPDMS frame
        best = None
        best_dt = float("inf")
        for erow in epdms[bag]:
            epdms_ts = float(erow["ts"])
            dt = abs(ts - epdms_ts)
            if dt < best_dt:
                best_dt = dt
                best = erow

        if best is None or best_dt > MATCH_TOLERANCE_SEC:
            unmatched += 1
            continue

        category = extract_category_from_path(ood_entry["npz_path"])
        maneuver = CATEGORY_TO_MANEUVER.get(category, "other")
        or_times = or_trans.get(bag, [])
        label = 1 if is_in_or_window(ts, or_times) else 0
        ood_residual = ood_entry["knn_mean"] - maneuver_medians.get(
            maneuver, maneuver_medians.get("straight", 0)
        )

        row = {
            "bag": bag,
            "ts_sec": f"{ts:.3f}",
            "category": category,
            "maneuver_type": maneuver,
            "label": label,
            "knn_mean": f"{ood_entry['knn_mean']:.6f}",
            "ood_residual": f"{ood_residual:.6f}",
            "is_override": 1,
            "ade_full": best.get("ade_full", ""),
            "fde_full": best.get("fde_full", ""),
        }
        for col in EPDMS_COLS:
            row[col] = best.get(col, "")

        rows.append(row)
        matched += 1

    print(f"  Override: matched={matched}, unmatched={unmatched}")

    if matched:
        unknown_count = sum(1 for r in rows if r["category"] == "unknown")
        unknown_frac = unknown_count / matched
        if unknown_frac > 0.10:
            print(
                f"  WARNING: {unknown_count}/{matched} ({unknown_frac:.1%}) override "
                "rows have category='unknown' — check CATEGORY_TO_MANEUVER coverage "
                "against actual directory names."
            )

    # Add normal frames (label=0, is_override=0)
    for ood_entry in normal_ood:
        maneuver = normal_maneuver.get(ood_entry["npz_path"], "straight")
        ood_residual = ood_entry["knn_mean"] - maneuver_medians.get(
            maneuver, maneuver_medians.get("straight", 0)
        )
        session_bag = extract_session_from_normal_path(ood_entry["npz_path"])

        row = {
            "bag": session_bag,
            "ts_sec": f"{ood_entry['ts_sec']:.3f}" if ood_entry["ts_sec"] else "",
            "category": "normal",
            "maneuver_type": maneuver,
            "label": 0,
            "knn_mean": f"{ood_entry['knn_mean']:.6f}",
            "ood_residual": f"{ood_residual:.6f}",
            "is_override": 0,
            "ade_full": "",
            "fde_full": "",
        }
        for col in EPDMS_COLS:
            row[col] = ""

        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")
    print(f"  Override frames: {sum(1 for r in rows if r['is_override'] == 1)}")
    print(f"  Normal frames: {sum(1 for r in rows if r['is_override'] == 0)}")
    print(f"  Positive labels: {sum(1 for r in rows if r['label'] == 1)}")


if __name__ == "__main__":
    main()
