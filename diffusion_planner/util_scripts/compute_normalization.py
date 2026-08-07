#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FAMILIES = (
    "position",
    "velocity",
    "size",
    "acceleration",
    "steering",
    "yaw_rate",
    "speed_limit",
)


class Reservoir:
    """Fixed-memory uniform reservoir for scalar float samples."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.values = np.empty(self.capacity, dtype=np.float64)
        self.seen = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        start = self.seen
        first = max(0, min(values.size, self.capacity - start))
        if first:
            self.values[start : start + first] = values[:first]
        tail = values[first:]
        if tail.size:
            # Vectorized form of classic reservoir replacement. For an item
            # with one-based global index j, it replaces a random slot iff a
            # uniform draw in [0, j) falls below capacity. Repeated slots are
            # intentional and match sequential reservoir replacement.
            global_index = np.arange(start + first + 1, start + values.size + 1, dtype=np.float64)
            draw = np.floor(self.rng.random(tail.size) * global_index).astype(np.int64)
            keep = draw < self.capacity
            self.values[draw[keep]] = tail[keep]
        self.seen += values.size

    def array(self) -> np.ndarray:
        return self.values[: min(self.seen, self.capacity)].copy()


def valid_rows(array: np.ndarray, dims: int) -> np.ndarray:
    return np.any(np.abs(array[..., :dims]) > 1e-7, axis=-1)


def scene_sample(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size <= limit:
        return values
    return values[rng.choice(values.size, size=limit, replace=False)]


def add_scene(
    store: dict[str, Reservoir], name: str, values: np.ndarray, limit: int, rng: np.random.Generator
) -> None:
    store[name].add(scene_sample(values, limit, rng))


def collect_npz(
    path: Path,
    store: dict[str, Reservoir],
    per_scene: int,
    rng: np.random.Generator,
    speed_counts: Counter[float] | None = None,
) -> None:
    with np.load(path, allow_pickle=False) as data:
        ego_past = data["ego_agent_past"]
        add_scene(store, "position", ego_past[..., :2], per_scene, rng)

        neighbors = data["neighbor_agents_past"]
        neighbor_valid = valid_rows(neighbors, 8)
        add_scene(store, "position", neighbors[..., :2][neighbor_valid], per_scene, rng)
        add_scene(store, "velocity", neighbors[..., 4:6][neighbor_valid], per_scene, rng)
        add_scene(store, "size", neighbors[..., 6:8][neighbor_valid], per_scene, rng)

        for key in ("lanes", "route_lanes"):
            lanes = data[key]
            lane_valid = valid_rows(lanes, 8)
            # x/y, dX/dY, left-bound and right-bound offsets are metre-valued.
            add_scene(store, "position", lanes[..., :8][lane_valid], per_scene, rng)

        for key in ("polygons", "line_strings"):
            lines = data[key]
            line_valid = valid_rows(lines, 2)
            add_scene(store, "position", lines[..., :2][line_valid], per_scene, rng)

        current = data["ego_current_state"]
        add_scene(store, "velocity", current[4:6], per_scene, rng)
        add_scene(store, "acceleration", current[6:8], per_scene, rng)
        add_scene(store, "steering", current[8], per_scene, rng)
        add_scene(store, "yaw_rate", current[9], per_scene, rng)
        # ego_shape is retained for diagnostics; it shares the metric scale.
        add_scene(store, "size", data["ego_shape"], per_scene, rng)

        speed = data["lanes_speed_limit"].reshape(-1)
        add_scene(store, "speed_limit", speed[speed > 0], per_scene, rng)
        if speed_counts is not None:
            values, counts = np.unique(speed[speed > 0], return_counts=True)
            for value, count in zip(values, counts):
                speed_counts[round(float(value), 9)] += int(count)


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def describe(name: str, values: np.ndarray) -> dict[str, float | int | str]:
    q25, q50, q75 = (quantile(values, q) for q in (0.25, 0.50, 0.75))
    iqr = q75 - q25
    return {
        "family": name,
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": q50,
        "iqr": iqr,
        "robust_scale_iqr_1.349": iqr / 1.349,
        "p01": quantile(values, 0.01),
        "p99": quantile(values, 0.99),
        "p95_abs": quantile(np.abs(values), 0.95),
        "zero_fraction": float(np.mean(np.isclose(values, 0.0))),
    }


def build_normalization(scales: dict[str, float]) -> dict[str, dict[str, list[float]]]:
    pos = scales["position"]
    vel = scales["velocity"]
    acc = scales["acceleration"]
    steer = scales["steering"]
    yaw = scales["yaw_rate"]
    speed = scales["speed_limit"]
    # All metre-valued quantities intentionally share pos, including size.
    size = pos
    zero4 = [0.0, 0.0, 0.0, 0.0]
    return {
        "ego": {"mean": zero4, "std": [pos, pos, 1.0, 1.0]},
        "neighbor": {"mean": zero4, "std": [pos, pos, 1.0, 1.0]},
        "ego_agent_past": {"mean": zero4, "std": [pos, pos, 1.0, 1.0]},
        "ego_current_state": {
            "mean": [0.0] * 10,
            "std": [pos, pos, 1.0, 1.0, vel, vel, acc, acc, steer, yaw],
        },
        "neighbor_agents_past": {
            "mean": [0.0] * 11,
            "std": [pos, pos, 1.0, 1.0, vel, vel, size, size, 1.0, 1.0, 1.0],
        },
        "static_objects": {
            "mean": [0.0] * 10,
            "std": [pos, pos, 1.0, 1.0, size, size, size, size, 1.0, 1.0],
        },
        "lanes": {"mean": [0.0] * 33, "std": [pos, pos, 1.0, 1.0, pos, pos, pos, pos] + [1.0] * 25},
        "lanes_speed_limit": {"mean": [0.0], "std": [speed]},
        "route_lanes": {
            "mean": [0.0] * 33,
            "std": [pos, pos, 1.0, 1.0, pos, pos, pos, pos] + [1.0] * 25,
        },
        "polygons": {"mean": [0.0, 0.0], "std": [pos, pos]},
        "line_strings": {"mean": [0.0, 0.0], "std": [pos, pos]},
        "route_lanes_speed_limit": {"mean": [0.0], "std": [speed]},
        # Masked by the model, but convention is kept future-proof.
        "goal_pose": {"mean": [0.0] * 4, "std": [pos, pos, 1.0, 1.0]},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="大規模NPZから物理単位ベースのnormalization.jsonを生成します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "実行例:\n"
            "  python notebooks/compute_scratch_normalization.py \\\n"
            "      --train-list combined_train.json \\\n"
            "      --output normalization_candidates/normalization_scratch_grouped.json\n\n"
            "再現性を保つには --seed を固定してください。\n"
            "本番の最終計算では、欠損確認のため --skip-missing は付けないでください。"
        ),
    )
    parser.add_argument("--train-list", type=Path, required=True, help="JSON list of NPZ paths")
    parser.add_argument("--output", type=Path, required=True, help="Output normalization JSON")
    parser.add_argument(
        "--per-scene", type=int, default=256, help="Maximum values per family/source/scene"
    )
    parser.add_argument(
        "--reservoir", type=int, default=300_000, help="Maximum values retained per family"
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_scene <= 0 or args.reservoir <= 0:
        raise ValueError("--per-scene and --reservoir must be positive")
    paths = json.loads(args.train_list.read_text())
    if not isinstance(paths, list):
        raise ValueError("train list must contain a JSON array")
    rng = np.random.default_rng(args.seed)
    store = {name: Reservoir(args.reservoir, rng) for name in FAMILIES}
    speed_counts: Counter[float] = Counter()
    missing = []
    for index, raw_path in enumerate(paths, 1):
        path = Path(raw_path)
        if not path.exists():
            missing.append(str(path))
            if not args.skip_missing:
                raise FileNotFoundError(path)
            continue
        collect_npz(path, store, args.per_scene, rng, speed_counts)
        if index % args.progress_every == 0 or index == len(paths):
            print(
                f"[{index:,}/{len(paths):,}] "
                + " ".join(f"{k}={store[k].seen:,}" for k in FAMILIES),
                flush=True,
            )

    if missing:
        print(f"warning: skipped {len(missing):,} missing files", flush=True)
    arrays = {name: reservoir.array() for name, reservoir in store.items()}
    summaries = {name: describe(name, values) for name, values in arrays.items()}
    for name, row in summaries.items():
        print(
            f"{name:>14}: n={row['n']:>8,} std={row['std']:.6g} robust={row['robust_scale_iqr_1.349']:.6g} p95_abs={row['p95_abs']:.6g}"
        )
    print("speed_limit unique values (m/s -> km/h, count):")
    for value, count in sorted(speed_counts.items()):
        print(f"  {value:.9f} -> {value * 3.6:.6f}, {count:,}")

    # Robust metric scale; dynamics use std because their zero-heavy IQR can degenerate.
    scales = {
        "position": max(summaries["position"]["robust_scale_iqr_1.349"], 1e-3),
        "velocity": max(summaries["velocity"]["std"], 1e-3),
        "acceleration": max(summaries["acceleration"]["std"], 1e-3),
        "steering": max(summaries["steering"]["std"], 1e-3),
        "yaw_rate": max(summaries["yaw_rate"]["std"], 1e-3),
        "size": max(summaries["position"]["robust_scale_iqr_1.349"], 1e-3),
        # Speed limits are stored in m/s; preserve current_speed/speed_limit
        # ratios by sharing the velocity scale.
        "speed_limit": max(summaries["velocity"]["std"], 1e-3),
    }
    result = build_normalization(scales)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent, text=True
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        Path(temp_name).replace(args.output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(f"wrote {args.output}")
    print("scales:", json.dumps(scales, sort_keys=True))


if __name__ == "__main__":
    main()
