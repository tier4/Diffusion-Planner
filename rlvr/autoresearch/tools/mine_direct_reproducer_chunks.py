"""Mine R2LPL scenes by running direct 8-second reproducer chunks.

This tool deliberately skips the old open-loop prefilter. It samples starts from
the input scene list at fixed global intervals (default: every 80 scenes), builds
only that local window, cuts the window before a path/frame discontinuity, and
runs the existing closed-loop MPC perception reproducer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from rlvr.autoresearch.tools.reproducer_danger_scorer import (
    build_reproducer_danger_scorer,
    load_credit_windows,
)
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import run_segments_batched
from scenario_generation.route_timeline import RouteTimeline

_FRAME_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class Chunk:
    key: str
    global_start_index: int
    global_end_index: int
    paths: tuple[Path, ...]
    start_frame: int
    end_frame: int
    end_reason: str
    is_full_length: bool

    @property
    def n_frames(self) -> int:
        return len(self.paths)


def _frame_index(path: Path) -> int:
    match = _FRAME_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse trailing frame index from {path}")
    return int(match.group(1))


def _prefix(path: Path) -> str:
    return _FRAME_RE.sub("", path.stem)


def _safe_name(value: str, limit: int = 64) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return (safe or "route")[-limit:]


def _chunk_key(start_index: int, first_path: Path) -> str:
    parent_hash = hashlib.sha1(str(first_path.parent).encode()).hexdigest()[:10]
    return f"g{start_index:012d}_{parent_hash}_{_safe_name(_prefix(first_path))}"


def _is_next_contiguous(prev_path: Path, next_path: Path, *, expected_frame_step: int) -> bool:
    if prev_path.parent != next_path.parent:
        return False
    if _prefix(prev_path) != _prefix(next_path):
        return False
    return _frame_index(next_path) - _frame_index(prev_path) == expected_frame_step


def _parse_scene_list_line(item: str) -> Path | None:
    item = item.strip()
    if not item or item in {"[", "]"}:
        return None
    if item.endswith(","):
        item = item[:-1].rstrip()
    if item.startswith("["):
        raise json.JSONDecodeError("compact JSON list", item, 0)
    if len(item) >= 2 and item[0] == '"' and item[-1] == '"':
        raw = item[1:-1] if "\\" not in item else json.loads(item)
    else:
        raw = json.loads(item)
    return Path(str(raw))


def _iter_scene_list(path: Path) -> Iterator[Path]:
    """Yield string entries from a JSON scene-list without materializing it.

    The project scene lists are pretty-printed JSON arrays with one path per line.
    A compact-list fallback is kept for small/debug files.
    """
    with open(path) as f:
        for line in f:
            try:
                parsed = _parse_scene_list_line(line)
            except json.JSONDecodeError:
                f.seek(0)
                loaded = json.load(f)
                if not isinstance(loaded, list):
                    raise ValueError(f"{path} must contain a JSON list of NPZ paths")
                for value in loaded:
                    yield Path(str(value))
                return
            if parsed is not None:
                yield parsed


def _sample_value(global_start: int, sample_seed: int) -> float:
    digest = hashlib.blake2b(f"{sample_seed}:{global_start}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def iter_direct_chunks(
    scene_list: Path,
    *,
    chunk_len: int = 80,
    start_stride: int = 80,
    expected_frame_step: int = 1,
    min_chunk_len: int | None = None,
    max_scenes: int | None = None,
    max_chunks: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    sample_fraction: float = 1.0,
    sample_seed: int = 0,
) -> Iterator[Chunk]:
    if chunk_len < 2:
        raise ValueError(f"chunk_len must be >= 2, got {chunk_len}")
    if start_stride < chunk_len:
        raise ValueError(
            f"start_stride ({start_stride}) must be >= chunk_len ({chunk_len}); "
            "overlapping chunks are intentionally unsupported in the streaming direct miner"
        )
    if expected_frame_step < 1:
        raise ValueError(f"expected_frame_step must be >= 1, got {expected_frame_step}")
    if min_chunk_len is None:
        min_chunk_len = chunk_len
    if min_chunk_len < 2:
        raise ValueError(f"min_chunk_len must be >= 2, got {min_chunk_len}")
    if min_chunk_len > chunk_len:
        raise ValueError(f"min_chunk_len ({min_chunk_len}) must be <= chunk_len ({chunk_len})")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if sample_fraction <= 0.0 or sample_fraction > 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    paths = _iter_scene_list(scene_list)
    candidate_idx = 0
    seen = 0
    yielded = 0
    while max_scenes is None or seen < max_scenes:
        block: list[Path] = []
        for _ in range(start_stride):
            if max_scenes is not None and seen >= max_scenes:
                break
            try:
                block.append(next(paths))
            except StopIteration:
                break
            seen += 1
        if not block:
            break

        global_start = candidate_idx * start_stride
        candidate_idx += 1
        if (global_start // start_stride) % num_shards != shard_index:
            continue
        if sample_fraction < 1.0 and _sample_value(global_start, sample_seed) >= sample_fraction:
            continue

        window = block[:chunk_len]
        kept = [window[0]]
        end_reason = "chunk_len"
        for next_path in window[1:]:
            if not _is_next_contiguous(
                kept[-1], next_path, expected_frame_step=expected_frame_step
            ):
                end_reason = "lineage_break"
                break
            kept.append(next_path)
        if len(kept) < min_chunk_len:
            continue
        if len(kept) < chunk_len and end_reason == "chunk_len":
            end_reason = "scene_list_tail"

        chunk = Chunk(
            key=_chunk_key(global_start, kept[0]),
            global_start_index=global_start,
            global_end_index=global_start + len(kept),
            paths=tuple(kept),
            start_frame=_frame_index(kept[0]),
            end_frame=_frame_index(kept[-1]),
            end_reason=end_reason,
            is_full_length=len(kept) == chunk_len,
        )
        yield chunk
        yielded += 1
        if max_chunks is not None and yielded >= max_chunks:
            break


def _chunk_row(chunk: Chunk) -> dict:
    return {
        "chunk_key": chunk.key,
        "global_start_index": chunk.global_start_index,
        "global_end_index": chunk.global_end_index,
        "n_frames": chunk.n_frames,
        "start_frame": chunk.start_frame,
        "end_frame": chunk.end_frame,
        "end_reason": chunk.end_reason,
        "is_full_length": chunk.is_full_length,
        "start_scene_path": str(chunk.paths[0]),
        "end_scene_path": str(chunk.paths[-1]),
    }


def _credit_rows_from_saved(save_dir: Path, out_jsonl: Path) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_jsonl, "w") as f:
        for scene_path in sorted(save_dir.rglob("credit*.npz")):
            manifest_path = scene_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            label = str(manifest.get("credit_label") or "perception_reproducer")
            row = {
                "scene_path": str(scene_path),
                "label": label,
                "labels": [label],
                "variant_kind": "direct_reproducer_chunk",
                "window_dir": str(scene_path.parent),
                "event_key": scene_path.parent.name,
                "offense_step": manifest.get("offense_step"),
                "offense_frame_id": manifest.get("offense_frame_id"),
                "segment": manifest.get("segment"),
                "segment_route_indices": manifest.get("segment_route_indices"),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n_rows += 1
    return n_rows


def _validate_timeline_continuity(
    tl: RouteTimeline,
    chunk: Chunk,
    *,
    expected_frame_step: int,
    max_pose_step_m: float,
    max_pose_speed_mps: float,
    max_yaw_step_rad: float,
) -> None:
    frame_steps = np.diff(tl.frame_indices)
    if frame_steps.size and not np.all(frame_steps == expected_frame_step):
        bad = int(np.flatnonzero(frame_steps != expected_frame_step)[0])
        raise ValueError(
            f"{chunk.key}: frame-id jump at local {bad}->{bad + 1}: "
            f"{int(tl.frame_indices[bad])}->{int(tl.frame_indices[bad + 1])}"
        )
    if len(tl.poses) < 2:
        return
    dxy = np.linalg.norm(np.diff(tl.poses[:, :2], axis=0), axis=1)
    if max_pose_step_m > 0 and bool(np.any(dxy > max_pose_step_m)):
        bad = int(np.argmax(dxy))
        raise ValueError(
            f"{chunk.key}: pose jump {float(dxy[bad]):.2f}m at local "
            f"{bad}->{bad + 1} (frames {int(tl.frame_indices[bad])}->"
            f"{int(tl.frame_indices[bad + 1])})"
        )
    if max_pose_speed_mps > 0 and frame_steps.size:
        dt = np.maximum(frame_steps, 1).astype(np.float64) * 0.1
        speeds = dxy / dt
        if bool(np.any(speeds > max_pose_speed_mps)):
            bad = int(np.argmax(speeds))
            raise ValueError(
                f"{chunk.key}: pose speed {float(speeds[bad]):.1f}m/s "
                f"from {float(dxy[bad]):.2f}m over {float(dt[bad]):.2f}s at local "
                f"{bad}->{bad + 1} (frames {int(tl.frame_indices[bad])}->"
                f"{int(tl.frame_indices[bad + 1])})"
            )
    yaw_steps = np.abs(np.diff(np.unwrap(tl.poses[:, 2])))
    if max_yaw_step_rad > 0 and bool(np.any(yaw_steps > max_yaw_step_rad)):
        bad = int(np.argmax(yaw_steps))
        raise ValueError(
            f"{chunk.key}: yaw jump {float(yaw_steps[bad]):.2f}rad at local "
            f"{bad}->{bad + 1} (frames {int(tl.frame_indices[bad])}->"
            f"{int(tl.frame_indices[bad + 1])})"
        )


def _build_work_unit(
    chunk: Chunk,
    sidecar_root: Path | None,
    prebuild_tracks: bool,
    *,
    expected_frame_step: int,
    max_pose_step_m: float,
    max_pose_speed_mps: float,
    max_yaw_step_rad: float,
):
    tl = RouteTimeline(list(chunk.paths), sidecar_dir=sidecar_root)
    _validate_timeline_continuity(
        tl,
        chunk,
        expected_frame_step=expected_frame_step,
        max_pose_step_m=max_pose_step_m,
        max_pose_speed_mps=max_pose_speed_mps,
        max_yaw_step_rad=max_yaw_step_rad,
    )
    if prebuild_tracks:
        from scenario_generation import reproducer_rollout

        tracks = reproducer_rollout._route_nbr_tracks(tl)
        if not tracks[0]:
            raise ValueError(f"{chunk.key}: no neighbor UUID tracks available")
    tl.prefetch([0])
    return tl, 0, chunk.n_frames


def _load_model(model_path: Path, lora_path: Path | None, device: str):
    from scenario_generation.simulate import load_model

    model, model_args = load_model(model_path, device)
    if lora_path is not None:
        from preference_optimization.lora_utils import load_lora_checkpoint

        model = load_lora_checkpoint(model, str(lora_path))
        model.eval()
    return model, model_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--lora_path", type=Path, default=None)
    parser.add_argument("--sidecar_root", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--out_jsonl", type=Path, default=None)
    parser.add_argument("--segments_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--chunk_len", type=int, default=80)
    parser.add_argument("--start_stride", type=int, default=80)
    parser.add_argument("--expected_frame_step", type=int, default=1)
    parser.add_argument(
        "--max_pose_step_m",
        type=float,
        default=10.0,
        help=(
            "skip a chunk if adjacent sidecar ego poses jump by more than this many metres. "
            "Set <=0 to disable. Checked after RouteTimeline loads sidecars, so it adds no "
            "extra large-data pass."
        ),
    )
    parser.add_argument(
        "--max_pose_speed_mps",
        type=float,
        default=20.0,
        help=(
            "skip a chunk if adjacent sidecar ego poses imply speed above this many m/s "
            "using frame-id deltas at 10 Hz. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--max_yaw_step_rad",
        type=float,
        default=1.57,
        help="skip a chunk if adjacent unwrapped sidecar yaw changes by more than this many radians; <=0 disables",
    )
    parser.add_argument(
        "--min_chunk_len",
        type=int,
        default=None,
        help=(
            "minimum frames required to simulate a chunk. Defaults to --chunk_len, "
            "so partial chunks caused by route changes, NPZ jumps, or scene-list tails "
            "are discarded. Lower for debugging/coverage experiments."
        ),
    )
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help=(
            "deterministically keep this fraction of every-80 candidate chunks before timeline "
            "construction. Sampling is by hash of global_start and --sample_seed, so shards are "
            "reproducible and non-overlapping."
        ),
    )
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--timeline_build_workers", type=int, default=8)
    parser.add_argument("--n_build_threads", type=int, default=16)
    parser.add_argument("--prefetch_ahead", type=int, default=2)
    parser.add_argument("--near_miss_thresh", type=float, default=0.5)
    parser.add_argument("--search_radius", type=float, default=1.5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps_mult", type=int, default=1)
    parser.add_argument("--goal_reach_m", type=float, default=0.0)
    parser.add_argument("--unstick_after", type=int, default=300)
    parser.add_argument("--unstick_advance_m", type=float, default=5.0)
    parser.add_argument("--tracker_mode", choices=["mpc", "perfect"], default="mpc")
    parser.add_argument("--timeline_progress_mode", choices=["clock", "pose"], default="clock")
    parser.add_argument("--neighbor_history_mode", choices=["sim", "recorded"], default="sim")
    parser.add_argument("--gpu_transform", action="store_true")
    parser.add_argument("--danger_reward_config", type=Path, default=None)
    parser.add_argument("--danger_threshold_config", type=Path, default=None)
    parser.add_argument("--danger_credit_window_config", type=Path, default=None)
    parser.add_argument("--danger_decluster_steps", type=int, default=10)
    parser.add_argument("--enable_conflict_detector", action="store_true")
    parser.add_argument("--labels", default="")
    parser.add_argument("--skip_bad_chunks", action="store_true")
    parser.add_argument(
        "--allow_existing_out_dir",
        action="store_true",
        help="allow appending/scanning an existing non-empty --out_dir; disabled by default to avoid stale credit rows",
    )
    parser.add_argument("--prebuild_neighbor_tracks", action="store_true", default=True)
    parser.add_argument(
        "--no_prebuild_neighbor_tracks",
        dest="prebuild_neighbor_tracks",
        action="store_false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plan_only and args.model_path is None:
        raise ValueError("--model_path is required unless --plan_only is set")
    if not args.plan_only:
        missing = [
            name
            for name, value in (
                ("out_dir", args.out_dir),
                ("out_jsonl", args.out_jsonl),
                ("danger_reward_config", args.danger_reward_config),
                ("danger_threshold_config", args.danger_threshold_config),
                ("danger_credit_window_config", args.danger_credit_window_config),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"direct reproducer mining is missing required args: {missing}")
        if (
            args.out_dir is not None
            and args.out_dir.exists()
            and any(args.out_dir.iterdir())
            and not args.allow_existing_out_dir
        ):
            raise ValueError(
                f"--out_dir is not empty: {args.out_dir}. Use a fresh directory or pass "
                "--allow_existing_out_dir to intentionally rescan existing saved windows."
            )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    allowed_labels = {label.strip() for label in args.labels.split(",") if label.strip()} or None
    danger_scorer = None
    danger_credit_windows = None
    model = model_args = None
    if not args.plan_only:
        model, model_args = _load_model(args.model_path, args.lora_path, device)
        danger_scorer = build_reproducer_danger_scorer(
            reward_config=args.danger_reward_config,
            threshold_config=args.danger_threshold_config,
            device=device,
            enable_conflict_detector=bool(args.enable_conflict_detector),
            allowed_labels=allowed_labels,
        )
        danger_credit_windows = load_credit_windows(args.danger_credit_window_config)

    args.segments_jsonl.parent.mkdir(parents=True, exist_ok=True)
    skipped_path = args.segments_jsonl.with_suffix(".skipped.jsonl")
    summary_path = args.summary_json or args.segments_jsonl.with_suffix(".summary.json")
    timers = Timers()
    n_chunks = 0
    n_simulated = 0
    n_skipped = 0
    t0 = time.perf_counter()

    def flush(chunks: list[Chunk], fout, fskip) -> None:
        nonlocal n_simulated, n_skipped
        if not chunks:
            return
        work_units = []
        kept_chunks = []

        def build(chunk: Chunk):
            return _build_work_unit(
                chunk,
                args.sidecar_root,
                args.prebuild_neighbor_tracks,
                expected_frame_step=args.expected_frame_step,
                max_pose_step_m=args.max_pose_step_m,
                max_pose_speed_mps=args.max_pose_speed_mps,
                max_yaw_step_rad=args.max_yaw_step_rad,
            )

        workers = max(1, int(args.timeline_build_workers))
        built = []
        if workers == 1:
            for chunk in chunks:
                try:
                    built.append((chunk, build(chunk), None))
                except Exception as exc:  # noqa: BLE001 - logged per chunk below
                    built.append((chunk, None, exc))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [(chunk, pool.submit(build, chunk)) for chunk in chunks]
                for chunk, fut in futures:
                    try:
                        built.append((chunk, fut.result(), None))
                    except Exception as exc:  # noqa: BLE001 - logged per chunk below
                        built.append((chunk, None, exc))

        for chunk, unit, error in built:
            if error is not None:
                row = {**_chunk_row(chunk), "error": str(error)}
                fskip.write(json.dumps(row, sort_keys=True) + "\n")
                fskip.flush()
                n_skipped += 1
                if not args.skip_bad_chunks:
                    raise RuntimeError(f"failed to build chunk {chunk.key}: {error}") from error
                continue
            kept_chunks.append(chunk)
            work_units.append(unit)
        if not work_units:
            return
        assert model is not None and model_args is not None
        results = run_segments_batched(
            model,
            model_args,
            work_units,
            device=device,
            batch_size=args.batch_size,
            near_miss_thresh=args.near_miss_thresh,
            search_radius=args.search_radius,
            warmup_steps=args.warmup_steps,
            goal_reach_m=args.goal_reach_m,
            max_steps_mult=args.max_steps_mult,
            n_build_threads=args.n_build_threads,
            prefetch_ahead=args.prefetch_ahead,
            timers=timers,
            route_keys=[chunk.key for chunk in kept_chunks],
            gpu_transform=args.gpu_transform,
            neighbor_history_mode=args.neighbor_history_mode,
            tracker_mode=args.tracker_mode,
            timeline_progress_mode=args.timeline_progress_mode,
            danger_save_dir=args.out_dir,
            danger_scorer=danger_scorer,
            danger_credit_windows=danger_credit_windows,
            danger_decluster_steps=args.danger_decluster_steps,
            unstick_after=args.unstick_after,
            unstick_advance_m=args.unstick_advance_m,
        )
        for chunk, result in zip(kept_chunks, results):
            row = {**_chunk_row(chunk), **result.metrics}
            fout.write(json.dumps(row, sort_keys=True, default=float) + "\n")
            n_simulated += 1
        fout.flush()

    with open(args.segments_jsonl, "w") as fout, open(skipped_path, "w") as fskip:
        pending: list[Chunk] = []
        for chunk in iter_direct_chunks(
            args.scene_list,
            chunk_len=args.chunk_len,
            start_stride=args.start_stride,
            expected_frame_step=args.expected_frame_step,
            min_chunk_len=args.min_chunk_len,
            max_scenes=args.max_scenes,
            max_chunks=args.max_chunks,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
        ):
            n_chunks += 1
            if args.plan_only:
                fout.write(json.dumps(_chunk_row(chunk), sort_keys=True) + "\n")
                continue
            pending.append(chunk)
            if len(pending) >= args.batch_size:
                flush(pending, fout, fskip)
                pending = []
        if args.plan_only:
            fout.flush()
        else:
            flush(pending, fout, fskip)

    n_credit_rows = 0
    if not args.plan_only and args.out_dir is not None and args.out_jsonl is not None:
        n_credit_rows = _credit_rows_from_saved(args.out_dir, args.out_jsonl)

    elapsed = time.perf_counter() - t0
    summary = {
        "scene_list": str(args.scene_list),
        "chunk_len": int(args.chunk_len),
        "start_stride": int(args.start_stride),
        "min_chunk_len": int(args.min_chunk_len or args.chunk_len),
        "expected_frame_step": int(args.expected_frame_step),
        "max_pose_step_m": float(args.max_pose_step_m),
        "max_pose_speed_mps": float(args.max_pose_speed_mps),
        "max_yaw_step_rad": float(args.max_yaw_step_rad),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "sample_fraction": float(args.sample_fraction),
        "sample_seed": int(args.sample_seed),
        "planned_chunks": int(n_chunks),
        "simulated_chunks": int(n_simulated),
        "skipped_chunks": int(n_skipped),
        "credit_rows": int(n_credit_rows),
        "elapsed_sec": round(elapsed, 3),
        "chunks_per_sec": n_simulated / elapsed if elapsed > 0 and n_simulated else 0.0,
        "timers": timers.report(max(1, n_simulated)) if n_simulated else "",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
