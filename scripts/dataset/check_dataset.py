"""Benchmark H5 dataset loading through a training-style DataLoader."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import argcomplete
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

from diffusion_planner.data import PlannerDataset


def benchmark_loading(
    dataset: PlannerDataset,
    *,
    jobs: int,
    batch_size: int,
    warmup_batches: int,
    limit: int | None,
    shuffle: bool,
) -> None:
    """Measure training-style loading, collation, and worker transfer throughput."""
    total = len(dataset) if limit is None else min(len(dataset), limit)
    source = dataset if shuffle else Subset(dataset, range(total))
    sampler = (
        RandomSampler(dataset, replacement=False, num_samples=total)
        if shuffle
        else None
    )
    loader = DataLoader(
        source,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=jobs,
        persistent_workers=jobs > 0,
    )

    print(
        f"benchmarking {total} frames with batch_size={batch_size}, jobs={jobs}, "
        f"shuffle={shuffle}"
    )
    started = time.perf_counter()
    iterator = iter(loader)
    measured_frames = 0
    measured_batches = 0
    measured_started: float | None = started if warmup_batches == 0 else None
    first_batch_seconds: float | None = None
    with tqdm(total=total, desc="benchmark", unit="frame", smoothing=0.0) as progress:
        for batch_index, batch in enumerate(iterator):
            batch_frames = len(next(iter(batch.values())))
            now = time.perf_counter()
            if first_batch_seconds is None:
                first_batch_seconds = now - started
            if batch_index >= warmup_batches:
                measured_frames += batch_frames
                measured_batches += 1
            elif batch_index + 1 == warmup_batches:
                measured_started = now
            progress.update(batch_frames)

    if measured_started is None or measured_batches == 0:
        raise ValueError(
            "benchmark did not reach a measured batch; increase --limit or reduce "
            "--warmup-batches"
        )
    finished = time.perf_counter()
    elapsed = finished - measured_started
    total_elapsed = finished - started
    frames_per_second = measured_frames / elapsed
    seconds_per_batch = elapsed / measured_batches
    epoch_seconds = len(dataset) / frames_per_second
    print(f"first batch: {first_batch_seconds:.3f} s")
    print(f"total: {total} frames in {total_elapsed:.3f} s")
    print(
        f"measured: {measured_frames} frames in {elapsed:.3f} s "
        f"({frames_per_second:.1f} frames/s, {seconds_per_batch:.3f} s/batch)"
    )
    if total == len(dataset):
        print(f"full epoch wall time: {total_elapsed / 60:.1f} min")
    else:
        print(f"estimated full epoch: {epoch_seconds / 60:.1f} min")


def main() -> None:
    """Benchmark loading all indexed frames or an explicitly limited subset."""
    parser = argparse.ArgumentParser(
        description="Benchmark training-style loading from an H5 frame index"
    )
    parser.add_argument("parquet", type=Path, help="H5 frame index to benchmark")
    parser.add_argument(
        "--jobs", type=int, default=8, help="worker processes (default: %(default)s)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="benchmark only N frames"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="benchmark batch size"
    )
    parser.add_argument(
        "--warmup-batches", type=int, default=2, help="untimed benchmark batches"
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle benchmark frames as training does (default: true)",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.jobs < 0:
        parser.error("--jobs must not be negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must not be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    dataset = PlannerDataset(args.parquet)
    benchmark_loading(
        dataset,
        jobs=args.jobs,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        limit=args.limit,
        shuffle=args.shuffle,
    )


if __name__ == "__main__":
    main()
