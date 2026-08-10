"""Orchestrate multi-process scoring across sharded path lists."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def split_into_shards(paths: list, n: int) -> list[list]:
    """Round-robin split paths into n shards."""
    shards: list[list] = [[] for _ in range(n)]
    for i, p in enumerate(paths):
        shards[i % n].append(p)
    return shards


def merge_shard_csvs(shard_paths: list[Path], output: Path) -> int:
    """Concatenate shard CSVs into a single CSV. Returns total row count."""
    fieldnames = None
    total = 0
    with open(output, "w", newline="") as out_f:
        writer = None
        for sp in shard_paths:
            if not sp.exists():
                continue
            with open(sp, newline="") as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    total += 1
    return total


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    p = argparse.ArgumentParser(description="Orchestrate parallel scoring.")
    p.add_argument("--npz_list", required=True, help="Full JSON path list")
    p.add_argument("--output_dir", required=True, help="Directory for shard CSVs and merged output")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mirror", default=None)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--monitor_interval", type=int, default=60, help="Seconds between progress checks"
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.npz_list) as f:
        all_paths = json.load(f)
    print(f"Total scenes: {len(all_paths)}")

    shards = split_into_shards(all_paths, args.num_workers)
    shard_lists = []
    shard_csvs = []
    for i, shard in enumerate(shards):
        shard_json = out_dir / f"shard_{i}_paths.json"
        with open(shard_json, "w") as f:
            json.dump(shard, f)
        shard_lists.append(shard_json)
        shard_csvs.append(out_dir / f"shard_{i}.csv")

    procs: list[subprocess.Popen] = []
    for i in range(args.num_workers):
        if not shards[i]:
            continue
        cmd = [
            sys.executable,
            "-m",
            "human_match_prototype.score_scenes",
            "--npz_list",
            str(shard_lists[i]),
            "--output",
            str(shard_csvs[i]),
            "--model_dir",
            args.model_dir,
            "--device",
            args.device,
            "--num_samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
            "--temperature",
            str(args.temperature),
            "--resume",
        ]
        if args.mirror:
            cmd.extend(["--mirror", args.mirror])
        print(f"Worker {i}: {len(shards[i])} scenes -> {shard_csvs[i].name}")
        proc = subprocess.Popen(cmd)
        procs.append(proc)

    print(f"\nLaunched {len(procs)} workers. Monitoring progress...")
    t0 = time.time()
    total_scenes = len(all_paths)

    try:
        while any(proc.poll() is None for proc in procs):
            time.sleep(args.monitor_interval)
            done = sum(_count_csv_rows(sc) for sc in shard_csvs)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_scenes - done) / rate / 3600 if rate > 0 else float("inf")
            print(
                f"  Progress: {done}/{total_scenes} ({done / total_scenes * 100:.1f}%) "
                f"| {rate:.1f} scenes/s | ETA: {eta:.1f}h"
            )
    except KeyboardInterrupt:
        print("\nInterrupted — killing workers...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
        print("Workers stopped. Re-run with same args to resume.")
        sys.exit(1)

    failures = sum(1 for proc in procs if proc.returncode != 0)
    if failures:
        print(f"WARNING: {failures}/{len(procs)} workers exited with errors")

    merged = out_dir / "scores_full.csv"
    n = merge_shard_csvs(shard_csvs, merged)
    elapsed = time.time() - t0
    print(f"\nDone. Merged {n} rows -> {merged} in {elapsed / 3600:.1f}h")


if __name__ == "__main__":
    main()
