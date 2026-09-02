"""§7.5 throughput: shard loader (C sweep) vs legacy npz loader; prints samples/s, TTFB, peak worker RSS."""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

from torch.utils.data import DataLoader, DistributedSampler

from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.shard_dataset import ShardDatasetConfig, make_shard_dataloader


def _measure(loader, steps: int, batch_size: int) -> dict:
    t0 = time.perf_counter()
    it = iter(loader)
    next(it)
    ttfb = time.perf_counter() - t0
    t1 = time.perf_counter()
    n = 1
    for _ in range(steps - 1):
        next(it)
        n += 1
    dt = time.perf_counter() - t1
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    return {
        "samples_per_s": (n - 1) * batch_size / dt if dt > 0 else float("nan"),
        "ttfb_s": ttfb,
        "peak_worker_rss_mb": rss,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="latest")
    ap.add_argument("--keyset", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--C", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--legacy-path-list", help="JSON list of npz paths for the npz-path comparison")
    a = ap.parse_args(argv)
    for C in (int(c) for c in a.C.split(",")):
        cfg = ShardDatasetConfig(
            root=Path(a.dataset_root),
            version=a.version,
            keyset_path=Path(a.keyset),
            batch_size=a.batch_size,
            world_size=1,
            rank=0,
            num_workers=a.workers,
            seed=0,
            shards_in_flight=C,
        )
        print(
            f"shards C={C}:",
            _measure(make_shard_dataloader(cfg, pin_memory=True), a.steps, a.batch_size),
        )
    if a.legacy_path_list:
        ds = DiffusionPlannerData(a.legacy_path_list)
        dl = DataLoader(
            ds,
            sampler=DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True),
            batch_size=a.batch_size,
            num_workers=a.workers,
            pin_memory=True,
            drop_last=True,
        )
        print("npz legacy:", _measure(dl, a.steps, a.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
