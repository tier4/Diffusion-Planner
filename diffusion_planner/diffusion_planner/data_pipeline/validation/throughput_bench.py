"""§7.5 throughput: shard loader (C sweep) vs legacy npz loader; prints samples/s, TTFB, peak worker RSS."""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

from torch.utils.data import DataLoader, DistributedSampler

from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.shard_dataset import ShardDatasetConfig, make_shard_dataloader


def _vm_hwm_kb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])  # kB
    except OSError:
        return None
    return None


def _peak_worker_rss_mb(loader_iter) -> tuple[float | None, str | None]:
    """Peak RSS (VmHWM) across the DataLoader's worker processes, read while they are still alive.

    This relies on two implementation details, not a public API: the private
    `_MultiProcessingDataLoaderIter._workers` attribute (a list of `multiprocessing.Process`
    handles) and Linux's `/proc/<pid>/status`. `RUSAGE_CHILDREN` is not usable here — it reads
    ~0 while workers are still alive and, worse, accumulates across the whole C sweep rather than
    reporting one config's peak. If either implementation detail is unavailable (num_workers=0,
    a non-Linux host, or a torch version that renamed the attribute) this returns `(None, note)`
    rather than a misleading number.
    """
    workers = getattr(loader_iter, "_workers", None)
    if not workers:
        return (
            None,
            "no worker processes (num_workers=0, or this DataLoader iterator has no `_workers`)",
        )
    hwm_kb = []
    for w in workers:
        pid = getattr(w, "pid", None)
        if pid is None:
            continue
        v = _vm_hwm_kb(pid)
        if v is not None:
            hwm_kb.append(v)
    if not hwm_kb:
        return (
            None,
            "/proc/<pid>/status VmHWM unavailable (non-Linux host, or workers already exited)",
        )
    return max(hwm_kb) / 1024, None


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
    peak_worker_rss_mb, note = _peak_worker_rss_mb(it)  # read before the iterator is torn down
    main_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = {
        "samples_per_s": (n - 1) * batch_size / dt if dt > 0 else float("nan"),
        "ttfb_s": ttfb,
        "peak_worker_rss_mb": peak_worker_rss_mb,
        "main_rss_mb": main_rss_mb,
    }
    if note is not None:
        result["peak_worker_rss_note"] = note
    del it  # release worker processes/pipes now, before the caller also drops `loader`
    return result


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
        loader = make_shard_dataloader(cfg, pin_memory=True)
        print(f"shards C={C}:", _measure(loader, a.steps, a.batch_size))
        del loader  # tear down this C's workers before the next DataLoader is constructed
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
        del dl
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
