"""§7.3 mixing acceptance test: shard loader vs true global permutation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from diffusion_planner.utils.shard_dataset import ShardDataset, ShardDatasetConfig


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _stats(seq: list[str], batch_size: int) -> tuple[float, int]:
    parts = sorted(set(seq))
    idx = {p: i for i, p in enumerate(parts)}
    glob = np.bincount([idx[s] for s in seq], minlength=len(parts)).astype(float)
    glob /= glob.sum()
    js, max_run = [], 0
    for b in range(0, len(seq) - batch_size + 1, batch_size):
        batch = seq[b : b + batch_size]
        h = np.bincount([idx[s] for s in batch], minlength=len(parts)).astype(float)
        h /= h.sum()
        js.append(_js(h, glob))
        run = 1
        for a, c in zip(batch, batch[1:]):
            run = run + 1 if a == c else 1
            max_run = max(max_run, run)
    return float(np.mean(js)), max_run


def _loader_sequence(cfg: ShardDatasetConfig, epoch: int) -> list[str]:
    ds = ShardDataset(cfg)
    ds.set_epoch(epoch)
    seq = []
    for worker in range(max(cfg.num_workers, 1)):
        seq += [
            o.shard.partition_id
            for o, _ in ds._buffered(ds._slot_occurrences(worker, epoch), epoch, worker)
        ]
    return seq


def run(
    root,
    version,
    keyset,
    *,
    world_size,
    workers,
    batch_size,
    C_values=(1, 4),
    seeds=(0, 1, 2),
    epochs=2,
    baseline_seeds=5,
) -> dict:
    # max_pad_fraction=1.0: this diagnostic never trains on the padding replays, so a few
    # duplicated tail samples (unavoidable on small selections that don't divide the batch size
    # evenly across a single slot) cannot bias the mixing statistic; real §7-scale selections are
    # far too large for the pad fraction to matter either way.
    base_cfg = ShardDatasetConfig(
        root=Path(root),
        version=version,
        keyset_path=Path(keyset),
        batch_size=batch_size,
        world_size=world_size,
        rank=0,
        num_workers=workers,
        seed=0,
        max_pad_fraction=1.0,
    )
    ds0 = ShardDataset(base_cfg)
    population = [s.partition_id for s, idx in ds0.selection.items() for _ in idx]
    base = [
        _stats(list(np.random.default_rng(s).permutation(population)), batch_size)
        for s in range(baseline_seeds)
    ]
    b_js, b_run = np.array([b[0] for b in base]), np.array([b[1] for b in base])
    out, ok = {}, True
    for C in C_values:
        js_vals, run_vals = [], []
        for seed in seeds:
            for epoch in range(epochs):
                cfg = ShardDatasetConfig(
                    **{**base_cfg.__dict__, "seed": seed, "shards_in_flight": C}
                )
                js, run_len = _stats(_loader_sequence(cfg, epoch), batch_size)
                js_vals.append(js)
                run_vals.append(run_len)
        js_pass = np.mean(js_vals) <= b_js.mean() + 3 * b_js.std()
        run_pass = np.mean(run_vals) <= b_run.mean() + 3 * b_run.std()
        ok &= bool(js_pass and run_pass)
        out[C] = {
            "js_mean": float(np.mean(js_vals)),
            "js_baseline": (float(b_js.mean()), float(b_js.std())),
            "run_mean": float(np.mean(run_vals)),
            "run_baseline": (float(b_run.mean()), float(b_run.std())),
            "pass": bool(js_pass and run_pass),
        }
    return {"pass": ok, "per_C": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="latest")
    ap.add_argument("--keyset", required=True)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--C", default="1,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args(argv)
    rep = run(
        a.dataset_root,
        a.version,
        a.keyset,
        world_size=a.world_size,
        workers=a.workers,
        batch_size=a.batch_size,
        C_values=tuple(int(c) for c in a.C.split(",")),
        seeds=tuple(int(s) for s in a.seeds.split(",")),
        epochs=a.epochs,
    )
    for C, r in rep["per_C"].items():
        print(
            f"C={C}: JS {r['js_mean']:.4f} (baseline {r['js_baseline'][0]:.4f}±{r['js_baseline'][1]:.4f}) "
            f"run {r['run_mean']:.1f} (baseline {r['run_baseline'][0]:.1f}±{r['run_baseline'][1]:.1f}) -> {'PASS' if r['pass'] else 'FAIL'}"
        )
    print("OVERALL", "PASS" if rep["pass"] else "FAIL")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
