"""§7.3 mixing acceptance test: shard loader vs true global permutation.

Measures both within-batch composition and cross-rank composition: every rank in
`0..world_size-1` gets its own `ShardDataset` (mirroring how each real DDP rank instantiates its
own dataset with its own `cfg.rank` in production), and every rank's per-batch partition-id
histogram is compared against the *same* dataset-wide reference histogram used by the baseline
permutation — never against a histogram derived from that rank's own (necessarily partial)
stream, which would hide any skew introduced by how chunks are assigned across ranks/workers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diffusion_planner.data_pipeline.defaults import MAX_PAD_FRACTION
from diffusion_planner.utils.shard_dataset import ShardDataset, ShardDatasetConfig


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon *distance*: sqrt of the base-2 JS divergence. Symmetric in (p, q); bins
    where a side is zero are excluded from that side's KL term (0*log(0) := 0)."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    divergence = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(divergence, 0.0)))  # clamp tiny negative float noise


def _stats(
    seq: list[str],
    batch_size: int,
    reference_hist: np.ndarray,
    part_index: dict[str, int],
) -> tuple[float, int]:
    """Per-batch JS distance of `seq` against the shared `reference_hist` (the dataset-wide
    partition-id distribution, not one derived from `seq` itself), plus the max same-partition
    run length within any batch."""
    js, max_run = [], 0
    for b in range(0, len(seq) - batch_size + 1, batch_size):
        batch = seq[b : b + batch_size]
        h = np.bincount([part_index[s] for s in batch], minlength=len(reference_hist)).astype(float)
        h /= h.sum()
        js.append(_js(h, reference_hist))
        run = 1
        for a, c in zip(batch, batch[1:]):
            run = run + 1 if a == c else 1
            max_run = max(max_run, run)
    return float(np.mean(js)), max_run


def _loader_sequence(cfg: ShardDatasetConfig, epoch: int) -> list[list[str]]:
    """One sequence per rank in `0..cfg.world_size-1` (that rank's workers concatenated in worker
    order). A separate `ShardDataset` is built per rank — planning is rank-independent, but
    `cfg.rank` is baked into a `ShardDataset` instance at construction and drives both slot
    selection and the per-slot RNG streams, so this is how each real DDP rank would see its own
    stream."""
    per_rank = []
    for rank in range(max(cfg.world_size, 1)):
        rank_cfg = ShardDatasetConfig(**{**cfg.__dict__, "rank": rank})
        ds = ShardDataset(rank_cfg)
        ds.set_epoch(epoch)
        seq = []
        for worker in range(max(cfg.num_workers, 1)):
            seq += [
                o.shard.partition_id
                for o, _ in ds._buffered(ds._slot_occurrences(worker, epoch), epoch, worker)
            ]
        per_rank.append(seq)
    return per_rank


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
    max_pad_fraction=MAX_PAD_FRACTION,
) -> dict:
    base_cfg = ShardDatasetConfig(
        root=Path(root),
        version=version,
        keyset_path=Path(keyset),
        batch_size=batch_size,
        world_size=world_size,
        rank=0,
        num_workers=workers,
        seed=0,
        max_pad_fraction=max_pad_fraction,
    )
    ds0 = ShardDataset(base_cfg)
    population = [s.partition_id for s, idx in ds0.selection.items() for _ in idx]
    total_members = sum(len(idx) for idx in ds0.selection.values())
    total_padding = sum(len(p) for p in ds0.plan.padding)
    pad_fraction = total_padding / total_members if total_members else 0.0

    parts = sorted(set(population))
    part_index = {p: i for i, p in enumerate(parts)}
    reference_hist = np.bincount([part_index[s] for s in population], minlength=len(parts)).astype(
        float
    )
    reference_hist /= reference_hist.sum()

    base = [
        _stats(
            list(np.random.default_rng(s).permutation(population)),
            batch_size,
            reference_hist,
            part_index,
        )
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
                for rank_seq in _loader_sequence(cfg, epoch):
                    js, run_len = _stats(rank_seq, batch_size, reference_hist, part_index)
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
    return {"pass": ok, "per_C": out, "pad_fraction": pad_fraction}


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
    ap.add_argument("--max-pad-fraction", type=float, default=MAX_PAD_FRACTION)
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
        max_pad_fraction=a.max_pad_fraction,
    )
    for C, r in rep["per_C"].items():
        print(
            f"C={C}: JS {r['js_mean']:.4f} (baseline {r['js_baseline'][0]:.4f}±{r['js_baseline'][1]:.4f}) "
            f"run {r['run_mean']:.1f} (baseline {r['run_baseline'][0]:.1f}±{r['run_baseline'][1]:.1f}) -> {'PASS' if r['pass'] else 'FAIL'}"
        )
    print(f"pad_fraction: {rep['pad_fraction']:.4f}")
    print("OVERALL", "PASS" if rep["pass"] else "FAIL")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
