"""Deterministic (rank, worker) planning over shard chunks (spec §5.4–5.7). No torch here."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from diffusion_planner.data_pipeline.errors import PlanError


@dataclass(frozen=True, order=True)
class ShardRef:
    partition_id: str
    data_rev: str
    shard_id: int


@dataclass(frozen=True)
class Chunk:
    shard: ShardRef
    indices: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class Occurrence:
    shard: ShardRef
    index: int
    occurrence: int


def n_slots(world_size: int, workers_per_rank: int) -> int:
    return world_size * max(workers_per_rank, 1)


def slot_id(rank: int, worker: int, workers_per_rank: int) -> int:
    return rank * max(workers_per_rank, 1) + worker


def build_chunks(selection: dict[ShardRef, np.ndarray], chunk_size: int) -> list[Chunk]:
    chunks = []
    for shard in sorted(selection):
        idx = np.unique(np.asarray(selection[shard], dtype=np.int64))
        for start in range(0, len(idx), chunk_size):
            chunks.append(Chunk(shard, tuple(int(i) for i in idx[start : start + chunk_size])))
    return chunks


def assign_slots(chunks: list[Chunk], n: int) -> list[list[Chunk]]:
    if len(chunks) < n:
        raise PlanError(
            f"fewer chunks ({len(chunks)}) than (rank, worker) slots ({n}); "
            "use fewer workers or a smaller --chunk-size"
        )
    slots: list[list[Chunk]] = [[] for _ in range(n)]
    loads = [0] * n
    for c in sorted(chunks, key=lambda c: (-c.n, c.shard, c.indices[0])):
        s = min(range(n), key=lambda i: (loads[i], i))
        slots[s].append(c)
        loads[s] += c.n
    return slots


def pad_plan(
    slots: list[list[Chunk]], batch_size: int, max_pad_fraction: float
) -> tuple[list[list[Occurrence]], int]:
    counts = [sum(c.n for c in s) for s in slots]
    target = math.ceil(max(counts) / batch_size) * batch_size
    padding: list[list[Occurrence]] = []
    for s, count in zip(slots, counts):
        need = target - count
        members = [(c.shard, i) for c in s for i in c.indices]
        pad = [
            Occurrence(sh, i, 1 + k // len(members))
            for k, (sh, i) in enumerate(members[j % len(members)] for j in range(need))
        ]
        padding.append(pad)
    total_pad = sum(len(p) for p in padding)
    total = sum(counts)
    if total and total_pad / total > max_pad_fraction:
        raise PlanError(
            f"padding would duplicate {total_pad}/{total} samples "
            f"({100 * total_pad / total:.2f}% > {100 * max_pad_fraction:.2f}% max pad fraction); "
            "use a smaller --chunk-size, fewer workers, or a smaller batch"
        )
    return padding, target


def epoch_order(chunks: list[Chunk], seed: int, epoch: int, rank: int, worker: int) -> list[Chunk]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, rank, worker]))
    return [chunks[i] for i in rng.permutation(len(chunks))]


@dataclass
class Plan:
    slots: list[list[Chunk]]
    padding: list[list[Occurrence]]
    target_per_slot: int
    batch_size: int
    workers_per_rank: int
    world_size: int

    @property
    def steps_per_rank(self) -> int:
        return self.target_per_slot // self.batch_size * max(self.workers_per_rank, 1)

    @property
    def samples_per_rank(self) -> int:
        return self.target_per_slot * max(self.workers_per_rank, 1)

    def for_slot(self, rank: int, worker: int) -> tuple[list[Chunk], list[Occurrence]]:
        s = slot_id(rank, worker, self.workers_per_rank)
        return self.slots[s], self.padding[s]


def make_plan(
    selection: dict[ShardRef, np.ndarray],
    *,
    world_size: int,
    workers_per_rank: int,
    batch_size: int,
    chunk_size: int,
    max_pad_fraction: float,
) -> Plan:
    if not selection or all(len(v) == 0 for v in selection.values()):
        raise PlanError("empty selection")
    chunks = build_chunks(selection, chunk_size)
    slots = assign_slots(chunks, n_slots(world_size, workers_per_rank))
    padding, target = pad_plan(slots, batch_size, max_pad_fraction)
    return Plan(slots, padding, target, batch_size, workers_per_rank, world_size)
