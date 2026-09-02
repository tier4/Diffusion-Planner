"""ShardDataset: DDP-safe IterableDataset over versioned tar shards (spec §5)."""

from __future__ import annotations

import functools
import hashlib
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from diffusion_planner.data_pipeline import encoding
from diffusion_planner.data_pipeline import tar_shards as T
from diffusion_planner.data_pipeline.defaults import (
    CHUNK_SIZE,
    MAX_PAD_FRACTION,
    READ_RETRIES,
    SEEK_THRESHOLD,
    SHARDS_IN_FLIGHT,
    SHUFFLE_BUFFER_BYTES,
    SHUFFLE_BUFFER_ITEMS,
)
from diffusion_planner.data_pipeline.errors import IntegrityError, PlanError
from diffusion_planner.data_pipeline.keyset import keyset_digest, load_keyset
from diffusion_planner.data_pipeline.manifest import read_manifest
from diffusion_planner.data_pipeline.versioning import DatasetRoot, Version
from diffusion_planner.utils.shard_plan import (
    Chunk,
    Occurrence,
    Plan,
    ShardRef,
    epoch_order,
    make_plan,
)


@dataclass
class ShardDatasetConfig:
    root: Path
    version: str
    keyset_path: Path
    batch_size: int
    world_size: int
    rank: int
    num_workers: int
    seed: int
    shuffle: bool = True
    shards_in_flight: int = SHARDS_IN_FLIGHT
    shuffle_buffer_items: int = SHUFFLE_BUFFER_ITEMS
    shuffle_buffer_bytes: int = SHUFFLE_BUFFER_BYTES
    chunk_size: int = CHUNK_SIZE
    seek_threshold: float = SEEK_THRESHOLD
    max_pad_fraction: float = MAX_PAD_FRACTION
    read_retries: int = READ_RETRIES
    verify_reads: bool = False
    sample_meta_columns: tuple[str, ...] = ()  # existing manifest columns only (spec §5.9)


def load_selection(
    root: DatasetRoot, version: Version, keyset: pa.Table
) -> dict[ShardRef, np.ndarray]:
    pids = keyset.column("partition_id").to_numpy(zero_copy_only=False)
    sids = keyset.column("shard_id").to_numpy()
    idxs = keyset.column("sample_index_in_shard").to_numpy()
    out: dict[ShardRef, list[int]] = {}
    for p, s, i in zip(pids, sids, idxs):
        out.setdefault(ShardRef(str(p), version.partitions[str(p)].data_rev, int(s)), []).append(
            int(i)
        )
    return {k: np.array(sorted(v), dtype=np.int32) for k, v in out.items()}


class ShardDataset(IterableDataset):
    def __init__(self, cfg: ShardDatasetConfig, plan: Plan | None = None):
        self.cfg = cfg
        self.root = DatasetRoot(cfg.root)
        self.version = self.root.read_version(cfg.version)  # preflight: version resolves
        self.version_hash = self.root.version_hash(self.version.tag)
        keyset = load_keyset(
            cfg.keyset_path, self.root, self.version.tag
        )  # preflight: bound to this version
        self.keyset_digest = keyset_digest(keyset)
        self.selection = load_selection(self.root, self.version, keyset)
        self.plan = plan or make_plan(
            self.selection,
            world_size=cfg.world_size,
            workers_per_rank=cfg.num_workers,
            batch_size=cfg.batch_size,
            chunk_size=cfg.chunk_size,
            max_pad_fraction=cfg.max_pad_fraction,
        )  # raises PlanError when empty
        self._epoch = mp.Value("i", 0)
        self._sha: dict[ShardRef, dict[int, bytes]] = {}
        self._meta: dict[ShardRef, dict[int, dict[str, int]]] = {}
        # Build code dictionaries ONCE in the main process from manifest columns,
        # so codes are identical across workers/ranks/epochs (finding #4).
        self._meta_codes: dict[str, dict] = {}
        if cfg.sample_meta_columns:
            for _pid, e in sorted(self.version.partitions.items()):
                mpath = self.root.manifest_path_for(e.pid, e.data_rev, e.meta_rev)
                t = read_manifest(mpath, columns=list(cfg.sample_meta_columns))
                for col in cfg.sample_meta_columns:
                    codes = self._meta_codes.setdefault(col, {})
                    for v in sorted(set(t.column(col).to_pylist()), key=lambda x: (x is None, x)):
                        codes.setdefault(v, len(codes))

    @property
    def meta_dictionary(self) -> dict[str, dict]:
        return self._meta_codes

    # ---- bookkeeping -------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return self.plan.samples_per_rank

    @property
    def steps_per_epoch(self) -> int:
        return self.plan.steps_per_rank

    def run_record(self) -> dict:
        return {
            **asdict(self.cfg),
            "version_hash": self.version_hash,
            "keyset_digest": self.keyset_digest,
            "root": str(self.cfg.root),
            "keyset_path": str(self.cfg.keyset_path),
        }

    # ---- per-shard metadata (cached per worker) -----------------------------------------
    @functools.lru_cache(maxsize=64)
    def _shard_table(self, shard: ShardRef) -> tuple[Path, dict[int, tuple[int, int]], int]:
        """(shard path, {member index: (offset, size)}, members in shard). Loads only planning columns,
        plus `payload_sha256` when verify_reads, plus `sample_meta_columns` codes when requested."""
        e = self.version.partitions[shard.partition_id]
        cols = ["shard_id", "sample_index_in_shard", "offset", "size"]
        cols += ["payload_sha256"] if self.cfg.verify_reads else []
        cols += list(self.cfg.sample_meta_columns)
        t = read_manifest(self.root.manifest_path_for(e.pid, e.data_rev, e.meta_rev), columns=cols)
        mask = t.column("shard_id").to_numpy() == shard.shard_id
        sub = t.filter(pa.array(mask))
        idx = sub.column("sample_index_in_shard").to_numpy()
        off = sub.column("offset").to_numpy()
        size = sub.column("size").to_numpy()
        if self.cfg.verify_reads:
            self._sha[shard] = dict(
                zip((int(i) for i in idx), sub.column("payload_sha256").to_pylist())
            )
        if (
            self.cfg.sample_meta_columns
        ):  # integer codes; one dictionary per column for the whole run
            per_member: dict[int, dict[str, int]] = {int(i): {} for i in idx}
            for col in self.cfg.sample_meta_columns:
                codes = self._meta_codes.setdefault(col, {})
                for i, v in zip(idx, sub.column(col).to_pylist()):
                    per_member[int(i)][col] = codes.setdefault(v, len(codes))
            self._meta[shard] = per_member
        path = self.root.shards_dir_for(e.pid, e.data_rev) / e.shards[shard.shard_id]
        return path, {int(i): (int(o), int(s)) for i, o, s in zip(idx, off, size)}, int(mask.sum())

    # ---- reading -----------------------------------------------------------------------
    def _retrying(self, fn, *args):
        last: Exception | None = None
        for _ in range(self.cfg.read_retries + 1):
            try:
                return fn(*args)
            except (OSError, IntegrityError) as e:  # ESTALE, short read, checksum
                last = e
        raise IntegrityError(f"read failed after {self.cfg.read_retries} retries: {last}")

    def _seek_read(self, path: Path, offset: int, size: int) -> bytes:
        def _do():
            with open(path, "rb") as f:
                return T.read_member(f, offset, size)

        return self._retrying(_do)

    def _chunk_stream(self, chunk: Chunk) -> Iterator[tuple[Occurrence, bytes]]:
        path, table, n_members = self._shard_table(chunk.shard)
        wanted = set(chunk.indices)
        if chunk.n / max(n_members, 1) >= self.cfg.seek_threshold:
            # Skim path: stream the tar and assert offset/size integrity for wanted members.
            # On OSError/IntegrityError mid-skim, fall back to _seek_read for remaining indices.
            remaining = set(chunk.indices)
            last = chunk.indices[-1]
            try:
                for idx, offset_data, sz, payload in T.iter_members(path):
                    if idx in wanted:
                        expect_off, expect_sz = table[idx]
                        if (offset_data, sz) != (expect_off, expect_sz):
                            raise IntegrityError(
                                f"skim: member {idx} offset/size ({offset_data}, {sz}) "
                                f"!= manifest ({expect_off}, {expect_sz})"
                            )
                        remaining.discard(idx)
                        yield Occurrence(chunk.shard, idx, 0), payload
                    if idx >= last:
                        break
            except (OSError, IntegrityError):
                # Fall back to seek-read for any remaining wanted indices
                for idx in sorted(remaining):
                    off, size = table[idx]
                    yield Occurrence(chunk.shard, idx, 0), self._seek_read(path, off, size)
        else:
            for idx in chunk.indices:
                off, size = table[idx]
                yield Occurrence(chunk.shard, idx, 0), self._seek_read(path, off, size)

    def _decode(self, occ: Occurrence, payload: bytes) -> dict[str, np.ndarray]:
        if self.cfg.verify_reads:
            self._shard_table(occ.shard)
            if hashlib.sha256(payload).digest() != self._sha[occ.shard][occ.index]:
                raise IntegrityError(f"payload sha mismatch for {occ.shard} member {occ.index}")
        try:
            sample = encoding.decode_for_training(
                payload
            )  # zstd frame checksum is verified on every read
        except IntegrityError:
            # Decode failure: re-read the member via _seek_read (which retries) and try again.
            path, table, _ = self._shard_table(occ.shard)
            off, size = table[occ.index]
            payload = self._seek_read(path, off, size)
            sample = encoding.decode_for_training(payload)
        if self.cfg.sample_meta_columns:
            self._shard_table(occ.shard)
            sample["meta"] = dict(
                self._meta[occ.shard][occ.index]
            )  # integer codes; not fed to the model
        return sample

    # ---- iteration ---------------------------------------------------------------------
    def _slot_occurrences(self, worker: int, epoch: int) -> Iterator[tuple[Occurrence, bytes]]:
        chunks, padding = self.plan.for_slot(self.cfg.rank, worker)
        cfg = self.cfg
        queue = (
            epoch_order(chunks, cfg.seed, epoch, cfg.rank, worker) if cfg.shuffle else list(chunks)
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, epoch, cfg.rank, worker, 1])
        )  # "interleave"
        streams: list[tuple[Iterator, list[int]]] = []  # (iterator, [remaining])
        qi = 0
        while qi < len(queue) and len(streams) < max(cfg.shards_in_flight, 1):
            streams.append((self._chunk_stream(queue[qi]), [queue[qi].n]))
            qi += 1
        while streams:
            total = sum(r[0] for _, r in streams)
            if cfg.shuffle:
                u = int(rng.integers(0, total))
                k = 0
                while u >= streams[k][1][0]:
                    u -= streams[k][1][0]
                    k += 1
            else:
                k = 0
            it, rem = streams[k]
            try:
                item = next(it)
                rem[0] -= 1
                yield item
            except StopIteration:
                rem[0] = 0
            if rem[0] <= 0:
                streams.pop(k)
                if qi < len(queue):
                    streams.append((self._chunk_stream(queue[qi]), [queue[qi].n]))
                    qi += 1
        for occ in padding:  # replays: fetched by seek (spec §5.6)
            path, table, _ = self._shard_table(occ.shard)
            off, size = table[occ.index]
            yield occ, self._seek_read(path, off, size)

    def _buffered(self, items: Iterator[tuple[Occurrence, bytes]], epoch: int, worker: int):
        cfg = self.cfg
        if not cfg.shuffle:
            yield from items
            return
        rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, epoch, cfg.rank, worker, 2])
        )  # "buffer"
        buf: list[tuple[Occurrence, bytes]] = []
        nbytes = 0
        for item in items:
            if len(item[1]) > cfg.shuffle_buffer_bytes:
                yield item  # oversized: admitted alone
                continue
            buf.append(item)
            nbytes += len(item[1])
            if len(buf) > cfg.shuffle_buffer_items or nbytes > cfg.shuffle_buffer_bytes:
                j = int(rng.integers(0, len(buf)))
                out = buf[j]
                buf[j] = buf[-1]
                buf.pop()
                nbytes -= len(out[1])
                yield out
        while buf:
            j = int(rng.integers(0, len(buf)))
            out = buf[j]
            buf[j] = buf[-1]
            buf.pop()
            yield out

    def _iter_occurrences(self) -> Iterator[tuple[Occurrence, dict[str, np.ndarray]]]:
        info = get_worker_info()
        worker = info.id if info is not None else 0
        epoch = int(self._epoch.value)
        for occ, payload in self._buffered(self._slot_occurrences(worker, epoch), epoch, worker):
            yield occ, self._decode(occ, payload)

    def __iter__(self):
        for _, sample in self._iter_occurrences():
            yield sample


def make_shard_dataloader(cfg: ShardDatasetConfig, *, pin_memory: bool) -> DataLoader:
    ds = ShardDataset(cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        in_order=True,
        persistent_workers=cfg.num_workers > 0,
    )


# ---- test/debug helpers (keys are never needed in the hot loop) -----------------------------
def _key_lookup(ds: ShardDataset) -> dict[tuple[str, int, int], str]:
    out = {}
    for pid, e in ds.version.partitions.items():
        t = read_manifest(
            ds.root.manifest_path_for(e.pid, e.data_rev, e.meta_rev),
            columns=["key", "shard_id", "sample_index_in_shard"],
        )
        for k, s, i in zip(
            t.column("key").to_pylist(),
            t.column("shard_id").to_pylist(),
            t.column("sample_index_in_shard").to_pylist(),
        ):
            out[(pid, s, i)] = k
    return out


def _debug_slot_keys(ds: ShardDataset, worker: int) -> list[str]:
    lookup = _key_lookup(ds)
    return [
        lookup[(o.shard.partition_id, o.shard.shard_id, o.index)]
        for o, _ in ds._buffered(
            ds._slot_occurrences(worker, int(ds._epoch.value)), int(ds._epoch.value), worker
        )
    ]


def _debug_iter_with_keys(ds: ShardDataset):
    lookup = _key_lookup(ds)
    for o, sample in ds._iter_occurrences():
        yield lookup[(o.shard.partition_id, o.shard.shard_id, o.index)], sample
