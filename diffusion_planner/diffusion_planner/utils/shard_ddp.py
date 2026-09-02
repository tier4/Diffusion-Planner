"""Wiring between TrainConfig / DDP and ShardDataset (spec §5 preflight, §6 integration)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist

from diffusion_planner.data_pipeline.keyset import materialize_keyset
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from diffusion_planner.utils.shard_dataset import ShardDataset, ShardDatasetConfig


def validate_args(args) -> str:
    if args.dataset_root and args.train_set_list:
        raise ValueError("--dataset_root and --train_set_list are mutually exclusive")
    if not args.dataset_root:
        return "npz"
    for split in ("train", "valid"):
        ks, flt = getattr(args, f"{split}_key_set"), getattr(args, f"{split}_shard_filter")
        if bool(ks) == bool(flt):
            raise ValueError(
                f"shard mode needs exactly one of --{split}_key_set / --{split}_shard_filter"
            )
    return "shards"


def _initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def resolve_keysets(args, save_dir: Path, rank: int) -> tuple[Path, Path]:
    save_dir = Path(save_dir)
    out = []
    for split in ("train", "valid"):
        ks, flt = getattr(args, f"{split}_key_set"), getattr(args, f"{split}_shard_filter")
        if ks:
            out.append(Path(ks))
            continue
        target = save_dir / f"{split}_keyset.parquet"
        if rank == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            materialize_keyset(DatasetRoot(args.dataset_root), args.dataset_version, flt, target)
        out.append(target)
    if _initialized():
        dist.barrier()
    return out[0], out[1]


def build_loaders(args, rank: int, world_size: int, batch_size_per_rank: int, save_dir: Path):
    """Construct the train/valid ShardDatasets and their DataLoaders (spec §5/§6 preflight).

    `set_epoch` must be called on the returned datasets before each epoch; persistent workers
    read the shared epoch value at `__iter__`.
    """
    train_ks, valid_ks = resolve_keysets(args, save_dir, rank)
    common = dict(
        root=Path(args.dataset_root),
        version=args.dataset_version,
        batch_size=batch_size_per_rank,
        world_size=world_size,
        rank=rank,
        num_workers=args.num_workers,
        seed=args.seed,
        shards_in_flight=args.shards_in_flight,
        shuffle_buffer_items=args.shuffle_buffer,
        shuffle_buffer_bytes=args.shuffle_buffer_bytes,
        chunk_size=args.shard_chunk_size,
        seek_threshold=args.shard_seek_threshold,
        max_pad_fraction=args.shard_max_pad_fraction,
    )
    train_cfg = ShardDatasetConfig(keyset_path=train_ks, shuffle=True, **common)
    valid_cfg = ShardDatasetConfig(keyset_path=valid_ks, shuffle=False, **common)
    train_ds = ShardDataset(train_cfg)  # preflight on every rank, before any collective
    valid_ds = ShardDataset(valid_cfg)
    if _initialized():  # all ranks must have resolved the same version bytes
        h = torch.tensor(
            [int(train_ds.version_hash[:15], 16)],
            dtype=torch.int64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        h_max = h.clone()
        dist.all_reduce(h_max, op=dist.ReduceOp.MAX)
        h_min = h.clone()
        dist.all_reduce(h_min, op=dist.ReduceOp.MIN)
        if not torch.equal(h_max, h_min):
            raise RuntimeError("ranks resolved different dataset versions")
    if rank == 0:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "shard_run_record.json").write_text(
            json.dumps(train_ds.run_record(), indent=2, default=str)
        )

    def _dl(ds: ShardDataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size_per_rank,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            in_order=True,
            persistent_workers=args.num_workers > 0,
        )

    return _dl(train_ds), _dl(valid_ds), train_ds, valid_ds


def coordinated_abort(exc: BaseException) -> None:
    """Log, tear down the process group best-effort, and re-raise. No collectives: a failed
    rank cannot rendezvous with peers blocked in DDP's gradient all-reduce, and torchrun
    terminates the remaining ranks on our non-zero exit."""
    print(
        f"[shard loader] fatal: {exc!r} — aborting; torchrun will terminate the other ranks",
        flush=True,
    )
    if _initialized():
        try:
            dist.destroy_process_group()
        except Exception as teardown_exc:  # never mask the original failure
            print(f"[shard loader] destroy_process_group failed: {teardown_exc!r}", flush=True)
    raise exc
