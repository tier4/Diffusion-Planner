import argparse

import pytest
from diffusion_planner.utils import shard_ddp


def _args(**kw):
    base = dict(
        train_set_list="",
        valid_set_list="",
        dataset_root="",
        dataset_version="latest",
        train_key_set="",
        valid_key_set="",
        train_shard_filter="",
        valid_shard_filter="",
        pin_mem=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_train_config_has_shard_fields():
    from diffusion_planner.config import TrainConfig

    names = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    for n in (
        "dataset_root",
        "dataset_version",
        "train_key_set",
        "valid_key_set",
        "train_shard_filter",
        "valid_shard_filter",
        "shards_in_flight",
        "shuffle_buffer",
        "shard_chunk_size",
        "shard_max_pad_fraction",
    ):
        assert n in names
    assert TrainConfig.__dataclass_fields__["dataset_root"].metadata.get("cli") is True


def test_validate_args_modes():
    assert shard_ddp.validate_args(_args(train_set_list="a.json", valid_set_list="b.json")) == "npz"
    assert (
        shard_ddp.validate_args(
            _args(dataset_root="/d", train_key_set="t.parquet", valid_shard_filter="1=1")
        )
        == "shards"
    )
    with pytest.raises(ValueError):
        shard_ddp.validate_args(
            _args(train_set_list="a.json", dataset_root="/d", train_key_set="t", valid_key_set="v")
        )
    with pytest.raises(ValueError):
        shard_ddp.validate_args(
            _args(dataset_root="/d", train_key_set="t", train_shard_filter="1=1", valid_key_set="v")
        )
    with pytest.raises(ValueError):
        shard_ddp.validate_args(
            _args(dataset_root="/d", train_key_set="t")
        )  # valid selection missing


def test_build_loaders_single_rank(tmp_path):
    from diffusion_planner.data_pipeline import packer as PK
    from diffusion_planner.data_pipeline.partition import PartitionRule
    from tests.dp_fixtures import make_tree

    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, [("pA/mX/manual/2026-01-01/t1/r", 120, "full")])
    PK.pack(
        PK.PackOptions(
            source=src,
            dest=dst,
            base="none",
            tag="v1",
            rule=PartitionRule(depth=4),
            shard_size_bytes=64 * 1024,
        )
    )
    args = _args(
        dataset_root=str(dst),
        train_shard_filter="is_skipped IS NOT TRUE",
        valid_shard_filter="is_skipped IS NOT TRUE",
        shards_in_flight=2,
        shuffle_buffer=20,
        shuffle_buffer_bytes=512 << 20,
        shard_chunk_size=16,
        shard_seek_threshold=0.2,
        shard_max_pad_fraction=0.5,
        seed=1,
        num_workers=0,
    )
    tl, vl, tds, vds = shard_ddp.build_loaders(
        args, rank=0, world_size=1, batch_size_per_rank=8, save_dir=tmp_path / "run"
    )
    assert (tmp_path / "run/train_keyset.parquet").exists() and len(tl) == tds.steps_per_epoch
    assert sum(1 for _ in vl) == len(vl) and vds.cfg.shuffle is False
