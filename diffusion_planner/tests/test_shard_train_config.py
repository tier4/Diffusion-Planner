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
        valid_num_workers=0,
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
        "valid_num_workers",
    ):
        assert n in names
    assert TrainConfig.__dataclass_fields__["dataset_root"].metadata.get("cli") is True


# The six shard-loader tuning fields that used to be plain dataclass fields (invisible to
# build_parser) and are now cli()-exposed, plus the new valid_num_workers knob.
SHARD_TUNING_CLI_FIELDS = (
    "shards_in_flight",
    "shuffle_buffer",
    "shuffle_buffer_bytes",
    "shard_chunk_size",
    "shard_seek_threshold",
    "shard_max_pad_fraction",
)


def test_shard_tuning_fields_are_cli_marked():
    """The six shard-loader tuning knobs must be cli()-declared, or build_parser silently
    drops them and an operator has no flag to respond to a PlanError with."""
    from diffusion_planner.config import TrainConfig

    for name in SHARD_TUNING_CLI_FIELDS + ("valid_num_workers",):
        f = TrainConfig.__dataclass_fields__[name]
        assert f.metadata.get("cli") is True, f"{name} is not cli()-marked"
        assert f.metadata.get("help"), f"{name} has no help string"


def test_shard_tuning_flags_accepted_on_command_line():
    """Each of the six flags is parsed off the command line and lands in the built config."""
    from diffusion_planner.config import TrainConfig, build_config, build_parser

    parser = build_parser(TrainConfig, "test")
    argv = [
        "--exp_name",
        "t",
        "--shards_in_flight",
        "9",
        "--shuffle_buffer",
        "111",
        "--shuffle_buffer_bytes",
        "222",
        "--shard_chunk_size",
        "333",
        "--shard_seek_threshold",
        "0.44",
        "--shard_max_pad_fraction",
        "0.05",
    ]
    ns = parser.parse_args(argv)
    cfg = build_config(TrainConfig, ns)
    assert cfg.shards_in_flight == 9
    assert cfg.shuffle_buffer == 111
    assert cfg.shuffle_buffer_bytes == 222
    assert cfg.shard_chunk_size == 333
    assert cfg.shard_seek_threshold == 0.44
    assert cfg.shard_max_pad_fraction == 0.05


def test_shard_tuning_flags_omitted_yield_previous_defaults():
    """Omitting the flags must reproduce exactly today's (pre-cli()) defaults — i.e. the same
    constants ShardDatasetConfig itself defaults to."""
    from diffusion_planner.config import TrainConfig, build_config, build_parser
    from diffusion_planner.data_pipeline.defaults import (
        CHUNK_SIZE,
        MAX_PAD_FRACTION,
        SEEK_THRESHOLD,
        SHARDS_IN_FLIGHT,
        SHUFFLE_BUFFER_BYTES,
        SHUFFLE_BUFFER_ITEMS,
    )

    parser = build_parser(TrainConfig, "test")
    ns = parser.parse_args(["--exp_name", "t"])
    cfg = build_config(TrainConfig, ns)
    assert cfg.shards_in_flight == SHARDS_IN_FLIGHT
    assert cfg.shuffle_buffer == SHUFFLE_BUFFER_ITEMS
    assert cfg.shuffle_buffer_bytes == SHUFFLE_BUFFER_BYTES
    assert cfg.shard_chunk_size == CHUNK_SIZE
    assert cfg.shard_seek_threshold == SEEK_THRESHOLD
    assert cfg.shard_max_pad_fraction == MAX_PAD_FRACTION


def test_valid_num_workers_defaults_to_zero_and_is_cli_flag():
    from diffusion_planner.config import TrainConfig, build_config, build_parser

    parser = build_parser(TrainConfig, "test")
    ns = parser.parse_args(["--exp_name", "t"])
    cfg = build_config(TrainConfig, ns)
    assert cfg.valid_num_workers == 0

    ns2 = parser.parse_args(["--exp_name", "t", "--valid_num_workers", "3"])
    cfg2 = build_config(TrainConfig, ns2)
    assert cfg2.valid_num_workers == 3


def test_resolve_valid_num_workers_zero_inherits_num_workers():
    """0 (the default) means validation inherits the training worker count."""
    assert shard_ddp.resolve_valid_num_workers(num_workers=8, valid_num_workers=0) == 8
    assert shard_ddp.resolve_valid_num_workers(num_workers=0, valid_num_workers=0) == 0


def test_resolve_valid_num_workers_positive_overrides_for_validation_only():
    """A positive override applies to validation; training's own num_workers is untouched by
    this helper — the caller (build_loaders) is responsible for keeping args.num_workers as
    the training value."""
    assert shard_ddp.resolve_valid_num_workers(num_workers=8, valid_num_workers=1) == 1
    assert shard_ddp.resolve_valid_num_workers(num_workers=8, valid_num_workers=8) == 8


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


def test_build_loaders_valid_num_workers_overrides_only_validation(tmp_path):
    """valid_num_workers must feed BOTH the validation ShardDatasetConfig (plan slots) and
    the validation DataLoader, while training keeps num_workers in both places."""
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
        valid_num_workers=1,
    )
    tl, vl, tds, vds = shard_ddp.build_loaders(
        args, rank=0, world_size=1, batch_size_per_rank=8, save_dir=tmp_path / "run"
    )
    assert tds.cfg.num_workers == 0 and vds.cfg.num_workers == 1
    assert tl.num_workers == 0 and vl.num_workers == 1


def test_coordinated_abort_reraises_without_dist():
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError) as exc_info:
        shard_ddp.coordinated_abort(boom)
    assert exc_info.value is boom
