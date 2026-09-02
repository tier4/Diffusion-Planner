from collections import Counter

import numpy as np
import pytest
import torch
from diffusion_planner.data_pipeline import keyset as K
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.encoding import arrays_bitexact, load_npz_bytes
from diffusion_planner.data_pipeline.errors import IntegrityError, KeysetMismatchError, PlanError
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from diffusion_planner.utils import shard_dataset as SD
from tests.dp_fixtures import make_tree
from torch.utils.data import DataLoader

LAYOUT = [
    ("pA/mX/manual/2026-01-01/t1/r", 260, "full"),
    ("pB/mY/manual/2026-01-02/t1/r", 130, "psim"),
]


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sd")
    src, dst = tmp / "src", tmp / "dst"
    keys = make_tree(src, LAYOUT)
    PK.pack(
        PK.PackOptions(
            source=src,
            dest=dst,
            base="none",
            tag="v1",
            rule=PartitionRule(depth=4),
            shard_size_bytes=96 * 1024,
        )
    )
    ks = K.materialize_keyset(DatasetRoot(dst), "v1", "is_skipped IS NOT TRUE", tmp / "all.parquet")
    return src, dst, keys, ks


def _cfg(dst, ks, **kw):
    base = dict(
        root=dst,
        version="v1",
        keyset_path=ks,
        batch_size=8,
        world_size=2,
        rank=0,
        num_workers=2,
        seed=3,
        chunk_size=32,
        shards_in_flight=2,
        shuffle_buffer_items=50,
        max_pad_fraction=0.2,
    )
    base.update(kw)
    return SD.ShardDatasetConfig(**base)


def _keys_seen(ds):
    return [d["__key__"] for d in SD._debug_iter_with_keys(ds)]


def test_output_contract_and_bitexact(packed):
    src, dst, keys, ks = packed
    ds = SD.ShardDataset(_cfg(dst, ks, world_size=1, num_workers=0, shuffle=False))
    seen = 0
    for key, sample in SD._debug_iter_with_keys(ds):
        assert (
            set(sample) == set(load_npz_bytes((src / f"{key}.npz").read_bytes())) - {"version"}
            and len(sample) == 17
        )
        assert arrays_bitexact(
            sample | {"version": load_npz_bytes((src / f"{key}.npz").read_bytes())["version"]},
            load_npz_bytes((src / f"{key}.npz").read_bytes()),
        )
        seen += 1
    assert seen == ds.plan.samples_per_rank == len(ds)


def test_ranks_partition_all_samples_with_bounded_padding(packed):
    src, dst, keys, ks = packed
    occ = Counter()
    for rank in range(2):
        ds = SD.ShardDataset(_cfg(dst, ks, rank=rank))
        for worker in range(2):
            occ.update(SD._debug_slot_keys(ds, worker))
    assert set(occ) == set(keys)  # every selected sample appears at least once
    dup = sum(v - 1 for v in occ.values())
    assert dup <= 0.2 * len(keys)


def test_skim_and_seek_yield_identical_multisets(packed):
    src, dst, keys, ks = packed
    a = SD.ShardDataset(
        _cfg(dst, ks, world_size=1, num_workers=0, seek_threshold=0.0)
    )  # always skim
    b = SD.ShardDataset(
        _cfg(dst, ks, world_size=1, num_workers=0, seek_threshold=1.01)
    )  # always seek
    assert Counter(SD._debug_slot_keys(a, 0)) == Counter(SD._debug_slot_keys(b, 0))


def test_epoch_reshuffle_is_deterministic_and_changes_order(packed):
    src, dst, keys, ks = packed
    ds = SD.ShardDataset(_cfg(dst, ks, world_size=1, num_workers=0))
    ds.set_epoch(0)
    e0 = SD._debug_slot_keys(ds, 0)
    ds.set_epoch(0)
    assert SD._debug_slot_keys(ds, 0) == e0
    ds.set_epoch(1)
    e1 = SD._debug_slot_keys(ds, 0)
    assert Counter(e0) == Counter(e1) and e0 != e1


def test_dataloader_len_matches_steps(packed):
    src, dst, keys, ks = packed
    ds = SD.ShardDataset(_cfg(dst, ks, world_size=2, rank=1, num_workers=2))
    dl = SD.make_shard_dataloader(
        _cfg(dst, ks, world_size=2, rank=1, num_workers=2), pin_memory=False
    )
    batches = list(dl)
    assert len(batches) == len(dl) == ds.plan.steps_per_rank
    assert all(b["lanes"].shape[0] == 8 for b in batches) and isinstance(
        batches[0]["lanes"], torch.Tensor
    )


def test_preflight_errors(packed, tmp_path):
    src, dst, keys, ks = packed
    with pytest.raises(PlanError):
        SD.ShardDataset(
            _cfg(
                dst,
                K.materialize_keyset(
                    DatasetRoot(dst),
                    "v1",
                    "project_id = 'projA' AND timestamp < 0",
                    tmp_path / "e.parquet",
                ),
            )
        )
    PK.remove(dst, "v1", "v2", ["pB/mY/manual/2026-01-02"])
    with pytest.raises(KeysetMismatchError):
        SD.ShardDataset(_cfg(dst, ks, version="v2"))


def test_metadata_passthrough_is_opt_in(packed):
    src, dst, keys, ks = packed
    ds = SD.ShardDataset(
        _cfg(dst, ks, world_size=1, num_workers=0, sample_meta_columns=("project_id",))
    )
    key, sample = next(SD._debug_iter_with_keys(ds))
    assert len(sample) == 18 and set(sample["meta"]) == {"project_id"}
    expected_value = "projA" if key.startswith("pA") else None
    assert ds.meta_dictionary["project_id"][expected_value] == sample["meta"]["project_id"]


def test_persistent_read_fault_raises_after_retries(packed, monkeypatch):
    src, dst, keys, ks = packed
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise OSError(116, "Stale file handle")

    monkeypatch.setattr(SD.T, "read_member", boom)
    ds = SD.ShardDataset(
        _cfg(dst, ks, world_size=1, num_workers=0, seek_threshold=1.01, read_retries=2)
    )
    with pytest.raises(IntegrityError):
        list(SD._debug_iter_with_keys(ds))
    assert calls["n"] == 3  # 1 attempt + 2 retries
