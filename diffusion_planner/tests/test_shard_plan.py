import numpy as np
import pytest
from diffusion_planner.data_pipeline.errors import PlanError
from diffusion_planner.utils import shard_plan as SP


def _sel(sizes, dup_shards=1):
    sel = {}
    for i, n in enumerate(sizes):
        for d in range(dup_shards):
            sel[SP.ShardRef(f"p{i}", "rev", d)] = np.arange(n, dtype=np.int32)
    return sel


def _all_occ(plan):
    out = []
    for slot, pad in zip(plan.slots, plan.padding):
        out += [SP.Occurrence(c.shard, i, 0) for c in slot for i in c.indices] + pad
    return out


def test_chunks_and_disjoint_ownership():
    plan = SP.make_plan(
        _sel([5000, 3000, 10]),
        world_size=2,
        workers_per_rank=4,
        batch_size=64,
        chunk_size=1024,
        max_pad_fraction=0.05,
    )
    primary = [o for o in _all_occ(plan) if o.occurrence == 0]
    assert len(primary) == len(set((o.shard, o.index) for o in primary)) == 8010
    counts = [sum(c.n for c in s) for s in plan.slots]
    assert max(counts) - min(counts) <= 1024  # imbalance ≤ one chunk
    assert all(len(s) > 0 for s in plan.slots)


def test_padding_gives_whole_batches_and_equal_ranks():
    plan = SP.make_plan(
        _sel([700, 650, 640]),
        world_size=2,
        workers_per_rank=2,
        batch_size=32,
        chunk_size=128,
        max_pad_fraction=0.2,
    )
    for slot, pad in zip(plan.slots, plan.padding):
        assert (sum(c.n for c in slot) + len(pad)) == plan.target_per_slot
    assert plan.target_per_slot % 32 == 0
    assert plan.steps_per_rank == plan.target_per_slot // 32 * 2
    assert all(o.occurrence >= 1 for pad in plan.padding for o in pad)
    r0 = plan.for_slot(0, 0)
    r1 = plan.for_slot(1, 1)
    assert sum(c.n for c in r0[0]) + len(r0[1]) == sum(c.n for c in r1[0]) + len(r1[1])


def test_pad_bound_and_slot_starvation_raise():
    with pytest.raises(PlanError, match="pad"):
        SP.make_plan(
            {SP.ShardRef("p", "r", 0): np.arange(7000), SP.ShardRef("q", "r", 0): np.arange(1)},
            world_size=1,
            workers_per_rank=2,
            batch_size=64,
            chunk_size=8192,
            max_pad_fraction=0.01,
        )
    with pytest.raises(PlanError, match="fewer chunks"):
        SP.make_plan(
            _sel([10]),
            world_size=2,
            workers_per_rank=4,
            batch_size=1,
            chunk_size=1024,
            max_pad_fraction=1.0,
        )


def test_num_workers_zero_is_one_slot_per_rank():
    plan = SP.make_plan(
        _sel([300, 300]),
        world_size=2,
        workers_per_rank=0,
        batch_size=10,
        chunk_size=64,
        max_pad_fraction=0.5,
    )
    assert len(plan.slots) == 2 and plan.steps_per_rank == plan.target_per_slot // 10


def test_epoch_order_changes_but_ownership_does_not():
    plan_a = SP.make_plan(
        _sel([2000, 2000]),
        world_size=1,
        workers_per_rank=2,
        batch_size=16,
        chunk_size=256,
        max_pad_fraction=0.1,
    )
    plan_b = SP.make_plan(
        _sel([2000, 2000]),
        world_size=1,
        workers_per_rank=2,
        batch_size=16,
        chunk_size=256,
        max_pad_fraction=0.1,
    )
    assert plan_a == plan_b  # deterministic
    e0 = SP.epoch_order(plan_a.slots[0], seed=1, epoch=0, rank=0, worker=0)
    e1 = SP.epoch_order(plan_a.slots[0], seed=1, epoch=1, rank=0, worker=0)
    assert sorted(map(hash, e0)) == sorted(map(hash, e1)) and e0 != e1
    assert SP.epoch_order(plan_a.slots[0], 1, 0, 0, 0) == e0
