"""Tests for the closed-loop DDP shard/merge path.

The multi-rank tests spawn real ``gloo`` process groups on CPU, so they exercise the actual
barrier / all-reduce / merge code rather than a mock of it, and still run in seconds without a
GPU. They drive :class:`FullRouteClosedLoopEvaluation` with ``run_job`` / ``discover_jobs``
stubbed out: everything below those two -- shard-file naming, the streaming writer,
``collect_ddp_shards``, ``merge_ddp_shards``, ``aggregate`` -- is production code.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from scenario_generation.closed_loop_ddp import balance_items, plan_fingerprint, shard_items
from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    FullRouteRouteJob,
    JobRunResult,
    RolloutParams,
    run_evaluations_ddp,
)

# Route frame counts from the production manifest (path_list_closed_loop_x2.json), measured
# 2026-07-31: sites of 2 / 1 / 14 / 2 routes x 2 object modes = 8 combos, 38 jobs. Real numbers
# because the skew (396..6353 frames) is what the sharding has to cope with.
_SPECS = [
    {"name": "fukuyama", "routes": {"fk_a": 3010, "fk_b": 6353}},
    {"name": "nishishinjuku", "routes": {"ns_a": 890}},
    {
        "name": "tsukuba",
        "routes": {
            f"tk_{i:02d}": n
            for i, n in enumerate(
                [396, 541, 571, 599, 615, 670, 763, 917, 987, 989, 999, 1027, 1344, 1548]
            )
        },
    },
    {"name": "odaiba", "routes": {"od_a": 4370, "od_b": 5118}},
]
_SPECS = [
    {"name": f"{s['name']}{suffix}", "routes": s["routes"]}
    for suffix in ("", "__noobj")
    for s in _SPECS
]
# The unit of work is (combo, route): the same route key appears under both object modes and
# those are two independent rollouts.
_ALL_UNITS = sorted(f"{s['name']}/{r}" for s in _SPECS for r in s["routes"])

_SLEEP_PER_FRAME_SEC = 2e-5


def _segment_row(route: str, n_steps: int) -> dict:
    """Minimal row carrying every nested block ``aggregate`` requires."""
    vals = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    block = {
        "miss_thresh_m": 0.5,
        "collision_steps": 1,
        "collision_count": 1,
        "miss_steps": 2,
        "miss_count": 1,
        "clearance_min_m": 0.1,
        "clearance_mean_m": float(vals.mean()),
        "clearance_p5_m": float(np.percentile(vals, 5)),
        "clearance_finite_steps": int(vals.size),
    }
    return {
        "route": route,
        "n_steps_run": n_steps,
        "terminated": "goal",
        "route_completion": 1.0,
        "mean_gt_deviation_m": 0.5,
        "object": dict(block),
        "road_border": dict(block),
        "red_light_violation": {"steps": 1, "count": 1},
        "strong_brake": {
            "thresh_mps2": -2.5,
            "strongest_mps2": float("inf"),
            "steps": 0,
            "count": 0,
        },
        "reproducer": {
            "expand_count": 1,
            "snap_count": 0,
            "normal_steps": n_steps - 1,
            "repeat_steps": 1,
        },
    }


class StubEvaluation(FullRouteClosedLoopEvaluation):
    """Real shard writing / merging; ``run_job`` replaced by a canned row.

    ``fail_on`` makes exactly one route raise, to test that one rank's exception surfaces on
    every rank instead of parking them at the barrier.
    """

    def __init__(self, *args, route_specs=None, fail_on=None, combo_name="", **kwargs):
        super().__init__(*args, **kwargs)
        self._route_specs = route_specs or {}
        self._fail_on = fail_on
        self.combo_name = combo_name
        self.ran: list[str] = []
        self.ran_cost = 0.0

    def discover_jobs(self):
        return [
            FullRouteRouteJob(
                job_id=key,
                npz_root=Path("/nonexistent"),
                route_key=key,
                route_paths=[Path(f"{key}_{i}.npz") for i in range(n)],
            )
            for key, n in sorted(self._route_specs.items())
        ]

    def run_job(self, job, *, segments_file=None, digest_file=None):
        assert isinstance(job, FullRouteRouteJob)
        if self._fail_on is not None and job.route_key == self._fail_on:
            raise ValueError(f"synthetic failure on {job.route_key}")
        self.ran.append(f"{self.combo_name}/{job.route_key}")
        self.ran_cost += len(job.route_paths)
        # Stand-in for the rollout, proportional to route length (1.3s for the whole 38-job set).
        time.sleep(len(job.route_paths) * _SLEEP_PER_FRAME_SEC)
        row = _segment_row(job.route_key, len(job.route_paths))
        if segments_file is not None:
            segments_file.write(json.dumps(row, default=float) + "\n")
            segments_file.flush()
        return JobRunResult(rows=[row])


def _make_evaluators(out_root: Path, specs: list[dict], rank: int, world: int, fail_on=None):
    return [
        StubEvaluation(
            None,
            None,
            ClosedLoopEvalConfig(
                out_dir=out_root / spec["name"],
                params=RolloutParams(device="cpu"),
                verbose=False,
                max_jobs=spec.get("max_jobs"),
            ),
            Path("/nonexistent"),
            ddp_rank=rank,
            ddp_world_size=world,
            route_specs=spec["routes"],
            fail_on=fail_on,
            combo_name=spec["name"],
        )
        for spec in specs
    ]


def _worker(rank: int, world: int, out_root: str, init_file: str, mode: str, q):
    """Run every evaluator through ``run_distributed`` (one barrier per evaluator)."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            world_size=world,
            rank=rank,
            timeout=datetime.timedelta(seconds=60),
        )
        fail_on = "tk_07" if mode == "fail" else None
        evaluators = _make_evaluators(Path(out_root), _SPECS, rank, world, fail_on=fail_on)
        error = ""
        summaries = []
        try:
            for ev in evaluators:
                summaries.append(ev.run_distributed())
        except Exception as exc:  # noqa: BLE001 - reported back to the parent
            error = f"{type(exc).__name__}: {exc}"
        q.put(
            {
                "rank": rank,
                "error": error,
                "ran": sorted(u for ev in evaluators for u in ev.ran),
                "n_segments": [s.get("n_segments") for s in summaries],
                "routes_merged": [
                    sorted({row["route"] for row in (s.get("segments") or [])}) for s in summaries
                ],
            }
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_world(tmp_path: Path, world: int, mode: str = "ok", worker=_worker) -> list[dict]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    out_root = tmp_path / "out"
    init_file = tmp_path / "pg_init"
    procs = [
        ctx.Process(target=worker, args=(r, world, str(out_root), str(init_file), mode, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = []
    try:
        for _ in procs:
            # Generous but finite: a regression here is a hang, and the point of the test is
            # that a hang fails the suite instead of wedging it.
            results.append(q.get(timeout=120))
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
    return sorted(results, key=lambda r: r["rank"])


@pytest.mark.parametrize("world", [2, 4])
def test_run_distributed_merges_every_shard(tmp_path, world):
    """Happy path, so the failure tests below cannot pass vacuously."""
    results = _run_world(tmp_path, world)
    assert [r["error"] for r in results] == [""] * world
    rank0 = results[0]
    assert rank0["n_segments"] == [len(s["routes"]) for s in _SPECS]
    for spec, routes in zip(_SPECS, rank0["routes_merged"]):
        assert routes == sorted(spec["routes"])
    for r in results[1:]:
        assert r["n_segments"] == [None] * len(_SPECS), "non-zero rank returned a summary"
    ran = [unit for r in results for unit in r["ran"]]
    assert sorted(ran) == _ALL_UNITS


def test_rank_failure_raises_everywhere_instead_of_hanging(tmp_path):
    """One rank raising must fail the whole group, promptly.

    Without the barrier-then-all-reduce, the healthy ranks sit on the barrier until the
    process-group timeout (10000s in production) -- so this asserts on the *error text* of every
    rank, not merely that they errored: a barrier timeout would also produce an error.
    """
    started = time.perf_counter()
    results = _run_world(tmp_path, 4, mode="fail")
    elapsed = time.perf_counter() - started
    for r in results:
        assert "rank(s) failed" in r["error"], f"rank {r['rank']} reported {r['error']!r}"
    assert elapsed < 45, f"took {elapsed:.0f}s -- ranks waited out a barrier instead of failing"


def test_rank_failure_does_not_publish_a_truncated_merge(tmp_path):
    """The quiet half of the bug: the dead rank's partial shard file still exists.

    ``execute_jobs`` writes inside a ``with``, so a raising rank's ``segments_{rank}.jsonl`` is
    flushed and closed on the way out. ``collect_ddp_shards`` only checks for *missing* files, so
    without the all-reduce rank-0 merges the truncation and reports it as a clean summary.
    """
    results = _run_world(tmp_path, 4, mode="fail")
    failing_idx = [s["name"] for s in _SPECS].index("tsukuba")
    for r in results:
        # Every rank stopped at the same evaluator, and none of them produced a summary for it.
        assert len(r["n_segments"]) == failing_idx, (
            f"rank {r['rank']} produced {len(r['n_segments'])} summaries, "
            f"expected to abort at evaluator {failing_idx}"
        )
    # The truncated shard really is on disk -- i.e. the missing-file check could not have saved
    # us, and something had to actively refuse the merge.
    shards = sorted((tmp_path / "out" / "tsukuba").glob("segments_*.jsonl"))
    assert shards, "expected the failing combo's shard files to exist on disk"
    assert not (tmp_path / "out" / "tsukuba" / "summary.json").exists(), (
        "rank-0 published a summary built from a truncated shard"
    )


def test_missing_process_group_fails_before_running_anything(tmp_path):
    """``world_size>1`` with no process group: a misconfigured launch should cost a second.

    Without a barrier there is nothing to guarantee the other ranks finished writing their
    shards, so merging would be unsound -- and discovering that only *after* a 48-minute
    evaluation is a pointless way to spend a GPU allocation.
    """
    assert not dist.is_initialized()
    evaluator = _make_evaluators(tmp_path / "out", _SPECS[:1], rank=0, world=4)[0]
    with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
        evaluator.run_distributed()
    assert evaluator.ran == [], "rollouts ran before the process-group check"


def test_world_size_one_uses_the_sequential_path(tmp_path):
    """The default for anyone not on multiple GPUs: no barrier, no rank-suffixed shards."""
    assert not dist.is_initialized()
    evaluators = _make_evaluators(tmp_path / "out", _SPECS, rank=0, world=1)
    summaries = [ev.run_distributed() for ev in evaluators]
    assert [s["n_segments"] for s in summaries] == [len(s["routes"]) for s in _SPECS]
    for spec in _SPECS:
        d = tmp_path / "out" / spec["name"]
        assert (d / "segments.jsonl").is_file()
        assert not list(d.glob("segments_*.jsonl")), "world=1 must not write rank-suffixed shards"
        assert (d / "summary.json").is_file()


def test_world_size_one_propagates_exceptions_unchanged(tmp_path):
    """No collectives exist to coordinate with, so the original exception must not be swallowed."""
    evaluators = _make_evaluators(tmp_path / "out", _SPECS[2:3], rank=0, world=1, fail_on="tk_07")
    with pytest.raises(ValueError, match="synthetic failure on tk_07"):
        evaluators[0].run_distributed()


# ---------------------------------------------------------------------------------------------
# Pooled, cost-balanced sharding across every evaluator (one barrier for the whole set)
# ---------------------------------------------------------------------------------------------


def _cost(item):
    return item[1]


def _key(item):
    return item[0]


def test_balance_items_is_a_partition():
    items = [(f"j{i:02d}", float(i % 7 + 1)) for i in range(23)]
    for world in (1, 2, 3, 4, 8, 32):
        buckets = balance_items(items, world, cost=_cost, key=_key)
        assert len(buckets) == max(world, 1)
        flat = [it for b in buckets for it in b]
        assert sorted(flat, key=_key) == sorted(items, key=_key)


def test_balance_items_is_deterministic_regardless_of_input_order():
    """Ranks enumerate independently; only agreeing plans form a partition."""
    items = [(f"j{i:02d}", float(i % 5)) for i in range(20)]
    base = balance_items(items, 4, cost=_cost, key=_key)
    assert balance_items(list(reversed(items)), 4, cost=_cost, key=_key) == base
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = [items[i] for i in rng.permutation(len(items))]
        assert balance_items(perm, 4, cost=_cost, key=_key) == base


def test_balance_items_allows_empty_buckets():
    """Fewer jobs than ranks is normal (a 1-route site on 4 GPUs), not an error."""
    assert [len(b) for b in balance_items([("only", 1.0)], 4, cost=_cost, key=_key)] == [1, 0, 0, 0]


def test_pooled_balancing_beats_per_combo_round_robin():
    """The reason this exists, on the real manifest's route lengths.

    Baseline = the pre-existing wiring: ``shard_items`` round-robin inside each combo with a
    barrier per combo, so the makespan is ``sum over combos of max over ranks``. A 1-route combo
    cannot be split under that scheme no matter how many GPUs are attached.
    """
    world = 4
    per_combo = [[(f"{s['name']}/{r}", float(n)) for r, n in s["routes"].items()] for s in _SPECS]
    baseline = sum(
        max(sum(c for _, c in shard_items(sorted(cb, key=_key), r, world)) for r in range(world))
        for cb in per_combo
    )
    pooled = [it for cb in per_combo for it in cb]
    new = max(sum(c for _, c in b) for b in balance_items(pooled, world, cost=_cost, key=_key))
    total = sum(c for _, c in pooled)
    assert new <= (total / world) * 1.05, f"pooled makespan {new} vs ideal {total / world}"
    # Measured on these numbers: per-combo round-robin 1.96x, pooled 3.95x of sequential cost.
    assert total / baseline < 2.2
    assert total / new > 3.5


def test_plan_fingerprint_is_order_sensitive_and_stable():
    assert plan_fingerprint(["a", "b"]) == plan_fingerprint(["a", "b"])
    assert plan_fingerprint(["a", "b"]) != plan_fingerprint(["b", "a"])
    assert plan_fingerprint(["a", "b"]) != plan_fingerprint(["a", "b", "c"])
    assert 0 <= plan_fingerprint(["a"]) < 2**63


def test_full_route_job_cost_is_route_length():
    job = FullRouteRouteJob(job_id="r", route_paths=[Path(f"{i}.npz") for i in range(37)])
    assert job.cost() == 37.0


def _worker_pooled(rank: int, world: int, out_root: str, init_file: str, mode: str, q):
    """Run every evaluator through the pooled path (ONE barrier for the whole set)."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            world_size=world,
            rank=rank,
            timeout=datetime.timedelta(seconds=60),
        )
        fail_on = "tk_07" if mode == "fail" else None
        evaluators = _make_evaluators(Path(out_root), _SPECS, rank, world, fail_on=fail_on)
        error = ""
        summaries = []
        try:
            summaries = run_evaluations_ddp(evaluators, rank=rank, world_size=world)
        except Exception as exc:  # noqa: BLE001 - reported back to the parent
            error = f"{type(exc).__name__}: {exc}"
        q.put(
            {
                "rank": rank,
                "error": error,
                "ran": sorted(u for ev in evaluators for u in ev.ran),
                "cost": sum(ev.ran_cost for ev in evaluators),
                "n_segments": [s.get("n_segments") for s in summaries],
                "elapsed_sec": [s.get("elapsed_sec") for s in summaries],
                "routes_merged": [
                    sorted({row["route"] for row in (s.get("segments") or [])}) for s in summaries
                ],
            }
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_pooled_runs_every_job_exactly_once(tmp_path, world):
    """The partition invariant, end to end: nothing run twice, nothing skipped."""
    results = _run_world(tmp_path, world, worker=_worker_pooled)
    assert [r["error"] for r in results] == [""] * world
    ran = [unit for r in results for unit in r["ran"]]
    assert len(ran) == len(set(ran)), "a job was executed on more than one rank"
    assert sorted(ran) == _ALL_UNITS


@pytest.mark.parametrize("world", [2, 4])
def test_pooled_merge_matches_the_full_route_set(tmp_path, world):
    results = _run_world(tmp_path, world, worker=_worker_pooled)
    rank0 = results[0]
    assert rank0["n_segments"] == [len(s["routes"]) for s in _SPECS]
    for spec, routes in zip(_SPECS, rank0["routes_merged"]):
        assert routes == sorted(spec["routes"])
    for r in results[1:]:
        assert r["n_segments"] == [None] * len(_SPECS), "non-zero rank returned a summary"


@pytest.mark.parametrize("world", [2, 4])
def test_pooled_balances_cost_across_ranks(tmp_path, world):
    """One barrier means the slowest rank sets the wall clock, so balance is the whole point."""
    results = _run_world(tmp_path, world, worker=_worker_pooled)
    costs = [r["cost"] for r in results]
    ideal = sum(costs) / world
    assert max(costs) <= ideal * 1.05, f"per-rank costs {costs}, ideal {ideal:.0f}"


def test_pooled_survives_ranks_with_no_jobs(tmp_path):
    """world=8 over combos of 1-2 routes: most ranks idle in most combos.

    A rank with zero jobs for a combo still has to create its empty ``segments_{rank}.jsonl``,
    or ``collect_ddp_shards``'s missing-file check turns ordinary imbalance into a crash.
    """
    results = _run_world(tmp_path, 8, worker=_worker_pooled)
    assert [r["error"] for r in results] == [""] * 8
    assert results[0]["n_segments"] == [len(s["routes"]) for s in _SPECS]
    for spec in _SPECS:
        shards = sorted((tmp_path / "out" / spec["name"]).glob("segments_*.jsonl"))
        assert len(shards) == 8, f"{spec['name']}: {len(shards)} shard files, expected 8"


def test_pooled_elapsed_sec_is_attributed_per_combo(tmp_path):
    """Each site's ``elapsed_sec`` must stay "what this site cost", not the call's makespan.

    Pooling shares one ``t0`` across every evaluator, so the naive version hands the same number
    to every ``merge_ddp_shards`` and all 8 sites report an identical (and far too large) time,
    which then flows into the per-site wandb metric and the training log.
    """
    results = _run_world(tmp_path, 4, worker=_worker_pooled)
    elapsed = results[0]["elapsed_sec"]
    assert len(set(elapsed)) == len(elapsed), f"all combos reported the same time: {elapsed}"
    # The stub sleeps a fixed amount per frame, so seconds-per-frame must come out the same for
    # every combo. A shared makespan would blow this apart by ~13x largest-to-smallest.
    frames = [sum(s["routes"].values()) for s in _SPECS]
    rates = [e / f for e, f in zip(elapsed, frames)]
    assert max(rates) / min(rates) < 1.25, f"elapsed {elapsed} not proportional to frames {frames}"


def test_pooled_rank_failure_raises_everywhere(tmp_path):
    started = time.perf_counter()
    results = _run_world(tmp_path, 4, mode="fail", worker=_worker_pooled)
    elapsed = time.perf_counter() - started
    for r in results:
        assert "rank(s) failed" in r["error"], f"rank {r['rank']} reported {r['error']!r}"
    assert all(r["n_segments"] == [] for r in results), "a rank published a summary anyway"
    assert elapsed < 45, f"took {elapsed:.0f}s -- ranks waited out a barrier instead of failing"


def test_pooled_world_size_one_uses_the_sequential_path(tmp_path):
    assert not dist.is_initialized()
    evaluators = _make_evaluators(tmp_path / "out", _SPECS, rank=0, world=1)
    summaries = run_evaluations_ddp(evaluators, rank=0, world_size=1)
    assert [s["n_segments"] for s in summaries] == [len(s["routes"]) for s in _SPECS]
    for spec in _SPECS:
        d = tmp_path / "out" / spec["name"]
        assert (d / "segments.jsonl").is_file()
        assert not list(d.glob("segments_*.jsonl")), "world=1 must not write rank-suffixed shards"


def test_pooled_world_size_gt_one_without_process_group_refuses(tmp_path):
    evaluators = _make_evaluators(tmp_path / "out", _SPECS[:1], rank=0, world=4)
    with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
        run_evaluations_ddp(evaluators, rank=0, world_size=4)


def test_pooled_max_jobs_is_honoured(tmp_path):
    """``run()`` is bypassed, so ``max_jobs`` must be applied when building the pooled plan."""
    specs = [{"name": "tsukuba", "routes": _SPECS[2]["routes"], "max_jobs": 3}]
    evaluators = _make_evaluators(tmp_path / "out", specs, rank=0, world=1)
    assert run_evaluations_ddp(evaluators, rank=0, world_size=1)[0]["n_segments"] == 3
