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

from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    FullRouteRouteJob,
    JobRunResult,
    RolloutParams,
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
