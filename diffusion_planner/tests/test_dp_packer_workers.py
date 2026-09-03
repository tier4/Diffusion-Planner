import hashlib
import io
import os
import pickle
import re
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import pytest
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.errors import PackWorkerError, PlanError
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from tests.dp_fixtures import make_tree

LAYOUT = [
    ("pA/mX/manual/2026-01-01/t1/route_0", 12, "full"),
    ("pA/mX/manual/2026-01-02/t1/route_0", 5, "skip"),
    ("psim/loc_seed_1/manual/seed_1/bag_0/r", 4, "psim"),
    ("pB/mY/manual/2026-01-03/t1/route_0", 3, "none"),
]


def _opts(src, dst, tag, base="none", **kw):
    return PK.PackOptions(
        source=src,
        dest=dst,
        base=base,
        tag=tag,
        rule=PartitionRule(depth=4),
        shard_size_bytes=64 * 1024,
        **kw,
    )


def _fake_job(pid: str) -> PK.WorkerJob:
    """A minimal WorkerJob for tests that exercise `_run_builds` directly, against a
    monkeypatched `_build_partition` that never touches the source tree."""
    return PK.WorkerJob(
        build_dir=Path("/unused"),
        partition_id=pid,
        samples=[],
        base_entry=None,
        base_manifest_path=None,
        shard_size_bytes=1,
        seed=0,
        drop_skipped=True,
        with_neighbor_ids=False,
        force=False,
    )


def _sleepy_worker(job):
    """Sleeps longer for earlier-indexed jobs, so completion order is the reverse of job
    (submission) order — proving that any pass/fail of the ordering assertion downstream
    is about `_run_builds`'s reassembly, not an accident of scheduling.
    """
    idx = int(job.partition_id.rsplit("-", 1)[-1])
    time.sleep(0.2 * (3 - idx))
    return SimpleNamespace(entry=SimpleNamespace(partition_id=job.partition_id))


def _die_worker(job):
    """Kills its own process hard, with no exception and no cleanup — the shape of an
    OOM kill, as opposed to a Python exception raised inside the worker."""
    os._exit(1)


def _boom_worker(job):
    """Stand-in for `_build_partition` that always fails.

    Defined at module scope (not nested in the test) so it survives pickling under the
    `spawn` context: pickle serializes a plain function by (module, qualname) reference,
    and a closure defined inside a test function has no such stable path — the child
    process fails to reconstruct it (PicklingError), which would test pickling, not the
    worker-failure path this test is meant to exercise.
    """
    raise ValueError(f"synthetic failure in {job.partition_id}")


def test_worker_job_excludes_path_list_and_is_picklable(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    huge = [f"pA/mX/manual/2026-01-01/t1/route_0/{i:08d}.npz" for i in range(3)]
    opts = _opts(src, tmp_path / "dst", "v1", path_list=huge)
    root = DatasetRoot(opts.dest)
    root.ensure_layout()
    job = PK._job_for(
        opts,
        tmp_path / "build",
        None,
        root,
        "pA/mX/manual/2026-01-01",
        [],
    )
    fields = set(vars(job))
    assert "path_list" not in fields
    assert "include" not in fields and "exclude" not in fields
    assert "partitions" not in fields and "sync" not in fields
    blob = pickle.dumps(job)
    assert pickle.loads(blob).partition_id == "pA/mX/manual/2026-01-01"
    # the job must not drag the path list along by any route
    assert b"00000000.npz" not in blob


def _shard_digests(root, version):
    out = {}
    for e in version.partitions.values():
        d = root.shards_dir_for(e.pid, e.data_rev)
        for name in e.shards:
            out[f"{e.partition_id}/{name}"] = hashlib.sha256((d / name).read_bytes()).hexdigest()
    return out


def test_parallel_pack_is_byte_identical_to_serial(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    PK.pack(_opts(src, tmp_path / "a", "v1"))
    PK.pack(_opts(src, tmp_path / "b", "v1", workers=4))
    ra, rb = DatasetRoot(tmp_path / "a"), DatasetRoot(tmp_path / "b")
    va, vb = ra.read_version("v1"), rb.read_version("v1")
    assert set(va.partitions) == set(vb.partitions)
    for pid, ea in va.partitions.items():
        eb = vb.partitions[pid]
        assert (ea.data_rev, ea.meta_rev, ea.sample_count) == (
            eb.data_rev,
            eb.meta_rev,
            eb.sample_count,
        )
    assert _shard_digests(ra, va) == _shard_digests(rb, vb)


def test_worker_failure_publishes_nothing_and_names_the_partition(tmp_path, monkeypatch):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    dst = tmp_path / "dst"

    monkeypatch.setattr(PK, "_build_partition", _boom_worker)
    with pytest.raises(PackWorkerError) as ei:
        PK.pack(_opts(src, dst, "v1", workers=2))
    assert "synthetic failure" in str(ei.value)
    assert not (DatasetRoot(dst).versions_dir / "v1.json").exists()


def test_pool_uses_spawn(tmp_path, monkeypatch):
    seen = {}
    real = PK.futures.ProcessPoolExecutor

    class Spy(real):
        def __init__(self, *a, **kw):
            seen["ctx"] = type(kw.get("mp_context")).__name__
            super().__init__(*a, **kw)

    monkeypatch.setattr(PK.futures, "ProcessPoolExecutor", Spy)
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    PK.pack(_opts(src, tmp_path / "dst", "v1", workers=2))
    assert seen["ctx"] == "SpawnContext"


def test_run_builds_returns_job_order_not_completion_order(monkeypatch):
    """`_run_builds` must return results in job order regardless of completion order.

    Nothing downstream of `pack()` would catch a regression to completion order (the
    version's `partitions` dict is built by partition id, not position, and per-partition
    RNG is seeded independently), so this asserts the invariant directly against
    `_run_builds`, using jobs whose completion order is deliberately the reverse of their
    submission order.
    """
    monkeypatch.setattr(PK, "_build_partition", _sleepy_worker)
    jobs = [_fake_job(f"job-{i}") for i in range(4)]
    out = PK._run_builds(jobs, 4, lambda job, build: None)
    assert [b.entry.partition_id for b in out] == [j.partition_id for j in jobs]


def test_worker_death_publishes_nothing(tmp_path, monkeypatch):
    """A worker that dies without raising (the OOM-killer shape, via `os._exit`) must
    surface as `PackWorkerError` through the `BrokenProcessPool` path, not hang or crash
    the parent, and must not publish anything."""
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    dst = tmp_path / "dst"

    monkeypatch.setattr(PK, "_build_partition", _die_worker)
    with pytest.raises(PackWorkerError):
        PK.pack(_opts(src, dst, "v1", workers=2))
    assert not (DatasetRoot(dst).versions_dir / "v1.json").exists()


def test_require_sidecars_fails_before_publishing(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)  # the "none" variant partition has no sidecars
    dst = tmp_path / "dst"
    with pytest.raises(PlanError) as ei:
        PK.pack(_opts(src, dst, "v1", require_sidecars=True))
    assert "sidecar" in str(ei.value)
    root = DatasetRoot(dst)
    assert not (root.versions_dir / "v1.json").exists()
    assert root.latest() is None


def test_missing_sidecars_still_allowed_by_default(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    v = PK.pack(_opts(src, tmp_path / "dst", "v1"))
    assert v.tag == "v1"


def test_progress_reports_samples_and_projection():
    buf = io.StringIO()
    prog = PK._Progress(total_partitions=2, total_samples=10, out=buf, every_s=0.0)

    class J:
        def __init__(self, n):
            self.samples = [None] * n

    class B:
        pass

    prog(J(4), B())
    prog(J(6), B())
    text = buf.getvalue()
    assert "1/2 partitions" in text
    assert "4/10 samples" in text
    assert "2/2 partitions" in text and "10/10 samples" in text
    assert re.search(r"eta \d+:\d\d:\d\d", text)


def _slow_worker(job):
    """Sleeps longer than the short heartbeat timeout the heartbeat test injects, so
    `_run_builds`'s `futures.wait` is guaranteed to time out at least once before this
    worker's result arrives. Module-level (not a nested closure), same reasoning as
    `_sleepy_worker`/`_boom_worker` above: it must survive pickling under `spawn`.
    """
    time.sleep(0.3)
    return SimpleNamespace(entry=SimpleNamespace(partition_id=job.partition_id))


def test_heartbeat_emitted_when_pool_goes_quiet(monkeypatch):
    """A hung/slow worker must not leave `_run_builds` printing nothing: with a short
    injected `heartbeat_timeout`, at least one heartbeat line fires on the progress object
    before the (still correct) result comes back.
    """
    monkeypatch.setattr(PK, "_build_partition", _slow_worker)
    buf = io.StringIO()
    prog = PK._Progress(total_partitions=1, total_samples=1, out=buf, every_s=0.0)
    jobs = [_fake_job("job-0")]
    out = PK._run_builds(jobs, 2, prog, heartbeat_timeout=0.05)
    text = buf.getvalue()
    assert "HEARTBEAT" in text
    assert text.index("HEARTBEAT") < text.index("1/1 partitions")
    assert [b.entry.partition_id for b in out] == ["job-0"]


def test_null_progress_is_safe_when_progress_disabled(tmp_path):
    """`_NullProgress` (what `pack()` hands to `_run_builds` when `opts.progress` is False)
    must tolerate both the per-partition call and the heartbeat call with no output and no
    error — it stands in for a plain lambda that has no `heartbeat` method.
    """
    prog = PK._NullProgress()
    prog(_fake_job("job-0"), object())
    prog.heartbeat(3)

    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    v = PK.pack(_opts(src, tmp_path / "dst", "v1", progress=False))
    assert v.tag == "v1"


def test_pid_collision_is_refused_before_packing(tmp_path, monkeypatch):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    monkeypatch.setattr(PK.P, "pid_of", lambda _pid: "collide")
    with pytest.raises(PlanError) as ei:
        PK.pack(_opts(src, tmp_path / "dst", "v1"))
    assert "pid collision" in str(ei.value)


def test_pid_injective_accepts_distinct_ids():
    PK._check_pid_injective(["a/b", "a/c", "d/e"])


def test_pack_rejects_workers_below_one_at_the_api_level(tmp_path):
    """I8: `PackOptions`/`pack()` must validate `--workers` themselves, not rely on the CLI's
    early check alone. `pack_shards.py` rejects `workers < 1` before ever calling `pack()`,
    but `pack_bench` builds `PackOptions` directly and got no validation at all -- `--workers 0`
    silently ran serially while reporting `workers: 0`, and `--workers 0,8` divided by
    `baseline_w == 0` downstream. The rejection must happen before any dest scaffolding, same
    as the CLI's own early check.
    """
    src = tmp_path / "src"
    make_tree(src, LAYOUT[:1])
    dst = tmp_path / "dst"
    with pytest.raises(ValueError, match="workers"):
        PK.pack(_opts(src, dst, "v1", workers=0))
    assert not dst.exists()


def _quick_worker(job):
    """Returns immediately -- no staggered sleep needed for
    `test_broken_pool_on_refill_submit_becomes_pack_worker_error`: with a 5-job list and
    `workers=2`, the initial fill always consumes exactly 4 jobs (`workers * 2`) via 4
    strictly sequential `submit()` calls made by a plain `for` loop, before the pool's own
    concurrency can matter at all. Whichever future completes first, the *one* possible
    refill (of the 1 job left over) is therefore always the 5th `submit()` call -- regardless
    of completion order or scheduling.
    """
    return SimpleNamespace(entry=SimpleNamespace(partition_id=job.partition_id))


def test_broken_pool_on_refill_submit_becomes_pack_worker_error(monkeypatch):
    """I2: the refill `ex.submit(...)` inside `_run_builds`'s per-completed-future loop sits
    outside the `try`/`except` that wraps `fut.result()`. If the pool has already broken by
    the time a future is drained and refilled, `submit()` itself raises `BrokenProcessPool`
    synchronously, and (before this fix) that escaped `_run_builds` entirely as a raw
    `BrokenProcessPool`, not the `PackWorkerError` the CLI, the spec, and the runbook's abort
    criteria all promise.

    Reproduced by forcing the 5th `submit()` call to raise `BrokenProcessPool` directly (see
    `_quick_worker` for why that call is deterministically the first refill), rather than
    actually killing a worker -- killing a worker only reliably exercises the `fut.result()`
    path, already covered by `test_worker_death_publishes_nothing`.
    """
    monkeypatch.setattr(PK, "_build_partition", _quick_worker)
    real = PK.futures.ProcessPoolExecutor

    class BreaksOnRefillSubmit(real):
        def __init__(self, *a, **kw):
            self._n = 0
            super().__init__(*a, **kw)

        def submit(self, *a, **kw):
            self._n += 1
            if self._n == 5:
                raise BrokenProcessPool("simulated break on refill submit")
            return super().submit(*a, **kw)

    monkeypatch.setattr(PK.futures, "ProcessPoolExecutor", BreaksOnRefillSubmit)
    jobs = [_fake_job(f"job-{i}") for i in range(5)]
    with pytest.raises(PackWorkerError):
        PK._run_builds(jobs, 2, lambda job, build: None)
