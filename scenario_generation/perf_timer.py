"""Lightweight always-on timing for the perception-reproducer pipeline.

Speed is a first-class requirement (we mine millions of scenes), so every hot
method is wrapped in a ``Timers`` block and the per-stage wall-clock is reported
at the end of a run. The overhead is a single ``time.perf_counter()`` pair per
block — negligible and free of any rendering / matplotlib dependency.

Usage::

    timers = Timers()
    with timers("model_forward"):
        ...
    print(timers.report(n_steps))
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import NamedTuple

# Per-rank imbalance heuristic for the DDP multi-rank report: ranks run concurrently, so the
# slowest rank (not the sum or the average) sets the real wall-clock duration of the run. This
# is a fixed diagnostic constant, not a config knob -- it only decides whether a warning line is
# printed, never any actual timing math.
_DDP_IMBALANCE_RATIO = 1.3


class Timers:
    """Accumulates wall-clock time and call counts per named stage.

    Thread-safe: the same instance is shared across the batched rollout's build
    threads, so the read-modify-write of the counters (and the report reads) are
    guarded by a lock. The overhead is negligible vs the timed work."""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._lock = threading.Lock()

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            with self._lock:
                self.total[name] = self.total.get(name, 0.0) + dt
                self.calls[name] = self.calls.get(name, 0) + 1

    def add(self, name: str, seconds: float, n: int = 1) -> None:
        """Manually fold in a measured duration (e.g. from a child run)."""
        with self._lock:
            self.total[name] = self.total.get(name, 0.0) + seconds
            self.calls[name] = self.calls.get(name, 0) + n

    def merge(self, other: "Timers") -> None:
        for name, sec in other.total.items():
            self.add(name, sec, other.calls.get(name, 1))

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, float]]) -> "Timers":
        """Reconstruct a ``Timers`` from an ``as_dict()``-shaped mapping (round-trips
        ``total_s``/``calls``; ``ms_per_call`` is re-derived, not stored). Used to re-merge
        per-rank timing data that was serialized to disk for a DDP run."""
        t = cls()
        for name, stats in d.items():
            t.add(name, float(stats["total_s"]), int(stats.get("calls", 1)))
        return t

    def as_dict(self) -> dict[str, dict[str, float]]:
        with self._lock:  # consistent snapshot of both dicts
            total = dict(self.total)
            calls = dict(self.calls)
        return {
            name: {
                "total_s": total[name],
                "calls": calls[name],
                "ms_per_call": 1e3 * total[name] / max(1, calls[name]),
            }
            for name in sorted(total, key=lambda k: -total[k])
        }

    def report(self, n_steps: int | None = None) -> str:
        d_all = self.as_dict()
        lines = ["timing (slowest first):"]
        for name, d in d_all.items():
            lines.append(
                f"  {name:24s} {d['total_s']:8.3f}s  "
                f"{d['calls']:6d} calls  {d['ms_per_call']:7.2f} ms/call"
            )
        if n_steps:
            grand = sum(d["total_s"] for d in d_all.values())
            lines.append(f"  {'TOTAL (summed)':24s} {grand:8.3f}s  over {n_steps} steps")
        return "\n".join(lines)


class MultiRankReport(NamedTuple):
    """``format_multi_rank_report``'s output: ``full`` (aggregate + every rank's own
    breakdown) for the on-disk report, ``aggregate`` (just the cross-rank section, no
    per-rank detail) for a console print that stays readable with many ranks."""

    full: str
    aggregate: str


def format_multi_rank_report(per_rank: list[dict], n_steps: int | None = None) -> MultiRankReport:
    """Render a DDP timing report from one dict per responding rank, each shaped like
    ``{"rank": int, "world_size": int, "elapsed_sec": float, "n_jobs": int, "stages": <Timers.as_dict()>}``
    (see ``ClosedLoopEvaluation._persist_rank_timers``/``collect_ddp_timers``).

    Two sections, because ranks run CONCURRENTLY on separate GPUs -- a stage's total_s
    summed across ranks is total compute-seconds spent by all workers, not wall-clock time,
    so it must never be divided by a single elapsed_sec and presented as a percentage:

    1. aggregate: cross-rank sum per stage, labeled as compute-seconds (not wall-clock),
       plus avg/rank, plus min/max/mean of each rank's own elapsed_sec so a slow straggler
       rank (which is what actually determines the run's wall-clock duration, via the
       end-of-run barrier) isn't averaged away.
    2. per-rank: each rank's own stage total_s as a fraction of THAT rank's own
       elapsed_sec -- the only statistically valid percentage here, since it's a
       same-process, same-clock ratio.
    """
    agg = Timers()
    for pr in per_rank:
        agg.merge(Timers.from_dict(pr.get("stages", {})))
    agg_dict = agg.as_dict()

    elapsed = [float(pr["elapsed_sec"]) for pr in per_rank if pr.get("elapsed_sec") is not None]
    n_ranks = len(per_rank)

    agg_lines = [
        f"=== aggregate across {n_ranks} rank(s): compute-seconds "
        "(SUM across ranks, NOT wall-clock) ==="
    ]
    for name, d in agg_dict.items():
        agg_lines.append(
            f"  {name:24s} {d['total_s']:8.3f}s  {d['calls']:6d} calls  "
            f"{d['ms_per_call']:7.2f} ms/call  {d['total_s'] / max(1, n_ranks):7.3f}s/rank avg"
        )
    if n_steps:
        grand = sum(d["total_s"] for d in agg_dict.values())
        agg_lines.append(f"  {'TOTAL (summed)':24s} {grand:8.3f}s  over {n_steps} steps")
    if elapsed:
        lo, hi, mean = min(elapsed), max(elapsed), sum(elapsed) / len(elapsed)
        agg_lines.append(
            f"  per-rank elapsed_sec: min={lo:.3f}s max={hi:.3f}s mean={mean:.3f}s"
        )
        if lo > 0 and hi / lo > _DDP_IMBALANCE_RATIO:
            agg_lines.append(
                f"  ** rank imbalance: slowest/fastest = {hi / lo:.2f}x (> {_DDP_IMBALANCE_RATIO}x) "
                "-- see per-rank breakdown below for which rank is the straggler **"
            )
    aggregate_text = "\n".join(agg_lines)

    per_rank_lines = ["", "=== per-rank breakdown (stage % of that rank's own wall time) ==="]
    for pr in sorted(per_rank, key=lambda p: p.get("rank", 0)):
        rank = pr.get("rank")
        rank_elapsed = float(pr.get("elapsed_sec") or 0.0)
        n_jobs = pr.get("n_jobs")
        per_rank_lines.append(f"rank {rank} (elapsed={rank_elapsed:.3f}s, n_jobs={n_jobs}):")
        stages = pr.get("stages") or {}
        for name in sorted(stages, key=lambda k: -stages[k]["total_s"]):
            d = stages[name]
            pct = 100.0 * d["total_s"] / rank_elapsed if rank_elapsed > 0 else float("nan")
            per_rank_lines.append(
                f"    {name:24s} {d['total_s']:8.3f}s  {d['calls']:6d} calls  "
                f"{d['ms_per_call']:7.2f} ms/call  {pct:5.1f}% of rank wall time"
            )
    full_text = aggregate_text + "\n".join(per_rank_lines)
    return MultiRankReport(full=full_text, aggregate=aggregate_text)
