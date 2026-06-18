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
