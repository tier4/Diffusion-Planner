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

import time
from contextlib import contextmanager


class Timers:
    """Accumulates wall-clock time and call counts per named stage."""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.total[name] = self.total.get(name, 0.0) + dt
            self.calls[name] = self.calls.get(name, 0) + 1

    def add(self, name: str, seconds: float, n: int = 1) -> None:
        """Manually fold in a measured duration (e.g. from a child run)."""
        self.total[name] = self.total.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + n

    def merge(self, other: "Timers") -> None:
        for name, sec in other.total.items():
            self.add(name, sec, other.calls.get(name, 1))

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "total_s": self.total[name],
                "calls": self.calls[name],
                "ms_per_call": 1e3 * self.total[name] / max(1, self.calls[name]),
            }
            for name in sorted(self.total, key=lambda k: -self.total[k])
        }

    def report(self, n_steps: int | None = None) -> str:
        lines = ["timing (slowest first):"]
        for name, d in self.as_dict().items():
            lines.append(
                f"  {name:24s} {d['total_s']:8.3f}s  "
                f"{d['calls']:6d} calls  {d['ms_per_call']:7.2f} ms/call"
            )
        if n_steps:
            grand = sum(self.total.values())
            lines.append(f"  {'TOTAL (summed)':24s} {grand:8.3f}s  over {n_steps} steps")
        return "\n".join(lines)
