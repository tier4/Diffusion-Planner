from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def sustained_true_indices(mask: np.ndarray, min_steps: int) -> set[int]:
    """Return indices inside true-runs that last at least ``min_steps``."""
    if min_steps < 1:
        raise ValueError(f"min_steps must be >= 1, got {min_steps}")
    sat = np.asarray(mask, dtype=bool)
    out: set[int] = set()
    i = 0
    n = len(sat)
    while i < n:
        if sat[i]:
            j = i
            while j < n and sat[j]:
                j += 1
            if j - i >= min_steps:
                out.update(range(i, j))
            i = j
        else:
            i += 1
    return out


def decluster_indices(indices: Iterable[int], window: int) -> list[int]:
    """Keep the first index in each cluster closer than ``window`` steps."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    idx = sorted(int(i) for i in indices)
    if not idx:
        return []
    kept = [idx[0]]
    for k in idx[1:]:
        if k - kept[-1] >= window:
            kept.append(k)
    return kept


class OnlineEventSelector:
    """Stateful rising-edge selector for online closed-loop mining.

    It emits a label only when the label first appears after being absent.
    ``decluster_steps`` also suppresses quick clear/reappear flicker.
    """

    def __init__(self, decluster_steps: int = 10):
        if decluster_steps < 1:
            raise ValueError(f"decluster_steps must be >= 1, got {decluster_steps}")
        self.decluster_steps = int(decluster_steps)
        self.active_labels: set[str] = set()
        self.last_emitted_step: dict[str, int] = {}

    def update(self, step: int, labels: Iterable[str]) -> list[str]:
        current = {str(label) for label in labels if str(label) != "clean"}
        emitted: list[str] = []
        for label in sorted(current):
            if label in self.active_labels:
                continue
            last = self.last_emitted_step.get(label)
            if last is not None and int(step) - last < self.decluster_steps:
                continue
            emitted.append(label)
            self.last_emitted_step[label] = int(step)
        self.active_labels = current
        return emitted
