"""Closed-loop turn-indicator evaluation per GT interval.

The recorded (GT) turn indicator, read at its own 10 Hz frames and independent of
``replan_interval``, is smoothed first: a GT run shorter than ``MIN_INTERVAL_FRAMES`` frames is
absorbed into the previous run (into the next one at the very start). The remaining maximal runs
of one class are the "intervals".

Every step (see ``reproducer_rollout._score_turn_indicator``) is scored in the interval that
contains its frame, with the prediction the system outputs at that step: a fresh one on an
inference step, the held last one in between. An interval no step falls into (a cursor skip)
cannot be measured and is left out. Per interval:

* accuracy: share of scored steps whose prediction equals the interval's GT class;
* time to correct: simulation time (steps x ``FRAME_DT_S``) from the interval's first scored step
  to the first scored step whose prediction is correct (None if there is none, i.e. accuracy is
  0). It counts steps, not GT frames, so frames the replay cursor skipped over (or repeated)
  are not counted as the model's delay.

Classes are folded to the three the metric reports: NONE (0 NONE and 1 DISABLE), LEFT (2) and
RIGHT (3). Every interval counts, including the first (data start) and the last (data end, or
an unstick teleport cut).

Per-segment intervals are stored as plain rows and pooled across segments / groups through the
summable per-class sums of ``interval_stats``, so a mean is always over intervals, never a mean
of segment means.
"""

from dataclasses import dataclass, field

MIN_INTERVAL_FRAMES = 3
FRAME_DT_S = 0.1

CLASS_NAMES = ("none", "left", "right")
_FOLD = {0: 0, 1: 0, 2: 1, 3: 2}


def fold_class(code: int) -> int:
    """Raw turn-indicator code {0 NONE, 1 DISABLE, 2 LEFT, 3 RIGHT} -> index into CLASS_NAMES."""
    return _FOLD[int(code)]


def smooth_runs(values: list, min_run: int = MIN_INTERVAL_FRAMES) -> list:
    """Absorb every run shorter than ``min_run`` into the previous run (the next one at the very
    start), repeating until none is left. The values keep their length."""
    runs: list[list] = []
    for v in values:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    while len(runs) > 1:
        i = next((i for i, (_, n) in enumerate(runs) if n < min_run), None)
        if i is None:
            break
        runs[i - 1 if i > 0 else 1][1] += runs[i][1]
        del runs[i]
        merged: list[list] = []
        for v, n in runs:
            if merged and merged[-1][0] == v:
                merged[-1][1] += n
            else:
                merged.append([v, n])
        runs = merged
    return [v for v, n in runs for _ in range(n)]


@dataclass
class TurnIntervalTracker:
    """Records the scored steps (``reset`` starts a new chunk at an unstick teleport) and builds
    the interval rows ``[class_idx, n_scored, n_correct, time_to_correct_s | None, start_frame]``
    on demand, because the GT intervals and the smoothing need the frames around each step."""

    _chunks: list = field(default_factory=lambda: [[]])

    def reset(self) -> None:
        """Cut at a teleport: intervals and smoothing never span the environment jump."""
        if self._chunks[-1]:
            self._chunks.append([])

    def observe(self, idx: int, pred: int) -> None:
        self._chunks[-1].append((idx, fold_class(pred)))

    def rows(self, gt_at) -> list:
        """``gt_at(frame_idx)`` is the recorded turn-indicator code of that 10 Hz frame."""
        return [row for chunk in self._chunks for row in _chunk_rows(chunk, gt_at)]


def _chunk_rows(chunk: list, gt_at) -> list:
    if not chunk:
        return []
    lo, hi = min(i for i, _ in chunk), max(i for i, _ in chunk)
    gt = smooth_runs([fold_class(gt_at(i)) for i in range(lo, hi + 1)])
    # Smoothed GT intervals on the 10 Hz frame axis: frame -> interval id, plus each class.
    interval_of: dict[int, int] = {}
    classes: list[int] = []
    i = 0
    while i < len(gt):
        j = i
        while j < len(gt) and gt[j] == gt[i]:
            interval_of[lo + j] = len(classes)
            j += 1
        classes.append(gt[i])
        i = j
    steps: list[list] = [[] for _ in classes]
    for idx, pred in chunk:
        steps[interval_of[idx]].append((idx, pred))
    rows = []
    for gt_c, scored in zip(classes, steps):
        if not scored:
            continue
        first_hit = next((n for n, (_, pred) in enumerate(scored) if pred == gt_c), None)
        n_hits = sum(pred == gt_c for _, pred in scored)
        ttc = round(first_hit * FRAME_DT_S, 3) if first_hit is not None else None
        rows.append([gt_c, len(scored), n_hits, ttc, int(scored[0][0])])
    return rows


def _empty_class_stats() -> dict:
    return {"count": 0, "accuracy_sum": 0.0, "ttc_sum": 0.0, "ttc_count": 0, "zero_count": 0}


def _with_means(st: dict) -> dict:
    n = st["count"]
    st["accuracy_mean"] = st["accuracy_sum"] / n if n else None
    st["ttc_mean_s"] = st["ttc_sum"] / st["ttc_count"] if st["ttc_count"] else None
    st["zero_rate"] = st["zero_count"] / n if n else None
    return st


def interval_stats(rows: list) -> dict:
    """Per-class summable sums plus derived means from interval rows (None = nothing to measure)."""
    out = {name: _empty_class_stats() for name in CLASS_NAMES}
    for cls, n, n_correct, ttc, _start in rows:
        st = out[CLASS_NAMES[cls]]
        st["count"] += 1
        st["accuracy_sum"] += n_correct / n
        st["zero_count"] += int(n_correct == 0)
        if ttc is not None:
            st["ttc_sum"] += ttc
            st["ttc_count"] += 1
    return {name: _with_means(st) for name, st in out.items()}


def pool_interval_stats(stats_list: list) -> dict:
    """Merge ``interval_stats`` dicts (e.g. one per group) by their sums, then recompute means."""
    out = {name: _empty_class_stats() for name in CLASS_NAMES}
    for stats in stats_list:
        for name in CLASS_NAMES:
            src = (stats or {}).get(name)
            if not src:
                continue
            for k in out[name]:
                out[name][k] += src.get(k, 0) or 0
    return {name: _with_means(st) for name, st in out.items()}


def format_interval_stats(stats: dict) -> str:
    def fmt(v, spec):
        return "N/A" if v is None else format(v, spec)

    return "  ".join(
        f"{name}: n={st['count']} acc={fmt(st['accuracy_mean'], '.3f')} "
        f"ttc={fmt(st['ttc_mean_s'], '.2f')}s zero={fmt(st['zero_rate'], '.3f')}"
        for name, st in stats.items()
    )
