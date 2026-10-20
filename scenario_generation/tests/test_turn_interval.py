import random

from scenario_generation.turn_interval import (
    TurnIntervalTracker,
    interval_stats,
    pool_interval_stats,
    smooth_runs,
)


def _rows(gt, pred, every=1):
    """Interval rows for a GT list (10 Hz frames) scored on every ``every``-th frame."""
    tr = TurnIntervalTracker()
    for i in range(0, len(gt), every):
        tr.observe(i, pred[i])
    return tr.rows(gt.__getitem__)


def test_smooth_runs_absorbs_short_runs_into_previous_then_next():
    assert smooth_runs([0, 0, 0, 2, 0, 0, 0]) == [0] * 7
    assert smooth_runs([0, 0, 0, 2, 2, 0, 0, 0]) == [0] * 8
    assert smooth_runs([2, 0, 0, 0, 0]) == [0] * 5  # no previous run: absorbed into the next
    assert smooth_runs([0, 0, 0, 2, 2, 2, 0, 0, 0]) == [0] * 3 + [2] * 3 + [0] * 3
    # absorbing a run can leave the neighbour short; it keeps going until none is left
    assert smooth_runs([0, 0, 0, 2, 3, 3, 0, 0, 0]) == [0] * 9


def test_short_gt_runs_are_absorbed_into_the_previous_state():
    gt = [0] * 4 + [2] * 6 + [0] * 5 + [3] * 2 + [0] * 3
    pred = [0] * 4 + [0, 0, 2, 2, 2, 2] + [2, 2, 0, 0, 0] + [3, 3] + [0, 0, 0]
    # the 2-frame RIGHT GT run is absorbed into the NONE before it, which then joins the NONE after
    assert _rows(gt, pred) == [
        [0, 4, 4, 0.0, 0],
        [1, 6, 4, 0.2, 4],
        [0, 10, 6, 0.2, 10],
    ]


def test_a_gt_flicker_inside_a_constant_segment_is_ignored():
    gt = [0, 0, 0, 2, 0, 0, 0, 0]
    assert _rows(gt, [0] * 8) == [[0, 8, 8, 0.0, 0]]
    gt = [0] * 5 + [2] * 2 + [0] * 5
    assert _rows(gt, gt) == [[0, 12, 10, 0.0, 0]]


def test_time_to_correct_counts_steps_not_gt_frames():
    gt = [0] * 5 + [2] * 10
    pred = [0] * 8 + [2] * 7
    assert _rows(gt, pred) == [[0, 5, 5, 0.0, 0], [1, 10, 7, 0.3, 5]]
    # the replay cursor skipped frames 8..11 mid-interval: two wrong steps, not six wrong frames
    tr = TurnIntervalTracker()
    for i, p in [(5, 0), (6, 0), (7, 2), (12, 2), (13, 2), (14, 2)]:
        tr.observe(i, p)
    assert tr.rows(gt.__getitem__) == [[1, 6, 4, 0.2, 5]]


def test_gt_intervals_do_not_depend_on_the_inference_cadence():
    # a 3-frame GT blink between two inference steps (every 8th frame) is a valid interval but
    # nothing was scored in it, so it is left out; the NONE intervals around it still count
    gt = [0] * 20 + [2] * 3 + [0] * 20
    assert _rows(gt, [0] * len(gt), every=8) == [[0, 3, 3, 0.0, 0], [0, 3, 3, 0.0, 24]]
    # when an inference step does land in a 3-frame blink, it is scored there
    gt = [0] * 7 + [2] * 3 + [0] * 10
    assert _rows(gt, [0] * len(gt), every=8)[1] == [1, 1, 0, None, 8]


def test_disable_folds_into_none_and_never_correct_has_no_ttc():
    gt = [0] * 3 + [1, 0, 1, 0] + [2] * 4 + [0] * 3
    pred = [0] * 3 + [0, 1, 0, 1] + [0, 0, 3, 0] + [0] * 3
    rows = _rows(gt, pred)
    # NONE/DISABLE are one interval: LEFT starts at 7, not at 3 or 4
    assert rows == [[0, 7, 7, 0.0, 0], [1, 4, 0, None, 7], [0, 3, 3, 0.0, 11]]
    st = interval_stats(rows)
    assert st["left"]["zero_rate"] == 1.0 and st["left"]["ttc_mean_s"] is None


def test_reset_cuts_intervals():
    tr = TurnIntervalTracker()
    for i in range(4):
        tr.observe(i, 2)
    tr.reset()
    for i in range(10, 14):
        tr.observe(i, 2)
    assert tr.rows(lambda i: 2) == [[1, 4, 4, 0.0, 0], [1, 4, 4, 0.0, 10]]


def test_time_to_correct_is_defined_exactly_when_accuracy_is_nonzero():
    rng = random.Random(0)
    for _ in range(50):
        gt = [rng.choice([0, 2, 3]) for _ in range(5)] * 8
        pred = [rng.choice([0, 2, 3]) for _ in gt]
        stats = interval_stats(_rows(gt, pred))
        for st in stats.values():
            assert st["ttc_count"] + st["zero_count"] == st["count"]


def test_pool_is_over_intervals_not_group_means():
    a = interval_stats([[1, 4, 4, 0.0, 0]])
    b = interval_stats([[1, 4, 0, None, 0], [1, 4, 0, None, 9]])
    pooled = pool_interval_stats([a, b, None])
    assert pooled["left"]["count"] == 3
    assert abs(pooled["left"]["accuracy_mean"] - 1 / 3) < 1e-9
    assert abs(pooled["left"]["zero_rate"] - 2 / 3) < 1e-9
