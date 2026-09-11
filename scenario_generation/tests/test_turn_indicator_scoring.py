"""Tests for closed-loop turn-indicator transition scoring.

Covers the two PR #403 review fixes:
- the transition condition must compare GT across SCORED steps, not within one frame's own
  two-tick window (misses a transition that happens entirely between scored steps under
  ``replan_interval > 1`` or a cursor skip/repeat);
- ``run_segments_batched`` must resolve/append/score BEFORE ``_advance_step``, so a same-tick
  unstick teleport (which re-seeds ``turn_hist``/``last_turn_indicator``/
  ``turn_indicator_prev_scored_gt`` from the teleport target's GT) never corrupts the score for
  the step that just ran -- matching ``render_segment``'s already-correct ordering.

Also covers the spurious-transition ("false positive") counters added alongside the transition
counters: they must fire only when GT holds steady across scored steps AND the model's own
resolved prediction flips from its own previous scored prediction, must stay untouched by
GT-transition steps and by held/cached-plan steps, and must reseed correctly on an unstick
teleport -- exactly the same set of edge cases as the transition counters, since both live in
the same scoring function and partition the same population of scored steps.

Route-building helper mirrors ``test_reproducer_unstick.py``'s ``_make_route``, but lets each
frame's own ``turn_indicators`` window be built from an explicit per-tick GT signal (rather than
all zeros), so the frame-local ``[-2]``/``[-1]`` window can be distinguished from the
cross-scored-step comparison in tests.
"""

import json

import numpy as np
import pytest

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import (
    _advance_step,
    _hold_turn_indicator,
    _score_turn_indicator,
    _seed_state,
)
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.simulate import resolve_keep_turn_indicator

EGO_SHAPE = np.array([4.76, 7.24, 2.29], dtype=np.float32)
STEP_M = 0.01  # ~0.1 m/s => below the 0.5 m/s "stuck" threshold, so unstick fires


def _ti_window(raw_signal: list[int], i: int) -> np.ndarray:
    """The (31,) turn_indicators window a real recorded frame ``i`` would carry: a proper
    sliding history of ``raw_signal`` ending at ``i``, padded with ``raw_signal[0]`` for
    ticks before the start of the route (mimics how a real bag's early frames are padded).
    """
    w = np.empty(31, dtype=np.int64)
    for k in range(31):
        j = i - (30 - k)
        w[k] = raw_signal[j] if j >= 0 else raw_signal[0]
    return w


def _make_route(tmp_path, raw_signal: list[int]) -> RouteTimeline:
    """A near-stationary straight route along +x, one frame per ``raw_signal`` entry, each
    frame's ``turn_indicators`` a real sliding window of ``raw_signal`` (see ``_ti_window``).
    """
    n = len(raw_signal)
    past = np.zeros((31, 3), dtype=np.float32)
    past[:, 0] = (np.arange(31) - 30) * STEP_M
    paths = []
    for i in range(n):
        p = tmp_path / f"route_{i:010d}.npz"
        np.savez_compressed(
            p,
            ego_agent_past=past,
            ego_shape=EGO_SHAPE,
            turn_indicators=_ti_window(raw_signal, i),
        )
        sidecar = {
            "timestamp": float(i),
            "x": float(i * STEP_M),
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
        (tmp_path / f"route_{i:010d}.json").write_text(json.dumps(sidecar))
        paths.append(p)
    return RouteTimeline(paths)


def test_resolve_keep_turn_indicator_basic():
    """KEEP(4) resolves to the previous state; any real class passes through unchanged."""
    assert resolve_keep_turn_indicator(4, 2) == 2  # KEEP -> prev (LEFT)
    assert resolve_keep_turn_indicator(4, 0) == 0  # KEEP -> prev (NONE)
    assert resolve_keep_turn_indicator(3, 2) == 3  # RIGHT passes through regardless of prev
    assert resolve_keep_turn_indicator(0, 2) == 0  # NONE passes through regardless of prev


def test_transition_missed_by_frame_local_window_but_caught_across_scored_steps(tmp_path):
    """GT flips from NONE to RIGHT between raw ticks 3 and 4. Scoring steps 3 then 6 (as
    ``replan_interval`` or a cursor skip/repeat would) never visits frame 4 or 5, so neither
    scored frame's OWN ``[-2]``/``[-1]`` window shows a change (frame 6's own window is
    RIGHT/RIGHT) -- the old same-frame check would miss this transition entirely. The
    cross-scored-step comparison (against ``turn_indicator_prev_scored_gt``) catches it.
    """
    raw = [0, 0, 0, 0, 3, 3, 3, 3, 3, 3]
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        len(raw),
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
    )
    assert s.turn_indicator_prev_scored_gt == 0  # seeded from raw[0]

    # Confirm frame 6's OWN window shows no change (the old bug's blind spot).
    assert int(_ti_window(raw, 6)[-2]) == int(_ti_window(raw, 6)[-1]) == 3

    s.last_turn_indicator = 0  # model correctly predicts NONE at the (unscored) frames before
    _score_turn_indicator(s, 3)
    assert s.turn_indicator_transition_total == 0  # no transition yet (0 -> 0)

    s.last_turn_indicator = 3  # model correctly predicts RIGHT by the time it's scored again
    _score_turn_indicator(s, 6)
    assert s.turn_indicator_transition_total == 1, "transition between scored steps must be caught"
    assert s.turn_indicator_transition_correct == 1
    assert s.turn_indicator_prev_scored_gt == 3


def test_score_turn_indicator_counts_fp_on_gt_steady_pred_flip(tmp_path):
    """When GT holds steady across scored steps, a resolved prediction that changes from the
    model's OWN previous scored prediction is a spurious transition -- counted as a false
    positive, independent of whether it happens to match GT.
    """
    raw = [0] * 10
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        len(raw),
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
    )
    s.last_turn_indicator = 0  # matches the seeded baseline: no flip
    _score_turn_indicator(s, 3)
    assert (s.turn_indicator_fp_total, s.turn_indicator_fp_count) == (1, 0)

    s.last_turn_indicator = 1  # spurious flip: GT is still 0 at idx 6
    _score_turn_indicator(s, 6)
    assert (s.turn_indicator_fp_total, s.turn_indicator_fp_count) == (2, 1)
    assert s.turn_indicator_transition_total == 0, "GT never changed, so no transition either"


def test_score_turn_indicator_no_fp_when_pred_stable(tmp_path):
    """When GT holds steady AND the model's own prediction also stays put, no false positive
    is counted -- only the opportunity (``fp_total``) accumulates.
    """
    raw = [0] * 10
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        len(raw),
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
    )
    s.last_turn_indicator = 0
    _score_turn_indicator(s, 3)
    _score_turn_indicator(s, 6)
    assert (s.turn_indicator_fp_total, s.turn_indicator_fp_count) == (2, 0)


def test_score_turn_indicator_gt_change_steps_do_not_touch_fp_counters(tmp_path):
    """A scored step where GT changes must accumulate only the transition counters, never the
    spurious-transition FP counters -- the two counter pairs partition the same population of
    scored steps and must stay disjoint (``transition_total + fp_total`` == scored-step count).
    """
    raw = [0, 0, 0, 0, 3, 3, 3, 3, 3, 3]
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        len(raw),
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
    )
    s.last_turn_indicator = 0
    _score_turn_indicator(s, 3)  # GT unchanged (0 -> 0): a genuine FP opportunity
    assert s.turn_indicator_fp_total == 1
    fp_before = (s.turn_indicator_fp_count, s.turn_indicator_fp_total)

    s.last_turn_indicator = 3  # a real GT transition (0 -> 3)
    _score_turn_indicator(s, 6)
    assert s.turn_indicator_transition_total == 1
    assert (s.turn_indicator_fp_count, s.turn_indicator_fp_total) == fp_before
    assert s.turn_indicator_transition_total + s.turn_indicator_fp_total == 2  # disjoint partition


def test_hold_turn_indicator_does_not_touch_transition_counters(tmp_path):
    """A cached-plan step (``replan_interval > 1``, no fresh inference) must never be scored:
    ``_hold_turn_indicator`` only re-appends the held signal, it must not touch the transition
    accumulators or the scored-GT tracking field.
    """
    raw = [2] * 10
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        len(raw),
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
    )
    before = (
        s.turn_indicator_transition_correct,
        s.turn_indicator_transition_total,
        s.turn_indicator_prev_scored_gt,
    )
    _hold_turn_indicator(s)
    after = (
        s.turn_indicator_transition_correct,
        s.turn_indicator_transition_total,
        s.turn_indicator_prev_scored_gt,
    )
    assert before == after


def test_hold_turn_indicator_does_not_touch_fp_counters(tmp_path):
    """A cached-plan step must never touch the spurious-transition FP counters or the model's
    own-prediction baseline (``turn_indicator_prev_scored_pred``) either -- only ``turn_hist``
    changes.
    """
    raw = [2] * 10
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        len(raw),
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
    )
    before = (
        s.turn_indicator_fp_count,
        s.turn_indicator_fp_total,
        s.turn_indicator_prev_scored_pred,
    )
    _hold_turn_indicator(s)
    after = (
        s.turn_indicator_fp_count,
        s.turn_indicator_fp_total,
        s.turn_indicator_prev_scored_pred,
    )
    assert before == after


def test_teleport_reseeds_transition_tracking_from_target_gt(tmp_path):
    """An unstick teleport must re-seed ``turn_indicator_prev_scored_gt`` (like ``turn_hist``/
    ``last_turn_indicator``) from the teleport target's recorded GT -- otherwise the next
    scored step would compare the pre-teleport GT against the target's GT and count the
    environment jump itself as a spurious transition.
    """
    n = 20
    raw = [0] * (n // 2) + [2] * (n - n // 2)  # a real transition partway through the route
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        n,
        search_radius=1.5,
        warmup_steps=1000,  # recorded-pose branch (no tracker needed)
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.05,
        unstick_radius_mult=1.0,  # disable gentle widen: exercise the teleport path directly
    )
    pred = np.zeros((80, 4), dtype=np.float32)

    for i in range(n):
        s.cursor.last_was_repeat = True  # stuck repeating
        _advance_step(s, pred, idx=i, device="cpu", timers=timers)
        if s.snap_count > 0:
            break
    else:
        pytest.fail("unstick never fired on the synthetic stalled route")

    # Find the teleport target the same way test_reproducer_unstick.py does.
    matches = np.where(np.all(np.isclose(tl.poses, s.live_pose), axis=1))[0]
    assert len(matches) == 1
    tgt = int(matches[0])
    tgt_gt = int(np.asarray(tl.npz(tgt)["turn_indicators"]).reshape(-1)[-1])

    assert s.last_turn_indicator == tgt_gt
    assert s.turn_indicator_prev_scored_gt == tgt_gt


def test_teleport_reseeds_fp_tracking_from_target_pred(tmp_path):
    """An unstick teleport must re-seed ``turn_indicator_prev_scored_pred`` (like
    ``turn_indicator_prev_scored_gt``) from the teleport target's recorded GT -- otherwise the
    next scored step would compare the model's pre-teleport prediction baseline against the
    target's context and count the environment jump itself as a spurious flip.
    """
    n = 20
    raw = [0] * (n // 2) + [2] * (n - n // 2)  # a real transition partway through the route
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        n,
        search_radius=1.5,
        warmup_steps=1000,  # recorded-pose branch (no tracker needed)
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.05,
        unstick_radius_mult=1.0,  # disable gentle widen: exercise the teleport path directly
    )
    pred = np.zeros((80, 4), dtype=np.float32)

    for i in range(n):
        s.cursor.last_was_repeat = True  # stuck repeating
        _advance_step(s, pred, idx=i, device="cpu", timers=timers)
        if s.snap_count > 0:
            break
    else:
        pytest.fail("unstick never fired on the synthetic stalled route")

    matches = np.where(np.all(np.isclose(tl.poses, s.live_pose), axis=1))[0]
    assert len(matches) == 1
    tgt = int(matches[0])
    tgt_gt = int(np.asarray(tl.npz(tgt)["turn_indicators"]).reshape(-1)[-1])

    assert s.turn_indicator_prev_scored_pred == tgt_gt


def test_resolve_append_score_before_advance_survives_same_tick_teleport(tmp_path):
    """Regression for the batched-path ordering bug: resolving a KEEP prediction, appending
    to ``turn_hist`` and scoring must all use the PRE-teleport state, even when
    ``_advance_step`` (called right after, matching both ``render_segment`` and the fixed
    ``run_segments_batched``) teleports on that same tick. A KEEP prediction must resolve
    against the PRE-teleport ``last_turn_indicator``, not whatever the teleport resets it to.

    The GT deliberately differs between the pre-teleport region (RIGHT) and wherever the
    teleport lands (LEFT, ``unstick_advance_m`` is large enough to jump well past the split) --
    if the two states matched, resolving against either one would look identical and the
    ordering bug would have nothing to expose.
    """
    n = 20
    raw = [3] * 10 + [2] * 10  # RIGHT for the first half, LEFT for the second
    tl = _make_route(tmp_path, raw)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        n,
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.15,  # jump well past the raw-signal split (~15 frames at STEP_M=0.01)
        unstick_radius_mult=1.0,
    )
    pred = np.zeros((80, 4), dtype=np.float32)
    s.last_turn_indicator = 3  # matches raw signal from the start

    teleported = False
    for i in range(n):
        s.cursor.last_was_repeat = True
        pre_teleport_last = s.last_turn_indicator
        # The fixed ordering, shared by render_segment and run_segments_batched: resolve +
        # append + score BEFORE _advance_step.
        s.last_turn_indicator = resolve_keep_turn_indicator(4, s.last_turn_indicator)  # KEEP
        assert s.last_turn_indicator == pre_teleport_last, (
            "KEEP must resolve against the PRE-teleport state, not whatever _advance_step "
            "(called next) might reset it to"
        )
        s.turn_hist = np.append(s.turn_hist[1:], np.int64(s.last_turn_indicator))
        _score_turn_indicator(s, i)
        _advance_step(s, pred, idx=i, device="cpu", timers=timers)
        if s.snap_count > 0:
            teleported = True
            break
    if not teleported:
        pytest.fail("unstick never fired on the synthetic stalled route")

    # Confirm the teleport actually landed somewhere the GT differs from the pre-teleport
    # region -- otherwise this test couldn't have distinguished the two orderings at all.
    matches = np.where(np.all(np.isclose(tl.poses, s.live_pose), axis=1))[0]
    assert len(matches) == 1
    tgt_gt = int(np.asarray(tl.npz(int(matches[0]))["turn_indicators"]).reshape(-1)[-1])
    assert tgt_gt == 2, "test setup must land the teleport in the differing-GT region"
