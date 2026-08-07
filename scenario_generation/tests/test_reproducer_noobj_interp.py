"""The invariant that makes ``render_segment`` safe to skip the track-anchor scan under
``drop_objects``: ``_apply_neighbor_interp`` leaves all-zero neighbor rows alone."""

from __future__ import annotations

import numpy as np

from scenario_generation.reproducer_rollout import _apply_neighbor_interp


def _anchors_that_would_move_everything(uuids: list[str]) -> dict:
    return {
        u: (
            np.array([0, 100]),
            np.array([[1000.0, 1000.0], [2000.0, 2000.0]]),
            np.array([0.5, 1.5]),
        )
        for u in uuids
    }


def _np_dict(n_slots: int = 4, *, zeroed: bool) -> dict:
    nb = np.zeros((1, n_slots, 31, 11), dtype=np.float64)
    if not zeroed:
        # a non-zero among the first six columns is what makes a row "present"
        nb[0, :, -1, 0] = 3.0
        nb[0, :, -1, 1] = 4.0
        nb[0, :, -1, 2] = 1.0
    return {"neighbor_agents_past": nb}


def test_apply_is_a_noop_on_zeroed_neighbors():
    uuids = [f"track{i}" for i in range(4)]
    d = _np_dict(zeroed=True)
    before = d["neighbor_agents_past"].copy()

    _apply_neighbor_interp(
        d, uuids, np.array([0.0, 0.0, 0.0]), 50, _anchors_that_would_move_everything(uuids)
    )

    assert np.array_equal(d["neighbor_agents_past"], before)


def test_apply_does_move_present_neighbors():
    uuids = [f"track{i}" for i in range(4)]
    d = _np_dict(zeroed=False)
    before = d["neighbor_agents_past"].copy()

    _apply_neighbor_interp(
        d, uuids, np.array([0.0, 0.0, 0.0]), 50, _anchors_that_would_move_everything(uuids)
    )

    assert not np.array_equal(d["neighbor_agents_past"], before)


def test_empty_anchor_dict_is_a_noop():
    uuids = [f"track{i}" for i in range(4)]
    d = _np_dict(zeroed=False)
    before = d["neighbor_agents_past"].copy()

    _apply_neighbor_interp(d, uuids, np.array([0.0, 0.0, 0.0]), 50, {})

    assert np.array_equal(d["neighbor_agents_past"], before)
