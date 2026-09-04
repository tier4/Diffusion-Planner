"""Tests for the traffic-light acceptance gate.

The gate logic is tested without a model: traffic-light classification reads the scene,
and the verdict is a pure function of the measured spans.
"""

import numpy as np
import pytest
import torch
from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_YELLOW,
)

from rlvr.autoresearch.ego_shape_diag import _common
from rlvr.autoresearch.ego_shape_diag.check_tl_gate import (
    route_tl_class,
    untested_halves,
    verdict,
)


def _scene(*onehot_indices: int, valid: bool = True) -> dict:
    """One scene whose route lanes carry the given traffic-light one-hots."""
    route = np.zeros((max(len(onehot_indices), 1), 20, 33), dtype=np.float32)
    for lane, index in enumerate(onehot_indices):
        route[lane, :, 0] = 1.0  # mark the lane valid
        route[lane, 0, index] = 1.0
    if not valid:
        route[:] = 0.0
    return {"route_lanes": torch.from_numpy(route)}


def test_route_tl_class_reads_each_state():
    assert route_tl_class(_scene(TRAFFIC_LIGHT_GREEN)) == "green"
    assert route_tl_class(_scene(TRAFFIC_LIGHT_YELLOW)) == "amber"
    assert route_tl_class(_scene(TRAFFIC_LIGHT_RED)) == "red"


def test_route_tl_class_unsignalled_is_none():
    # a valid lane carrying no traffic-light one-hot, and an all-zero (absent) route
    assert route_tl_class(_scene()) == "none"
    assert route_tl_class(_scene(TRAFFIC_LIGHT_GREEN, valid=False)) == "none"


def test_route_tl_class_mixed_states_are_ambiguous():
    """route_lanes can hold a perpendicular red while the ego's own approach is green.

    Picking either one would be a guess, and guessing "red" makes this gate fail a normal
    plan for running a light the ego never faced. Such scenes are excluded instead.
    """
    assert route_tl_class(_scene(TRAFFIC_LIGHT_GREEN, TRAFFIC_LIGHT_RED)) == "ambiguous"
    assert route_tl_class(_scene(TRAFFIC_LIGHT_GREEN, TRAFFIC_LIGHT_YELLOW)) == "ambiguous"
    # agreeing lanes are not ambiguous
    assert route_tl_class(_scene(TRAFFIC_LIGHT_RED, TRAFFIC_LIGHT_RED)) == "red"


def test_verdict_passes_when_it_goes_on_green_and_holds_on_red():
    spans = {"green": [16.0, 18.0], "amber": [0.1], "red": [0.04], "none": []}
    assert verdict(spans, min_green_m=15.0, max_red_m=2.0) == []


def test_verdict_catches_a_model_that_will_not_go():
    spans = {"green": [1.0, 2.0], "amber": [], "red": [0.04], "none": []}
    (failure,) = verdict(spans, min_green_m=15.0, max_red_m=2.0)
    assert "does not commit to go" in failure


def test_verdict_catches_a_model_that_runs_the_light():
    """The failure an unstratified span gate scores as healthy."""
    spans = {"green": [20.0], "amber": [], "red": [19.0], "none": []}
    (failure,) = verdict(spans, min_green_m=15.0, max_red_m=2.0)
    assert "runs the light" in failure


def test_verdict_refuses_a_vacuous_pass():
    """No signalled scene means the gate proved nothing; it must not report PASS."""
    with pytest.raises(SystemExit, match="no green and no red scenes"):
        verdict({"green": [], "amber": [1.0], "red": [], "none": [5.0]}, 15.0, 2.0)


def test_verdict_refuses_a_green_only_set():
    """The case that matters: a green-only set cannot detect a red-runner."""
    green_only = {"green": [16.0], "amber": [], "red": [], "none": []}
    with pytest.raises(SystemExit, match="no red scenes"):
        verdict(green_only, 15.0, 2.0)
    # the override grades what is there, and the untested half is named for the caller
    assert verdict(green_only, 15.0, 2.0, allow_one_sided=True) == []
    assert untested_halves(green_only) == ["red"]


def test_verdict_refuses_a_red_only_set():
    """The mirror case: a red-only set cannot detect a checkpoint that will not move."""
    red_only = {"green": [], "amber": [], "red": [0.04], "none": []}
    with pytest.raises(SystemExit, match="no green scenes"):
        verdict(red_only, 15.0, 2.0)
    assert verdict(red_only, 15.0, 2.0, allow_one_sided=True) == []
    assert untested_halves(red_only) == ["green"]


def test_one_sided_override_still_fails_a_bad_checkpoint():
    """The override relaxes coverage, never the thresholds."""
    runs_reds = {"green": [], "amber": [], "red": [19.0], "none": []}
    (failure,) = verdict(runs_reds, 15.0, 2.0, allow_one_sided=True)
    assert "runs the light" in failure


def test_untested_halves_empty_when_both_covered():
    assert untested_halves({"green": [16.0], "amber": [], "red": [0.04], "none": []}) == []


def test_amber_is_reported_but_never_gated():
    holds = {"green": [16.0], "amber": [0.1], "red": [0.04], "none": []}
    proceeds = {"green": [16.0], "amber": [17.0], "red": [0.04], "none": []}
    assert verdict(holds, 15.0, 2.0) == verdict(proceeds, 15.0, 2.0) == []


def test_load_deployable_model_refuses_a_training_milestone(tmp_path):
    """A milestone holds raw AND EMA weights; load_model would silently take the raw."""
    milestone = tmp_path / "milestone.pth"
    torch.save({"model": {"w": torch.zeros(1)}, "ema_state_dict": {"w": torch.ones(1)}}, milestone)
    with pytest.raises(SystemExit, match="training milestone"):
        _common.load_deployable_model(str(milestone), torch.device("cpu"))


def test_set_shape_replaces_dimensions_without_touching_the_original():
    scene = {"ego_shape": torch.tensor([2.75, 4.34, 1.84])}
    out = _common.set_shape(scene, 4.76, 7.24, 2.43)
    assert torch.allclose(out["ego_shape"], torch.tensor([4.76, 7.24, 2.43]))
    assert torch.allclose(scene["ego_shape"], torch.tensor([2.75, 4.34, 1.84]))


def test_parse_shape_rejects_a_wrong_length():
    assert _common.parse_shape("4.76,7.24,2.43") == (4.76, 7.24, 2.43)
    with pytest.raises(SystemExit, match="wheelbase,length,width"):
        _common.parse_shape("4.76,7.24")
