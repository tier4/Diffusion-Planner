"""FlipAugmentation: mirror the scene across the ego longitudinal axis (y -> -y),
swapping left/right lane boundaries, line types and turn indicators."""

from unittest.mock import patch

import torch
from diffusion_planner.dimensions import (
    LB_X,
    LB_Y,
    LINE_TYPE_LEFT_START,
    LINE_TYPE_NUM,
    LINE_TYPE_RIGHT_START,
    RB_X,
    RB_Y,
    SEGMENT_POINT_DIM,
    TRAFFIC_LIGHT,
    Y,
    dY,
)
from diffusion_planner.utils.data_augmentation import FlipAugmentation

B, N, T, P, V = 2, 3, 5, 4, 6


def _make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    inputs = {
        "ego_current_state": r(B, 10),
        "ego_agent_past": r(B, T, 4),
        "goal_pose": r(B, 4),
        "neighbor_agents_past": r(B, N, T, 11),
        "static_objects": r(B, P, 10),
        "polygons": r(B, P, V, 3),
        "line_strings": r(B, P, V, 4),
        "lanes": r(B, P, V, SEGMENT_POINT_DIM),
        "route_lanes": r(B, P, V, SEGMENT_POINT_DIM),
        "turn_indicators": torch.tensor([[0, 1, 2, 3, 2, 1], [1, 1, 3, 3, 0, 2]]),
    }
    # zero padding rows to verify they stay zero
    inputs["neighbor_agents_past"][:, -1] = 0.0
    inputs["lanes"][:, -1] = 0.0
    ego_future = r(B, T, 3)
    neighbors_future = r(B, N, T, 3)
    neighbors_future[:, -1] = 0.0
    return inputs, ego_future, neighbors_future


def _clone_batch(inputs, ego_future, neighbors_future):
    return (
        {k: v.clone() for k, v in inputs.items()},
        ego_future.clone(),
        neighbors_future.clone(),
    )


def test_flip_negates_lateral_components():
    inputs, ego_future, neighbors_future = _make_batch()
    orig = _clone_batch(inputs, ego_future, neighbors_future)[0]
    orig_ego_future = ego_future.clone()
    orig_neighbors_future = neighbors_future.clone()

    aug = FlipAugmentation(flip_prob=1.0, device="cpu")
    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    # ego_current_state: y, sin, vy, ay, steering, yaw_rate negated; x, cos, vx, ax kept
    for i in [1, 3, 5, 7, 8, 9]:
        assert torch.equal(inputs["ego_current_state"][:, i], -orig["ego_current_state"][:, i])
    for i in [0, 2, 4, 6]:
        assert torch.equal(inputs["ego_current_state"][:, i], orig["ego_current_state"][:, i])

    # (x, y, cos, sin) tensors: y and sin negated
    for key in ["ego_agent_past", "goal_pose", "neighbor_agents_past", "static_objects"]:
        assert torch.equal(inputs[key][..., 1], -orig[key][..., 1])
        assert torch.equal(inputs[key][..., 3], -orig[key][..., 3])
        assert torch.equal(inputs[key][..., 0], orig[key][..., 0])
        assert torch.equal(inputs[key][..., 2], orig[key][..., 2])

    # neighbor vy negated, vx kept
    assert torch.equal(
        inputs["neighbor_agents_past"][..., 5], -orig["neighbor_agents_past"][..., 5]
    )
    assert torch.equal(inputs["neighbor_agents_past"][..., 4], orig["neighbor_agents_past"][..., 4])

    # polygons / line_strings: y negated, types kept
    for key in ["polygons", "line_strings"]:
        assert torch.equal(inputs[key][..., 1], -orig[key][..., 1])
        assert torch.equal(inputs[key][..., 0], orig[key][..., 0])
        assert torch.equal(inputs[key][..., 2:], orig[key][..., 2:])

    # futures: y and heading negated
    assert torch.equal(ego_future[..., 1], -orig_ego_future[..., 1])
    assert torch.equal(ego_future[..., 2], -orig_ego_future[..., 2])
    assert torch.equal(ego_future[..., 0], orig_ego_future[..., 0])
    assert torch.equal(neighbors_future[..., 1], -orig_neighbors_future[..., 1])
    assert torch.equal(neighbors_future[..., 2], -orig_neighbors_future[..., 2])
    assert torch.equal(inputs["ego_agent_future"], ego_future)


def test_flip_swaps_lane_boundaries_and_line_types():
    inputs, ego_future, neighbors_future = _make_batch()
    orig = {k: v.clone() for k, v in inputs.items()}

    aug = FlipAugmentation(flip_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    for key in ["lanes", "route_lanes"]:
        new, old = inputs[key], orig[key]
        assert torch.equal(new[..., Y], -old[..., Y])
        assert torch.equal(new[..., dY], -old[..., dY])
        # left boundary becomes mirrored right boundary and vice versa
        assert torch.equal(new[..., LB_X], old[..., RB_X])
        assert torch.equal(new[..., LB_Y], -old[..., RB_Y])
        assert torch.equal(new[..., RB_X], old[..., LB_X])
        assert torch.equal(new[..., RB_Y], -old[..., LB_Y])
        # line type one-hots swap sides
        left = slice(LINE_TYPE_LEFT_START, LINE_TYPE_LEFT_START + LINE_TYPE_NUM)
        right = slice(LINE_TYPE_RIGHT_START, LINE_TYPE_RIGHT_START + LINE_TYPE_NUM)
        assert torch.equal(new[..., left], old[..., right])
        assert torch.equal(new[..., right], old[..., left])
        # traffic light one-hot is side-agnostic
        assert torch.equal(
            new[..., TRAFFIC_LIGHT:LINE_TYPE_LEFT_START],
            old[..., TRAFFIC_LIGHT:LINE_TYPE_LEFT_START],
        )


def test_flip_swaps_turn_indicators():
    inputs, ego_future, neighbors_future = _make_batch()

    aug = FlipAugmentation(flip_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    # 2 (ENABLE_LEFT) <-> 3 (ENABLE_RIGHT); 0 / 1 untouched
    expected = torch.tensor([[0, 1, 3, 2, 3, 1], [1, 1, 2, 2, 0, 3]])
    assert torch.equal(inputs["turn_indicators"], expected)


def test_flip_keeps_padding_rows_zero():
    inputs, ego_future, neighbors_future = _make_batch()

    aug = FlipAugmentation(flip_prob=1.0, device="cpu")
    inputs, _, neighbors_future = aug(inputs, ego_future, neighbors_future)

    assert torch.equal(
        inputs["neighbor_agents_past"][:, -1],
        torch.zeros_like(inputs["neighbor_agents_past"][:, -1]),
    )
    assert torch.equal(inputs["lanes"][:, -1], torch.zeros_like(inputs["lanes"][:, -1]))
    assert torch.equal(neighbors_future[:, -1], torch.zeros_like(neighbors_future[:, -1]))


def test_flip_prob_zero_is_identity():
    inputs, ego_future, neighbors_future = _make_batch()
    orig_inputs, orig_ego_future, orig_neighbors_future = _clone_batch(
        inputs, ego_future, neighbors_future
    )

    aug = FlipAugmentation(flip_prob=0.0, device="cpu")
    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    for key, val in orig_inputs.items():
        assert torch.equal(inputs[key], val), key
    assert torch.equal(ego_future, orig_ego_future)
    assert torch.equal(neighbors_future, orig_neighbors_future)


def test_flip_is_in_place_and_updates_future_entries():
    inputs, ego_future, neighbors_future = _make_batch()
    input_refs = {key: value for key, value in inputs.items()}

    aug = FlipAugmentation(flip_prob=1.0, device="cpu")
    inputs, returned_ego, returned_neighbors = aug(inputs, ego_future, neighbors_future)

    for key, ref in input_refs.items():
        assert inputs[key] is ref, key
    assert returned_ego is ego_future
    assert returned_neighbors is neighbors_future
    assert inputs["ego_agent_future"] is ego_future
    assert inputs["neighbor_agents_future"] is neighbors_future


def test_mixed_batch_only_flips_selected_samples():
    inputs, ego_future, neighbors_future = _make_batch()
    orig_inputs, orig_ego_future, orig_neighbors_future = _clone_batch(
        inputs, ego_future, neighbors_future
    )

    aug = FlipAugmentation(flip_prob=0.5, device="cpu")
    with patch(
        "diffusion_planner.utils.data_augmentation.torch.rand",
        return_value=torch.tensor([0.1, 0.9]),
    ):
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    assert torch.equal(ego_future[0, :, 1:], -orig_ego_future[0, :, 1:])
    assert torch.equal(ego_future[1], orig_ego_future[1])
    assert torch.equal(neighbors_future[0, ..., 1:], -orig_neighbors_future[0, ..., 1:])
    assert torch.equal(neighbors_future[1], orig_neighbors_future[1])
    assert torch.equal(inputs["ego_current_state"][1], orig_inputs["ego_current_state"][1])


def test_four_component_futures_negate_sin_not_cos():
    inputs, ego_future, neighbors_future = _make_batch()
    ego_future = torch.cat(
        [ego_future[..., :2], ego_future[..., 2:].cos(), ego_future[..., 2:].sin()], dim=-1
    )
    neighbors_future = torch.cat(
        [
            neighbors_future[..., :2],
            neighbors_future[..., 2:].cos(),
            neighbors_future[..., 2:].sin(),
        ],
        dim=-1,
    )
    orig_ego_future = ego_future.clone()
    orig_neighbors_future = neighbors_future.clone()

    aug = FlipAugmentation(flip_prob=1.0, device="cpu")
    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    assert torch.equal(ego_future[..., 0], orig_ego_future[..., 0])
    assert torch.equal(ego_future[..., 1], -orig_ego_future[..., 1])
    assert torch.equal(ego_future[..., 2], orig_ego_future[..., 2])
    assert torch.equal(ego_future[..., 3], -orig_ego_future[..., 3])
    assert torch.equal(neighbors_future[..., 2], orig_neighbors_future[..., 2])
    assert torch.equal(neighbors_future[..., 3], -orig_neighbors_future[..., 3])


def test_invalid_flip_probability_is_rejected():
    for flip_prob in (-0.1, 1.1):
        try:
            FlipAugmentation(flip_prob=flip_prob, device="cpu")
        except ValueError:
            pass
        else:
            raise AssertionError(f"flip_prob={flip_prob} was accepted")


def test_double_flip_is_identity():
    inputs, ego_future, neighbors_future = _make_batch()
    orig_inputs, orig_ego_future, orig_neighbors_future = _clone_batch(
        inputs, ego_future, neighbors_future
    )

    aug = FlipAugmentation(flip_prob=1.0, device="cpu")
    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)
    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    for key, val in orig_inputs.items():
        assert torch.equal(inputs[key], val), key
    assert torch.equal(ego_future, orig_ego_future)
    assert torch.equal(neighbors_future, orig_neighbors_future)
