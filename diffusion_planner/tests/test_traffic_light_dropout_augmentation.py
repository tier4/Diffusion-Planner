"""TrafficLightDropoutAugmentation: replace known traffic light colors with the
unknown state to simulate misdetection, per lane, across the whole polyline."""

import torch
from diffusion_planner.dimensions import (
    SEGMENT_POINT_DIM,
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_WHITE,
    TRAFFIC_LIGHT_YELLOW,
)
from diffusion_planner.utils.data_augmentation import TrafficLightDropoutAugmentation

B, P, V = 2, 4, 6


def _make_lanes(seed=0):
    """4 lanes per sample: green, red, no-traffic-light, padding."""
    g = torch.Generator().manual_seed(seed)
    lanes = torch.randn(B, P, V, SEGMENT_POINT_DIM, generator=g)
    lanes[..., TRAFFIC_LIGHT_GREEN : TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT + 1] = 0.0
    lanes[:, 0, :, TRAFFIC_LIGHT_GREEN] = 1.0
    lanes[:, 1, :, TRAFFIC_LIGHT_RED] = 1.0
    lanes[:, 2, :, TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT] = 1.0
    lanes[:, 3] = 0.0  # padding lane
    # last point of lane 0 is padding within a valid lane
    lanes[:, 0, -1] = 0.0
    return lanes


def _make_batch(seed=0):
    inputs = {"lanes": _make_lanes(seed), "route_lanes": _make_lanes(seed + 1)}
    ego_future = torch.randn(B, 5, 3)
    neighbors_future = torch.randn(B, 3, 5, 3)
    return inputs, ego_future, neighbors_future


def test_known_colors_become_unknown():
    inputs, ego_future, neighbors_future = _make_batch()
    orig = {k: v.clone() for k, v in inputs.items()}

    aug = TrafficLightDropoutAugmentation(dropout_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    for key in ["lanes", "route_lanes"]:
        new = inputs[key]
        for lane in [0, 1]:  # green and red lanes
            valid = torch.sum(torch.ne(orig[key][:, lane, :, :8], 0), dim=-1) > 0
            assert torch.equal(
                new[:, lane, :, TRAFFIC_LIGHT_GREEN : TRAFFIC_LIGHT_RED + 1],
                torch.zeros_like(new[:, lane, :, TRAFFIC_LIGHT_GREEN : TRAFFIC_LIGHT_RED + 1]),
            )
            assert torch.equal(new[:, lane, :, TRAFFIC_LIGHT_WHITE], valid.to(new.dtype))
            # geometry and line types untouched
            assert torch.equal(new[:, lane, :, :8], orig[key][:, lane, :, :8])
            assert torch.equal(
                new[:, lane, :, TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT + 1 :],
                orig[key][:, lane, :, TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT + 1 :],
            )


def test_no_traffic_light_and_padding_lanes_untouched():
    inputs, ego_future, neighbors_future = _make_batch()
    orig = {k: v.clone() for k, v in inputs.items()}

    aug = TrafficLightDropoutAugmentation(dropout_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    for key in ["lanes", "route_lanes"]:
        assert torch.equal(inputs[key][:, 2], orig[key][:, 2])  # no traffic light
        assert torch.equal(inputs[key][:, 3], torch.zeros_like(inputs[key][:, 3]))  # padding


def test_padding_points_stay_zero_in_dropped_lane():
    inputs, ego_future, neighbors_future = _make_batch()

    aug = TrafficLightDropoutAugmentation(dropout_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    # last point of lane 0 is a padding row; the unknown flag must not be set there
    pad_point = inputs["lanes"][:, 0, -1]
    assert torch.equal(pad_point, torch.zeros_like(pad_point))


def test_dropout_prob_zero_is_identity():
    inputs, ego_future, neighbors_future = _make_batch()
    orig = {k: v.clone() for k, v in inputs.items()}

    aug = TrafficLightDropoutAugmentation(dropout_prob=0.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    for key, val in orig.items():
        assert torch.equal(inputs[key], val), key


def test_yellow_is_also_dropped():
    inputs, ego_future, neighbors_future = _make_batch()
    inputs["lanes"][:, 0, :, TRAFFIC_LIGHT_GREEN] = 0.0
    inputs["lanes"][:, 0, :-1, TRAFFIC_LIGHT_YELLOW] = 1.0

    aug = TrafficLightDropoutAugmentation(dropout_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    yellow = inputs["lanes"][:, 0, :, TRAFFIC_LIGHT_YELLOW]
    assert torch.equal(yellow, torch.zeros_like(yellow))
    assert (inputs["lanes"][:, 0, 0, TRAFFIC_LIGHT_WHITE] == 1.0).all()


def test_sample_dropout_is_shared_by_lanes_and_route_lanes(monkeypatch):
    """One per-NPZ-sample decision must apply consistently to both lane tensors."""
    inputs, ego_future, neighbors_future = _make_batch()

    # The same physical lane can appear at different indices in the surrounding-lane
    # and route-lane tensors. Make route lane 1 an exact copy of surrounding lane 0.
    inputs["route_lanes"][:, 1] = inputs["lanes"][:, 0].clone()
    orig = {key: value.clone() for key, value in inputs.items()}

    def fake_rand(*size, **kwargs):
        assert size == (B,)
        # Drop every known light in sample 0 and keep every light in sample 1.
        return torch.tensor([0.1, 0.9], device=kwargs.get("device"))

    monkeypatch.setattr(torch, "rand", fake_rand)

    aug = TrafficLightDropoutAugmentation(dropout_prob=0.5, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    traffic_light = slice(TRAFFIC_LIGHT_GREEN, TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT + 1)
    lane_traffic_light = inputs["lanes"][:, 0, :, traffic_light]
    route_traffic_light = inputs["route_lanes"][:, 1, :, traffic_light]
    assert torch.equal(lane_traffic_light, route_traffic_light)
    assert (lane_traffic_light[0, :-1, TRAFFIC_LIGHT_WHITE - TRAFFIC_LIGHT_GREEN] == 1).all()
    assert (lane_traffic_light[1, :-1, TRAFFIC_LIGHT_GREEN - TRAFFIC_LIGHT_GREEN] == 1).all()

    # The mask is sample-wide, not limited to the duplicated lane: every other known
    # green/red lane in sample 0 is unknown, while sample 1 remains unchanged.
    for key in ["lanes", "route_lanes"]:
        for lane in [0, 1]:
            valid = torch.sum(torch.ne(orig[key][0, lane, :, :8], 0), dim=-1) > 0
            assert torch.equal(
                inputs[key][0, lane, :, TRAFFIC_LIGHT_WHITE], valid.to(inputs[key].dtype)
            )
            assert torch.equal(inputs[key][1, lane], orig[key][1, lane])
