"""LineEncoder geometry fixes.

Before the fix, pos was taken from the center vertex index — which is padding
for short elements such as 2-point stop lines — and its heading came from
atan2 over the type one-hot columns, so it flipped with the element type
instead of following the geometry. The diff at the last-valid-point -> padding
boundary was also non-zero (a spurious segment back toward the origin).

pos must instead be computed from valid-point geometry (centroid + summed diff
direction) and the padding-boundary diff must be zeroed."""

import torch
from diffusion_planner.dimensions import (
    LINE_STRING_TYPE_NUM,
    POINTS_PER_LINE_STRING,
)
from diffusion_planner.model.module.encoder import (
    CLASS_TYPE_LINE_STRING,
    CLASS_TYPE_NUM,
    LineEncoder,
)

POINT_DIM = 2 + LINE_STRING_TYPE_NUM  # (x, y, oh_stop_line, oh_road_border)
STOP_LINE = 0
ROAD_BORDER = 1


def _make_encoder():
    torch.manual_seed(0)
    encoder = LineEncoder(
        line_len=POINTS_PER_LINE_STRING,
        class_type=CLASS_TYPE_LINE_STRING,
        drop_path_rate=0.0,
        hidden_dim=32,
        depth=1,
        point_dim=POINT_DIM,
    )
    encoder.eval()
    return encoder


def _stop_line_input(type_index=STOP_LINE):
    """Element 0: a 2-point stop line p0=(15, -2), p1=(15, 2) in ego frame,
    normalized with x -> (x - 10) / 20, y -> y / 20 as in normalization.json.
    Element 1: all-zero padding."""
    x = torch.zeros(1, 2, POINTS_PER_LINE_STRING, POINT_DIM)
    x[0, 0, 0, :2] = torch.tensor([0.25, -0.1])
    x[0, 0, 1, :2] = torch.tensor([0.25, 0.1])
    x[0, 0, :2, 2 + type_index] = 1.0
    return x


def _capture_mixer_input(encoder, x):
    """Return the (B*P, V, POINT_DIM + 2) features fed into channel_pre_project."""
    captured = {}

    def hook(module, inputs):
        captured["x"] = inputs[0].detach().clone()

    handle = encoder.channel_pre_project.register_forward_pre_hook(hook)
    with torch.no_grad():
        encoder(x)
    handle.remove()
    return captured["x"]


def test_pos_is_centroid_and_segment_direction():
    encoder = _make_encoder()
    with torch.no_grad():
        _, _, pos = encoder(_stop_line_input())

    assert pos.shape == (1, 2, 4 + CLASS_TYPE_NUM)
    # Centroid of the two valid points, not the (padded) center index.
    assert torch.allclose(pos[0, 0, :2], torch.tensor([0.25, 0.0]), atol=1e-6)
    # (cos, sin) of the segment direction p0 -> p1 = (0, +0.2), i.e. +y.
    assert torch.allclose(pos[0, 0, 2:4], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_pos_does_not_depend_on_type_one_hot():
    # Before the fix, heading was atan2 over the one-hot columns, so pos flipped
    # between (1, 0) and (0, 1) purely based on the element type.
    encoder = _make_encoder()
    with torch.no_grad():
        _, _, pos_stop = encoder(_stop_line_input(STOP_LINE))
        _, _, pos_border = encoder(_stop_line_input(ROAD_BORDER))
    assert torch.allclose(pos_stop[..., :4], pos_border[..., :4], atol=1e-6)


def test_no_spurious_diff_at_padding_boundary():
    encoder = _make_encoder()
    features = _capture_mixer_input(encoder, _stop_line_input())

    # Layout is (x, y, dx, dy, one-hot...): row 0 carries the real segment diff.
    assert torch.allclose(features[0, 0, 2:4], torch.tensor([0.0, 0.2]), atol=1e-6)
    # Row 1 is the last valid point: its diff toward the zero padding must be 0
    # (before the fix it was -p1 = (-0.25, -0.1)).
    assert torch.all(features[0, 1, 2:4] == 0)
    # Padding rows carry no signal at all.
    assert torch.all(features[0, 2:] == 0)


def test_mixer_input_feature_layout():
    # Regression guard for the concat order: (x, y, dx, dy, type one-hot...).
    encoder = _make_encoder()
    x = _stop_line_input()
    features = _capture_mixer_input(encoder, x)

    assert encoder.channel_pre_project.fc1.in_features == POINT_DIM + 2
    assert features.shape == (2, POINTS_PER_LINE_STRING, POINT_DIM + 2)
    assert torch.allclose(features[0, :, :2], x[0, 0, :, :2])
    assert torch.allclose(features[0, :, 4:], x[0, 0, :, 2:])


def test_padding_element_is_masked_and_finite():
    encoder = _make_encoder()
    with torch.no_grad():
        tokens, mask_p, pos = encoder(_stop_line_input())

    assert mask_p.tolist() == [[False, True]]
    assert torch.all(tokens[0, 1] == 0)
    assert torch.all(torch.isfinite(pos))
    # Padding falls back to the deterministic direction (1, 0) at the origin.
    assert torch.allclose(pos[0, 1, :4], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_origin_point_with_zero_type_is_not_padding():
    # Legacy D=2 sources zero-pad the type channel, so a real mid-polyline
    # vertex landing exactly on the normalized origin is an all-zero row.
    # Padding is always a zero *suffix*, so this row must stay valid: its
    # diffs to both neighbors are real segments, not padding boundaries.
    encoder = _make_encoder()
    x = torch.zeros(1, 1, POINTS_PER_LINE_STRING, POINT_DIM)
    x[0, 0, 0, :2] = torch.tensor([-0.1, 0.0])
    # x[0, 0, 1] stays all-zero: a real vertex at the normalized origin.
    x[0, 0, 2, :2] = torch.tensor([0.1, 0.0])

    features = _capture_mixer_input(encoder, x)
    assert torch.allclose(features[0, 0, 2:4], torch.tensor([0.1, 0.0]), atol=1e-6)
    assert torch.allclose(features[0, 1, 2:4], torch.tensor([0.1, 0.0]), atol=1e-6)
    assert torch.all(features[0, 2, 2:4] == 0)

    with torch.no_grad():
        _, mask_p, pos = encoder(x)
    assert mask_p.tolist() == [[False]]
    # Centroid over 3 valid points (the origin vertex counts), direction +x.
    assert torch.allclose(pos[0, 0, :4], torch.tensor([0.0, 0.0, 1.0, 0.0]), atol=1e-6)


def test_closed_polyline_direction_falls_back_deterministically():
    # A closed loop (like intersection polygons) has diffs summing to ~0; the
    # direction must fall back to (1, 0) instead of being numerical noise / NaN.
    encoder = _make_encoder()
    x = torch.zeros(1, 1, POINTS_PER_LINE_STRING, POINT_DIM)
    square = torch.tensor([[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]])
    x[0, 0, :5, :2] = square
    x[0, 0, :5, 2 + ROAD_BORDER] = 1.0

    with torch.no_grad():
        _, _, pos = encoder(x)
    assert torch.all(torch.isfinite(pos))
    assert torch.allclose(pos[0, 0, :2], torch.tensor([0.02, 0.02]), atol=1e-6)
    assert torch.allclose(pos[0, 0, 2:4], torch.tensor([1.0, 0.0]), atol=1e-6)
