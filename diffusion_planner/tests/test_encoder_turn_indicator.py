"""turn_indicators input: raw TurnIndicatorsReport codes must be one-hot encoded inside the
encoder (0=unset, 1=DISABLE, 2=ENABLE_LEFT, 3=ENABLE_RIGHT), with out-of-range codes mapping
to all-zero rows, and the full Encoder must consume the widened feature without shape errors."""

import torch
from diffusion_planner.dimensions import (
    EGO_SHAPE_SHAPE,
    GOAL_POSE_SHAPE,
    INPUT_T,
    LANES_HAS_SPEED_LIMIT_SHAPE,
    LANES_SHAPE,
    LANES_SPEED_LIMIT_SHAPE,
    LINE_STRING_TYPE_NUM,
    NEIGHBOR_SHAPE,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    POLYGON_TYPE_NUM,
    POSE_DIM,
    ROUTE_LANES_HAS_SPEED_LIMIT_SHAPE,
    ROUTE_LANES_SHAPE,
    ROUTE_LANES_SPEED_LIMIT_SHAPE,
    STATIC_OBJECTS_SHAPE,
    TURN_INDICATOR_INPUT_ONE_HOT_DIM,
)
from diffusion_planner.model.module.encoder import Encoder, one_hot_turn_indicators


def test_one_hot_maps_each_report_code_to_its_own_class():
    codes = torch.tensor([[0, 1, 2, 3]], dtype=torch.float32)
    out = one_hot_turn_indicators(codes)
    assert out.shape == (1, 4 * TURN_INDICATOR_INPUT_ONE_HOT_DIM)
    one_hot = out.view(1, 4, TURN_INDICATOR_INPUT_ONE_HOT_DIM)
    assert one_hot[0].tolist() == [
        [1, 0, 0, 0],  # 0: unset
        [0, 1, 0, 0],  # 1: DISABLE
        [0, 0, 1, 0],  # 2: ENABLE_LEFT
        [0, 0, 0, 1],  # 3: ENABLE_RIGHT
    ]


def test_one_hot_left_and_right_are_orthogonal():
    left = one_hot_turn_indicators(torch.tensor([[2.0]]))
    right = one_hot_turn_indicators(torch.tensor([[3.0]]))
    assert torch.dot(left.flatten(), right.flatten()).item() == 0.0


def test_out_of_range_codes_become_all_zero():
    codes = torch.tensor([[4, 7, -1]], dtype=torch.float32)
    out = one_hot_turn_indicators(codes)
    assert torch.all(out == 0)


def test_int_input_dtype_is_accepted():
    codes = torch.tensor([[1, 2]], dtype=torch.int32)
    out = one_hot_turn_indicators(codes)
    one_hot = out.view(1, 2, TURN_INDICATOR_INPUT_ONE_HOT_DIM)
    assert one_hot[0, 0].tolist() == [0, 1, 0, 0]
    assert one_hot[0, 1].tolist() == [0, 0, 1, 0]


def _build_dummy_encoder_inputs():
    # Built locally (mirroring onnx_export.build_dummy_inputs) so the test does not import
    # the full model stack and its heavier dependencies.
    return {
        "ego_agent_past": torch.randn(1, INPUT_T + 1, POSE_DIM),
        "neighbor_agents_past": torch.randn(*NEIGHBOR_SHAPE),
        "static_objects": torch.randn(*STATIC_OBJECTS_SHAPE),
        "lanes": torch.randn(*LANES_SHAPE),
        "lanes_speed_limit": torch.randn(*LANES_SPEED_LIMIT_SHAPE),
        "lanes_has_speed_limit": torch.ones(*LANES_HAS_SPEED_LIMIT_SHAPE, dtype=torch.bool),
        "route_lanes": torch.randn(*ROUTE_LANES_SHAPE),
        "route_lanes_speed_limit": torch.randn(*ROUTE_LANES_SPEED_LIMIT_SHAPE),
        "route_lanes_has_speed_limit": torch.ones(
            *ROUTE_LANES_HAS_SPEED_LIMIT_SHAPE, dtype=torch.bool
        ),
        "polygons": torch.randn(1, NUM_POLYGONS, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM),
        "line_strings": torch.randn(
            1, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM
        ),
        "goal_pose": torch.randn(*GOAL_POSE_SHAPE),
        "ego_shape": torch.randn(*EGO_SHAPE_SHAPE),
        "turn_indicators": torch.randint(0, 4, (1, INPUT_T + 1), dtype=torch.int32),
    }


def test_encoder_forward_consumes_one_hot_turn_indicators():
    from diffusion_planner.train_config import TrainConfig

    config = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
    )
    encoder = Encoder(config)
    encoder.eval()

    inputs = _build_dummy_encoder_inputs()
    # (B, INPUT_T + 1): current frame is dropped by the encoder, the rest is one-hot encoded
    inputs["turn_indicators"] = torch.randint(0, 4, (1, INPUT_T + 1), dtype=torch.float32)

    with torch.no_grad():
        out = encoder(inputs)
    assert out.shape == (1, encoder.token_num, config.hidden_dim)

    # use_turn_indicators=False must still run with the widened feature width
    encoder.use_turn_indicators = False
    with torch.no_grad():
        out_disabled = encoder(inputs)
    assert out_disabled.shape == (1, encoder.token_num, config.hidden_dim)
