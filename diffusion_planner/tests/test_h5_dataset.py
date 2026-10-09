"""Contract tests for training off the new-architecture H5 frame dataset."""

import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from diffusion_planner.utils.h5_dataset import (
    H5_FORMAT,
    H5_FORMAT_VERSION,
    REQUIRED_H5_KEYS,
    H5FrameData,
    convert_h5_frame,
    load_wheel_base_by_project,
)

from diffusion_planner import dimensions as dim

PAST_STEPS = dim.INPUT_T + 1
NUM_STOP_LINES = 30
NUM_ROAD_BORDERS = 30


def _h5_frame(rng: np.random.Generator, num_valid_neighbors: int = 3) -> dict:
    """Build one synthetic H5 frame with realistic padding and one-hot attributes."""
    frame = {
        "ego_agent_past": np.zeros((PAST_STEPS, 6), np.float32),
        "ego_agent_future": np.zeros((dim.OUTPUT_T, 6), np.float32),
        "neighbor_agents_past": np.zeros((dim.MAX_NUM_NEIGHBORS, PAST_STEPS, 4), np.float32),
        "neighbor_agents_future": np.zeros((dim.MAX_NUM_NEIGHBORS, dim.OUTPUT_T, 4), np.float32),
        "agent_shape": np.zeros((dim.MAX_NUM_NEIGHBORS, 2), np.float32),
        "agent_label": np.zeros((dim.MAX_NUM_NEIGHBORS, 3), np.float32),
        "lanes": np.zeros((dim.NUM_SEGMENTS_IN_LANE, dim.POINTS_PER_LANELET, 6), np.float32),
        "lane_types": np.zeros((dim.NUM_SEGMENTS_IN_LANE, 2 * dim.LINE_TYPE_NUM), np.float32),
        "lanes_speed_limit": np.zeros((dim.NUM_SEGMENTS_IN_LANE, 1), np.float32),
        "lane_traffic_light_past": np.zeros((dim.NUM_SEGMENTS_IN_LANE, PAST_STEPS, 6), np.float32),
        "route_lanes": np.zeros((dim.NUM_SEGMENTS_IN_ROUTE, dim.POINTS_PER_LANELET, 6), np.float32),
        "route_lane_types": np.zeros(
            (dim.NUM_SEGMENTS_IN_ROUTE, 2 * dim.LINE_TYPE_NUM), np.float32
        ),
        "route_lanes_speed_limit": np.zeros((dim.NUM_SEGMENTS_IN_ROUTE, 1), np.float32),
        "route_traffic_light_past": np.zeros(
            (dim.NUM_SEGMENTS_IN_ROUTE, PAST_STEPS, 6), np.float32
        ),
        "intersection_area": np.zeros((dim.NUM_POLYGONS, dim.POINTS_PER_POLYGON, 2), np.float32),
        "stop_lines": np.zeros((NUM_STOP_LINES, 2, 2), np.float32),
        "road_borders": np.zeros((NUM_ROAD_BORDERS, dim.POINTS_PER_LINE_STRING, 2), np.float32),
        "goal_pose": np.zeros(4, np.float32),
        "ego_shape": np.array([5.71111, 7.2369, 2.42741], np.float32),
        "turn_indicators": np.full(PAST_STEPS, 1.0, np.float32),
    }

    yaw = rng.uniform(-np.pi, np.pi, PAST_STEPS).astype(np.float32)
    frame["ego_agent_past"][:, 0] = np.linspace(-20.0, 0.0, PAST_STEPS)
    frame["ego_agent_past"][:, 1] = rng.normal(0.0, 0.2, PAST_STEPS)
    frame["ego_agent_past"][:, 2] = np.cos(yaw)
    frame["ego_agent_past"][:, 3] = np.sin(yaw)
    frame["ego_agent_past"][:, 4] = 6.0
    frame["ego_agent_past"][:, 5] = 0.03

    future_yaw = rng.uniform(-np.pi, np.pi, dim.OUTPUT_T).astype(np.float32)
    frame["ego_agent_future"][:, 0] = np.linspace(1.0, 40.0, dim.OUTPUT_T)
    frame["ego_agent_future"][:, 2] = np.cos(future_yaw)
    frame["ego_agent_future"][:, 3] = np.sin(future_yaw)

    for agent in range(num_valid_neighbors):
        frame["neighbor_agents_past"][agent, :, 0] = 10.0 + agent
        frame["neighbor_agents_past"][agent, :, 1] = -3.0 * agent - 1.0
        frame["neighbor_agents_past"][agent, :, 2] = 1.0
        frame["neighbor_agents_future"][agent, :, 0] = 12.0 + agent
        frame["neighbor_agents_future"][agent, :, 2] = 1.0
        frame["agent_shape"][agent] = (1.9, 4.5)
        frame["agent_label"][agent, agent % 3] = 1.0

    # One valid lane whose points walk forward, plus a red light and both boundary types.
    frame["lanes"][0, :, 0] = np.linspace(0.0, 19.0, dim.POINTS_PER_LANELET)
    frame["lanes"][0, :, 1] = 0.5
    frame["lanes"][0, :, 2:4] = (0.0, 1.75)
    frame["lanes"][0, :, 4:6] = (0.0, -1.75)
    frame["lane_types"][0, dim.LINE_TYPE_NUM - 1] = 1.0
    frame["lane_types"][0, dim.LINE_TYPE_NUM] = 1.0
    frame["lanes_speed_limit"][0, 0] = 13.89
    frame["lane_traffic_light_past"][0, :, 2] = 1.0

    frame["route_lanes"][0] = frame["lanes"][0]
    frame["route_lane_types"][0] = frame["lane_types"][0]
    frame["route_lanes_speed_limit"][0, 0] = 13.89
    frame["route_traffic_light_past"][0, :, 0] = 1.0

    frame["intersection_area"][0, :, 0] = np.linspace(5.0, 45.0, dim.POINTS_PER_POLYGON)
    frame["intersection_area"][0, :, 1] = 2.0
    frame["stop_lines"][0] = ((8.0, -2.0), (8.0, 2.0))
    frame["road_borders"][0, :, 0] = np.linspace(0.0, 19.0, dim.POINTS_PER_LINE_STRING)
    frame["road_borders"][0, :, 1] = 4.0
    frame["goal_pose"][:] = (40.0, 0.0, 1.0, 0.0)
    return frame


def _write_shard(directory, frame: dict, project_id: str = "x2_dev"):
    """Write a one-frame H5 shard plus the Parquet index addressing it."""
    shard_dir = directory / "x2_dev" / "0001_area" / "train" / "2026-09-03" / "10-00-00"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "frames.h5"
    with h5py.File(shard_path, "w") as file:
        file.attrs["format"] = H5_FORMAT
        file.attrs["format_version"] = H5_FORMAT_VERSION
        file.attrs["num_frames"] = 1
        file.attrs["project_id"] = project_id
        frames = file.create_group("frames")
        for key, value in frame.items():
            frames.create_dataset(key, data=value[None, ...])

    index_dir = directory / "indexes"
    index_dir.mkdir()
    index_path = index_dir / "train.parquet"
    relative = shard_path.relative_to(index_dir.parent)
    pq.write_table(
        pa.table(
            {
                "h5_path": pa.array([f"../{relative.as_posix()}"]),
                "frame_index": pa.array([0], pa.int64()),
                "frame_time_ns": pa.array([1], pa.int64()),
            }
        ),
        index_path,
    )
    return index_path


@pytest.fixture
def frame():
    return _h5_frame(np.random.default_rng(0))


def test_converted_shapes_match_the_canonical_contract(frame):
    """Every emitted tensor matches the shape contract the NPZ converter is held to.

    The contract is imported rather than restated so that a change to the model's tensor
    dimensions shows up here instead of drifting silently. It lives in ``rlvr``, which is a
    sibling workspace package, so the assertion is skipped where only this package is present.
    """
    scene_features = pytest.importorskip("rlvr.autoresearch.scene_features")
    converted = convert_h5_frame(frame)
    expected = scene_features._canonical_expected_shapes()

    assert set(expected).issubset(converted)
    for key, expected_shape in expected.items():
        actual = converted[key].shape
        assert len(actual) == len(expected_shape), f"{key}: {actual} vs {expected_shape}"
        for axis, (got, want) in enumerate(zip(actual, expected_shape)):
            if want is None:
                continue
            allowed = want if isinstance(want, tuple) else (want,)
            assert got in allowed, f"{key} axis {axis}: {got} not in {allowed}"


def test_ego_state_and_futures_use_the_layouts_downstream_code_reads(frame):
    """The ego current state and both futures follow the NPZ layout, not the H5 one."""
    converted = convert_h5_frame(frame)

    current = converted["ego_current_state"]
    assert current.shape == (10,)
    np.testing.assert_allclose(current[: dim.EGOSTATE.SIN + 1], frame["ego_agent_past"][-1, :4])
    assert current[dim.EGOSTATE.VX] == pytest.approx(6.0)
    assert current[dim.EGOSTATE.YAW_RATE] == pytest.approx(0.03)
    # H5 carries no lateral velocity, acceleration or steering angle.
    assert current[dim.EGOSTATE.VY] == 0.0
    assert current[dim.EGOSTATE.AX] == 0.0
    assert current[dim.EGOSTATE.AY] == 0.0
    assert current[dim.EGOSTATE.STEERING] == 0.0

    # StatePerturbation reads column 2 of the ego future as a raw heading; the neighbour
    # future is fixed at cos/sin by the canonical contract.
    assert converted["ego_agent_future"].shape[-1] == 3
    assert converted["neighbor_agents_future"].shape[-1] == dim.POSE_DIM
    np.testing.assert_allclose(converted["neighbor_agents_future"], frame["neighbor_agents_future"])
    h5_future = frame["ego_agent_future"]
    np.testing.assert_allclose(
        converted["ego_agent_future"][:, 2],
        np.arctan2(h5_future[:, 3], h5_future[:, 2]),
        atol=1e-6,
    )
    # The past and the goal keep cos/sin, which heading_to_cos_sin passes through.
    assert converted["ego_agent_past"].shape[-1] == dim.POSE_DIM
    assert converted["goal_pose"].shape[-1] == dim.POSE_DIM


def test_neighbor_channels_carry_shape_and_class_only_where_valid(frame):
    """Neighbour width/length/class are broadcast over time and masked by each step."""
    converted = convert_h5_frame(frame)
    neighbors = converted["neighbor_agents_past"]

    np.testing.assert_allclose(neighbors[..., : dim.POSE_DIM], frame["neighbor_agents_past"])
    # vx and vy have no H5 source, and NeighborEncoder zeroes them regardless.
    assert not neighbors[..., 4:6].any()
    np.testing.assert_allclose(neighbors[0, -1, 6:8], frame["agent_shape"][0])
    np.testing.assert_allclose(neighbors[0, -1, 8:11], frame["agent_label"][0])
    # A padded slot stays all-zero across the full row.
    assert not neighbors[100].any()


def test_lane_rows_are_reassembled_and_padding_stays_zero(frame):
    """Lane geometry, lights and boundary types land in this branch's 33-wide layout."""
    converted = convert_h5_frame(frame)
    lanes = converted["lanes"]
    source = frame["lanes"]

    assert lanes.shape[-1] == dim.SEGMENT_POINT_DIM
    np.testing.assert_allclose(lanes[0, :, dim.X : dim.Y + 1], source[0, :, 0:2])
    np.testing.assert_allclose(lanes[0, :, dim.LB_X : dim.LB_Y + 1], source[0, :, 2:4])
    np.testing.assert_allclose(lanes[0, :, dim.RB_X : dim.RB_Y + 1], source[0, :, 4:6])

    # dx, dy step to the next centreline sample; the final point has no successor.
    np.testing.assert_allclose(lanes[0, :-1, dim.dX], np.ones(dim.POINTS_PER_LANELET - 1))
    assert lanes[0, -1, dim.dX] == 0.0
    assert lanes[0, -1, dim.dY] == 0.0

    # A red H5 light becomes a red light here, and the arrow flag has no slot.
    light = lanes[0, 0, dim.TRAFFIC_LIGHT : dim.TRAFFIC_LIGHT + dim.TRAFFIC_LIGHT_ONE_HOT_DIM]
    np.testing.assert_allclose(light, [0.0, 0.0, 1.0, 0.0, 0.0])

    left = lanes[0, 0, dim.LINE_TYPE_LEFT_START : dim.LINE_TYPE_LEFT_START + dim.LINE_TYPE_NUM]
    right = lanes[0, 0, dim.LINE_TYPE_RIGHT_START :]
    np.testing.assert_allclose(left, frame["lane_types"][0, : dim.LINE_TYPE_NUM])
    np.testing.assert_allclose(right, frame["lane_types"][0, dim.LINE_TYPE_NUM :])

    # Padded segments must stay all-zero, or the encoder reads them as valid lanes.
    assert not lanes[1:].any()
    assert converted["route_lanes"][1:].any() == False  # noqa: E712


def test_traffic_light_unknown_and_no_light_share_one_slot(frame):
    """H5 'unknown' and 'white or no light' both map onto no_traffic_light."""
    frame["lane_traffic_light_past"][0, :, :] = 0.0
    frame["lane_traffic_light_past"][0, :, 3] = 1.0
    unknown = convert_h5_frame(frame)["lanes"][0, 0, dim.TRAFFIC_LIGHT :][
        : dim.TRAFFIC_LIGHT_ONE_HOT_DIM
    ]
    assert unknown[dim.TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT - dim.TRAFFIC_LIGHT] == 1.0

    frame["lane_traffic_light_past"][0, :, :] = 0.0
    frame["lane_traffic_light_past"][0, :, 4] = 1.0
    no_light = convert_h5_frame(frame)["lanes"][0, 0, dim.TRAFFIC_LIGHT :][
        : dim.TRAFFIC_LIGHT_ONE_HOT_DIM
    ]
    np.testing.assert_allclose(no_light, unknown)


def test_speed_limit_flag_is_boolean_and_keyed_on_availability(frame):
    """A zero H5 speed limit means unknown, which is what the flag has to say."""
    converted = convert_h5_frame(frame)
    flag = converted["lanes_has_speed_limit"]

    assert flag.dtype == np.bool_
    assert bool(flag[0, 0])
    assert not bool(flag[1, 0])
    # torch.where wants a bool condition; uint8 only survives as a deprecation warning.
    assert torch.from_numpy(flag).dtype == torch.bool


def test_stop_lines_and_road_borders_merge_into_flagged_line_strings(frame):
    """Both H5 map arrays land in one tensor whose flags say which is which."""
    line_strings = convert_h5_frame(frame)["line_strings"]

    assert line_strings.shape == (
        dim.NUM_LINE_STRINGS,
        dim.POINTS_PER_LINE_STRING,
        2 + dim.LINE_STRING_TYPE_NUM,
    )
    np.testing.assert_allclose(line_strings[0, :2, :2], frame["stop_lines"][0])
    assert line_strings[0, :2, dim.LINESTRING.STOP_LINE_FLAG].all()
    assert not line_strings[0, :2, dim.LINESTRING.ROAD_BORDER_FLAG].any()
    # Stop lines carry two points, so the rest of the row stays padding.
    assert not line_strings[0, 2:].any()

    border = line_strings[NUM_STOP_LINES]
    np.testing.assert_allclose(border[:, :2], frame["road_borders"][0])
    assert border[:, dim.LINESTRING.ROAD_BORDER_FLAG].all()
    # compute_road_border_penalty selects borders on exactly this column.
    assert bool((line_strings[..., 3] > 0.5).any(axis=-1)[NUM_STOP_LINES])


def test_polygons_get_their_type_flag_only_on_real_points(frame):
    polygons = convert_h5_frame(frame)["polygons"]

    assert polygons.shape == (dim.NUM_POLYGONS, dim.POINTS_PER_POLYGON, 2 + dim.POLYGON_TYPE_NUM)
    np.testing.assert_allclose(polygons[0, :, :2], frame["intersection_area"][0])
    assert polygons[0, :, 2].all()
    assert not polygons[1:].any()


def test_static_objects_are_emitted_as_maskable_padding(frame):
    """H5 has no static objects, and an all-zero block is what StaticEncoder masks out."""
    static = convert_h5_frame(frame)["static_objects"]
    assert static.shape == (dim.NUM_STATIC_OBJECTS, dim.STATIC_OBJECTS_SHAPE[-1])
    assert not static.any()


def test_wheel_base_slot_is_exact_when_supplied_and_geometric_otherwise(frame):
    """Slot 0 is a real wheelbase here, so a supplied value must win over the estimate."""
    base_link_to_front, length, width = frame["ego_shape"]

    estimated = convert_h5_frame(frame)["ego_shape"]
    assert estimated[dim.EGOSHAPE.WHEEL_BASE] == pytest.approx(
        2.0 * (base_link_to_front - 0.5 * length), abs=1e-5
    )
    assert estimated[dim.EGOSHAPE.LENGTH] == pytest.approx(length)
    assert estimated[dim.EGOSHAPE.WIDTH] == pytest.approx(width)

    supplied = convert_h5_frame(frame, wheel_base=4.76012)["ego_shape"]
    assert supplied[dim.EGOSHAPE.WHEEL_BASE] == pytest.approx(4.76012)


def test_load_wheel_base_by_project_reads_the_converter_param_file(tmp_path):
    path = tmp_path / "data_converter_param.json"
    path.write_text(
        '{"x2_dev": {"ego_wheel_base": 4.76012, "ego_length": 7.2369},'
        ' "no_wheel_base": {"ego_length": 1.0}}',
        encoding="utf-8",
    )
    assert load_wheel_base_by_project(path) == {"x2_dev": 4.76012}


def test_dataset_reads_a_shard_through_its_parquet_index(tmp_path, frame):
    index_path = _write_shard(tmp_path, frame)
    dataset = H5FrameData(index_path)
    try:
        assert len(dataset) == 1
        item = dataset[0]
        assert set(convert_h5_frame(frame)) == set(item)
        np.testing.assert_allclose(
            item["ego_agent_past"], frame["ego_agent_past"][:, : dim.POSE_DIM]
        )
        # Without a converter param file the wheelbase falls back to the estimate.
        assert item["ego_shape"][dim.EGOSHAPE.WHEEL_BASE] == pytest.approx(2.09266 * 2, abs=1e-3)
    finally:
        dataset.close()


def test_dataset_applies_the_per_project_wheel_base(tmp_path, frame):
    index_path = _write_shard(tmp_path, frame, project_id="x2_dev")
    dataset = H5FrameData(index_path, wheel_base_by_project={"x2_dev": 4.76012})
    try:
        assert dataset[0]["ego_shape"][dim.EGOSHAPE.WHEEL_BASE] == pytest.approx(4.76012)
    finally:
        dataset.close()


def test_subsample_keeps_paths_and_frame_indices_aligned(tmp_path, frame):
    """Thinning must slice the shard paths and the frame indices together."""
    index_path = _write_shard(tmp_path, frame)
    index_dir = index_path.parent
    shard = (index_dir.parent / "x2_dev/0001_area/train/2026-09-03/10-00-00/frames.h5").resolve()
    with h5py.File(shard, "r+") as file:
        file.attrs["num_frames"] = 4
        for key in REQUIRED_H5_KEYS:
            data = np.repeat(file["frames"][key][...], 4, axis=0)
            del file["frames"][key]
            file["frames"].create_dataset(key, data=data)
    relative = f"../{shard.relative_to(index_dir.parent).as_posix()}"
    pq.write_table(
        pa.table(
            {
                "h5_path": pa.array([relative] * 4),
                "frame_index": pa.array([0, 1, 2, 3], pa.int64()),
                "frame_time_ns": pa.array([0, 1, 2, 3], pa.int64()),
            }
        ),
        index_path,
    )

    dataset = H5FrameData(index_path)
    try:
        dataset.subsample(2)
        assert len(dataset) == 2
        assert len(dataset.data_list) == 2
        assert dataset[1] is not None
    finally:
        dataset.close()


def test_shard_with_the_wrong_format_version_is_rejected(tmp_path, frame):
    index_path = _write_shard(tmp_path, frame)
    shard = index_path.parent.parent / "x2_dev/0001_area/train/2026-09-03/10-00-00/frames.h5"
    with h5py.File(shard, "r+") as file:
        file.attrs["format_version"] = H5_FORMAT_VERSION - 1

    dataset = H5FrameData(index_path)
    with pytest.raises(ValueError, match="format version"):
        dataset[0]


def test_missing_parquet_index_fails_before_any_worker_starts(tmp_path):
    with pytest.raises(FileNotFoundError):
        H5FrameData(tmp_path / "absent.parquet")


def test_converted_batch_trains_the_unmodified_model(tmp_path):
    """A converted H5 batch runs the real training step with no change to the network.

    This is the point of re-assembling the H5 tensors instead of teaching the encoder a second
    input contract: the augmenter, the observation normalizer, the encoder and the decoder all
    take the batch as-is, and gradients reach the parameters.
    """
    from diffusion_planner.config.train_config import TrainConfig
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    from diffusion_planner.model.module.decoder import compute_training_loss
    from diffusion_planner.train_epoch import heading_to_cos_sin
    from diffusion_planner.utils.data_augmentation import StatePerturbation
    from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

    normalization_path = _write_normalization(tmp_path)
    rng = np.random.default_rng(0)
    frames = [convert_h5_frame(_h5_frame(rng, num_valid_neighbors=4)) for _ in range(2)]
    inputs = {
        key: torch.stack([torch.from_numpy(np.asarray(item[key])) for item in frames])
        for key in frames[0]
    }

    args = TrainConfig(exp_name="h5_smoke")
    args.device = "cpu"
    args.normalization_file_path = str(normalization_path)
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(str(normalization_path))

    model = Diffusion_Planner(args)
    model.train()

    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
    ego_future = inputs["ego_agent_future"]
    neighbors_future = inputs["neighbor_agents_future"]

    aug = StatePerturbation(
        augment_prob=1.0,
        device="cpu",
        num_refine=args.num_refine,
        ego_past_noise_std=args.ego_past_noise_std,
        use_smoothing_future_trajectory=args.use_smoothing_future_trajectory,
    )
    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    ego_future = heading_to_cos_sin(ego_future)
    mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = heading_to_cos_sin(neighbors_future)
    neighbors_future[mask] = 0.0
    inputs = args.observation_normalizer(inputs)

    loss = compute_training_loss(model, inputs, (ego_future, neighbors_future, mask), args)
    total = (
        args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
        + args.alpha_planning_loss * loss["ego_planning_loss"]
        + loss["turn_indicator_loss"]
    )
    assert torch.isfinite(total)
    total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def _write_normalization(directory):
    """Write a normalization JSON covering every key the observation normalizer touches."""
    widths = {
        "ego": dim.POSE_DIM,
        "neighbor": dim.POSE_DIM,
        "ego_agent_past": dim.POSE_DIM,
        "ego_current_state": dim.EGO_CURRENT_STATE_SHAPE[-1],
        "neighbor_agents_past": dim.NEIGHBOR_SHAPE[-1],
        "lanes": dim.SEGMENT_POINT_DIM,
        "lanes_speed_limit": 1,
        "route_lanes": dim.SEGMENT_POINT_DIM,
        "route_lanes_speed_limit": 1,
        "polygons": 2 + dim.POLYGON_TYPE_NUM,
        "line_strings": 2 + dim.LINE_STRING_TYPE_NUM,
        "goal_pose": dim.POSE_DIM,
    }
    path = directory / "normalization.json"
    payload = {key: {"mean": [0.0] * width, "std": [1.0] * width} for key, width in widths.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_auxiliary_losses_read_the_converted_columns(frame):
    """The road-border and neighbour-collision penalties fire on converted tensors.

    Both losses reach into specific columns this converter has to get right -- the road-border
    flag in line_strings column 3, and neighbour width/length in columns 6/7 -- and both are
    hinges, so a silently mislaid column shows up as a penalty that is always zero rather than
    as a crash. Driving the ego onto the feature and then far away pins down both ends.
    """
    from diffusion_planner.loss import (
        compute_ego_edge_points,
        compute_neighbor_collision_penalty,
        compute_road_border_penalty,
    )

    converted = convert_h5_frame(frame)
    ego_shape = torch.from_numpy(converted["ego_shape"])[None]
    line_strings = torch.from_numpy(converted["line_strings"])[None]
    neighbors_past = torch.from_numpy(converted["neighbor_agents_past"])[None]

    steps = 8
    far = torch.zeros(1, steps, 4)
    far[..., 0] = torch.linspace(0.0, 19.0, steps)
    far[..., 1] = 400.0
    far[..., 2] = 1.0
    edge_far = compute_ego_edge_points(far, ego_shape, n_interp=2)

    # The synthetic road border runs along y = 4.0.
    on_border = far.clone()
    on_border[..., 1] = 4.0
    edge_border = compute_ego_edge_points(on_border, ego_shape, n_interp=2)
    assert compute_road_border_penalty(edge_border, line_strings, margin=0.5).sum() > 0.0
    assert compute_road_border_penalty(edge_far, line_strings, margin=0.5).sum() == 0.0

    # Neighbour 0 sits at (10, -1) with the width and length taken from agent_shape.
    neighbors_future = torch.zeros(1, dim.MAX_NUM_NEIGHBORS, steps, dim.POSE_DIM)
    neighbors_future[..., 2] = 1.0
    neighbors_future[0, :4, :, 0] = 10.0
    neighbors_future[0, :4, :, 1] = -1.0
    valid = torch.zeros(1, dim.MAX_NUM_NEIGHBORS, steps, dtype=torch.bool)
    valid[0, :4] = True
    on_agent = far.clone()
    on_agent[..., 0] = 10.0
    on_agent[..., 1] = -1.0
    edge_agent = compute_ego_edge_points(on_agent, ego_shape, n_interp=2)
    margins = {
        "margin_vehicle": 0.3,
        "margin_pedestrian": 0.3,
        "margin_bicycle": 0.3,
    }
    assert (
        compute_neighbor_collision_penalty(
            edge_agent, neighbors_future, valid, neighbors_past, **margins
        ).sum()
        > 0.0
    )
    assert (
        compute_neighbor_collision_penalty(
            edge_far, neighbors_future, valid, neighbors_past, **margins
        ).sum()
        == 0.0
    )
