import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from diffusion_planner.dimensions import SEGMENT_POINT_DIM
from diffusion_planner.utils.forced_augmentation import (
    build_aug_pipeline,
    validate_forced_aug_flags,
)
from diffusion_planner.utils.online_augmentations import (
    FlipAugmentation,
    NeighborDropoutAugmentation,
    NeighborNoiseAugmentation,
    TurnIndicatorDropoutAugmentation,
)
from train_predictor import get_args


def _batch(batch_size=2):
    return {
        "ego_current_state": torch.ones(batch_size, 10),
        "ego_agent_past": torch.ones(batch_size, 2, 4),
        "goal_pose": torch.ones(batch_size, 4),
        "neighbor_agents_past": torch.ones(batch_size, 2, 2, 11),
        "static_objects": torch.ones(batch_size, 1, 10),
        "polygons": torch.ones(batch_size, 1, 2, 3),
        "line_strings": torch.ones(batch_size, 1, 2, 4),
        "lanes": torch.ones(batch_size, 1, 2, SEGMENT_POINT_DIM),
        "route_lanes": torch.ones(batch_size, 1, 2, SEGMENT_POINT_DIM),
        "turn_indicators": torch.ones(batch_size, 2, dtype=torch.long),
    }


def test_forced_flip_overrides_zero_probability():
    inputs = _batch()
    ego_future = torch.ones(2, 2, 3)
    neighbors_future = torch.ones(2, 2, 2, 3)

    FlipAugmentation(0.0, "cpu")(
        inputs, ego_future, neighbors_future, force=torch.tensor([True, False])
    )

    assert torch.equal(ego_future[:, 0, 1], torch.tensor([-1.0, 1.0]))


def test_forced_neighbor_dropout_drops_one_when_probability_is_zero():
    inputs = _batch()
    ego_future = torch.ones(2, 2, 3)
    neighbors_future = torch.ones(2, 2, 2, 3)

    NeighborDropoutAugmentation(0.0, "cpu")(
        inputs, ego_future, neighbors_future, force=torch.tensor([True, False])
    )

    assert torch.count_nonzero(inputs["neighbor_agents_past"][0, 0]) == 0
    assert torch.count_nonzero(inputs["neighbor_agents_past"][1]) > 0


def test_forced_turn_indicator_dropout_preserves_gt():
    inputs = _batch()
    original = inputs["turn_indicators"].clone()

    TurnIndicatorDropoutAugmentation(0.0, "cpu")(
        inputs,
        torch.ones(2, 2, 3),
        torch.ones(2, 2, 2, 3),
        force=torch.tensor([True, False]),
    )

    assert torch.count_nonzero(inputs["turn_indicators"][0]) == 0
    assert torch.equal(inputs["turn_indicators"][1], original[1])
    assert torch.equal(inputs["turn_indicators_gt_source"], original)


def test_real_classes_build_requested_pipeline_and_cli_flags_parse():
    classes = {
        "flip": FlipAugmentation(0.5, "cpu"),
        "neighbor_dropout": NeighborDropoutAugmentation(0.1, "cpu"),
        "neighbor_noise": NeighborNoiseAugmentation(0.2, 0.3, 0.05, "cpu"),
        "turn_indicator": TurnIndicatorDropoutAugmentation(0.1, "cpu"),
    }
    assert [name for name, _ in build_aug_pipeline(classes)] == list(classes)

    args = get_args(
        [
            "--exp_name",
            "test",
            "--save_dir",
            "/tmp/test",
            "--train_set_list",
            "train.json",
            "--valid_set_list",
            "valid.json",
            "--cluster_json",
            "clusters.json",
            "--force_aug_on_repeat",
            "True",
            "--repeat_aug_pool",
            "flip,neighbor_dropout,neighbor_noise,turn_indicator",
            "--use_flip_augment",
            "True",
            "--use_neighbor_dropout",
            "True",
            "--use_neighbor_noise",
            "True",
            "--use_turn_indicator_dropout",
            "True",
        ]
    )
    assert args.use_flip_augment
    assert args.use_neighbor_dropout
    assert args.use_neighbor_noise
    assert args.use_turn_indicator_dropout
    pool, warnings = validate_forced_aug_flags(args, build_aug_pipeline(classes))
    assert pool == ["flip", "neighbor_dropout", "neighbor_noise", "turn_indicator"]
    assert warnings == []
