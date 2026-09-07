"""Tests for the vector-input MLP-Mixer encoders."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.data.dimensions import (
    AGENT_LABEL_DIM,
    EGO_HISTORY_LENGTH,
    INTERSECTION_AREA_LENGTH,
    LANE_LENGTH,
    ROAD_BORDER_LENGTH,
    TRAFFIC_LIGHT_FUTURE_LENGTH,
    TRAFFIC_LIGHT_PAST_LENGTH,
)
from diffusion_planner.models.encoder import (
    EGO_VELOCITY_INDEX,
    EgoHistoryEncoder,
    EgoShapeEncoder,
    FloatVectorEncoder,
    FusionEncoder,
    GoalPoseEncoder,
    IntersectionAreaEncoder,
    LaneEncoder,
    NeighborAgentEncoder,
    RoadBorderEncoder,
    SceneEncoder,
    StopLineEncoder,
)


class FusionEncoderTest(unittest.TestCase):
    def test_masks_padded_tokens(self) -> None:
        encoder = FusionEncoder(hidden_dim=8, num_heads=2, depth=1)
        tokens = torch.randn(2, 4, 8)
        mask = torch.tensor([[False, False, True, True], [False, True, True, True]])

        features = encoder(tokens, mask)

        self.assertEqual(features.shape, tokens.shape)
        torch.testing.assert_close(features[mask], torch.zeros_like(features[mask]))


class SceneEncoderTest(unittest.TestCase):
    def test_tokenizes_and_fuses_input_data_map(self) -> None:
        encoder = SceneEncoder(
            hidden_dim=16,
            num_heads=4,
            fusion_depth=1,
            encoder_depth=1,
            mixer_hidden_dim=8,
        )
        input_data = {
            "ego_agent_past": torch.zeros(2, EGO_HISTORY_LENGTH, 6),
            "neighbor_agents_past": torch.zeros(2, 3, EGO_HISTORY_LENGTH, 4),
            "agent_shape": torch.zeros(2, 3, 2),
            "agent_label": torch.zeros(2, 3, AGENT_LABEL_DIM),
            "lanes": torch.zeros(2, 2, LANE_LENGTH, 6),
            "lane_types": torch.zeros(2, 2, 20),
            "lanes_speed_limit": torch.zeros(2, 2, 1),
            "lane_traffic_light_past": torch.zeros(2, 2, TRAFFIC_LIGHT_PAST_LENGTH, 6),
            "lane_traffic_light_future": torch.zeros(
                2, 2, TRAFFIC_LIGHT_FUTURE_LENGTH, 6
            ),
            "route_lanes": torch.zeros(2, 1, LANE_LENGTH, 6),
            "route_lane_types": torch.zeros(2, 1, 20),
            "route_lanes_speed_limit": torch.zeros(2, 1, 1),
            "route_traffic_light_past": torch.zeros(2, 1, TRAFFIC_LIGHT_PAST_LENGTH, 6),
            "route_traffic_light_future": torch.zeros(
                2, 1, TRAFFIC_LIGHT_FUTURE_LENGTH, 6
            ),
            "intersection_area": torch.zeros(2, 2, INTERSECTION_AREA_LENGTH, 2),
            "stop_lines": torch.zeros(2, 2, 2, 2),
            "road_borders": torch.zeros(2, 2, ROAD_BORDER_LENGTH, 2),
            "goal_pose": torch.zeros(2, 4),
            "ego_shape": torch.ones(2, 3),
        }
        input_data["neighbor_agents_past"][:, 0, :, 2] = 1.0
        input_data["lanes"][:, 0, :, 0] = 1.0
        input_data["route_lanes"][:, 0, :, 0] = 1.0
        input_data["intersection_area"][:, 0, :, 0] = 1.0
        input_data["stop_lines"][:, 0, :, 0] = 1.0
        input_data["road_borders"][:, 0, :, 0] = 1.0

        tokens, padding_mask = encoder(input_data)

        self.assertEqual(tokens.shape, (2, 15, 16))
        self.assertEqual(padding_mask.shape, (2, 15))
        torch.testing.assert_close(
            tokens[padding_mask], torch.zeros_like(tokens[padding_mask])
        )


class SceneVectorEncoderTest(unittest.TestCase):
    def test_goal_pose_encoder(self) -> None:
        encoder = GoalPoseEncoder(hidden_dim=16)
        goal_pose = torch.tensor(
            [[0.2, -0.04, 1.0, 0.0], [0.08, 0.06, 0.0, 1.0]],
            requires_grad=True,
        )

        features, mask = encoder(goal_pose)

        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(mask.shape, (2,))
        self.assertFalse(mask.any())
        features.sum().backward()
        self.assertIsNotNone(goal_pose.grad)

    def test_goal_pose_encoder_masks_goals_beyond_100_meters(self) -> None:
        encoder = GoalPoseEncoder(hidden_dim=16, max_distance=2.0)
        goal_pose = torch.tensor([[2.0, 0.0, 1.0, 0.0], [2.01, 0.0, 1.0, 0.0]])

        features, mask = encoder(goal_pose)

        torch.testing.assert_close(mask, torch.tensor([False, True]))
        torch.testing.assert_close(features[1], torch.zeros_like(features[1]))

    def test_ego_shape_encoder(self) -> None:
        encoder = EgoShapeEncoder(hidden_dim=16)
        ego_shape = torch.tensor([[3.8, 4.9, 1.9], [5.7, 7.2, 2.4]], requires_grad=True)

        features, mask = encoder(ego_shape)

        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(mask.shape, (2,))
        self.assertFalse(mask.any())
        features.sum().backward()
        self.assertIsNotNone(ego_shape.grad)


class EgoHistoryEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = EgoHistoryEncoder(
            velocity_threshold=0.1,
            drop_path_rate=0.0,
            hidden_dim=16,
            depth=1,
            mixer_hidden_dim=8,
        )

    def test_encodes_velocity_threshold_history(self) -> None:
        ego_history = torch.zeros(2, EGO_HISTORY_LENGTH, 6)
        ego_history[0, :5, 4] = torch.tensor([0.0, 0.1, 0.2, 1.0, 0.05])
        ego_history[1, :, 4] = 1.0

        features, mask = self.encoder(ego_history)

        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(mask.shape, (2,))
        self.assertFalse(torch.equal(features[0], features[1]))

    def test_ignores_features_other_than_velocity(self) -> None:
        baseline = torch.zeros(1, EGO_HISTORY_LENGTH, 6)
        changed = baseline.clone()
        changed[..., :4] = torch.randn(1, EGO_HISTORY_LENGTH, 4)
        changed[..., 5] = torch.randn(1, EGO_HISTORY_LENGTH)

        baseline_features, baseline_mask = self.encoder(baseline)
        changed_features, changed_mask = self.encoder(changed)
        torch.testing.assert_close(baseline_features, changed_features)
        torch.testing.assert_close(baseline_mask, changed_mask)

    def test_encodes_only_current_velocity_as_continuous_value(self) -> None:
        baseline = torch.ones(1, EGO_HISTORY_LENGTH, 6)
        changed_past = baseline.clone()
        changed_past[:, :-1, EGO_VELOCITY_INDEX] = 2.0
        changed_current = baseline.clone()
        changed_current[:, -1, EGO_VELOCITY_INDEX] = 2.0

        baseline_features, _ = self.encoder(baseline)
        changed_past_features, _ = self.encoder(changed_past)
        changed_current_features, _ = self.encoder(changed_current)

        torch.testing.assert_close(baseline_features, changed_past_features)
        self.assertFalse(torch.equal(baseline_features, changed_current_features))


class FloatVectorEncoderTest(unittest.TestCase):
    def test_preserves_prefix_dimensions(self) -> None:
        encoder = FloatVectorEncoder(vector_dim=3, output_dim=8)
        values = torch.ones(2, 4, 3, requires_grad=True)

        features = encoder(values)

        self.assertEqual(features.shape, (2, 4, 8))
        features.sum().backward()
        self.assertIsNotNone(values.grad)


class MapLineEncoderTest(unittest.TestCase):
    def test_intersection_area_encoder_masks_padding(self) -> None:
        encoder = IntersectionAreaEncoder(
            drop_path_rate=0.0,
            hidden_dim=16,
            depth=1,
            mixer_hidden_dim=8,
        )
        intersection_area = torch.zeros(
            2, 3, INTERSECTION_AREA_LENGTH, 2, requires_grad=True
        )
        with torch.no_grad():
            intersection_area[0, 0, :4] = torch.tensor(
                [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
            )

        features, mask = encoder(intersection_area)

        self.assertEqual(features.shape, (2, 3, 16))
        self.assertEqual(mask.shape, (2, 3))
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        torch.testing.assert_close(features[0, 1], torch.zeros(16))
        features.sum().backward()
        self.assertIsNotNone(intersection_area.grad)

    def test_road_border_encoder_masks_padding(self) -> None:
        encoder = RoadBorderEncoder(
            drop_path_rate=0.0,
            hidden_dim=16,
            depth=1,
            mixer_hidden_dim=8,
        )
        road_borders = torch.zeros(2, 3, ROAD_BORDER_LENGTH, 2, requires_grad=True)
        with torch.no_grad():
            road_borders[0, 0, :, 0] = torch.arange(ROAD_BORDER_LENGTH)

        features, mask = encoder(road_borders)

        self.assertEqual(features.shape, (2, 3, 16))
        self.assertEqual(mask.shape, (2, 3))
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        torch.testing.assert_close(features[0, 1], torch.zeros(16))
        features.sum().backward()
        self.assertIsNotNone(road_borders.grad)

    def test_stop_line_encoder_masks_padding(self) -> None:
        encoder = StopLineEncoder(
            drop_path_rate=0.0,
            hidden_dim=8,
            depth=0,
            mixer_hidden_dim=4,
        )
        stop_lines = torch.zeros(1, 2, 2, 2)
        stop_lines[0, 0] = torch.tensor([[1.0, -1.0], [1.0, 1.0]])

        features, mask = encoder(stop_lines)

        self.assertEqual(features.shape, (1, 2, 8))
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        torch.testing.assert_close(features[0, 1], torch.zeros(8))


class NeighborAgentEncoderTest(unittest.TestCase):
    def test_encodes_history_shape_and_label(self) -> None:
        encoder = NeighborAgentEncoder(
            drop_path_rate=0.0,
            hidden_dim=16,
            depth=1,
            mixer_hidden_dim=8,
        )
        history = torch.zeros(2, 3, EGO_HISTORY_LENGTH, 4, requires_grad=True)
        with torch.no_grad():
            history[0, 0, :, 0] = torch.arange(EGO_HISTORY_LENGTH)
            history[0, 0, :, 2] = 1.0
        shape = torch.zeros(2, 3, 2)
        shape[0, 0] = torch.tensor([2.0, 4.5])
        label = torch.zeros(2, 3, AGENT_LABEL_DIM)
        label[0, 0, 0] = 1.0

        features, mask = encoder(history, shape, label)

        self.assertEqual(features.shape, (2, 3, 16))
        self.assertEqual(mask.shape, (2, 3))
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        torch.testing.assert_close(features[0, 1], torch.zeros(16))
        features.sum().backward()
        self.assertIsNotNone(history.grad)


class LaneEncoderTest(unittest.TestCase):
    """Lane encoders consume the separated format-version-3 schema."""

    def setUp(self) -> None:
        self.encoder = LaneEncoder(
            drop_path_rate=0.0,
            hidden_dim=16,
            depth=1,
            mixer_hidden_dim=8,
        )

    def test_encodes_all_lane_attributes_and_masks_padding(self) -> None:
        lanes = torch.zeros(2, 3, LANE_LENGTH, 6, requires_grad=True)
        with torch.no_grad():
            lanes[0, 0, :, 0] = torch.arange(LANE_LENGTH)
        lane_types = torch.zeros(2, 3, 20)
        lane_types[0, 0, 4] = 1.0
        lane_types[0, 0, 14] = 1.0
        speed_limits = torch.zeros(2, 3, 1)
        speed_limits[0, 0] = 10.0
        traffic_lights = torch.zeros(2, 3, TRAFFIC_LIGHT_PAST_LENGTH, 6)
        traffic_lights[0, 0, -1, 0] = 1.0
        traffic_light_future = torch.zeros(2, 3, TRAFFIC_LIGHT_FUTURE_LENGTH, 6)
        traffic_light_future[0, 0, -1, 2] = 1.0

        features, mask = self.encoder(
            lanes,
            lane_types,
            speed_limits,
            traffic_lights,
            traffic_light_future,
        )

        self.assertEqual(features.shape, (2, 3, 16))
        self.assertEqual(mask.shape, (2, 3))
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        torch.testing.assert_close(features[0, 1], torch.zeros(16))
        features.sum().backward()
        self.assertIsNotNone(lanes.grad)

    def test_uses_the_complete_past_sequence(self) -> None:
        lanes = torch.ones(1, 1, LANE_LENGTH, 6)
        lane_types = torch.zeros(1, 1, 20)
        speed_limits = torch.ones(1, 1, 1)
        baseline = torch.zeros(1, 1, TRAFFIC_LIGHT_PAST_LENGTH, 6)
        changed = baseline.clone()
        changed[0, 0, 0, 2] = 1.0
        future = torch.zeros(1, 1, TRAFFIC_LIGHT_FUTURE_LENGTH, 6)

        baseline_features, _ = self.encoder(
            lanes, lane_types, speed_limits, baseline, future
        )
        changed_features, _ = self.encoder(
            lanes, lane_types, speed_limits, changed, future
        )

        self.assertFalse(torch.equal(baseline_features, changed_features))

    def test_uses_the_complete_future_sequence(self) -> None:
        lanes = torch.ones(1, 1, LANE_LENGTH, 6)
        lane_types = torch.zeros(1, 1, 20)
        speed_limits = torch.ones(1, 1, 1)
        past = torch.zeros(1, 1, TRAFFIC_LIGHT_PAST_LENGTH, 6)
        baseline = torch.zeros(1, 1, TRAFFIC_LIGHT_FUTURE_LENGTH, 6)
        changed = baseline.clone()
        changed[0, 0, 0, 0] = 1.0

        baseline_features, _ = self.encoder(
            lanes, lane_types, speed_limits, past, baseline
        )
        changed_features, _ = self.encoder(
            lanes, lane_types, speed_limits, past, changed
        )

        self.assertFalse(torch.equal(baseline_features, changed_features))


if __name__ == "__main__":
    unittest.main()
