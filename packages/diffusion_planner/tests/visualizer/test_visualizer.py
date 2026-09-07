"""Tests for frame adaptation and Plotly figure construction."""

from __future__ import annotations

import unittest

import numpy as np

from diffusion_planner.data.dimensions import AGENT_LABEL_DIM
from diffusion_planner.visualizer import FrameData, FramePlotOptions, plot_frame


def make_frame() -> dict[str, np.ndarray]:
    """Create a compact frame using the production tensor layouts."""
    ego = np.zeros((3, 6), dtype=np.float32)
    ego[:, 0] = [-2.0, -1.0, 0.0]
    ego[:, 2] = 1.0

    neighbors = np.zeros((2, 3, 4), dtype=np.float32)
    neighbors[0, :, 0] = [3.0, 4.0, 5.0]
    neighbors[0, :, 1] = 2.0
    neighbors[0, :, 2] = 1.0
    agent_shape = np.zeros((2, 2), dtype=np.float32)
    agent_shape[0] = [1.8, 4.5]
    agent_label = np.zeros((2, AGENT_LABEL_DIM), dtype=np.float32)
    agent_label[0, 0] = 1.0

    lanes = np.zeros((2, 3, 6), dtype=np.float32)
    lanes[0, :, 0] = [0.0, 5.0, 10.0]
    lanes[0, :, 3] = -2.0
    lanes[0, :, 5] = 2.0

    intersection_area = np.zeros((1, 4, 2), dtype=np.float32)
    intersection_area[0] = [[5.0, 5.0], [7.0, 5.0], [7.0, 7.0], [5.0, 7.0]]

    stop_lines = np.zeros((1, 2, 2), dtype=np.float32)
    stop_lines[0] = [[8.0, -2.0], [8.0, 2.0]]

    road_borders = np.zeros((1, 3, 2), dtype=np.float32)
    road_borders[0] = [[-2.0, -5.0], [0.0, -5.0], [2.0, -5.0]]

    traffic = np.zeros((2, 3, 6), dtype=np.float32)
    traffic[0, -1, 2] = 1.0

    return {
        "ego_agent_past": ego,
        "neighbor_agents_past": neighbors,
        "agent_shape": agent_shape,
        "agent_label": agent_label,
        "lanes": lanes,
        "lane_types": np.zeros((2, 20), dtype=np.float32),
        "lanes_speed_limit": np.array([[10.0], [0.0]], dtype=np.float32),
        "route_lanes": lanes[:1].copy(),
        "route_lane_types": np.zeros((1, 20), dtype=np.float32),
        "route_lanes_speed_limit": np.array([[10.0]], dtype=np.float32),
        "lane_traffic_light_past": traffic,
        "route_traffic_light_past": traffic[:1].copy(),
        "intersection_area": intersection_area,
        "stop_lines": stop_lines,
        "road_borders": road_borders,
        "goal_pose": np.array([20.0, 1.0, 1.0, 0.0], dtype=np.float32),
        "ego_shape": np.array([3.5, 4.8, 1.8], dtype=np.float32),
    }


class FrameDataTest(unittest.TestCase):
    """Verify shape normalization and masks."""

    def test_selects_optional_batch_dimension(self) -> None:
        """An optional leading batch dimension is indexed and removed."""
        data = make_frame()
        data["ego_agent_past"] = np.stack(
            (data["ego_agent_past"], data["ego_agent_past"] + 1)
        )

        frame = FrameData.from_mapping(data, batch_index=1)

        self.assertEqual(frame["ego_agent_past"].shape, (3, 6))
        self.assertEqual(frame["ego_agent_past"][0, 0], -1.0)

    def test_rejects_missing_required_key(self) -> None:
        """Required visualization inputs are checked eagerly."""
        data = make_frame()
        del data["lanes"]

        with self.assertRaisesRegex(KeyError, "lanes"):
            FrameData.from_mapping(data)

    def test_future_validity_is_per_step(self) -> None:
        """All-zero future poses are treated as missing observations."""
        future = np.zeros((2, 4), dtype=np.float32)
        future[1, 2] = 1.0

        np.testing.assert_array_equal(FrameData.valid_steps(future), [False, True])


class PlotFrameTest(unittest.TestCase):
    """Verify the composed Plotly figure."""

    def test_builds_expected_layers_and_equal_axes(self) -> None:
        """The default composition contains core layers and equal axes."""
        figure = plot_frame(
            make_frame(), options=FramePlotOptions(show_speed_limits=True)
        )
        trace_names = {trace.name for trace in figure.data}

        self.assertIn("Lanes", trace_names)
        self.assertIn("Route", trace_names)
        self.assertIn("Ego past", trace_names)
        self.assertIn("Neighbors past", trace_names)
        self.assertIn("Ego footprint", trace_names)
        self.assertIn("Goal pose", trace_names)
        self.assertIn("Red light", trace_names)
        self.assertIn("Stop lines", trace_names)
        self.assertIn("Road borders", trace_names)
        self.assertEqual(figure.layout.yaxis.scaleanchor, "x")
        route = next(trace for trace in figure.data if trace.name == "Route")
        self.assertEqual(route.fill, "toself")
        self.assertEqual(route.fillcolor, "rgba(37, 99, 235, 0.18)")

    def test_recovers_lane_boundary_offsets(self) -> None:
        """Relative lane boundary coordinates are restored to ego coordinates."""
        figure = plot_frame(make_frame())
        boundary = next(
            trace for trace in figure.data if trace.name == "Lanes boundaries"
        )

        self.assertEqual(list(boundary.y[:3]), [-2.0, -2.0, -2.0])

    def test_future_layers_are_optional(self) -> None:
        """Input-only frames render successfully without training labels."""
        figure = plot_frame(make_frame())

        self.assertNotIn("Ego future", {trace.name for trace in figure.data})

    def test_draws_ego_prediction_footprints(self) -> None:
        prediction = np.zeros((3, 80, 4), dtype=np.float32)
        prediction[0, :, 0] = np.arange(80, dtype=np.float32) * 0.1
        prediction[0, :, 2] = 1.0

        figure = plot_frame(
            make_frame(),
            predicted_trajectory=prediction,
            options=FramePlotOptions(
                show_prediction_footprints=True,
                prediction_footprint_stride=20,
            ),
        )

        trace_names = {trace.name for trace in figure.data}
        self.assertIn("Ego prediction footprints", trace_names)


if __name__ == "__main__":
    unittest.main()
