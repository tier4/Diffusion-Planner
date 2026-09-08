"""Behavioral checks for traffic-light future forward-filling."""

from __future__ import annotations

import unittest

import numpy as np

from diffusion_planner.data.transforms.traffic_light import (
    UNKNOWN_INDEX,
    fill_unknown_traffic_light_futures,
)

GREEN_INDEX = 0
RED_INDEX = 1


def _states(*indices: int) -> np.ndarray:
    """Build a one-hot traffic-light sequence with shape `(len(indices), 6)`."""
    states = np.zeros((len(indices), 6), dtype=np.float32)
    states[np.arange(len(indices)), list(indices)] = 1.0
    return states


class FillUnknownTrafficLightFuturesTest(unittest.TestCase):
    def _fill(self, past: np.ndarray, future: np.ndarray) -> np.ndarray:
        frame = {
            "lane_traffic_light_past": past[None],
            "lane_traffic_light_future": future[None],
        }
        filled = fill_unknown_traffic_light_futures(frame)
        self.assertIs(
            filled["lane_traffic_light_past"], frame["lane_traffic_light_past"]
        )
        result = filled["lane_traffic_light_future"]
        self.assertEqual(result.shape, future[None].shape)
        self.assertEqual(result.dtype, future.dtype)
        return result[0]

    def test_keeps_a_future_without_unknown_states(self) -> None:
        past = _states(GREEN_INDEX, GREEN_INDEX)
        future = _states(GREEN_INDEX, RED_INDEX, RED_INDEX)

        np.testing.assert_array_equal(self._fill(past, future), future)

    def test_fills_leading_unknown_states_from_the_last_past_state(self) -> None:
        past = _states(GREEN_INDEX, RED_INDEX)
        future = _states(UNKNOWN_INDEX, UNKNOWN_INDEX, GREEN_INDEX)

        np.testing.assert_array_equal(
            self._fill(past, future), _states(RED_INDEX, RED_INDEX, GREEN_INDEX)
        )

    def test_fills_trailing_unknown_states_from_the_last_known_future(self) -> None:
        past = _states(RED_INDEX)
        future = _states(GREEN_INDEX, UNKNOWN_INDEX, UNKNOWN_INDEX)

        np.testing.assert_array_equal(
            self._fill(past, future), _states(GREEN_INDEX, GREEN_INDEX, GREEN_INDEX)
        )

    def test_fills_each_element_independently(self) -> None:
        past = np.stack((_states(GREEN_INDEX), _states(RED_INDEX)))
        future = np.stack(
            (
                _states(UNKNOWN_INDEX, RED_INDEX),
                _states(UNKNOWN_INDEX, UNKNOWN_INDEX),
            )
        )
        frame = {
            "route_traffic_light_past": past,
            "route_traffic_light_future": future,
        }

        filled = fill_unknown_traffic_light_futures(frame)

        np.testing.assert_array_equal(
            filled["route_traffic_light_future"],
            np.stack(
                (
                    _states(GREEN_INDEX, RED_INDEX),
                    _states(RED_INDEX, RED_INDEX),
                )
            ),
        )

    def test_keeps_padded_elements_zero(self) -> None:
        past = np.zeros((3, 6), dtype=np.float32)
        future = np.zeros((4, 6), dtype=np.float32)

        np.testing.assert_array_equal(self._fill(past, future), future)

    def test_leaves_a_frame_without_traffic_lights_unchanged(self) -> None:
        frame = {"lanes": np.zeros((2, 2), dtype=np.float32)}

        filled = fill_unknown_traffic_light_futures(frame)

        self.assertEqual(tuple(filled), ("lanes",))
        self.assertIs(filled["lanes"], frame["lanes"])


if __name__ == "__main__":
    unittest.main()
