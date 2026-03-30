"""Unit tests for rlvr.closed_loop.state_update.

Uses synthetic tensors (no model needed).
Run: python -m rlvr.closed_loop.test_state_update
"""

from __future__ import annotations

import math

import torch

from rlvr.closed_loop.state_update import (
    advance_neighbor_past,
    build_transform_matrix,
    transform_positions_to_ego_frame,
    update_scene_state,
)


def _make_scene_data(device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
    """Create a minimal synthetic scene for testing."""
    data: dict[str, torch.Tensor] = {}

    # Ego at origin, heading forward
    data["ego_current_state"] = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        device=device,
    )  # [1, 10]

    # Ego past: 31 timesteps, last = current = [0, 0, cos(0), sin(0)]
    past = torch.zeros(1, 31, 4, device=device)
    for i in range(31):
        t = (i - 30) * 0.1  # -3.0s to 0.0s
        past[0, i, 0] = t * 5.0  # x = speed * time (going backward)
        past[0, i, 2] = 1.0      # cos(0)
    data["ego_agent_past"] = past

    # One neighbor at (10, 3) heading forward, speed 5 m/s
    nb = torch.zeros(1, 1, 31, 11, device=device)
    nb[0, 0, -1, 0] = 10.0   # x
    nb[0, 0, -1, 1] = 3.0    # y
    nb[0, 0, -1, 2] = 1.0    # cos(0)
    nb[0, 0, -1, 3] = 0.0    # sin(0)
    nb[0, 0, -1, 4] = 5.0    # vx
    nb[0, 0, -1, 5] = 0.0    # vy
    nb[0, 0, -1, 6] = 2.0    # width
    nb[0, 0, -1, 7] = 4.5    # length
    data["neighbor_agents_past"] = nb

    # Lanes: one lane segment going straight ahead
    lanes = torch.zeros(1, 140, 20, 33, device=device)
    for pt in range(20):
        x = pt * 2.0
        lanes[0, 0, pt, 0] = x       # center X
        lanes[0, 0, pt, 1] = 0.0     # center Y
        lanes[0, 0, pt, 2] = 1.0     # direction dX
        lanes[0, 0, pt, 4] = x       # left boundary X
        lanes[0, 0, pt, 5] = 1.75    # left boundary Y
        lanes[0, 0, pt, 6] = x       # right boundary X
        lanes[0, 0, pt, 7] = -1.75   # right boundary Y
    data["lanes"] = lanes

    # Route lanes (same structure)
    data["route_lanes"] = torch.zeros(1, 25, 20, 33, device=device)

    # Line strings with road borders
    ls = torch.zeros(1, 60, 20, 4, device=device)
    for pt in range(20):
        x = pt * 2.0
        ls[0, 0, pt, 0] = x       # x
        ls[0, 0, pt, 1] = 3.0     # y (left border)
        ls[0, 0, pt, 3] = 1.0     # road_border flag
        ls[0, 1, pt, 0] = x       # x
        ls[0, 1, pt, 1] = -3.0    # y (right border)
        ls[0, 1, pt, 3] = 1.0     # road_border flag
    data["line_strings"] = ls

    # Polygons, static objects
    data["polygons"] = torch.zeros(1, 10, 40, 3, device=device)
    data["static_objects"] = torch.zeros(1, 5, 10, device=device)

    # Goal: 50m ahead
    data["goal_pose"] = torch.tensor([[50.0, 0.0, 1.0, 0.0]], device=device)

    # Ego shape
    data["ego_shape"] = torch.tensor([[2.79, 4.34, 1.70]], device=device)

    return data


def test_transform_matrix():
    """Verify transform matrix convention matches centric_transform."""
    # Heading = 0 (forward) => identity-like rotation
    T = build_transform_matrix(1.0, 0.0, torch.device("cpu"))
    assert T.shape == (1, 2, 2), f"Expected (1,2,2), got {T.shape}"
    assert torch.allclose(T, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), atol=1e-6)

    # Heading = 90 degrees (pi/2) => cos=0, sin=1
    T = build_transform_matrix(0.0, 1.0, torch.device("cpu"))
    expected = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    assert torch.allclose(T, expected, atol=1e-6), f"90-degree transform mismatch: {T}"
    print("  PASS: transform_matrix")


def test_straight_forward_step():
    """Ego moves 0.5m forward (straight) — scene should shift back by 0.5m."""
    data = _make_scene_data()

    # Trajectory: ego moves 0.5m forward, heading unchanged
    trajectory = torch.zeros(80, 4)
    for t in range(80):
        trajectory[t, 0] = (t + 1) * 0.5  # x
        trajectory[t, 2] = 1.0             # cos(0)

    new_data, (dx, dy, dh) = update_scene_state(data, trajectory, step_idx=0)

    # Check ego moved
    assert abs(dx - 0.5) < 1e-5, f"dx={dx}, expected 0.5"
    assert abs(dy) < 1e-5, f"dy={dy}, expected 0.0"
    assert abs(dh) < 1e-5, f"dh={dh}, expected 0.0"

    # New ego should be at origin, heading forward
    assert abs(new_data["ego_current_state"][0, 0].item()) < 1e-5, "ego x not 0"
    assert abs(new_data["ego_current_state"][0, 1].item()) < 1e-5, "ego y not 0"
    assert abs(new_data["ego_current_state"][0, 2].item() - 1.0) < 1e-5, "ego cos not 1"
    assert abs(new_data["ego_current_state"][0, 3].item()) < 1e-5, "ego sin not 0"

    # Neighbor should have shifted: from (10, 3) to (9.5, 3)
    nb_x = new_data["neighbor_agents_past"][0, 0, -1, 0].item()
    nb_y = new_data["neighbor_agents_past"][0, 0, -1, 1].item()
    assert abs(nb_x - 9.5) < 1e-4, f"neighbor x={nb_x}, expected 9.5"
    assert abs(nb_y - 3.0) < 1e-4, f"neighbor y={nb_y}, expected 3.0"

    # Goal should have shifted: from (50, 0) to (49.5, 0)
    goal_x = new_data["goal_pose"][0, 0].item()
    assert abs(goal_x - 49.5) < 1e-4, f"goal x={goal_x}, expected 49.5"

    # Lane points should have shifted back by 0.5m
    lane_x_0 = new_data["lanes"][0, 0, 0, 0].item()
    assert abs(lane_x_0 - (-0.5)) < 1e-4, f"lane x={lane_x_0}, expected -0.5"

    print("  PASS: straight_forward_step")


def test_turn_step():
    """Ego turns 30 degrees to the left — scene should rotate clockwise."""
    data = _make_scene_data()

    angle = math.radians(30)
    # Ego moves to (1, 0.5) with heading 30 degrees
    trajectory = torch.zeros(80, 4)
    trajectory[0, 0] = 1.0
    trajectory[0, 1] = 0.5
    trajectory[0, 2] = math.cos(angle)
    trajectory[0, 3] = math.sin(angle)

    new_data, (dx, dy, dh) = update_scene_state(data, trajectory, step_idx=0)

    assert abs(dh - angle) < 1e-5, f"dh={dh}, expected {angle}"

    # Ego at origin
    assert abs(new_data["ego_current_state"][0, 0].item()) < 1e-5
    assert abs(new_data["ego_current_state"][0, 1].item()) < 1e-5
    assert abs(new_data["ego_current_state"][0, 2].item() - 1.0) < 1e-5
    assert abs(new_data["ego_current_state"][0, 3].item()) < 1e-5

    # Neighbor was at (10, 3). After translating by -(1, 0.5) => (9, 2.5)
    # Then rotating by -30 degrees:
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    expected_x = cos_a * 9.0 + sin_a * 2.5
    expected_y = -sin_a * 9.0 + cos_a * 2.5
    nb_x = new_data["neighbor_agents_past"][0, 0, -1, 0].item()
    nb_y = new_data["neighbor_agents_past"][0, 0, -1, 1].item()

    # The transform matrix is [[cos, sin], [-sin, cos]] applied to (pos - center)
    # So: R @ (pos - center) where R = [[cos(30), sin(30)], [-sin(30), cos(30)]]
    cos30 = math.cos(angle)
    sin30 = math.sin(angle)
    rel = (9.0, 2.5)
    exp_x = cos30 * rel[0] + sin30 * rel[1]
    exp_y = -sin30 * rel[0] + cos30 * rel[1]
    assert abs(nb_x - exp_x) < 1e-3, f"neighbor x={nb_x}, expected {exp_x}"
    assert abs(nb_y - exp_y) < 1e-3, f"neighbor y={nb_y}, expected {exp_y}"

    print("  PASS: turn_step")


def test_ego_past_roll():
    """Verify ego_agent_past rolls correctly: old [0,0,1,0] appended, oldest dropped."""
    data = _make_scene_data()

    trajectory = torch.zeros(80, 4)
    trajectory[0, 0] = 0.5  # forward
    trajectory[0, 2] = 1.0  # cos(0)

    old_past_first = data["ego_agent_past"][0, 0].clone()
    old_past_second = data["ego_agent_past"][0, 1].clone()

    new_data, _ = update_scene_state(data, trajectory)

    # The old first entry should be gone
    # The old second entry should now be first (after transform)
    # The last entry should be the old [0, 0, 1, 0] (transformed to new frame)
    # After straight forward step: old [0,0,1,0] transforms to [-0.5, 0, 1, 0]
    new_last = new_data["ego_agent_past"][0, -1]
    assert abs(new_last[0].item() - (-0.5)) < 1e-4, f"past last x={new_last[0].item()}, expected -0.5"
    assert abs(new_last[1].item()) < 1e-4
    assert abs(new_last[2].item() - 1.0) < 1e-4  # cos(0)
    assert abs(new_last[3].item()) < 1e-4  # sin(0)

    print("  PASS: ego_past_roll")


def test_advance_neighbor():
    """Verify advance_neighbor_past rolls and inserts correctly."""
    data = _make_scene_data()

    # New neighbor position: (10.5, 3.0, cos(0), sin(0))
    new_pos = torch.tensor([[10.5, 3.0, 1.0, 0.0]])

    advance_neighbor_past(data, new_pos, dt=0.1)

    # Check the last entry is the new position
    nb_last = data["neighbor_agents_past"][0, 0, -1]
    assert abs(nb_last[0].item() - 10.5) < 1e-4
    assert abs(nb_last[1].item() - 3.0) < 1e-4
    # Check velocity: (10.5 - 10.0) / 0.1 = 5.0, (3.0 - 3.0) / 0.1 = 0.0
    assert abs(nb_last[4].item() - 5.0) < 1e-3, f"vx={nb_last[4].item()}, expected 5.0"
    assert abs(nb_last[5].item()) < 1e-3

    print("  PASS: advance_neighbor")


def test_transform_positions_to_ego_frame():
    """Verify coordinate transform from original to ego frame."""
    # Ego at (5, 0) heading 0 in original frame
    # Point at (10, 3) in original frame -> (5, 3) in ego frame
    positions = torch.tensor([[10.0, 3.0, 0.0]])  # (x, y, heading_rad)
    result = transform_positions_to_ego_frame(
        positions, ego_x=5.0, ego_y=0.0, ego_heading=0.0,
        device=torch.device("cpu"),
    )
    assert abs(result[0, 0].item() - 5.0) < 1e-4
    assert abs(result[0, 1].item() - 3.0) < 1e-4

    # Ego at (0, 0) heading pi/2 in original frame
    # Point at (0, 5) in original frame -> (5, 0) in ego frame (rotated)
    result = transform_positions_to_ego_frame(
        torch.tensor([[0.0, 5.0, math.pi / 2]]),
        ego_x=0.0, ego_y=0.0, ego_heading=math.pi / 2,
        device=torch.device("cpu"),
    )
    assert abs(result[0, 0].item() - 5.0) < 1e-4, f"x={result[0, 0].item()}"
    assert abs(result[0, 1].item()) < 1e-4, f"y={result[0, 1].item()}"

    print("  PASS: transform_positions_to_ego_frame")


def test_no_original_mutation():
    """Verify update_scene_state does not modify the original data dict."""
    data = _make_scene_data()
    orig_ego_x = data["ego_current_state"][0, 0].item()
    orig_nb_x = data["neighbor_agents_past"][0, 0, -1, 0].item()

    trajectory = torch.zeros(80, 4)
    trajectory[0, 0] = 1.0
    trajectory[0, 2] = 1.0

    new_data, _ = update_scene_state(data, trajectory)

    # Original should be unchanged
    assert data["ego_current_state"][0, 0].item() == orig_ego_x, "Original ego mutated!"
    assert data["neighbor_agents_past"][0, 0, -1, 0].item() == orig_nb_x, "Original neighbor mutated!"

    print("  PASS: no_original_mutation")


if __name__ == "__main__":
    print("Running state_update tests...")
    test_transform_matrix()
    test_straight_forward_step()
    test_turn_step()
    test_ego_past_roll()
    test_advance_neighbor()
    test_transform_positions_to_ego_frame()
    test_no_original_mutation()
    print("\nAll state_update tests passed!")
