"""Unit tests for the simulated neighbor-history mode + the shared turn-indicator decode.

Covers the invariants Copilot flagged on PR #164:
  - ``decode_turn_indicator`` KEEP-bias behavior + scalar/batched shapes.
  - ``SimNeighborTracker`` derives neighbor velocity from the SHOWN motion, not the recorded
    velocity field: a neighbor whose position is held constant reads v approx 0 even when its
    recorded velocity column says it is moving fast (the phantom-collision the sim mode fixes),
    while a neighbor whose shown position advances reads a real, non-zero velocity.
"""

import json

import numpy as np
import torch

from scenario_generation.reproducer_rollout import SimNeighborTracker, _build_nbr_world_tracks
from scenario_generation.route_timeline import RouteTimeline

# decode_turn_indicator is defined in simulate.py (shared by _predict_batch + the reproducer);
# import it from there so these tests unambiguously cover that canonical implementation.
from scenario_generation.simulate import decode_turn_indicator

EGO_SHAPE = np.array([4.76, 7.24, 2.29], dtype=np.float32)
N_FRAMES = 40
FROZEN = "frozen-uuid"
MOVER = "mover-uuid"
MOVER_STEP_M = 0.5  # shown +x per frame (0.1 s) -> ~5 m/s in the rebuilt history


def _nbr_row(x, y, recorded_vx):
    """One neighbor_agents_past row: [x, y, cos, sin, vx, vy, width, length, veh, ped, bike].

    ``recorded_vx`` is the (ignored-by-sim-mode) recorded velocity column — set it large for the
    frozen track to prove the tracker does NOT read it.
    """
    return np.array([x, y, 1.0, 0.0, recorded_vx, 0.0, 2.0, 4.5, 1.0, 0.0, 0.0], dtype=np.float32)


def _make_neighbor_route(tmp_path):
    """Ego static at the origin (yaw 0, so ego-frame == world); two tracked neighbors:

    slot 0 FROZEN — constant ego-frame position but a big fake recorded velocity (11 m/s);
    slot 1 MOVER  — ego-frame x advances ``MOVER_STEP_M``/frame, recorded velocity column 0.
    """
    paths = []
    for i in range(N_FRAMES):
        nb = np.zeros((320, 31, 11), dtype=np.float32)
        frozen_row = _nbr_row(10.0, 0.0, recorded_vx=11.0)  # held still, "moving" per recorded vx
        mover_row = _nbr_row(10.0 + MOVER_STEP_M * i, 3.0, recorded_vx=0.0)  # moves, recorded vx 0
        nb[0, :, :] = (
            frozen_row  # whole 31-step history is the current row (neighbor_last = [:,-1])
        )
        nb[1, :, :] = mover_row
        p = tmp_path / f"route_{i:010d}.npz"
        np.savez_compressed(
            p,
            neighbor_agents_past=nb,
            ego_agent_past=np.zeros((31, 3), dtype=np.float32),
            ego_shape=EGO_SHAPE,
        )
        sidecar = {
            "timestamp": float(i),
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "neighbor_ids": [FROZEN, MOVER] + [""] * 318,
        }
        (tmp_path / f"route_{i:010d}.json").write_text(json.dumps(sidecar))
        paths.append(p)
    return RouteTimeline(paths)


def test_decode_turn_indicator_keep_bias():
    # KEEP (class 4) wins the raw argmax (1.0) but loses to class 2 (0.9) after the 0.25 bias.
    logit = torch.tensor([0.0, 0.0, 0.9, 0.0, 1.0])
    assert int(decode_turn_indicator(logit, keep_bias=0.0)) == 4  # no bias -> KEEP
    assert int(decode_turn_indicator(logit, keep_bias=0.25)) == 2  # 0.9 > 1.0 - 0.25


def test_decode_turn_indicator_batched_shape():
    # Row 0: KEEP survives the bias (1.0 - 0.25 = 0.75 > 0). Row 1: class 2 dominates.
    batch = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.0, 1.0]])
    out = decode_turn_indicator(batch, keep_bias=0.25)
    assert out.shape == (2,)
    assert out.tolist() == [4, 2]


def test_sim_neighbor_velocity_from_shown_motion(tmp_path):
    tl = _make_neighbor_route(tmp_path)
    trk = SimNeighborTracker(tl, start=5, max_rec_advance=1.0)
    ego = np.array([0.0, 0.0, 0.0])  # static ego at origin
    for t in range(6, 20):
        trk.step(t, ego[:2])
    out, slot_uuids, _ = trk.build(ego)  # (1, 320, PAST, 11)

    assert FROZEN in slot_uuids and MOVER in slot_uuids
    nb = out[0]
    fi, mi = slot_uuids.index(FROZEN), slot_uuids.index(MOVER)
    # Velocity is rebuilt from the SHOWN positions (cols 4,5), regardless of the recorded vx.
    frozen_v = float(np.linalg.norm(nb[fi, -1, 4:6]))
    mover_v = float(np.linalg.norm(nb[mi, -1, 4:6]))
    assert frozen_v < 0.3, (
        f"held-still neighbor must read ~0 velocity (got {frozen_v}) despite recorded vx=11"
    )
    assert mover_v > 1.0, f"moving neighbor must read its shown velocity (got {mover_v})"
    # And the held-still neighbor's shown position stayed put.
    assert abs(float(nb[fi, -1, 0]) - 10.0) < 0.5 and abs(float(nb[fi, -1, 1])) < 0.5


def test_sim_neighbor_keeps_single_frame_track(tmp_path):
    """A neighbor present in only ONE recorded frame must still be tracked (kept as a constant,
    v~0) instead of dropped, so step 0 reproduces the recorded context."""
    blip = "blip-uuid"
    blip_frame = 3
    paths = []
    for i in range(6):
        nb = np.zeros((320, 31, 11), dtype=np.float32)
        ids = [""] * 320
        if i == blip_frame:  # the neighbor appears in exactly one frame
            nb[0, :, :] = _nbr_row(8.0, 1.0, recorded_vx=0.0)
            ids[0] = blip
        p = tmp_path / f"r_{i:010d}.npz"
        np.savez_compressed(
            p,
            neighbor_agents_past=nb,
            ego_agent_past=np.zeros((31, 3), dtype=np.float32),
            ego_shape=EGO_SHAPE,
        )
        (tmp_path / f"r_{i:010d}.json").write_text(
            json.dumps(
                {
                    "timestamp": float(i),
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                    "neighbor_ids": ids,
                }
            )
        )
        paths.append(p)
    tl = RouteTimeline(paths)
    interp, _attrs, span = _build_nbr_world_tracks(tl, 0, len(tl))
    assert blip in interp, "single-frame neighbor was dropped (should be kept as a constant track)"
    assert span[blip] == (blip_frame, blip_frame)
