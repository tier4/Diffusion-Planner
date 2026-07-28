"""Direct tests for the 4-col width machinery added with the canonical
``planner_metrics.scene_format.future_to_4col``:

- the canonical helper itself (padding semantics, idempotence, dtype),
- the 4-col branch of ``transform_positions_to_ego_frame``,
- ``neighbor_db._match_future_width`` in both directions,
- ``SyntheticColliderInjector`` writing at the batch's future width.
"""

import math

import numpy as np
import pytest
import torch
from diffusion_planner.utils.neighbor_db import _match_future_width
from diffusion_planner.utils.synthetic_neighbors import SyntheticColliderInjector

from planner_metrics.scene_format import future_to_4col
from rlvr.closed_loop.state_update import transform_positions_to_ego_frame


def test_future_to_4col_padding_and_idempotence():
    arr = np.zeros((2, 5, 3), dtype=np.float64)
    arr[0, :] = [1.0, 2.0, 0.5]
    out = future_to_4col(arr)
    assert out.shape == (2, 5, 4) and out.dtype == np.float32
    assert np.allclose(out[0, :, 2], np.cos(0.5)) and np.allclose(out[0, :, 3], np.sin(0.5))
    assert out[1].sum() == 0.0  # padding row stays fully zero
    assert future_to_4col(out) is out  # idempotent passthrough

    # Single-agent trajectories: the origin waypoint is real, not padding.
    ego = np.array([[0.0, 0.0, 0.3], [1.0, 0.0, 0.3]], dtype=np.float32)
    oe = future_to_4col(ego, zero_rows_are_padding=False)
    assert abs(oe[0, 2] - np.cos(0.3)) < 1e-6 and abs(oe[0, 3] - np.sin(0.3)) < 1e-6

    with pytest.raises(ValueError):
        future_to_4col(np.zeros((2, 5, 5), dtype=np.float32))


def test_transform_positions_to_ego_frame_4col_matches_3col():
    p3 = torch.tensor([[5.0, 3.0, 0.7], [-2.0, 1.0, -1.2]])
    p4 = torch.cat([p3[:, :2], p3[:, 2:3].cos(), p3[:, 2:3].sin()], dim=-1)
    r3 = transform_positions_to_ego_frame(p3, 1.0, 2.0, 0.3, "cpu")
    r4 = transform_positions_to_ego_frame(p4, 1.0, 2.0, 0.3, "cpu")
    assert torch.allclose(r3, r4, atol=1e-6)


def test_match_future_width_round_trip():
    f3 = torch.zeros(2, 5, 3)
    f3[0, :] = torch.tensor([1.0, 2.0, 0.5])
    f4 = _match_future_width(f3, 4)
    assert f4.shape[-1] == 4
    assert abs(f4[0, 0, 2].item() - math.cos(0.5)) < 1e-6
    assert f4[1].abs().sum() == 0  # padding preserved through widening
    back = _match_future_width(f4, 3)
    assert torch.allclose(back, f3, atol=1e-6)
    assert _match_future_width(f4, 4) is f4  # same-width passthrough
    with pytest.raises(ValueError):
        _match_future_width(torch.zeros(1, 5, 5), 4)


@pytest.mark.parametrize("width", [3, 4])
def test_synthetic_injector_matches_batch_width(width):
    torch.manual_seed(0)
    inj = SyntheticColliderInjector(
        pedestrian_prob=0.3, bicycle_prob=0.3, keep_clear_radius=3.0, straight_line=False
    )
    B, Pn = 2, 8
    batch = {
        "neighbor_agents_past": torch.zeros(B, Pn, 31, 11),
        "neighbor_agents_future": torch.zeros(B, Pn, 80, width),
        "ego_agent_future": torch.zeros(B, 80, 3),
    }
    batch["neighbor_agents_past"][:, :2] = 0.5
    batch["ego_agent_future"][:, :, 0] = torch.linspace(0, 40, 80)
    inj.inject(batch, inject_max=2, inject_prob=1.0)
    nf = batch["neighbor_agents_future"]
    assert nf.shape[-1] == width
    wrote = inj.last_injected_mask
    assert wrote.any()
    if width == 4:
        rows = nf[wrote]
        valid = rows[..., :2].abs().sum(-1) > 0
        norms = rows[valid][:, 2] ** 2 + rows[valid][:, 3] ** 2
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_synthetic_injector_rejects_bad_width():
    inj = SyntheticColliderInjector(
        pedestrian_prob=0.0, bicycle_prob=0.0, keep_clear_radius=3.0, straight_line=False
    )
    batch = {
        "neighbor_agents_past": torch.zeros(1, 4, 31, 11),
        "neighbor_agents_future": torch.zeros(1, 4, 80, 5),
        "ego_agent_future": torch.zeros(1, 80, 3),
    }
    with pytest.raises(ValueError):
        inj.inject(batch, inject_max=1, inject_prob=1.0)


def _augmentation_inputs():
    """Minimal batch for StatePerturbation.centric_transform."""
    B = 1
    inputs = {
        "ego_current_state": torch.zeros(B, 10),
        "ego_agent_past": torch.zeros(B, 31, 4),
        "ego_agent_future": torch.zeros(B, 80, 3),
        "neighbor_agents_past": torch.zeros(B, 4, 31, 11),
        "lanes": torch.zeros(B, 10, 20, 13),
        "route_lanes": torch.zeros(B, 5, 20, 13),
        "line_strings": torch.zeros(B, 6, 20, 4),
        "polygons": torch.zeros(B, 3, 40, 5),
        "static_objects": torch.zeros(B, 5, 10),
    }
    # Non-trivial ego pose so the re-centering rotation is a real one (~53 deg).
    inputs["ego_current_state"][0, :4] = torch.tensor([2.0, -1.0, 0.6, 0.8])
    ego_future = torch.zeros(B, 80, 3)
    ego_future[0, :, 0] = torch.linspace(1, 40, 80)
    return inputs, ego_future


@pytest.mark.parametrize("module", ["quintic", "bridge"])
def test_centric_transform_neighbor_future_3col_equals_4col(module):
    if module == "quintic":
        from diffusion_planner.utils.data_augmentation import StatePerturbation
    else:
        from diffusion_planner.utils.data_augmentation_bridge import StatePerturbation

    def clone_all(inputs, ego_future, nf):
        return (
            {k: v.clone() for k, v in inputs.items()},
            ego_future.clone(),
            nf.clone(),
        )

    sp = StatePerturbation.__new__(StatePerturbation)  # avoid full __init__
    sp._use_smoothing_future_trajectory = False

    nf3 = torch.zeros(1, 4, 80, 3)
    nf3[0, 0, :, 0] = torch.linspace(1, 30, 80)
    nf3[0, 0, :, 1] = 2.0
    nf3[0, 0, :, 2] = 0.9  # heading angle
    nf4 = torch.cat([nf3[..., :2], nf3[..., 2:3].cos(), nf3[..., 2:3].sin()], dim=-1)
    nf4[0, 1:] = 0.0  # padding slots stay zero in both

    in3, ef3 = _augmentation_inputs()
    in4, ef4 = _augmentation_inputs()
    _, _, out3 = sp.centric_transform(*clone_all(in3, ef3, nf3)[:2], nf3)
    _, _, out4 = sp.centric_transform(*clone_all(in4, ef4, nf4)[:2], nf4)

    assert out3.shape[-1] == 3 and out4.shape[-1] == 4
    # xy identical across widths
    assert torch.allclose(out3[..., :2], out4[..., :2], atol=1e-5)
    # 4-col heading vector must equal (cos, sin) of the 3-col transformed angle,
    # and stay unit-norm on the valid slot.
    cos3, sin3 = out3[0, 0, :, 2].cos(), out3[0, 0, :, 2].sin()
    assert torch.allclose(out4[0, 0, :, 2], cos3, atol=1e-5)
    assert torch.allclose(out4[0, 0, :, 3], sin3, atol=1e-5)
    norms = out4[0, 0, :, 2] ** 2 + out4[0, 0, :, 3] ** 2
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # padding slots remain zero
    assert out4[0, 1:].abs().sum() == 0


def test_build_repaired_targets_widens_3col_ego():
    from rlvr.autoresearch.tools.build_repaired_targets import _future_to_4col as brt_to_4col

    tr3 = np.zeros((80, 3), dtype=np.float32)
    tr3[:, 0] = np.arange(80) * 0.5
    tr3[:, 2] = 0.4
    out = brt_to_4col(tr3)
    assert out.shape == (80, 4)
    assert np.allclose(out[:, 2], np.cos(0.4)) and np.allclose(out[:, 3], np.sin(0.4))
    # origin waypoint is real for a single ego trajectory
    assert abs(out[0, 2] - np.cos(0.4)) < 1e-6
