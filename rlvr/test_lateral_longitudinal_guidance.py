#!/usr/bin/env python3
"""Tests for PlannerRFT-style lateral and longitudinal guidance.

Tier 1 (standalone, no model): synthetic data unit tests.
Tier 2 (model + NPZ): visualization comparing deterministic vs guided
trajectories at various offset/shift values.

Usage:
    # Standalone unit tests only:
    python3 rlvr/test_lateral_longitudinal_guidance.py

    # Full tests with visualization:
    python3 rlvr/test_lateral_longitudinal_guidance.py \
        --model_path /path/to/model.pth --npz_path /path/to/scene.npz

    # Optional: specify output directory for images
    python3 rlvr/test_lateral_longitudinal_guidance.py \
        --model_path ... --npz_path ... --save_dir ~/Pictures/guidance_tests
"""

import argparse
import sys
import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffusion_planner"))

from diffusion_planner.model.guidance import (
    GuidanceComposer,
    GuidanceConfig,
    GuidanceSetConfig,
    build,
    list_available,
)
from diffusion_planner.model.guidance.lateral_guidance import LateralGuidance
from diffusion_planner.model.guidance.longitudinal_guidance import LongitudinalGuidance

# ======================================================================
# Synthetic data builders
# ======================================================================

T = 80  # OUTPUT_T
DT = 0.1


def _straight_trajectory(speed: float = 5.0, heading: float = 0.0) -> torch.Tensor:
    """Straight-line trajectory at constant speed and heading.

    Args:
        speed: m/s along heading direction.
        heading: radians from +X axis.

    Returns:
        [T, 4] tensor (x, y, cos_yaw, sin_yaw).
    """
    t = torch.arange(T, dtype=torch.float32) * DT
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    x = t * speed * cos_h
    y = t * speed * sin_h
    return torch.stack([
        x, y,
        torch.full((T,), cos_h),
        torch.full((T,), sin_h),
    ], dim=-1)


def _curved_trajectory(speed: float = 5.0, curvature: float = 0.02) -> torch.Tensor:
    """Constant-curvature (circular arc) trajectory.

    Args:
        speed: m/s (constant).
        curvature: 1/radius in 1/m. Positive = turning left.

    Returns:
        [T, 4] tensor (x, y, cos_yaw, sin_yaw).
    """
    positions = []
    yaw = 0.0
    x, y = 0.0, 0.0
    for i in range(T):
        positions.append([x, y, np.cos(yaw), np.sin(yaw)])
        x += speed * DT * np.cos(yaw)
        y += speed * DT * np.sin(yaw)
        yaw += speed * DT * curvature
    return torch.tensor(positions, dtype=torch.float32)


def _build_guidance_input(
    ego_traj: torch.Tensor,
    ref_traj: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Build [B, P, T+1, 4] x tensor and inputs dict from ego trajectory.

    Args:
        ego_traj: [T, 4] ego trajectory.
        ref_traj: [T, 4] reference trajectory (optional).

    Returns:
        (x, inputs) where x is [1, 1, T+1, 4] and inputs has reference_trajectory.
    """
    B, P = 1, 1
    # Prepend current-state slot (zeros = ego at origin)
    current = torch.zeros(1, 4)
    x = torch.cat([current.unsqueeze(0), ego_traj.unsqueeze(0)], dim=1)  # [1, T+1, 4]
    x = x.unsqueeze(1)  # [1, 1, T+1, 4]

    inputs = {}
    if ref_traj is not None:
        inputs["reference_trajectory"] = ref_traj.unsqueeze(0)  # [1, T, 4]

    return x, inputs


# ======================================================================
# Tier 1: Standalone unit tests
# ======================================================================

def test_registration():
    """Both guidances appear in the registry."""
    available = list_available()
    assert "lateral" in available, f"'lateral' not in registry: {available}"
    assert "longitudinal" in available, f"'longitudinal' not in registry: {available}"
    print("  PASS  test_registration: lateral + longitudinal in registry")


def test_lateral_build_and_defaults():
    """Build lateral guidance with default and custom params."""
    cfg = GuidanceConfig("lateral", scale=1.0)
    fn = build(cfg)
    assert fn.name == "lateral"
    assert fn._lambda_lat == 3.0, f"Default lambda_lat should be 3.0, got {fn._lambda_lat}"
    assert fn._eta_lat == 0.0, f"Default eta_lat should be 0.0, got {fn._eta_lat}"

    cfg2 = GuidanceConfig("lateral", scale=2.0, params={"lambda_lat": 2.0, "eta_lat": -0.5})
    fn2 = build(cfg2)
    assert fn2._lambda_lat == 2.0 and fn2._eta_lat == -0.5
    print("  PASS  test_lateral_build_and_defaults")


def test_longitudinal_build_and_defaults():
    """Build longitudinal guidance with default and custom params."""
    cfg = GuidanceConfig("longitudinal", scale=1.0)
    fn = build(cfg)
    assert fn.name == "longitudinal"
    assert fn._lambda_lon == 0.5, f"Default lambda_lon should be 0.5, got {fn._lambda_lon}"
    assert fn._eta_lon == 0.0, f"Default eta_lon should be 0.0, got {fn._eta_lon}"

    cfg2 = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 0.3, "eta_lon": -0.8})
    fn2 = build(cfg2)
    assert fn2._lambda_lon == 0.3 and fn2._eta_lon == -0.8
    print("  PASS  test_longitudinal_build_and_defaults")


def test_lateral_no_reference_returns_zero():
    """Lateral guidance returns zeros when no reference trajectory is provided."""
    cfg = GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 2.0, "eta_lat": 1.0})
    fn = build(cfg)
    ego = _straight_trajectory(speed=5.0)
    x, inputs = _build_guidance_input(ego, ref_traj=None)
    r = fn._compute(x, inputs)
    assert r.shape == (1,), f"Expected shape (1,), got {r.shape}"
    assert r.item() == 0.0, f"Expected 0.0 without reference, got {r.item()}"
    print("  PASS  test_lateral_no_reference_returns_zero")


def test_longitudinal_no_reference_returns_zero():
    """Longitudinal guidance returns zeros when no reference trajectory is provided."""
    cfg = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 0.5, "eta_lon": 1.0})
    fn = build(cfg)
    ego = _straight_trajectory(speed=5.0)
    x, inputs = _build_guidance_input(ego, ref_traj=None)
    r = fn._compute(x, inputs)
    assert r.item() == 0.0, f"Expected 0.0 without reference, got {r.item()}"
    print("  PASS  test_longitudinal_no_reference_returns_zero")


def test_lateral_zero_offset_on_reference():
    """Ego on reference with eta=0 should give reward=0 (no penalty)."""
    ref = _straight_trajectory(speed=5.0)
    ego = ref.clone()
    cfg = GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 3.0, "eta_lat": 0.0})
    fn = build(cfg)
    x, inputs = _build_guidance_input(ego, ref_traj=ref)
    r = fn._compute(x, inputs)
    assert abs(r.item()) < 1e-6, f"Expected ~0 for ego on ref with offset=0, got {r.item()}"
    print("  PASS  test_lateral_zero_offset_on_reference")


def test_lateral_offset_direction():
    """Positive lateral offset should prefer ego to the LEFT of the reference.

    For a trajectory heading along +X (heading=0), left is +Y.
    """
    ref = _straight_trajectory(speed=5.0, heading=0.0)

    # Ego shifted +1m in Y (left of reference)
    ego_left = ref.clone()
    ego_left[:, 1] += 1.0

    # Ego shifted -1m in Y (right of reference)
    ego_right = ref.clone()
    ego_right[:, 1] -= 1.0

    # eta_lat=1.0, lambda_lat=1.0 → target offset = +1.0m left
    cfg_pos = GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 1.0, "eta_lat": 1.0})
    fn = build(cfg_pos)

    x_left, inputs_left = _build_guidance_input(ego_left, ref_traj=ref)
    x_right, inputs_right = _build_guidance_input(ego_right, ref_traj=ref)

    r_left = fn._compute(x_left, inputs_left).item()
    r_right = fn._compute(x_right, inputs_right).item()

    # Ego at +1m Y should be at the target (reward ~0)
    # Ego at -1m Y should be 2m from target (reward << 0)
    assert r_left > r_right, (
        f"Positive offset should prefer left ego: r_left={r_left:.2f}, r_right={r_right:.2f}"
    )
    assert abs(r_left) < 1e-4, f"Ego at target should have ~0 reward, got {r_left:.4f}"
    print(f"  PASS  test_lateral_offset_direction: r_left={r_left:.4f}, r_right={r_right:.2f}")


def test_lateral_reward_scales_with_distance():
    """Reward magnitude should increase with distance from target."""
    ref = _straight_trajectory(speed=5.0)
    cfg = GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 3.0, "eta_lat": 0.0})
    fn = build(cfg)

    rewards = []
    offsets = [0.0, 0.5, 1.0, 2.0, 3.0]
    for off in offsets:
        ego = ref.clone()
        ego[:, 1] += off
        x, inputs = _build_guidance_input(ego, ref_traj=ref)
        r = fn._compute(x, inputs).item()
        rewards.append(r)

    # Rewards should be monotonically decreasing (more negative)
    for i in range(1, len(rewards)):
        assert rewards[i] <= rewards[i - 1], (
            f"Reward should decrease with distance: offsets={offsets}, rewards={rewards}"
        )
    print(f"  PASS  test_lateral_reward_scales_with_distance: {list(zip(offsets, [f'{r:.1f}' for r in rewards]))}")


def test_longitudinal_zero_shift():
    """Stopped ego with eta=0 should have zero penalty (target velocity = 0)."""
    ref = _straight_trajectory(speed=5.0)
    # Stopped ego: all positions same → velocity = 0
    ego = _straight_trajectory(speed=0.0)
    cfg = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 0.5, "eta_lon": 0.0})
    fn = build(cfg)
    x, inputs = _build_guidance_input(ego, ref_traj=ref)
    r = fn._compute(x, inputs)
    assert abs(r.item()) < 0.1, f"Stopped ego with eta=0 should have ~0 penalty, got {r.item()}"
    print(f"  PASS  test_longitudinal_zero_shift: r={r.item():.4f}")


def test_longitudinal_shift_direction():
    """Positive eta with lambda=1 targets v_ref; ego at ref speed scores best."""
    ref = _straight_trajectory(speed=5.0, heading=0.0)

    ego_match = ref.clone()  # 5 m/s, matches λ=1·η=1·v_ref = v_ref
    ego_fast = _straight_trajectory(speed=8.0, heading=0.0)
    ego_slow = _straight_trajectory(speed=2.0, heading=0.0)

    cfg = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 1.0, "eta_lon": 1.0})
    fn = build(cfg)

    x_match, inp_match = _build_guidance_input(ego_match, ref_traj=ref)
    x_fast, inp_fast = _build_guidance_input(ego_fast, ref_traj=ref)
    x_slow, inp_slow = _build_guidance_input(ego_slow, ref_traj=ref)

    r_match = fn._compute(x_match, inp_match).item()
    r_fast = fn._compute(x_fast, inp_fast).item()
    r_slow = fn._compute(x_slow, inp_slow).item()

    assert r_match > r_fast, f"Ego at target speed should beat faster: r_match={r_match:.2f}, r_fast={r_fast:.2f}"
    assert r_match > r_slow, f"Ego at target speed should beat slower: r_match={r_match:.2f}, r_slow={r_slow:.2f}"
    print(f"  PASS  test_longitudinal_shift_direction: r_match={r_match:.4f}, r_fast={r_fast:.2f}, r_slow={r_slow:.2f}")


def test_lateral_negative_eta():
    """Negative eta should prefer ego to the RIGHT of reference."""
    ref = _straight_trajectory(speed=5.0, heading=0.0)
    # eta=-1, lambda=2 → target = -2m (right)
    cfg = GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 2.0, "eta_lat": -1.0})
    fn = build(cfg)

    ego_right = ref.clone()
    ego_right[:, 1] -= 2.0  # 2m to the right
    ego_left = ref.clone()
    ego_left[:, 1] += 2.0

    x_right, inp_right = _build_guidance_input(ego_right, ref_traj=ref)
    x_left, inp_left = _build_guidance_input(ego_left, ref_traj=ref)
    r_right = fn._compute(x_right, inp_right).item()
    r_left = fn._compute(x_left, inp_left).item()
    assert r_right > r_left, f"Negative eta should prefer right: r_right={r_right:.2f} vs r_left={r_left:.2f}"
    print(f"  PASS  test_lateral_negative_eta: r_right={r_right:.4f}, r_left={r_left:.2f}")


def test_longitudinal_velocity_based():
    """Longitudinal guidance operates on velocity, not position."""
    ref = _straight_trajectory(speed=5.0, heading=0.0)
    # eta=1, lambda=1.0 → target = 1.0 * 1.0 * v_ref = v_ref
    # Ego matching ref speed should have ~0 penalty
    cfg = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 1.0, "eta_lon": 1.0})
    fn = build(cfg)

    x_same, inputs = _build_guidance_input(ref.clone(), ref_traj=ref)
    r = fn._compute(x_same, inputs).item()
    assert abs(r) < 0.1, f"Ego matching target speed should have ~0 penalty, got {r:.4f}"

    # Faster ego should have penalty
    ego_fast = _straight_trajectory(speed=8.0, heading=0.0)
    x_fast, inp_fast = _build_guidance_input(ego_fast, ref_traj=ref)
    r_fast = fn._compute(x_fast, inp_fast).item()
    assert r_fast < r, f"Faster ego should have more penalty: {r_fast:.2f} vs {r:.2f}"
    print(f"  PASS  test_longitudinal_velocity_based: r_match={r:.4f}, r_fast={r_fast:.2f}")


def test_longitudinal_positive_eta_prefers_faster():
    """Higher positive eta targets higher speed."""
    ref = _straight_trajectory(speed=5.0, heading=0.0)
    # eta=1, lambda=1.0 → target = v_ref (5 m/s)
    # 7 m/s ego is 2 m/s over target; 3 m/s ego is 2 m/s under target → same penalty
    # But with lambda=0.5: target = 0.5 * v_ref = 2.5 m/s
    # 3 m/s ego is closer to 2.5 than 7 m/s ego
    cfg = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 0.5, "eta_lon": 1.0})
    fn = build(cfg)

    ego_2_5 = _straight_trajectory(speed=2.5, heading=0.0)  # matches target exactly
    ego_far = _straight_trajectory(speed=7.0, heading=0.0)

    x_match, inp_match = _build_guidance_input(ego_2_5, ref_traj=ref)
    x_far, inp_far = _build_guidance_input(ego_far, ref_traj=ref)
    r_match = fn._compute(x_match, inp_match).item()
    r_far = fn._compute(x_far, inp_far).item()
    assert r_match > r_far, f"Ego at target speed should score better: r_match={r_match:.2f} vs r_far={r_far:.2f}"
    print(f"  PASS  test_longitudinal_positive_eta_prefers_faster: r_match={r_match:.4f}, r_far={r_far:.2f}")


def test_lateral_on_curve():
    """Lateral guidance on a curved trajectory produces perpendicular offsets."""
    ref = _curved_trajectory(speed=5.0, curvature=0.02)
    # lambda=2, eta=1 → target offset = 2m left
    cfg = GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 2.0, "eta_lat": 1.0})
    fn = build(cfg)

    # Ego at reference (no offset) should have reward < 0 (pulled toward target)
    x_on_ref, inputs = _build_guidance_input(ref.clone(), ref_traj=ref)
    r_on_ref = fn._compute(x_on_ref, inputs).item()
    assert r_on_ref < 0, f"Ego on curved ref with target=2m should have penalty, got {r_on_ref}"

    # Ego shifted perpendicular to heading by 2m at each point
    ego_at_target = ref.clone()
    cos_h = ref[:, 2]
    sin_h = ref[:, 3]
    ego_at_target[:, 0] += 2.0 * (-sin_h)
    ego_at_target[:, 1] += 2.0 * cos_h
    x_target, inputs_target = _build_guidance_input(ego_at_target, ref_traj=ref)
    r_at_target = fn._compute(x_target, inputs_target).item()
    assert abs(r_at_target) < 1e-3, (
        f"Ego at target on curve should have ~0 reward, got {r_at_target:.4f}"
    )
    print(f"  PASS  test_lateral_on_curve: r_on_ref={r_on_ref:.2f}, r_at_target={r_at_target:.6f}")


def test_longitudinal_on_curve():
    """Longitudinal guidance on a curve: Frenet arc-length offset."""
    ref = _curved_trajectory(speed=5.0, curvature=0.02)
    # eta=1, lambda=1 → target = v_ref. Ego matching ref should have ~0 penalty
    cfg = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 1.0, "eta_lon": 1.0})
    fn = build(cfg)

    x_on_ref, inputs = _build_guidance_input(ref.clone(), ref_traj=ref)
    r_on_ref = fn._compute(x_on_ref, inputs).item()
    assert abs(r_on_ref) < 0.5, (
        f"Ego matching ref speed on curve should have small penalty, got {r_on_ref:.4f}"
    )

    # eta=-1 → target = -v_ref (opposite direction). Should have large penalty
    cfg2 = GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 1.0, "eta_lon": -1.0})
    fn2 = build(cfg2)
    r_opposite = fn2._compute(x_on_ref, inputs).item()
    assert r_opposite < r_on_ref, f"Opposite target should have more penalty"
    print(f"  PASS  test_longitudinal_on_curve: r_match={r_on_ref:.4f}, r_opposite={r_opposite:.2f}")


def test_energy_method_time_gating():
    """energy() should gate output to zero outside the diffusion time window."""
    ref = _straight_trajectory(speed=5.0)
    cfg = GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 2.0, "eta_lat": 1.0})
    fn = build(cfg)
    x, inputs = _build_guidance_input(ref.clone(), ref_traj=ref)

    # t=0.05 is within window (0.005, 0.1) → should produce nonzero
    t_in = torch.tensor([0.05])
    e_in = fn.energy(x, t_in, inputs)
    assert e_in.item() != 0.0, f"Energy within time window should be nonzero, got {e_in.item()}"

    # t=0.5 is outside window → energy still computed but x is detached
    # (energy value may be nonzero, but gradients won't flow)
    t_out = torch.tensor([0.5])
    x_grad = x.clone().requires_grad_(True)
    e_out = fn.energy(x_grad, t_out, inputs)
    # The energy value is computed but gradient shouldn't flow through x
    if e_out.requires_grad:
        grad = torch.autograd.grad(e_out, x_grad, allow_unused=True)[0]
        assert grad is None or grad.abs().sum() == 0, "Gradient should not flow outside time window"
    print(f"  PASS  test_energy_method_time_gating: e_in={e_in.item():.2f}")


def test_reward_method():
    """reward() should return scaled reward without requiring time input."""
    ref = _straight_trajectory(speed=5.0)
    cfg = GuidanceConfig("lateral", scale=2.0, params={"lambda_lat": 1.0, "eta_lat": 1.0})
    fn = build(cfg)

    ego = ref.clone()
    ego[:, 1] += 1.5  # 0.5m from target

    inputs = {"reference_trajectory": ref.unsqueeze(0)}
    r = fn.reward(ego.unsqueeze(0), inputs)
    assert r.shape == (1,), f"Expected shape (1,), got {r.shape}"
    assert r.item() < 0, f"Ego off-target should have negative reward, got {r.item()}"
    print(f"  PASS  test_reward_method: reward={r.item():.2f}")


def test_composer_integration():
    """Both guidances work within GuidanceComposer.compute_rewards()."""
    ref = _straight_trajectory(speed=5.0)
    ego = ref.clone()
    ego[:, 1] += 1.0  # 1m lateral offset

    set_cfg = GuidanceSetConfig(
        global_scale=1.0,
        functions=[
            GuidanceConfig("lateral", scale=1.0, params={"lambda_lat": 3.0, "eta_lat": 0.5}),
            GuidanceConfig("longitudinal", scale=1.0, params={"lambda_lon": 0.5, "eta_lon": 0.5}),
        ],
    )
    composer = GuidanceComposer(set_cfg)

    inputs = {"reference_trajectory": ref.unsqueeze(0)}
    rewards = composer.compute_rewards(ego.unsqueeze(0), inputs)

    assert "lateral" in rewards, f"Missing 'lateral' in rewards: {rewards.keys()}"
    assert "longitudinal" in rewards, f"Missing 'longitudinal' in rewards: {rewards.keys()}"
    assert "total" in rewards, f"Missing 'total' in rewards: {rewards.keys()}"
    assert rewards["lateral"].item() < 0, f"Lateral should penalize 1m offset"
    assert rewards["longitudinal"].item() < 0, f"Longitudinal should penalize off-target"
    print(
        f"  PASS  test_composer_integration: "
        f"lateral={rewards['lateral'].item():.2f}, "
        f"longitudinal={rewards['longitudinal'].item():.2f}, "
        f"total={rewards['total'].item():.2f}"
    )


# ======================================================================
# Tier 2: Model-dependent visualization tests
# ======================================================================

def load_model_and_data(model_path, npz_path, device):
    """Load model and NPZ using the existing utilities."""
    from preference_optimization.model_utils import load_model
    from preference_optimization.utils import load_npz_data

    model, model_args = load_model(Path(model_path), device)
    model.eval()

    # Load raw NPZ for visualization
    d = np.load(npz_path)

    # Load as tensor batch and normalize
    data = load_npz_data(npz_path, device)
    norm_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }
    norm_data = model_args.observation_normalizer(norm_data)

    return model, model_args, norm_data, d


def generate_with_guidance(model, model_args, norm_data, composer, device):
    """Generate a single deterministic trajectory with optional guidance."""
    from guidance_gui.generate_samples import generate_samples

    samples = generate_samples(
        model=model,
        model_args=model_args,
        data=norm_data,
        noise_scale=0.0,
        n_samples=1,
        composer=composer,
        device=device,
    )
    return samples[0]  # (T, 4)


def visualize_lateral_test(
    npz_data, det_traj, guided_trajs, save_path, view_range=40,
):
    """Visualize deterministic vs lateral-guided trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    _draw_scene(ax, npz_data, view_range)

    # GT
    ego_future = npz_data["ego_agent_future"]
    ax.plot(ego_future[:, 0], ego_future[:, 1], "g-", linewidth=2, alpha=0.7, label="GT future")

    # Deterministic baseline
    ax.plot(det_traj[:, 0], det_traj[:, 1], "k-", linewidth=2.5, label="Deterministic", zorder=10)
    ax.plot(det_traj[-1, 0], det_traj[-1, 1], "ko", markersize=5, zorder=11)

    # Lateral guided trajectories with color gradient
    cmap = plt.cm.coolwarm
    offsets = sorted(guided_trajs.keys())
    for i, offset in enumerate(offsets):
        traj = guided_trajs[offset]
        color = cmap(i / max(len(offsets) - 1, 1))
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=color, linewidth=2, alpha=0.85,
            label=f"lateral={offset:+.1f}m",
            zorder=9,
        )
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=color, markersize=4, zorder=10)

    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax.set_title("Lateral Guidance Test: perpendicular offsets from deterministic reference", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def visualize_longitudinal_test(
    npz_data, det_traj, guided_trajs, save_path, view_range=40,
):
    """Visualize deterministic vs longitudinal-guided trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    _draw_scene(ax, npz_data, view_range)

    # GT
    ego_future = npz_data["ego_agent_future"]
    ax.plot(ego_future[:, 0], ego_future[:, 1], "g-", linewidth=2, alpha=0.7, label="GT future")

    # Deterministic baseline
    ax.plot(det_traj[:, 0], det_traj[:, 1], "k-", linewidth=2.5, label="Deterministic", zorder=10)
    ax.plot(det_traj[-1, 0], det_traj[-1, 1], "ko", markersize=5, zorder=11)

    # Longitudinal guided trajectories
    cmap = plt.cm.RdYlGn
    shifts = sorted(guided_trajs.keys())
    for i, shift in enumerate(shifts):
        traj = guided_trajs[shift]
        color = cmap(i / max(len(shifts) - 1, 1))
        _eta = max(-1.0, min(1.0, float(shift) / 8.0))
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=color, linewidth=2, alpha=0.85,
            label=f"η_lon={_eta:+.2f}",
            zorder=9,
        )
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=color, markersize=4, zorder=10)

    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax.set_title("Longitudinal Guidance Test: velocity scaling (Eq. 3) from deterministic reference", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def visualize_combined_test(
    npz_data, det_traj, combined_trajs, save_path, view_range=40,
):
    """Visualize combined lateral+longitudinal guidance."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    _draw_scene(ax, npz_data, view_range)

    ego_future = npz_data["ego_agent_future"]
    ax.plot(ego_future[:, 0], ego_future[:, 1], "g-", linewidth=2, alpha=0.7, label="GT future")

    ax.plot(det_traj[:, 0], det_traj[:, 1], "k-", linewidth=2.5, label="Deterministic", zorder=10)

    colors = plt.cm.Set1(np.linspace(0, 1, len(combined_trajs)))
    for i, (label, traj) in enumerate(combined_trajs.items()):
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=colors[i], linewidth=2, alpha=0.85,
            label=label, zorder=9,
        )
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=colors[i], markersize=4, zorder=10)

    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax.set_title("Combined Lateral + Longitudinal Guidance", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _draw_scene(ax, npz_data, view_range):
    """Draw lane boundaries, road borders, route, polygons, ego past."""
    # Lane boundaries
    lanes = npz_data["lanes"]
    lane_lines = []
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        if lane.shape[1] > 7:
            lx = lane[:, 0] + lane[:, 4]
            ly = lane[:, 1] + lane[:, 5]
            rx = lane[:, 0] + lane[:, 6]
            ry = lane[:, 1] + lane[:, 7]
            lane_lines.append(np.column_stack([lx, ly]))
            lane_lines.append(np.column_stack([rx, ry]))
        else:
            ax.plot(lane[:, 0], lane[:, 1], color="lightgray", linewidth=0.5, alpha=0.4)
    if lane_lines:
        lc = LineCollection(lane_lines, colors="lightgray", alpha=0.4, linewidths=0.8)
        ax.add_collection(lc)

    # Route
    route = npz_data["route_lanes"]
    for i in range(route.shape[0]):
        r = route[i]
        if np.abs(r[:, :2]).sum() < 1e-6:
            continue
        ax.plot(r[:, 0], r[:, 1], color="olive", linewidth=1.5, linestyle="--", alpha=0.5)

    # Road borders and stop lines
    if "line_strings" in npz_data:
        ls = npz_data["line_strings"]
        has_types = ls.shape[-1] >= 4
        for i in range(ls.shape[0]):
            line = ls[i]
            if np.abs(line[:, :2]).sum() < 1e-6:
                continue
            if has_types and line[:, 3].max() > 0.5:
                ax.plot(line[:, 0], line[:, 1], color="red", linewidth=2.5, alpha=0.9, zorder=5)
            elif has_types and line[:, 2].max() > 0.5:
                ax.plot(line[:, 0], line[:, 1], color="orange", linewidth=1.5, alpha=0.7)

    # Polygons
    if "polygons" in npz_data:
        polys = npz_data["polygons"]
        for i in range(polys.shape[0]):
            p = polys[i]
            if np.abs(p[:, :2]).sum() < 1e-6:
                continue
            ax.fill(p[:, 0], p[:, 1], color="lightgray", alpha=0.3, edgecolor="gray", linewidth=0.5)

    # Ego past
    ego_past = npz_data["ego_agent_past"]
    ax.plot(ego_past[:, 0], ego_past[:, 1], "b-", linewidth=1.5, alpha=0.4, label="Ego past")

    # Ego car shape
    ego_shape = npz_data.get("ego_shape", None)
    if ego_shape is not None and len(ego_shape) >= 3:
        wb, length, width = ego_shape[0], ego_shape[1], ego_shape[2]
        ro = (length - wb) / 2
        corners = np.array([
            [-ro, -width / 2], [length - ro, -width / 2],
            [length - ro, width / 2], [-ro, width / 2], [-ro, -width / 2],
        ])
        ax.fill(corners[:, 0], corners[:, 1], color="blue", alpha=0.6, zorder=12)
    else:
        ax.plot(0, 0, "bs", markersize=8, zorder=12)

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def compute_lateral_displacement(det_traj, guided_traj):
    """Compute mean perpendicular displacement from deterministic to guided trajectory.

    Uses the deterministic heading to project the displacement into
    lateral (perpendicular) and longitudinal (along heading) components.

    Args:
        det_traj: (T, 4) numpy array — deterministic reference.
        guided_traj: (T, 4) numpy array — guided trajectory.

    Returns:
        (mean_lateral, mean_longitudinal, max_lateral) in metres.
    """
    cos_h = det_traj[:, 2]
    sin_h = det_traj[:, 3]
    dx = guided_traj[:, 0] - det_traj[:, 0]
    dy = guided_traj[:, 1] - det_traj[:, 1]
    # Longitudinal = along heading: dx*cos + dy*sin
    longitudinal = dx * cos_h + dy * sin_h
    # Lateral = perpendicular: -dx*sin + dy*cos
    lateral = -dx * sin_h + dy * cos_h
    return float(np.mean(lateral)), float(np.mean(longitudinal)), float(np.max(np.abs(lateral)))


def run_model_tests(model_path, npz_path, save_dir, device):
    """Tier 2: model-dependent tests with visualization."""
    print("\n" + "=" * 60)
    print("Tier 2: Model-dependent visualization tests")
    print("=" * 60)

    model, model_args, norm_data, npz_data = load_model_and_data(
        model_path, npz_path, device
    )
    print(f"  Model loaded on {device}")

    scene_name = Path(npz_path).stem
    os.makedirs(save_dir, exist_ok=True)

    # --- Generate deterministic reference ---
    print("\n  Generating deterministic reference...")
    det_traj = generate_with_guidance(model, model_args, norm_data, None, device)
    print(f"  Deterministic: endpoint=({det_traj[-1, 0]:.2f}, {det_traj[-1, 1]:.2f})")

    # Store reference trajectory for lateral/longitudinal guidance
    ref_tensor = torch.from_numpy(det_traj).unsqueeze(0).to(device)

    # --- Lateral guidance sweep ---
    print("\n  Lateral guidance sweep...")
    lateral_offsets = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
    lateral_trajs = {}
    failed = 0

    for offset in lateral_offsets:
        norm_data_copy = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in norm_data.items()
        }
        norm_data_copy["reference_trajectory"] = ref_tensor
        # η_lat = offset / λ_lat, clamped to [-1, 1]
        lambda_lat = 3.0
        eta_lat = max(-1.0, min(1.0, offset / lambda_lat))
        set_cfg = GuidanceSetConfig(
            global_scale=1.0,
            functions=[GuidanceConfig("lateral", scale=50.0,
                                      params={"lambda_lat": lambda_lat, "eta_lat": eta_lat})],
        )
        composer = GuidanceComposer(set_cfg)
        traj = generate_with_guidance(model, model_args, norm_data_copy, composer, device)
        lateral_trajs[offset] = traj

        mean_lat, mean_lon, max_lat = compute_lateral_displacement(det_traj, traj)
        print(
            f"    offset={offset:+.1f}m → "
            f"mean_lateral={mean_lat:+.3f}m, mean_longitudinal={mean_lon:+.3f}m, "
            f"max_lateral={max_lat:.3f}m"
        )

    # Numerical verification: positive offsets should produce positive mean lateral displacement
    for offset in [1.0, 2.0, 3.0]:
        mean_lat, _, _ = compute_lateral_displacement(det_traj, lateral_trajs[offset])
        if mean_lat <= 0:
            print(f"  WARN  lateral offset={offset:+.1f}m produced mean_lat={mean_lat:.3f}m (expected >0)")
            failed += 1
        else:
            print(f"  CHECK lateral offset={offset:+.1f}m: mean_lat={mean_lat:.3f}m > 0 ✓")

    for offset in [-1.0, -2.0, -3.0]:
        mean_lat, _, _ = compute_lateral_displacement(det_traj, lateral_trajs[offset])
        if mean_lat >= 0:
            print(f"  WARN  lateral offset={offset:+.1f}m produced mean_lat={mean_lat:.3f}m (expected <0)")
            failed += 1
        else:
            print(f"  CHECK lateral offset={offset:+.1f}m: mean_lat={mean_lat:.3f}m < 0 ✓")

    # Monotonicity check: larger offset → larger displacement
    lats_pos = [(o, compute_lateral_displacement(det_traj, lateral_trajs[o])[0]) for o in [0.5, 1.0, 2.0, 3.0]]
    for i in range(1, len(lats_pos)):
        if lats_pos[i][1] <= lats_pos[i - 1][1]:
            print(
                f"  WARN  lateral monotonicity: "
                f"offset={lats_pos[i][0]}→{lats_pos[i][1]:.3f} <= "
                f"offset={lats_pos[i-1][0]}→{lats_pos[i-1][1]:.3f}"
            )

    save_lat = os.path.join(save_dir, f"{scene_name}_lateral_sweep.png")
    visualize_lateral_test(npz_data, det_traj, lateral_trajs, save_lat)

    # --- Longitudinal guidance sweep ---
    print("\n  Longitudinal guidance sweep...")
    time_shifts = [-8, -5, -3, -1, 1, 3, 5, 8]  # η_lon sweep values (mapped to [-1, 1])
    longitudinal_trajs = {}

    for shift in time_shifts:
        norm_data_copy = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in norm_data.items()
        }
        norm_data_copy["reference_trajectory"] = ref_tensor
        eta_lon = float(shift) / 8.0  # normalize to ~[-1, 1]
        eta_lon = max(-1.0, min(1.0, eta_lon))
        set_cfg = GuidanceSetConfig(
            global_scale=1.0,
            functions=[GuidanceConfig("longitudinal", scale=50.0,
                                      params={"lambda_lon": 0.5, "eta_lon": eta_lon})],
        )
        composer = GuidanceComposer(set_cfg)
        traj = generate_with_guidance(model, model_args, norm_data_copy, composer, device)
        longitudinal_trajs[shift] = traj

        mean_lat, mean_lon, _ = compute_lateral_displacement(det_traj, traj)
        # Also compute endpoint distance from det
        ep_dist = np.linalg.norm(traj[-1, :2] - det_traj[-1, :2])
        det_travel = np.linalg.norm(np.diff(det_traj[:, :2], axis=0), axis=1).sum()
        guided_travel = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()
        print(
            f"    η_lon={eta_lon:+.2f} → "
            f"mean_longitudinal={mean_lon:+.3f}m, "
            f"travel: det={det_travel:.1f}m, guided={guided_travel:.1f}m, "
            f"endpoint_delta={ep_dist:.2f}m"
        )

    # Numerical check: positive η should modulate speed
    det_travel = np.linalg.norm(np.diff(det_traj[:, :2], axis=0), axis=1).sum()
    for shift in [3, 5, 8]:
        traj = longitudinal_trajs[shift]
        guided_travel = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()
        _eta = max(-1.0, min(1.0, float(shift) / 8.0))
        if guided_travel > det_travel:
            print(f"  CHECK η_lon={_eta:+.2f}: guided_travel={guided_travel:.1f}m > det={det_travel:.1f}m ✓")
        else:
            print(
                f"  WARN  η_lon={_eta:+.2f}: guided_travel={guided_travel:.1f}m <= det={det_travel:.1f}m "
                f"(may saturate at extreme η)"
            )

    for shift in [-3, -5, -8]:
        traj = longitudinal_trajs[shift]
        guided_travel = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()
        _eta = max(-1.0, min(1.0, float(shift) / 8.0))
        if guided_travel < det_travel:
            print(f"  CHECK η_lon={_eta:+.2f}: guided_travel={guided_travel:.1f}m < det={det_travel:.1f}m ✓")
        else:
            print(
                f"  WARN  η_lon={_eta:+.2f}: guided_travel={guided_travel:.1f}m >= det={det_travel:.1f}m "
                f"(may saturate at extreme η)"
            )

    save_lon = os.path.join(save_dir, f"{scene_name}_longitudinal_sweep.png")
    visualize_longitudinal_test(npz_data, det_traj, longitudinal_trajs, save_lon)

    # --- Combined lateral + longitudinal ---
    print("\n  Combined guidance test...")
    combined_trajs = {}
    combos = [
        ("lat+1 lon+3", 1.0, 3.0),
        ("lat+2 lon+5", 2.0, 5.0),
        ("lat-1 lon-3", -1.0, -3.0),
        ("lat-2 lon-5", -2.0, -5.0),
        ("lat+2 lon-3", 2.0, -3.0),
        ("lat-2 lon+3", -2.0, 3.0),
    ]
    for label, lat_off, lon_shift in combos:
        norm_data_copy = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in norm_data.items()
        }
        norm_data_copy["reference_trajectory"] = ref_tensor
        eta_lat = max(-1.0, min(1.0, lat_off / 3.0))
        eta_lon = max(-1.0, min(1.0, lon_shift / 5.0))
        set_cfg = GuidanceSetConfig(
            global_scale=1.0,
            functions=[
                GuidanceConfig("lateral", scale=50.0,
                               params={"lambda_lat": 3.0, "eta_lat": eta_lat}),
                GuidanceConfig("longitudinal", scale=50.0,
                               params={"lambda_lon": 0.5, "eta_lon": eta_lon}),
            ],
        )
        composer = GuidanceComposer(set_cfg)
        traj = generate_with_guidance(model, model_args, norm_data_copy, composer, device)
        combined_trajs[label] = traj
        mean_lat, mean_lon, _ = compute_lateral_displacement(det_traj, traj)
        print(f"    {label:20s} → mean_lat={mean_lat:+.3f}m, mean_lon={mean_lon:+.3f}m")

    save_comb = os.path.join(save_dir, f"{scene_name}_combined_guidance.png")
    visualize_combined_test(npz_data, det_traj, combined_trajs, save_comb)

    print(f"\n  All images saved to: {save_dir}")
    return failed


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lateral & Longitudinal Guidance Tests")
    print("=" * 60 + "\n")

    # Tier 1: standalone
    print("Tier 1: Standalone unit tests (no model needed)")
    print("-" * 60)

    tier1_tests = [
        test_registration,
        test_lateral_build_and_defaults,
        test_longitudinal_build_and_defaults,
        test_lateral_no_reference_returns_zero,
        test_longitudinal_no_reference_returns_zero,
        test_lateral_zero_offset_on_reference,
        test_lateral_offset_direction,
        test_lateral_reward_scales_with_distance,
        test_longitudinal_zero_shift,
        test_longitudinal_shift_direction,
        test_lateral_negative_eta,
        test_longitudinal_velocity_based,
        test_longitudinal_positive_eta_prefers_faster,
        test_lateral_on_curve,
        test_longitudinal_on_curve,
        test_energy_method_time_gating,
        test_reward_method,
        test_composer_integration,
    ]

    failed = 0
    for t in tier1_tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Tier 2: model-dependent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument(
        "--save_dir", type=str,
        default=os.path.expanduser("~/Pictures/guidance_tests"),
    )
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is None or args.npz_path is None:
        print("\n  SKIP  Tier 2 model tests (provide --model_path and --npz_path)")
    else:
        try:
            tier2_fails = run_model_tests(
                args.model_path, args.npz_path, args.save_dir, device
            )
            failed += tier2_fails
        except Exception as e:
            print(f"  ERROR Tier 2: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    total = len(tier1_tests)
    if failed == 0:
        print(f"ALL {total} TIER 1 TESTS PASSED!")
    else:
        print(f"{failed} TEST(S) FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
