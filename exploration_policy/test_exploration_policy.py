"""Unit tests for the Exploration Policy network.

Tests forward pass shapes, Beta distribution output ranges, deterministic mode,
gradient flow, and config serialization. No model checkpoint required.

Usage:
    python exploration_policy/test_exploration_policy.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from exploration_policy.model import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyOutput,
)
from exploration_policy.module.heads import GuidanceHead, ValueHead
from exploration_policy.module.ref_fusion import RefFusionAttention
from exploration_policy.module.ref_mixer import RefTrajectoryMixer


def test_ref_trajectory_mixer():
    """RefTrajectoryMixer: [B, T, 4] -> [B, H]."""
    B, T, H = 2, 80, 128
    mixer = RefTrajectoryMixer(seq_len=T, hidden_dim=H, n_layers=2, dropout=0.0)
    x_ref = torch.randn(B, T, 4)
    out = mixer(x_ref)
    assert out.shape == (B, H), f"Expected ({B}, {H}), got {out.shape}"
    print("  [PASS] RefTrajectoryMixer: shape OK")


def test_ref_fusion_attention():
    """RefFusionAttention: [B, H] + [B, N, D_enc] -> [B, H]."""
    B, H, D_enc, N = 2, 128, 256, 50
    fusion = RefFusionAttention(hidden_dim=H, encoder_dim=D_enc, n_heads=4, dropout=0.0)
    ref_token = torch.randn(B, H)
    scene = torch.randn(B, N, D_enc)
    out = fusion(ref_token, scene)
    assert out.shape == (B, H), f"Expected ({B}, {H}), got {out.shape}"
    print("  [PASS] RefFusionAttention: shape OK")


def test_guidance_head():
    """GuidanceHead: [B, H] -> two Beta distributions."""
    B, H = 2, 128
    head = GuidanceHead(H)
    fused = torch.randn(B, H)
    lat_dist, lon_dist = head(fused)

    assert lat_dist.batch_shape == (B,), f"lat batch_shape: {lat_dist.batch_shape}"
    assert lon_dist.batch_shape == (B,), f"lon batch_shape: {lon_dist.batch_shape}"

    # Beta params should be >= 1.0 (softplus + 1.0)
    assert (lat_dist.concentration1 >= 1.0).all(), "alpha_lat < 1.0"
    assert (lat_dist.concentration0 >= 1.0).all(), "beta_lat < 1.0"
    assert (lon_dist.concentration1 >= 1.0).all(), "alpha_lon < 1.0"
    assert (lon_dist.concentration0 >= 1.0).all(), "beta_lon < 1.0"
    print("  [PASS] GuidanceHead: distributions valid, params >= 1.0")


def test_value_head():
    """ValueHead: [B, H] -> [B]."""
    B, H = 2, 128
    head = ValueHead(H)
    fused = torch.randn(B, H)
    value = head(fused)
    assert value.shape == (B,), f"Expected ({B},), got {value.shape}"
    print("  [PASS] ValueHead: shape OK")


def test_exploration_policy_forward():
    """Full forward pass: scene_encoding + x_ref -> ExplorationPolicyOutput."""
    B, N, D_enc, T = 2, 50, 256, 80
    config = ExplorationPolicyConfig(
        hidden_dim=128,
        n_mixer_layers=2,
        n_attn_heads=4,
        dropout=0.0,
        encoder_hidden_dim=D_enc,
    )
    policy = ExplorationPolicy(config, ref_seq_len=T)

    scene = torch.randn(B, N, D_enc)
    x_ref = torch.randn(B, T, 4)

    output = policy(scene, x_ref, deterministic=False)
    assert isinstance(output, ExplorationPolicyOutput)
    assert output.eta_lat.shape == (B,)
    assert output.eta_lon.shape == (B,)
    assert output.log_prob_lat.shape == (B,)
    assert output.log_prob_lon.shape == (B,)
    assert output.value.shape == (B,)
    print("  [PASS] ExplorationPolicy forward: all shapes OK")


def test_eta_range():
    """Sampled eta values should be in [-1, 1]."""
    B, N, D_enc, T = 32, 50, 256, 80
    config = ExplorationPolicyConfig(
        hidden_dim=64,
        n_mixer_layers=1,
        n_attn_heads=4,
        dropout=0.0,
        encoder_hidden_dim=D_enc,
    )
    policy = ExplorationPolicy(config, ref_seq_len=T)

    scene = torch.randn(B, N, D_enc)
    x_ref = torch.randn(B, T, 4)

    # Sample many times to check range
    for _ in range(10):
        output = policy(scene, x_ref, deterministic=False)
        assert (output.eta_lat >= -1.0).all() and (output.eta_lat <= 1.0).all(), (
            f"eta_lat out of range: [{output.eta_lat.min()}, {output.eta_lat.max()}]"
        )
        assert (output.eta_lon >= -1.0).all() and (output.eta_lon <= 1.0).all(), (
            f"eta_lon out of range: [{output.eta_lon.min()}, {output.eta_lon.max()}]"
        )
    print("  [PASS] eta values in [-1, 1] across 10 samples")


def test_deterministic_mode():
    """Deterministic mode should produce identical outputs."""
    B, N, D_enc, T = 2, 50, 256, 80
    config = ExplorationPolicyConfig(
        hidden_dim=64,
        n_mixer_layers=1,
        n_attn_heads=4,
        dropout=0.0,
        encoder_hidden_dim=D_enc,
    )
    policy = ExplorationPolicy(config, ref_seq_len=T)
    policy.eval()

    scene = torch.randn(B, N, D_enc)
    x_ref = torch.randn(B, T, 4)

    out1 = policy(scene, x_ref, deterministic=True)
    out2 = policy(scene, x_ref, deterministic=True)

    assert torch.allclose(out1.eta_lat, out2.eta_lat), "Deterministic eta_lat differs"
    assert torch.allclose(out1.eta_lon, out2.eta_lon), "Deterministic eta_lon differs"
    assert torch.allclose(out1.value, out2.value), "Deterministic value differs"
    print("  [PASS] Deterministic mode: outputs identical")


def test_gradient_flow():
    """Gradients should flow through rsample() to policy parameters."""
    B, N, D_enc, T = 2, 50, 256, 80
    config = ExplorationPolicyConfig(
        hidden_dim=64,
        n_mixer_layers=1,
        n_attn_heads=4,
        dropout=0.0,
        encoder_hidden_dim=D_enc,
    )
    policy = ExplorationPolicy(config, ref_seq_len=T)

    scene = torch.randn(B, N, D_enc)
    x_ref = torch.randn(B, T, 4)

    output = policy(scene, x_ref, deterministic=False)

    # Use a simple loss combining eta and value
    loss = output.eta_lat.sum() + output.eta_lon.sum() + output.value.sum()
    loss.backward()

    has_grad = False
    for name, param in policy.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowing to policy parameters"
    print("  [PASS] Gradient flow: gradients reach policy parameters via rsample()")


def test_zero_initialization():
    """GuidanceHead should output eta with mean=0 at initialization (PlannerRFT 4.5).

    With zero-initialized last layer: raw=[0,0,0,0] -> softplus(0)+1 = ln(2)+1 ≈ 1.693
    -> alpha=beta -> Beta mean=0.5 in (0,1) -> eta mean=0.0 in (-1,1).
    """
    B, N, D_enc, T = 1, 50, 256, 80
    config = ExplorationPolicyConfig(
        hidden_dim=128,
        n_mixer_layers=2,
        n_attn_heads=4,
        dropout=0.0,
        encoder_hidden_dim=D_enc,
    )
    policy = ExplorationPolicy(config, ref_seq_len=T)
    policy.eval()

    # Check that the guidance head output layer is zero-initialized
    assert (policy.guidance_head.fc2.weight == 0).all(), "GuidanceHead output weight not zero"
    assert policy.guidance_head.fc2.bias is None, "GuidanceHead fc2 is bias=False by design"

    # Check that value head output layer is zero-initialized
    assert (policy.value_head.fc2.weight == 0).all(), "ValueHead output weight not zero"
    assert (policy.value_head.fc2.bias == 0).all(), "ValueHead output bias not zero"

    # With zero input to the output layer, raw = [0,0,0,0]
    # softplus(0) + 1.0 = ln(2) + 1 ≈ 1.6931
    import math

    expected_param = math.log(2) + 1.0

    # Feed zeros through just the guidance head to verify
    head = policy.guidance_head
    zero_fused = torch.zeros(B, 128)
    lat_dist, lon_dist = head(zero_fused)

    # alpha and beta should both equal softplus(0)+1 = ln(2)+1
    assert torch.allclose(lat_dist.concentration1, torch.tensor(expected_param), atol=1e-4), (
        f"alpha_lat={lat_dist.concentration1.item():.4f}, expected {expected_param:.4f}"
    )
    assert torch.allclose(lat_dist.concentration0, torch.tensor(expected_param), atol=1e-4), (
        f"beta_lat={lat_dist.concentration0.item():.4f}, expected {expected_param:.4f}"
    )

    # Mean of Beta(a,a) = 0.5, mapped to eta = 2*0.5 - 1 = 0.0
    eta_mean_01 = lat_dist.mean.item()
    eta_mean = 2.0 * eta_mean_01 - 1.0
    assert abs(eta_mean) < 1e-6, f"eta mean at init = {eta_mean:.6f}, expected 0.0"

    # Variance: Var(Beta(a,a)) = a²/((2a)²(2a+1)) = 1/(4(2a+1))
    # In (-1,1): Var_eta = 4 * Var_01
    var_01 = lat_dist.variance.item()
    var_eta = 4.0 * var_01
    std_eta = var_eta**0.5
    assert 0.3 < std_eta < 0.6, f"eta std at init = {std_eta:.3f}, expected ~0.48"

    print(f"  [PASS] Zero-initialization: eta_mean={eta_mean:.6f}, eta_std={std_eta:.3f}")
    print(f"         Beta params: alpha=beta={expected_param:.4f}")


def test_config_serialization():
    """Config should round-trip through JSON."""
    config = ExplorationPolicyConfig(
        hidden_dim=256,
        n_mixer_layers=4,
        n_attn_heads=8,
        dropout=0.2,
        learning_rate=3e-4,
        encoder_hidden_dim=512,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        loaded = ExplorationPolicyConfig.from_json(f.name)

    assert loaded.hidden_dim == 256
    assert loaded.n_mixer_layers == 4
    assert loaded.n_attn_heads == 8
    assert loaded.dropout == 0.2
    assert loaded.learning_rate == 3e-4
    assert loaded.encoder_hidden_dim == 512
    print("  [PASS] Config serialization: round-trip OK")


def test_parameter_count():
    """Verify parameter count is in expected range for small config."""
    config = ExplorationPolicyConfig(
        hidden_dim=128,
        n_mixer_layers=2,
        n_attn_heads=4,
        encoder_hidden_dim=256,
    )
    policy = ExplorationPolicy(config, ref_seq_len=80)

    n_params = sum(p.numel() for p in policy.parameters())
    # Small config should be roughly 200K-500K params
    assert 100_000 < n_params < 1_000_000, f"Unexpected param count: {n_params:,}"
    print(f"  [PASS] Parameter count: {n_params:,} (in expected range)")


def test_grpo_config_exploration_fields():
    """Verify GRPOConfig has exploration policy fields."""
    from rlvr.grpo_config import GRPOConfig

    config = GRPOConfig()
    assert hasattr(config, "use_exploration_policy")
    assert config.use_exploration_policy is False
    assert config.exploration_hidden_dim == 128
    assert config.exploration_n_mixer_layers == 2
    assert config.exploration_n_attn_heads == 4
    assert config.exploration_dropout == 0.1
    assert config.exploration_lr == 1e-4
    assert config.exploration_checkpoint_path is None

    # Test JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.use_exploration_policy = True
        config.exploration_hidden_dim = 256
        config.to_json(f.name)
        loaded = GRPOConfig.from_json(f.name)

    assert loaded.use_exploration_policy is True
    assert loaded.exploration_hidden_dim == 256
    print("  [PASS] GRPOConfig exploration fields: present and serializable")


if __name__ == "__main__":
    print("Running Exploration Policy unit tests...\n")

    test_ref_trajectory_mixer()
    test_ref_fusion_attention()
    test_guidance_head()
    test_value_head()
    test_exploration_policy_forward()
    test_eta_range()
    test_deterministic_mode()
    test_gradient_flow()
    test_zero_initialization()
    test_config_serialization()
    test_parameter_count()
    test_grpo_config_exploration_fields()

    print("\nAll tests passed!")


# ---------------------------------------------------------------------------
# Multi-head (config-driven heads) tests
# ---------------------------------------------------------------------------


def test_multi_head_config():
    """Three heads -> three Beta dists, etas dict keyed by head name."""
    config = ExplorationPolicyConfig(
        hidden_dim=64,
        n_attn_heads=2,
        encoder_hidden_dim=128,
        heads=["lateral", "collision", "stretch"],
    )
    policy = ExplorationPolicy(config, ref_seq_len=20)
    policy.eval()
    out = policy(torch.randn(2, 10, 128), torch.randn(2, 20, 4))
    assert set(out.etas.keys()) == {"lateral", "collision", "stretch"}
    assert policy.guidance_head.fc2.weight.shape[0] == 6
    for eta in out.etas.values():
        assert eta.shape == (2,)
        assert (eta >= -1).all() and (eta <= 1).all()
    # lateral accessor works; longitudinal must fail loudly (not configured)
    _ = out.eta_lat
    try:
        _ = out.eta_lon
        raise AssertionError("eta_lon should raise for a policy without that head")
    except KeyError:
        pass


def test_default_heads_checkpoint_backward_compat(tmp_path):
    """A default-config (2-head) state dict round-trips into the new layout."""
    config = ExplorationPolicyConfig(hidden_dim=64, n_attn_heads=2, encoder_hidden_dim=128)
    policy = ExplorationPolicy(config, ref_seq_len=20)
    path = tmp_path / "p.pth"
    torch.save(policy.state_dict(), path)
    fresh = ExplorationPolicy(
        ExplorationPolicyConfig(hidden_dim=64, n_attn_heads=2, encoder_hidden_dim=128),
        ref_seq_len=20,
    )
    fresh.load_state_dict(torch.load(path), strict=True)
    out = fresh(torch.randn(1, 5, 128), torch.randn(1, 20, 4), deterministic=True)
    assert abs(out.eta_lat.item()) < 1e-5  # zero-init -> inert mean
    assert abs(out.eta_lon.item()) < 1e-5


def test_multi_head_zero_init_inert():
    """Every head's deterministic action is exactly 0 at zero-init."""
    config = ExplorationPolicyConfig(
        hidden_dim=64,
        n_attn_heads=2,
        encoder_hidden_dim=128,
        heads=["lateral", "collision"],
    )
    policy = ExplorationPolicy(config, ref_seq_len=20)
    policy.eval()
    out = policy(torch.randn(4, 8, 128), torch.randn(4, 20, 4), deterministic=True)
    for name, eta in out.etas.items():
        assert torch.all(eta.abs() < 1e-5), f"head {name} not inert at init"


def test_loss_multi_head_dists():
    """compute_exploration_loss accepts a dists dict + action cost."""
    from torch.distributions import Beta

    from exploration_policy.loss import compute_exploration_loss

    dists = {
        "lateral": Beta(torch.tensor([2.0]), torch.tensor([1.5])),
        "collision": Beta(torch.tensor([1.7]), torch.tensor([1.7])),
    }
    adv = torch.tensor([1.0, -1.0, 0.5])
    lp = torch.tensor([-1.0, -2.0, -1.5], requires_grad=True)
    loss, metrics = compute_exploration_loss(
        advantages=adv,
        log_probs=lp,
        dists=dists,
        action_cost_coef=0.01,
    )
    assert loss.dim() == 0
    assert "exploration_eta_collision_mean" in metrics
    assert "exploration_eta_lat_mean" in metrics  # legacy alias
    assert metrics["exploration_action_cost"] > 0


# ---------------------------------------------------------------------------
# Strength gate (use_strength_head) + configurable Beta-concentration cap
# ---------------------------------------------------------------------------


def test_strength_head_disabled_by_default():
    """Default config has no strength head; forward emits strength=None."""
    config = ExplorationPolicyConfig(hidden_dim=64, n_attn_heads=2, encoder_hidden_dim=128)
    assert config.use_strength_head is False
    assert config.head_max_conc == 10.0
    policy = ExplorationPolicy(config, ref_seq_len=20)
    assert policy.strength_head is None
    out = policy(torch.randn(3, 8, 128), torch.randn(3, 20, 4), deterministic=True)
    assert out.strength is None


def test_strength_head_enabled_shape_and_range():
    """With the head on, strength is [B] in (0,1)."""
    config = ExplorationPolicyConfig(
        hidden_dim=64, n_attn_heads=2, encoder_hidden_dim=128, use_strength_head=True
    )
    policy = ExplorationPolicy(config, ref_seq_len=20)
    assert policy.strength_head is not None
    out = policy(torch.randn(5, 8, 128), torch.randn(5, 20, 4), deterministic=True)
    assert out.strength is not None
    assert out.strength.shape == (5,)
    assert (out.strength > 0).all() and (out.strength < 1).all()
    # zero-init output layer -> sigmoid(0) = 0.5 at init
    assert torch.allclose(out.strength, torch.full((5,), 0.5), atol=1e-5)


def test_strength_head_config_roundtrip(tmp_path):
    """use_strength_head + head_max_conc survive JSON round-trip and rebuild."""
    config = ExplorationPolicyConfig(
        hidden_dim=64,
        n_attn_heads=2,
        encoder_hidden_dim=128,
        use_strength_head=True,
        head_max_conc=50.0,
    )
    p = tmp_path / "cfg.json"
    config.to_json(p)
    loaded = ExplorationPolicyConfig.from_json(p)
    assert loaded.use_strength_head is True
    assert loaded.head_max_conc == 50.0
    policy = ExplorationPolicy(loaded, ref_seq_len=20)
    assert policy.strength_head is not None
    assert policy.guidance_head.max_conc == 50.0


def test_head_max_conc_raises_reachable_eta():
    """A higher concentration cap lets the deterministic |eta| exceed the
    default-cap ceiling (~0.82) when the head is pushed off zero-init."""
    from exploration_policy.module.heads import GuidanceHead

    torch.manual_seed(0)
    fused = torch.randn(64, 32)
    low = GuidanceHead(
        32, n_heads=2, init_mode="normal", init_std=1.0, raw_scale=10.0, max_conc=10.0
    )
    high = GuidanceHead(
        32, n_heads=2, init_mode="normal", init_std=1.0, raw_scale=10.0, max_conc=100.0
    )
    # copy weights so only the cap differs
    high.load_state_dict(low.state_dict())
    eta_low = (2.0 * low(fused)[0].mean - 1.0).abs().max().item()
    eta_high = (2.0 * high(fused)[0].mean - 1.0).abs().max().item()
    assert eta_low <= 0.83  # default cap ceiling
    assert eta_high > eta_low  # raising the cap extends reachable eta
