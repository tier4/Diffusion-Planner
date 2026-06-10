"""Unit tests for GRPOExplorationTrainer frozen-DiT / policy-only fixes.

Covers:
- reward_config_from_grpo propagates every shared field (incl. the sc_*
  static-collision family the old hand-copied list dropped)
- train_dit=False config validation + trainer guard
- frozen-DiT training: DiT params untouched, explorer params update
- pinned zero-eta slot 0 excluded from the log-prob policy gradient
- policy-only checkpoint round-trip
- aggregate_policy_eval metric aggregation

All tests are CPU-only and use a stub DiT (the real model is never invoked
because train_dit=False skips the DiT loss and tests call train_on_groups
with pre-built groups).
"""

from dataclasses import fields as dc_fields

import numpy as np
import pytest
import torch
from torch import nn

from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_exploration_trainer import (
    GRPOExplorationTrainer,
    aggregate_policy_eval,
    reward_config_from_grpo,
)
from rlvr.reward import RewardConfig

DEVICE = torch.device("cpu")
HIDDEN = 16
ENC_DIM = 32
FUTURE_LEN = 8
K = 4


class _StubArgs:
    hidden_dim = ENC_DIM
    future_len = FUTURE_LEN


class _StubDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


def _make_config(**overrides) -> GRPOConfig:
    base = dict(
        use_exploration_policy=True,
        train_dit=False,
        use_lora=False,
        random_guidance_mode="explorer",
        num_generations=K,
        noise_scale_range=[0.0, 0.0],
        exploration_hidden_dim=HIDDEN,
        exploration_n_attn_heads=2,
        exploration_lr=1e-2,
    )
    base.update(overrides)
    return GRPOConfig(**base)


def _make_trainer(tmp_path, config=None, dit_optimizer=None):
    config = config or _make_config()
    return GRPOExplorationTrainer(
        policy_model=_StubDiT(),
        model_args=_StubArgs(),
        dit_optimizer=dit_optimizer,
        device=DEVICE,
        run_dir=tmp_path,
        config=config,
        use_lora=False,
    )


def _make_group(advantages, eta_lat_01=None, eta_lon_01=None):
    eta = torch.tensor([0.5, 0.3, 0.7, 0.6])
    return {
        "npz_path": "fake.npz",
        "data": {},
        "norm_data": {},
        "trajectories": [np.zeros((FUTURE_LEN, 4), dtype=np.float32) for _ in range(K)],
        "reward_breakdowns": None,
        "advantages": np.asarray(advantages, dtype=np.float32),
        "policy_meta": None,
        "det_trajectory": np.zeros((FUTURE_LEN, 4), dtype=np.float32),
        "scene_encoding": torch.randn(1, 5, ENC_DIM),
        "x_ref": torch.randn(1, FUTURE_LEN, 4),
        "eta_lat_01": eta_lat_01 if eta_lat_01 is not None else eta.clone(),
        "eta_lon_01": eta_lon_01 if eta_lon_01 is not None else eta.clone(),
    }


# ---------------------------------------------------------------------------
# 0a. reward config field propagation
# ---------------------------------------------------------------------------

def test_reward_config_from_grpo_propagates_all_shared_fields():
    cfg = _make_config(
        static_collision_enabled=True,
        sc_gate_enabled=True,
        sc_penalty_mode="survival",
        sc_near_scale=15.0,
        sc_wide_scale=5.0,
        sc_cont_scale=1.0,
        sc_cross_thresh=0.25,
        sc_near_thresh=0.45,
        sc_wide_thresh=0.75,
        sc_cont_thresh=1.05,
        sc_neighbor_vel_thresh=0.15,
        sc_neighbor_disp_thresh=0.55,
        sc_ego_min_speed=1.1,
        progress_norm_scale=33.0,
        w_centerline=3.0,
        stopped_penalty=100.0,
    )
    rc = reward_config_from_grpo(cfg)

    shared = {f.name for f in dc_fields(RewardConfig)} & {f.name for f in dc_fields(GRPOConfig)}
    assert len(shared) >= 50, f"expected ~52 shared fields, got {len(shared)}"
    for name in sorted(shared):
        assert getattr(rc, name) == getattr(cfg, name), f"field {name} not propagated"

    # The fields the old hand-copied list silently dropped:
    assert rc.static_collision_enabled is True
    assert rc.sc_gate_enabled is True
    assert rc.sc_near_scale == 15.0
    assert rc.sc_cross_thresh == 0.25
    assert rc.progress_norm_scale == 33.0


def test_trainer_uses_full_reward_config(tmp_path):
    cfg = _make_config(static_collision_enabled=True, sc_gate_enabled=True, sc_near_scale=15.0)
    trainer = _make_trainer(tmp_path, cfg)
    assert trainer.reward_config.static_collision_enabled is True
    assert trainer.reward_config.sc_gate_enabled is True
    assert trainer.reward_config.sc_near_scale == 15.0


# ---------------------------------------------------------------------------
# 0b. train_dit flag
# ---------------------------------------------------------------------------

def test_train_dit_false_requires_exploration_policy():
    with pytest.raises(ValueError, match="train_dit=False"):
        _make_config(use_exploration_policy=False)


def test_train_dit_false_rejects_lora():
    with pytest.raises(ValueError, match="use_lora"):
        _make_config(use_lora=True)


def test_train_dit_false_rejects_non_explorer_mode():
    with pytest.raises(ValueError, match="random_guidance_mode"):
        _make_config(random_guidance_mode="uniform")


def test_trainer_requires_dit_optimizer_when_training_dit(tmp_path):
    cfg = _make_config(train_dit=True)
    with pytest.raises(ValueError, match="dit_optimizer"):
        _make_trainer(tmp_path, cfg, dit_optimizer=None)


def test_frozen_dit_unchanged_policy_updates(tmp_path):
    trainer = _make_trainer(tmp_path)
    dit_before = {k: v.clone() for k, v in trainer.policy_model.state_dict().items()}
    policy_before = {k: v.clone() for k, v in trainer.exploration_policy.state_dict().items()}

    group = _make_group([0.0, 1.0, -1.0, 0.5])
    metrics = trainer.train_on_groups([group], epoch=1)

    for k, v in trainer.policy_model.state_dict().items():
        assert torch.equal(v, dit_before[k]), f"frozen DiT param {k} changed"
    changed = any(
        not torch.equal(v, policy_before[k])
        for k, v in trainer.exploration_policy.state_dict().items()
    )
    assert changed, "exploration policy params did not update"
    assert "exploration_total_loss" in metrics


# ---------------------------------------------------------------------------
# 0d. pinned zero-eta sample excluded from policy gradient
# ---------------------------------------------------------------------------

def _capture_loss_args(monkeypatch, calls):
    import rlvr.grpo_exploration_trainer as get_mod

    def fake_loss(advantages, log_probs, lat_dist, lon_dist, entropy_coef, kl_coef):
        calls["advantages"] = advantages.detach().clone()
        calls["n_log_probs"] = log_probs.shape[0]
        loss = log_probs.sum() * 0.0 + lat_dist.mean.sum() * 0.0
        return loss, {"exploration_total_loss": 0.0}

    monkeypatch.setattr(get_mod, "compute_exploration_loss", fake_loss)


def test_pinned_slot_excluded_from_policy_gradient(tmp_path, monkeypatch):
    calls = {}
    _capture_loss_args(monkeypatch, calls)
    trainer = _make_trainer(tmp_path, _make_config(exploration_pin_zero_eta=True))
    advantages = [9.9, 1.0, -1.0, 0.5]  # huge pinned-slot advantage must be ignored
    trainer.train_on_groups([_make_group(advantages)], epoch=1)

    assert calls["n_log_probs"] == K - 1
    assert torch.allclose(
        calls["advantages"],
        torch.tensor(advantages[1:], dtype=torch.float32),
    )


def test_no_pin_uses_all_samples(tmp_path, monkeypatch):
    calls = {}
    _capture_loss_args(monkeypatch, calls)
    trainer = _make_trainer(tmp_path, _make_config(exploration_pin_zero_eta=False))
    trainer.train_on_groups([_make_group([0.0, 1.0, -1.0, 0.5])], epoch=1)
    assert calls["n_log_probs"] == K


# ---------------------------------------------------------------------------
# 0e. policy-only checkpointing
# ---------------------------------------------------------------------------

def test_policy_only_checkpoint_roundtrip(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.save_checkpoint(epoch=3, args_dict={})

    epoch_path = tmp_path / "exploration_policy_epoch_003.pth"
    assert epoch_path.exists()
    assert (tmp_path / "exploration_policy.pth").exists()
    assert (tmp_path / "policy_optimizer.pth").exists()
    assert (tmp_path / "exploration_policy_config.json").exists()
    assert (tmp_path / "grpo_config.json").exists()
    # Frozen DiT must NOT be serialized
    assert not (tmp_path / "latest.pth").exists()

    state = torch.load(epoch_path, map_location=DEVICE)
    fresh = ExplorationPolicy(
        ExplorationPolicyConfig(
            hidden_dim=HIDDEN, n_attn_heads=2, encoder_hidden_dim=ENC_DIM,
        ),
        ref_seq_len=FUTURE_LEN,
    )
    fresh.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# 0c. eval aggregation helper
# ---------------------------------------------------------------------------

def _row(sc_n_stopped, eta_lat, static_crossing=False, det_static_crossing=True,
         deviation=0.1, sc_min_dist=0.5):
    return {
        "npz_path": "x.npz", "eta_lat": eta_lat, "eta_lon": 0.0,
        "reward": 1.0, "det_reward": 0.0,
        "static_crossing": static_crossing, "det_static_crossing": det_static_crossing,
        "sc_min_dist": sc_min_dist, "det_sc_min_dist": 0.1,
        "rb_crossing": False, "collision": False,
        "sc_n_stopped": sc_n_stopped, "deviation": deviation,
    }


def test_aggregate_policy_eval_splits_scene_types():
    rows = [
        _row(sc_n_stopped=2, eta_lat=0.8),                          # avoidance
        _row(sc_n_stopped=1, eta_lat=-0.6, static_crossing=True),   # avoidance
        _row(sc_n_stopped=0, eta_lat=0.05, det_static_crossing=False),  # normal
        _row(sc_n_stopped=0, eta_lat=-0.01, det_static_crossing=False, deviation=0.02),
    ]
    out = aggregate_policy_eval(rows)
    assert out["n_scenes"] == 4
    assert out["n_avoidance_scenes"] == 2
    assert out["static_crossings"] == 1
    assert out["det_static_crossings"] == 2
    assert out["eta_lat_abs_avoid"] == pytest.approx(0.7)
    assert out["eta_lat_abs_normal"] == pytest.approx(0.03)
    assert out["rb_crossings"] == 0
    assert out["collision_rate"] == 0.0


def test_aggregate_policy_eval_empty():
    out = aggregate_policy_eval([])
    assert out["reward_mean"] == float("-inf")
    assert out["n_scenes"] == 0
