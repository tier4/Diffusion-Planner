"""Guidance-envelope persistence: the policy config carries the calibration its
etas are bound to, and make_composer resolves from it (not divergent per-tool
CLI defaults). Guards the v23 "non-reproduction" footgun."""

import json
from argparse import Namespace

from exploration_policy.model import V1_GUIDANCE_ENVELOPE, ExplorationPolicyConfig


def test_config_default_envelope_is_v1():
    cfg = ExplorationPolicyConfig()
    assert cfg.guidance_envelope == V1_GUIDANCE_ENVELOPE
    # a fresh dict, not the shared module constant (mutation isolation)
    assert cfg.guidance_envelope is not V1_GUIDANCE_ENVELOPE


def test_config_roundtrip_preserves_envelope(tmp_path):
    env = dict(V1_GUIDANCE_ENVELOPE, lambda_lat=7.0, col_scale=12.0)
    cfg = ExplorationPolicyConfig(heads=["lateral", "collision"], guidance_envelope=env)
    p = tmp_path / "exploration_policy_config.json"
    cfg.to_json(p)
    assert json.load(open(p))["guidance_envelope"]["lambda_lat"] == 7.0
    back = ExplorationPolicyConfig.from_json(p)
    assert back.guidance_envelope == env


def test_old_config_without_key_loads_as_v1(tmp_path):
    # Pre-2026-06-14 configs have no guidance_envelope key; they were trained at
    # v1, so the default must backfill to v1 (not crash, not a weak default).
    p = tmp_path / "exploration_policy_config.json"
    json.dump({"hidden_dim": 128, "heads": ["lateral", "collision"]}, open(p, "w"))
    cfg = ExplorationPolicyConfig.from_json(p)
    assert cfg.guidance_envelope == V1_GUIDANCE_ENVELOPE


def test_make_composer_prefers_persisted_envelope_over_missing_arg():
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    # args leave every envelope knob unset (None) -> the policy's persisted
    # envelope must be used, NOT a weak per-tool default.
    args = Namespace(
        lambda_lat=None,
        lat_scale=None,
        col_scale=None,
        col_range=None,
        lambda_spd=None,
        stretch_scale=None,
        guidance_scale=None,
        envelope=None,
        lambda_col=None,
        head_protect=0,
        slow_composer=False,
    )
    env = dict(V1_GUIDANCE_ENVELOPE, lambda_lat=5.0, lat_scale=2.0, col_scale=9.0)
    comp = make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=env)
    # the lateral head carries lambda_lat * scale from the persisted envelope
    lat = next(f for f in comp._functions if hasattr(f, "_lambda_lat"))
    assert abs(float(lat._lambda_lat) - 5.0) < 1e-9


def test_make_composer_explicit_arg_overrides_envelope():
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    args = Namespace(
        lambda_lat=3.3,
        lat_scale=None,
        col_scale=None,
        col_range=None,
        lambda_spd=None,
        stretch_scale=None,
        guidance_scale=None,
        envelope=None,
        lambda_col=None,
        head_protect=0,
        slow_composer=False,
    )
    env = dict(V1_GUIDANCE_ENVELOPE, lambda_lat=5.0)
    comp = make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=env)
    lat = next(f for f in comp._functions if hasattr(f, "_lambda_lat"))
    assert abs(float(lat._lambda_lat) - 3.3) < 1e-9  # CLI override wins


def test_make_composer_falls_back_to_v1_when_nothing_provided():
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    # no envelope dict, no args knobs -> the canonical v1 constant (the single
    # source of truth), never a divergent weak default.
    args = Namespace(head_protect=0, slow_composer=False)
    comp = make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=None)
    lat = next(f for f in comp._functions if hasattr(f, "_lambda_lat"))
    assert abs(float(lat._lambda_lat) - V1_GUIDANCE_ENVELOPE["lambda_lat"]) < 1e-9
