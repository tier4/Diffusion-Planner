"""Guidance-envelope persistence: the policy config carries the calibration its
etas are bound to, and make_composer resolves from it (not divergent per-tool
CLI defaults). Guards the v23 "non-reproduction" footgun."""

import ast
import json
from argparse import Namespace
from pathlib import Path

import pytest

from exploration_policy.model import V1_GUIDANCE_ENVELOPE, ExplorationPolicyConfig

# Guided eval/deploy tools that score or bake a policy and so MUST resolve the
# envelope from the policy's persisted calibration (CLI = override-only). Their
# envelope argparse defaults must be None — a non-None default silently shadows
# the persisted value and reintroduces the cross-tool drift bug.
_CERT_DEPLOY_TOOLS = [
    "rlvr/autoresearch/tools/eval_policy_avoidance.py",
    "rlvr/autoresearch/tools/eval_closedloop_avoidance.py",
    "rlvr/autoresearch/tools/valid_predictor_guided.py",
    "rlvr/autoresearch/tools/eval_policy_l2.py",
    "rlvr/autoresearch/tools/distill_guided_targets.py",
    "rlvr/autoresearch/tools/rollforward_avoidance_scenes.py",
]
_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rel", _CERT_DEPLOY_TOOLS)
def test_cert_deploy_tools_envelope_args_default_none(rel):
    """No envelope knob may carry a non-None argparse default in these tools."""
    tree = ast.parse((_REPO_ROOT / rel).read_text())
    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            name = str(node.args[0].value).lstrip("-")
            if name not in V1_GUIDANCE_ENVELOPE:
                continue
            default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
            # missing default kw == argparse None; only a non-None Constant is a bug.
            if isinstance(default, ast.Constant) and default.value is not None:
                offenders.append((name, default.value))
    assert not offenders, f"{rel}: envelope args with non-None default: {offenders}"


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
    # Use a NON-v1 lambda_lat so the assertion distinguishes "read the persisted
    # envelope" from "fell through to the v1 constant" (v1 lambda_lat is 5.0).
    env = dict(V1_GUIDANCE_ENVELOPE, lambda_lat=7.0)
    comp = make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=env)
    # the lateral head carries lambda_lat from the persisted envelope
    lat = next(f for f in comp._functions if hasattr(f, "_lambda_lat"))
    assert abs(float(lat._lambda_lat) - 7.0) < 1e-9


def _envelope_args(**overrides):
    base = dict(
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
        force_envelope_override=False,
    )
    base.update(overrides)
    return Namespace(**base)


def test_disagreeing_cli_without_force_hard_fails():
    # The core guard: a stale flag that disagrees with the persisted envelope
    # must HARD-FAIL (not silently mis-scale) unless deliberately forced.
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    args = _envelope_args(lambda_lat=4.0)  # old weak value vs persisted 7.0
    env = dict(V1_GUIDANCE_ENVELOPE, lambda_lat=7.0)
    with pytest.raises(RuntimeError, match="disagree with the policy's persisted"):
        make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=env)


def test_disagreeing_cli_with_force_overrides_and_warns(capsys):
    # --force_envelope_override downgrades the hard-fail to a loud warning and
    # lets the explicit value win (legitimate sweeps / v2 experiments).
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    args = _envelope_args(lambda_lat=3.3, force_envelope_override=True)
    env = dict(V1_GUIDANCE_ENVELOPE, lambda_lat=5.0)
    comp = make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=env)
    lat = next(f for f in comp._functions if hasattr(f, "_lambda_lat"))
    assert abs(float(lat._lambda_lat) - 3.3) < 1e-9  # forced override wins
    assert "OVERRIDE" in capsys.readouterr().err


def test_cli_matching_persisted_does_not_fail_or_warn(capsys):
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    args = _envelope_args(lambda_lat=7.0)  # explicit but AGREES -> no-op
    env = dict(V1_GUIDANCE_ENVELOPE, lambda_lat=7.0)
    comp = make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=env)
    lat = next(f for f in comp._functions if hasattr(f, "_lambda_lat"))
    assert abs(float(lat._lambda_lat) - 7.0) < 1e-9
    assert "OVERRIDE" not in capsys.readouterr().err


def test_make_composer_falls_back_to_v1_when_nothing_provided():
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    # no envelope dict, no args knobs -> the canonical v1 constant (the single
    # source of truth), never a divergent weak default.
    args = Namespace(head_protect=0, slow_composer=False)
    comp = make_composer({"lateral": 0.5, "collision": 0.5}, args, envelope=None)
    lat = next(f for f in comp._functions if hasattr(f, "_lambda_lat"))
    assert abs(float(lat._lambda_lat) - V1_GUIDANCE_ENVELOPE["lambda_lat"]) < 1e-9
