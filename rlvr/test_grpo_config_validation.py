"""Validation tests for GRPOConfig string-enum fields."""
from __future__ import annotations

import pytest

from rlvr.grpo_config import GRPOConfig


def test_ego_il_mode_valid():
    for v in ("gt", "baseline"):
        cfg = GRPOConfig(ego_il_mode=v)
        assert cfg.ego_il_mode == v


def test_ego_il_mode_invalid():
    with pytest.raises(ValueError, match="ego_il_mode"):
        GRPOConfig(ego_il_mode="gtbaseline")


def test_selective_mode_valid():
    for v in ("threshold", "advantage"):
        cfg = GRPOConfig(selective_mode=v)
        assert cfg.selective_mode == v


def test_selective_mode_invalid():
    with pytest.raises(ValueError, match="selective_mode"):
        GRPOConfig(selective_mode="AdvantagE")


def test_gt_fallback_mode_valid():
    for v in ("none", "skip", "il"):
        cfg = GRPOConfig(gt_fallback_mode=v)
        assert cfg.gt_fallback_mode == v


@pytest.mark.parametrize("bad", ["Skip", "IL", "ground_truth", "", "any"])
def test_gt_fallback_mode_invalid(bad):
    with pytest.raises(ValueError, match="gt_fallback_mode"):
        GRPOConfig(gt_fallback_mode=bad)


def test_gt_fallback_mode_default_none():
    cfg = GRPOConfig()
    assert cfg.gt_fallback_mode == "none"
    assert cfg.gt_fallback_margin == 0.0
