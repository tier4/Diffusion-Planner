"""Tests for WandbLogger no-op and auto-disable behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rlvr.grpo_config import GRPOConfig
from rlvr.wandb_logger import WandbLogger


def test_disabled_by_default():
    """Logger is a no-op when wandb_enabled=False."""
    config = GRPOConfig(wandb_enabled=False)
    logger = WandbLogger.from_config(config)
    assert not logger._enabled
    assert logger._run is None
    # All calls are no-ops
    logger.log_training(1, {"loss": 0.5})
    logger.log_eval(1, prob_result={"reward_mean": 10.0})
    logger.log_rank_analytics(1, {"summary": {"win_rates": {"a": 0.5}}})
    logger.finish({"best_epoch": 1})


def test_disabled_when_wandb_import_fails():
    """Logger auto-disables when wandb is not installed."""
    config = GRPOConfig(wandb_enabled=True)
    with patch.dict("sys.modules", {"wandb": None}):
        logger = WandbLogger.from_config(config)
    assert not logger._enabled


def test_disabled_when_wandb_init_fails():
    """Logger auto-disables when wandb.init() raises."""
    config = GRPOConfig(wandb_enabled=True)
    mock_wandb = MagicMock()
    mock_wandb.init.side_effect = RuntimeError("network error")
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        logger = WandbLogger.from_config(config)
    assert not logger._enabled
    assert logger._run is None


def test_safe_log_disables_on_failure():
    """_safe_log catches errors and disables logging."""
    logger = WandbLogger(enabled=True)
    logger._run = MagicMock()
    mock_wandb = MagicMock()
    mock_wandb.log.side_effect = RuntimeError("upload failed")
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        logger.log_training(1, {"loss": 0.5})
    assert not logger._enabled


def test_finish_clears_run_and_disables():
    """finish() sets _run=None and _enabled=False."""
    logger = WandbLogger(enabled=True)
    logger._run = MagicMock()
    mock_wandb = MagicMock()
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        logger.finish({"best_epoch": 3})
    assert logger._run is None
    assert not logger._enabled
    mock_wandb.finish.assert_called_once()


def test_finish_noop_when_no_run():
    """finish() is a no-op when no run was ever started."""
    logger = WandbLogger(enabled=False)
    logger.finish({"best_epoch": 1})  # should not raise
    assert not logger._enabled


def test_finish_best_effort_after_disable():
    """finish() still closes the run even if _enabled was set to False by _safe_log."""
    logger = WandbLogger(enabled=False)
    logger._run = MagicMock()  # run was started but logging got disabled
    mock_wandb = MagicMock()
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        logger.finish({"best_epoch": 5})
    assert logger._run is None
    assert not logger._enabled
    mock_wandb.finish.assert_called_once()


def test_finish_handles_wandb_finish_error():
    """finish() doesn't crash if wandb.finish() itself raises."""
    logger = WandbLogger(enabled=True)
    logger._run = MagicMock()
    mock_wandb = MagicMock()
    mock_wandb.finish.side_effect = RuntimeError("cleanup error")
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        logger.finish()  # should not raise
    assert logger._run is None
    assert not logger._enabled


def test_tags_from_config():
    """Auto-tags are set based on config fields."""
    config = GRPOConfig(wandb_enabled=True, ranked_sft_mode="gt_neighbor")
    mock_wandb = MagicMock()
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        WandbLogger.from_config(config, extra_tags=["test"])
    call_kwargs = mock_wandb.init.call_args[1]
    assert "ranked_sft" in call_kwargs["tags"]
    assert "test" in call_kwargs["tags"]
