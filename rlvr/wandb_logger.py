"""Lightweight wandb integration for RLVR training.

Provides a WandbLogger class that wraps wandb.init / wandb.log / wandb.finish
with graceful no-op behavior when wandb is disabled or unavailable.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from rlvr.grpo_config import GRPOConfig


class WandbLogger:
    """Thin wrapper around wandb with no-op fallback."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._run = None

    @classmethod
    def from_config(
        cls,
        config: GRPOConfig,
        run_dir: str | None = None,
        run_name: str | None = None,
        extra_tags: list[str] | None = None,
        extra_config: dict | None = None,
    ) -> WandbLogger:
        """Create logger from GRPOConfig. Returns no-op logger if disabled."""
        instance = cls(enabled=config.wandb_enabled)
        if not instance._enabled:
            return instance

        try:
            import wandb
        except ImportError:
            print("[wandb] wandb not installed, disabling logging")
            instance._enabled = False
            return instance

        tags = []
        if config.ranked_sft_mode != "none":
            tags.append("ranked_sft")
        if config.use_exploration_policy:
            tags.append("exploration")
        if config.use_closed_loop:
            tags.append("closed_loop")
        if extra_tags:
            tags.extend(extra_tags)

        wandb_config = asdict(config)
        if extra_config:
            wandb_config.update(extra_config)

        try:
            instance._run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity or os.environ.get("WANDB_ENTITY") or None,
                name=run_name,
                dir=run_dir,
                config=wandb_config,
                tags=tags,
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}, disabling logging")
            instance._enabled = False

        return instance

    def _safe_log(self, payload: dict[str, Any], step: int) -> None:
        """Log to wandb, disabling on failure so training continues."""
        try:
            import wandb

            wandb.log(payload, step=step)
        except Exception as e:
            print(f"[wandb] log failed: {e}, disabling logging")
            self._enabled = False

    def log_training(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log per-epoch training metrics."""
        if not self._enabled:
            return
        self._safe_log({f"train/{k}": v for k, v in metrics.items()}, step=epoch)

    def log_eval(
        self,
        epoch: int,
        prob_result: dict[str, Any] | None = None,
        val_result: dict[str, Any] | None = None,
    ) -> None:
        """Log per-epoch evaluation results."""
        if not self._enabled:
            return
        payload: dict[str, Any] = {}
        if prob_result:
            for k, v in prob_result.items():
                if isinstance(v, (int, float)):
                    payload[f"eval_prob/{k}"] = v
        if val_result:
            for k, v in val_result.items():
                if isinstance(v, (int, float)):
                    payload[f"eval_val/{k}"] = v
        if payload:
            self._safe_log(payload, step=epoch)

    def log_rank_analytics(self, epoch: int, analytics: dict) -> None:
        """Log rank analytics (win rates, category rates, dominant components)."""
        if not self._enabled:
            return
        payload: dict[str, Any] = {}
        summary = analytics.get("summary", analytics)

        for label, rate in summary.get("win_rates", {}).items():
            payload[f"rank/win_rate/{label}"] = rate
        for cat, rate in summary.get("category_rates", {}).items():
            payload[f"rank/category/{cat}"] = rate

        dom = summary.get("dominant_components", {})
        total_dom = sum(dom.values()) or 1
        for comp, cnt in dom.items():
            payload[f"rank/dominant/{comp}"] = cnt / total_dom

        for key in ("mean_winner_reward", "mean_det_reward", "mean_improvement"):
            if key in summary:
                payload[f"rank/{key}"] = summary[key]

        if payload:
            self._safe_log(payload, step=epoch)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        """Finalize the wandb run with optional summary metrics."""
        if self._run is None:
            self._enabled = False
            return

        try:
            if summary:
                for k, v in summary.items():
                    self._run.summary[k] = v
        except Exception as e:
            print(f"[wandb] failed to update summary during finish: {e}")

        try:
            import wandb

            wandb.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")
        finally:
            self._run = None
            self._enabled = False
