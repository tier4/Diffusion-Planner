"""Regression tests for ``log_closed_loop_to_wandb``'s config handling.

The callers (``run_all_groups_closed_loop.main`` and ``train.py``) pass a
``ClosedLoopConfig`` dataclass, not a dict. Reading it with ``.get`` raised
``AttributeError`` at the very end of a finished closed-loop run, which is the
worst place to fail -- every metric was already computed by then.
"""

import sys
from unittest import mock

# Mock wandb module if not installed in local environment
sys.modules.setdefault("wandb", mock.MagicMock())

from diffusion_planner.config.closed_loop_config import ClosedLoopConfig

from scenario_generation.wandb_closed_loop import log_closed_loop_to_wandb

_NAMES = ["site/all"]
_SUMMARIES = {"site/all": {"group": "all", "n_segments": 1, "total_steps": 10}}


def test_dataclass_config_reaches_wandb_init():
    """A ClosedLoopConfig must be read via attributes, not ``.get``."""
    cfg = ClosedLoopConfig(wandb_project_name="proj-x", exp_name="run-y")
    with mock.patch("scenario_generation.wandb_closed_loop.wandb") as wb:
        log_closed_loop_to_wandb(cfg, _NAMES, _SUMMARIES, run=None)

    assert wb.init.call_args.kwargs == {"project": "proj-x", "name": "run-y"}
    assert wb.finish.called, "a run we opened ourselves must be finished"


def test_mapping_config_still_supported():
    """Older callers passed a plain dict; keep that path working."""
    with mock.patch("scenario_generation.wandb_closed_loop.wandb") as wb:
        log_closed_loop_to_wandb(
            {"wandb_project_name": "p", "exp_name": "n"}, _NAMES, _SUMMARIES, run=None
        )

    assert wb.init.call_args.kwargs == {"project": "p", "name": "n"}


def test_empty_project_name_skips_upload():
    """``wandb_project_name`` is documented as "empty = disabled"."""
    with mock.patch("scenario_generation.wandb_closed_loop.wandb") as wb:
        log_closed_loop_to_wandb(ClosedLoopConfig(), _NAMES, _SUMMARIES, run=None)

    assert not wb.init.called


def test_none_config_skips_upload():
    with mock.patch("scenario_generation.wandb_closed_loop.wandb") as wb:
        log_closed_loop_to_wandb(None, _NAMES, _SUMMARIES, run=None)

    assert not wb.init.called


def test_supplied_run_logs_even_without_project_name():
    """train.py hands us its own run; the project name is irrelevant then."""
    run = mock.Mock()
    with (
        mock.patch("scenario_generation.wandb_closed_loop.log_metrics_tables") as tables,
        mock.patch("scenario_generation.wandb_closed_loop.log_cross_run_charts") as charts,
        mock.patch("scenario_generation.wandb_closed_loop.wandb") as wb,
    ):
        log_closed_loop_to_wandb(ClosedLoopConfig(), _NAMES, _SUMMARIES, run=run)

    assert tables.called and charts.called
    assert not wb.finish.called, "a caller-owned run must not be finished here"


def test_empty_summaries_is_a_no_op():
    with mock.patch("scenario_generation.wandb_closed_loop.wandb") as wb:
        log_closed_loop_to_wandb(ClosedLoopConfig(wandb_project_name="p"), [], {}, run=None)

    assert not wb.init.called
