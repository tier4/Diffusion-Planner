"""Tests for closed-loop validation cadence in train.closed_loop_validate."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import diffusion_planner.train as train_module
import pytest
import scenario_generation.closed_loop_evaluation as closed_loop_evaluation
import scenario_generation.closed_loop_html_report as closed_loop_html_report
import scenario_generation.site_discovery as site_discovery
import scenario_generation.wandb_closed_loop as wandb_closed_loop
from diffusion_planner.train import closed_loop_validate


def _make_args(**overrides):
    """Minimal args namespace covering every closed_loop_* field the function reads."""
    defaults = dict(
        closed_loop_npz_root="single/route",
        closed_loop_sites_npz_root="sites_manifest.json",
        closed_loop_npz_object_modes=["objects"],
        closed_loop_sites_object_modes=["objects"],
        closed_loop_near_miss_thresh=0.5,
        closed_loop_search_radius=1.5,
        closed_loop_warmup_steps=0,
        closed_loop_unstick_after=300,
        closed_loop_unstick_advance_m=5.0,
        closed_loop_unstick_radius_mult=10.0,
        closed_loop_unstick_teleport_after=300,
        closed_loop_draw_every=4,
        closed_loop_replan_interval=4,
        closed_loop_abort_deviation_m=50.0,
        closed_loop_abort_after=30,
        closed_loop_abort_max_snaps=0,
        closed_loop_fps=10,
        closed_loop_seg_len=100000,
        closed_loop_wandb_video_pick="worst",
        closed_loop_colormap_metrics=[],
        closed_loop_report_base_url="",
        device="cpu",
        ddp=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _FakeEvaluator:
    """Stand-in for FullRouteClosedLoopEvaluation; records the npz_root it was run on."""

    calls: list = []

    def __init__(self, net, args, cfg, npz_root, seg_len):
        self.npz_root = npz_root

    def run(self):
        _FakeEvaluator.calls.append(self.npz_root)
        return {
            "n_segments": 1,
            "elapsed_sec": 0.1,
            "video_mp4s": [],
            "segments": [],
            "mean_route_completion": 1.0,
        }


def _fake_discover_sites(_path):
    return {"site_a": "sites/site_a", "site_b": "sites/site_b"}


wandb_log_calls: list = []


def _fake_build_full_closed_loop_wandb_log(_summary, *, site=None, include_score_scalars=True, **_kw):
    wandb_log_calls.append((site, include_score_scalars))
    return {}


@pytest.fixture(autouse=True)
def _patched_dependencies(monkeypatch):
    """Replace the heavy scenario_generation calls with recording fakes."""
    _FakeEvaluator.calls = []
    wandb_log_calls.clear()
    monkeypatch.setattr(
        closed_loop_evaluation, "FullRouteClosedLoopEvaluation", _FakeEvaluator
    )
    monkeypatch.setattr(site_discovery, "discover_sites_from_json", _fake_discover_sites)
    monkeypatch.setattr(
        wandb_closed_loop,
        "build_full_closed_loop_wandb_log",
        _fake_build_full_closed_loop_wandb_log,
    )
    monkeypatch.setattr(
        wandb_closed_loop, "build_combined_episode_table", lambda *a, **k: None
    )
    monkeypatch.setattr(wandb_closed_loop, "build_sites_aggregate_log", lambda *a, **k: {})
    monkeypatch.setattr(wandb_closed_loop, "build_sites_score_bar_charts", lambda *a, **k: {})
    monkeypatch.setattr(closed_loop_html_report, "build_html_report", lambda *a, **k: None)
    monkeypatch.setattr(train_module.wandb, "log", lambda *a, **k: None)
    return _FakeEvaluator.calls


def _fake_model():
    model = MagicMock()
    model.training = False
    return model


def test_sites_npz_root_skipped_on_non_final_save(_patched_dependencies, tmp_path):
    """closed_loop_sites_npz_root must not run at all on a non-final cadence call."""
    closed_loop_validate(
        _fake_model(), _make_args(), epoch=0, out_dir=str(tmp_path), is_final_save=False
    )

    assert _patched_dependencies == ["single/route"]


def test_sites_npz_root_runs_on_final_save(_patched_dependencies, tmp_path):
    """closed_loop_sites_npz_root runs every discovered site once is_final_save fires."""
    closed_loop_validate(
        _fake_model(), _make_args(), epoch=0, out_dir=str(tmp_path), is_final_save=True
    )

    assert _patched_dependencies == ["single/route", "sites/site_a", "sites/site_b"]


def test_closed_loop_npz_root_runs_regardless_of_is_final_save(_patched_dependencies, tmp_path):
    """closed_loop_npz_root (not sites) is unaffected by is_final_save -- always fires."""
    for is_final_save in (False, True):
        _patched_dependencies.clear()
        closed_loop_validate(
            _fake_model(),
            _make_args(),
            epoch=0,
            out_dir=str(tmp_path),
            is_final_save=is_final_save,
        )
        assert "single/route" in _patched_dependencies


def test_only_sites_skip_the_per_epoch_score_scalars(_patched_dependencies, tmp_path):
    """main keeps its per-epoch score scalars; sites opt out (bar chart covers them instead)."""
    closed_loop_validate(
        _fake_model(), _make_args(), epoch=0, out_dir=str(tmp_path), is_final_save=True
    )

    calls = dict(wandb_log_calls)
    assert calls[None] is True  # closed_loop_npz_root ("main") keeps its scalar trend
    assert calls["site_a"] is False
    assert calls["site_b"] is False
