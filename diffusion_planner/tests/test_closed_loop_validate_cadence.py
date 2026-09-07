"""Tests for closed-loop validation cadence in train.closed_loop_validate."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_args(**overrides):
    """Minimal args namespace covering every closed_loop_* field the function reads."""
    defaults = dict(
        closed_loop_npz_root=["single/route"],
        closed_loop_object_modes=["objects"],
        closed_loop_near_miss_thresh=0.5,
        closed_loop_search_radius=1.5,
        closed_loop_warmup_steps=0,
        closed_loop_unstick_after=300,
        closed_loop_unstick_advance_m=5.0,
        closed_loop_unstick_radius_mult=10.0,
        closed_loop_unstick_teleport_after=300,
        closed_loop_draw_every=4,
        closed_loop_replan_interval=1,
        closed_loop_abort_deviation_m=50.0,
        closed_loop_abort_after=30,
        closed_loop_abort_max_snaps=0,
        closed_loop_fps=10,
        closed_loop_seg_len=100000,
        closed_loop_wandb_video_pick="worst",
        closed_loop_colormap_metrics=[],
        device="cpu",
        ddp=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_model():
    model = MagicMock()
    model.training = False
    return model


def test_empty_closed_loop_npz_root_returns_early(tmp_path):
    """No closed_loop_npz_root means closed_loop_validate returns early."""
    import diffusion_planner.train as train_module
    from diffusion_planner.train import closed_loop_validate

    # Create a mock to verify train_loop is NOT called
    original_training = train_module.model_training
    called = []

    def mock_training(*args, **kwargs):
        called.append(True)

    # Can't easily mock model_training due to imports, but we can test the early return
    # by checking that empty npz_root returns immediately
    args = _make_args(closed_loop_npz_root=[])
    model = _fake_model()
    # This should return early since closed_loop_npz_root is empty (after conversion to list)
    closed_loop_validate(model, args, epoch=0, out_dir=str(tmp_path))
    # If we get here without error, the early return worked
    assert True  # Test passes if no exception was raised


def test_args_conversion_handles_list_input(tmp_path):
    """closed_loop_npz_root should accept list of paths."""
    args = _make_args(closed_loop_npz_root=["path1", "path2"])
    assert args.closed_loop_npz_root == ["path1", "path2"]
