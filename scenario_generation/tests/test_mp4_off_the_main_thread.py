"""A segment's MP4 is encoded on the render pool, and the group still waits for it.

Only the rollout and ffmpeg are stubbed; ``run_job`` -- which decides between submitting the
encode and running it inline -- and ``execute_jobs`` -- which joins the encoders -- are real.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scenario_generation import closed_loop_evaluation as cle
from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    FullRouteRouteJob,
)


class _Pool:
    """Defers what it is handed until the result is asked for, and records the calls."""

    def __init__(self):
        self.submitted: list[tuple] = []

    def submit(self, fn, *args):
        self.submitted.append(args)
        return SimpleNamespace(result=lambda: fn(*args))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Timeline:
    """Two segments per route, so an encode has a following rollout to overlap."""

    def __init__(self, *_args, **_kwargs):
        pass

    def iter_segments(self, _seg_len):
        return [(0, 10), (10, 20)]


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    encoded: list[Path] = []

    def render_segment(_model, _model_args, _tl, _start, _end, png_dir, **_kwargs):
        png_dir.mkdir(parents=True, exist_ok=True)
        (png_dir / "000.png").write_bytes(b"png")
        return {"object": {"collision_steps": 0}}

    def build_mp4(png_dir, out_mp4, _fps):
        assert any(png_dir.glob("*.png")), f"{png_dir} was gone before ffmpeg read it"
        out_mp4.write_bytes(b"mp4")
        encoded.append(out_mp4)

    monkeypatch.setattr(cle, "RouteTimeline", _Timeline)
    monkeypatch.setattr(cle, "render_segment", render_segment)
    monkeypatch.setattr(cle, "build_mp4", build_mp4)

    ev = FullRouteClosedLoopEvaluation(
        model=None,
        model_args=None,
        config=ClosedLoopEvalConfig(
            out_dir=tmp_path / "out",
            params=SimpleNamespace(draw_workers=2, colormap_metrics=None, render_kwargs=dict),
            fps=10.0,
            verbose=False,
            profile=False,
        ),
        npz_root=tmp_path,
        seg_len=10,
        ddp_rank=0,
        ddp_world_size=1,
    )
    return ev, encoded


def _job(key: str) -> FullRouteRouteJob:
    return FullRouteRouteJob(job_id=key, route_key=key, route_paths=[], seg_len=10)


def _with_pool(ev, monkeypatch) -> _Pool:
    pool = _Pool()
    monkeypatch.setattr(cle, "render_pool", lambda _workers: pool)
    return pool


def test_a_pooled_run_submits_the_encode_instead_of_running_it(evaluation, monkeypatch):
    ev, encoded = evaluation
    pool = _with_pool(ev, monkeypatch)

    ev._mp4_futures = []
    result = ev.run_job(_job("r0"), draw_pool=pool, frames_root=ev.out_dir)

    assert len(pool.submitted) == 2, "the encodes were not handed to the render pool"
    assert encoded == [], "an encode ran on the calling thread"
    assert len(result.video_mp4s) == 2


def test_the_group_waits_for_every_encoder_before_it_returns(evaluation, monkeypatch):
    ev, _encoded = evaluation
    _with_pool(ev, monkeypatch)

    merged = ev.execute_jobs([_job("r0"), _job("r1")])

    assert len(merged.video_mp4s) == 4
    for mp4 in merged.video_mp4s:
        assert mp4.is_file(), f"{mp4.name} was still encoding when the group reported it"


def test_a_failing_encoder_is_raised_rather_than_dropped(evaluation, monkeypatch):
    ev, _encoded = evaluation
    _with_pool(ev, monkeypatch)

    def boom(*_args):
        raise RuntimeError("ffmpeg exited 1")

    monkeypatch.setattr(cle, "build_mp4", boom)

    with pytest.raises(RuntimeError, match="ffmpeg exited 1"):
        ev.execute_jobs([_job("r0")])
