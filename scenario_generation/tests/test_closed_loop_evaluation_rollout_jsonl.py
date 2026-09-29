"""``rollout.jsonl`` must survive ``FullRouteClosedLoopEvaluation.execute_jobs``.

``execute_jobs`` renders each segment's PNGs (and the per-step ``rollout.jsonl`` next to
them) into a scratch ``TemporaryDirectory`` that is deleted at the end of the run, so
anything that isn't pulled out into ``out_dir`` first is lost. Regression coverage for that:
Both legacy and native-H5 ``run_job`` implementations use ``_preserve_rollout_trace``
after colormap rendering.
"""

from pathlib import Path

import pytest

import scenario_generation.closed_loop_evaluation as cle
from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    FullRouteRouteJob,
    RolloutParams,
)


def _rollout_params(**overrides) -> RolloutParams:
    defaults = dict(
        device="cpu",
        near_miss_thresh=1.0,
        search_radius=50.0,
        warmup_steps=0,
        unstick_after=0,
        unstick_advance_m=0.0,
        unstick_radius_mult=1.0,
        unstick_teleport_after=0,
        draw_every=1,
        draw_workers=1,
        replan_interval=1,
        tracker_mode="gt",
        neighbor_history_mode="sim",
        yaw_gate=False,
        strong_brake_mps2=3.0,
        abort_deviation_m=0.0,
        abort_after=0,
        abort_max_snaps=0,
        drop_objects=False,
        goal_mode="route",
        title_prefix=None,
        distance_label_offset_m=0.0,
        view_half_m=50.0,
        max_stuck_steps=0,
        goal_reach_m=2.0,
        interpolate=True,
        color_by_uuid=False,
        window=None,
        max_steps=None,
        timeline_progress_mode="sim",
        deviation_collision_thresh_m=2.0,
        colormap_metrics=(),
    )
    defaults.update(overrides)
    return RolloutParams(**defaults)


class _FakeTimeline:
    """Stands in for ``RouteTimeline``: yields a single fixed segment."""

    def __init__(self, route_paths, sidecar_dir=None, timers=None):
        del route_paths, sidecar_dir, timers

    def iter_segments(self, seg_len):
        del seg_len
        yield (0, 5)


def _fake_render_segment(model, model_args, tl, start, end, png_dir, **kwargs):
    del model, model_args, tl, start, end, kwargs
    png_dir = Path(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    (png_dir / "rollout.jsonl").write_text('{"event": "start"}\n')
    return {}


@pytest.mark.parametrize("native_h5", [False, True])
@pytest.mark.parametrize("colormaps", [False, True])
def test_execute_jobs_preserves_rollout_jsonl(tmp_path, monkeypatch, native_h5, colormaps):
    monkeypatch.setattr(cle, "RouteTimeline", _FakeTimeline)
    monkeypatch.setattr(cle, "render_segment", _fake_render_segment)

    colormap_sources = []

    def fake_colormaps(png_dir, *args, **kwargs):
        source = Path(png_dir) / "rollout.jsonl"
        assert source.read_text() == '{"event": "start"}\n'
        colormap_sources.append(source)

    if colormaps:
        import scenario_generation.trajectory_colormap as tc

        monkeypatch.setattr(tc, "render_trajectory_colormaps", fake_colormaps)

    out_dir = tmp_path / "out"
    config = ClosedLoopEvalConfig(
        out_dir=out_dir,
        params=_rollout_params(colormap_metrics=("road_border",) if colormaps else ()),
        fps=10.0,
        verbose=False,
        profile=False,
    )
    evaluator = FullRouteClosedLoopEvaluation(
        model=None,
        model_args=None,
        config=config,
        npz_root=tmp_path / "npz",
        seg_len=100,
        ddp_rank=0,
        ddp_world_size=1,
    )
    job = FullRouteRouteJob(
        job_id="routeA",
        npz_root=tmp_path / "npz",
        route_key="routeA",
        route_paths=[],
        seg_len=100,
    )

    if native_h5:
        import new_dp_h5_eval.closed_loop as native

        class FakeNativeTimeline(_FakeTimeline):
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                pass

        monkeypatch.setattr(native, "NativeH5RouteTimeline", FakeNativeTimeline)
        monkeypatch.setattr(native, "render_segment", _fake_render_segment)
        evaluator = native.NativeH5FullRouteClosedLoopEvaluation(
            model=None,
            model_args=None,
            config=config,
            routes=[{"route_id": "routeA", "h5_path": str(tmp_path / "frames.h5")}],
            seg_len=100,
            ddp_rank=0,
            ddp_world_size=1,
        )
        job = evaluator.discover_jobs()[0]

    evaluator.execute_jobs([job])

    rollout_path = out_dir / "routeA_0_5.rollout.jsonl"
    assert rollout_path.is_file(), (
        "rollout.jsonl should be moved into out_dir, not lost with scratch frames"
    )
    assert rollout_path.read_text() == '{"event": "start"}\n'
    assert len(colormap_sources) == int(colormaps)
    assert all(not source.parent.exists() for source in colormap_sources)
