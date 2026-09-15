"""Adapters that let the existing reproducer and aggregators consume native H5."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401 - registers the zstd HDF5 filter
import numpy as np
import torch
from scipy.spatial import cKDTree

from scenario_generation.closed_loop_eval import (
    build_mp4,
    evaluate_segment_pass,
    segment_row_for_json,
    tdigest_sidecar_row,
)
from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    FullRouteRouteJob,
    JobRunResult,
    RolloutParams,
)
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline

from .model import (
    NewDpOnnxRunner,
    decode_onnx_outputs,
    legacy_feedback_turn_logits,
    seeded_initial_noise,
)
from .schema import H5_FORMAT, H5_FORMAT_VERSION, MODEL_INPUT_NAMES


class NativeH5RouteTimeline(RouteTimeline):
    """Closed-loop route backed entirely by one multi-frame native H5 shard."""

    native_h5 = True

    def __init__(
        self,
        h5_path: str | Path,
        frame_start: int = 0,
        frame_stop: int | None = None,
        timers: Timers | None = None,
    ) -> None:
        self.h5_path = Path(h5_path).expanduser().resolve()
        self._h5 = h5py.File(self.h5_path, "r")
        try:
            if self._h5.attrs.get("format") != H5_FORMAT:
                raise ValueError(f"unexpected H5 format: {self.h5_path}")
            if int(self._h5.attrs.get("format_version", -1)) != H5_FORMAT_VERSION:
                raise ValueError(f"unsupported H5 format version: {self.h5_path}")
            metadata = self._h5["metadata"]
            required = {"frame_time_ns", "ego_x", "ego_y", "ego_yaw"}
            missing = required.difference(metadata.keys())
            if missing:
                raise ValueError(
                    f"closed-loop H5 missing pose metadata {sorted(missing)}: {self.h5_path}"
                )
            missing = set(MODEL_INPUT_NAMES).difference(self._h5["frames"].keys())
            if missing:
                raise ValueError(
                    f"closed-loop H5 missing model fields {sorted(missing)}: {self.h5_path}"
                )
            total = int(self._h5.attrs["num_frames"])
            stop = total if frame_stop is None else int(frame_stop)
            start = int(frame_start)
            if not 0 <= start < stop <= total:
                raise ValueError(f"invalid H5 frame range [{start}, {stop}) for {total} frames")
            self._rows = np.arange(start, stop, dtype=np.int64)
            self.frame_indices = self._rows.copy()
            self.frame_times_ns = np.asarray(metadata["frame_time_ns"][start:stop], dtype=np.int64)
            self.poses = np.column_stack(
                [
                    metadata["ego_x"][start:stop],
                    metadata["ego_y"][start:stop],
                    metadata["ego_yaw"][start:stop],
                ]
            ).astype(np.float64)
            if not np.isfinite(self.poses).all():
                raise ValueError(f"non-finite closed-loop poses in {self.h5_path}")
        except BaseException:
            self._h5.close()
            raise

        self.timers = timers or Timers()
        # Several inherited helpers use only the length of npz_paths.
        self.npz_paths = [self.h5_path] * len(self._rows)
        self._sidecar_paths = self.npz_paths
        self.kdtree = cKDTree(self.poses[:, :2])
        self.speeds = self._compute_native_speeds()
        self._npz_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._npz_cache_max = 128
        self._cache_lock = threading.Lock()

    def _compute_native_speeds(self) -> np.ndarray:
        speeds = np.zeros(len(self._rows), dtype=np.float64)
        if len(speeds) < 2:
            return speeds
        dt = np.diff(self.frame_times_ns).astype(np.float64) * 1e-9
        if np.any(dt <= 0.0):
            raise ValueError(f"non-increasing frame times in {self.h5_path}")
        segment = np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1) / dt
        speeds[:-1] = segment
        speeds[-1] = segment[-1]
        return speeds

    def npz(self, idx: int) -> dict[str, np.ndarray]:
        with self._cache_lock:
            cached = self._npz_cache.get(idx)
            if cached is not None:
                self._npz_cache.move_to_end(idx)
                return cached
        row = int(self._rows[idx])
        with self.timers("timeline_load_h5"):
            frames = self._h5["frames"]
            missing = set(MODEL_INPUT_NAMES).difference(frames.keys())
            if missing:
                raise ValueError(f"native H5 missing model fields: {sorted(missing)}")
            data = {key: np.asarray(value[row]) for key, value in frames.items()}
        with self._cache_lock:
            self._npz_cache[idx] = data
            self._npz_cache.move_to_end(idx)
            while len(self._npz_cache) > self._npz_cache_max:
                self._npz_cache.popitem(last=False)
        return data

    def neighbor_last(self, idx: int) -> np.ndarray:
        frame = self.npz(idx)
        out = np.zeros((frame["neighbor_agents_past"].shape[0], 11), dtype=np.float32)
        out[:, :4] = frame["neighbor_agents_past"][:, -1]
        out[:, 6:8] = frame["agent_shape"]
        out[:, 8:11] = frame["agent_label"]
        return out

    def neighbor_ids(self, _idx: int) -> list[str]:
        # The current native preprocessing API does not expose selected-agent
        # UUIDs. Recorded-neighbor mode remains exact and does not require them.
        return []

    def sidecar_path(self, _idx: int) -> Path:
        return self.h5_path

    def close(self) -> None:
        self._h5.close()


class TorchNativeNormalizer:
    """Torch equivalent of new DP's PlannerDataNormalizer."""

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(data)
        for key in ("ego_agent_past", "neighbor_agents_past", "goal_pose"):
            if key in out:
                out[key] = out[key].clone()
                out[key][..., :2] /= 50.0
        for key in ("lanes", "route_lanes", "intersection_area", "stop_lines", "road_borders"):
            if key in out:
                out[key] = out[key] / 50.0
        for key in ("lanes_speed_limit", "route_lanes_speed_limit"):
            if key in out:
                out[key] = out[key] / 15.0
        for key in ("agent_shape", "ego_shape"):
            if key in out:
                out[key] = out[key] / 10.0
        return out


def model_args() -> SimpleNamespace:
    """The small model-argument surface used by the old reproducer."""
    return SimpleNamespace(
        new_dp_h5=True,
        observation_normalizer=TorchNativeNormalizer(),
        predicted_neighbor_num=320,
        future_len=80,
    )


class ReproducerOnnxModel:
    """Expose the old model call protocol while executing the new ONNX graph."""

    def __init__(
        self, model_path: str | Path, providers: list[str] | None = None, seed: int = 0
    ) -> None:
        self.runner = NewDpOnnxRunner(str(model_path), providers)
        self.seed = int(seed)
        self.calls = 0

    def __call__(self, data: dict[str, torch.Tensor]):
        device = data["ego_agent_past"].device
        feed = {key: data[key].detach().cpu().numpy() for key in MODEL_INPUT_NAMES}
        batch = feed["ego_agent_past"].shape[0]
        noise = seeded_initial_noise([self.seed + self.calls + i for i in range(batch)])
        self.calls += batch
        feed["initial_noise"] = noise
        trajectory, logits3 = self.runner.session.run(None, feed)
        trajectory, logits3 = decode_onnx_outputs(trajectory, logits3, batch_size=batch)
        outputs = {
            "prediction": torch.from_numpy(trajectory).to(device),
            "turn_indicator_logit": torch.from_numpy(legacy_feedback_turn_logits(logits3)).to(
                device
            ),
        }
        return None, outputs


class NativeH5FullRouteClosedLoopEvaluation(FullRouteClosedLoopEvaluation):
    """Standard full-route evaluator with native H5 route discovery.

    This deliberately inherits the normal project's evaluator so its artifact
    layout, DDP merge, t-digest sidecars, and summary aggregation cannot drift
    from ``run_all_groups_closed_loop.py``.  Only route discovery and timeline
    construction differ from the legacy NPZ implementation.
    """

    def __init__(self, *args, routes: list[dict], **kwargs) -> None:
        super().__init__(*args, npz_root="native_h5", **kwargs)
        self.routes = routes
        self._routes_by_id = {str(route["route_id"]): route for route in routes}

    def discover_jobs(self) -> list[FullRouteRouteJob]:
        return [
            FullRouteRouteJob(
                job_id=str(route["route_id"]),
                route_key=str(route["route_id"]),
                seg_len=self.seg_len,
            )
            for route in self.routes
        ]

    def run_job(
        self,
        job,
        *,
        segments_file=None,
        digest_file=None,
        draw_pool=None,
        frames_root: Path | None = None,
    ) -> JobRunResult:
        route = self._routes_by_id[job.job_id]
        h5_path = Path(route["h5_path"])
        frames_root = frames_root if frames_root is not None else self.out_dir
        timers = Timers()
        timeline = NativeH5RouteTimeline(
            h5_path,
            int(route.get("frame_start", 0)),
            route.get("frame_stop"),
            timers=timers,
        )
        rows: list[dict] = []
        video_mp4s: list[Path] = []
        try:
            for start, end in timeline.iter_segments(job.seg_len):
                png_dir = frames_root / f"{job.route_key}_{start}_{end}"
                metrics = render_segment(
                    self.model,
                    self.model_args,
                    timeline,
                    start,
                    end,
                    png_dir,
                    **self.config.params.render_kwargs(),
                    draw_pool=draw_pool,
                    timers=timers,
                )
                if self.config.params.colormap_metrics:
                    from scenario_generation.trajectory_colormap import render_trajectory_colormaps

                    render_trajectory_colormaps(
                        png_dir,
                        self.out_dir,
                        f"{job.route_key}_{start}_{end}",
                        metrics=self.config.params.colormap_metrics,
                        near_miss_thresh=self.config.params.near_miss_thresh,
                        strong_brake_mps2=self.config.params.strong_brake_mps2,
                        title=f"{job.route_key} [{start},{end}]",
                    )
                row = {"route": job.route_key, **metrics}
                if self.config.pass_condition is not None:
                    row["passed"] = evaluate_segment_pass(row, self.config.pass_condition)
                if segments_file is not None:
                    segments_file.write(json.dumps(segment_row_for_json(row), default=float) + "\n")
                    segments_file.flush()
                    if digest_file is not None:
                        side = tdigest_sidecar_row(row)
                        if side is not None:
                            digest_file.write(json.dumps(side, default=float) + "\n")
                            digest_file.flush()
                rows.append(row)
                if any(png_dir.glob("*.png")):
                    segment_mp4 = self.out_dir / f"{job.route_key}_{start}_{end}.mp4"
                    build_mp4(png_dir, segment_mp4, self.config.fps)
                    video_mp4s.append(segment_mp4)
        finally:
            timeline.close()
        return JobRunResult(rows=rows, video_mp4s=video_mp4s, extras={"timers": timers})


def rollout_params_from_closed_loop_config(
    cfg, *, drop_objects: bool, draw_every: int | None
) -> RolloutParams:
    """Map the project's single ClosedLoopConfig source of truth to native H5 rollout params."""
    return RolloutParams(
        device=cfg.device,
        near_miss_thresh=cfg.closed_loop_near_miss_thresh,
        search_radius=cfg.closed_loop_search_radius,
        warmup_steps=cfg.closed_loop_warmup_steps,
        unstick_after=cfg.closed_loop_unstick_after,
        unstick_advance_m=cfg.closed_loop_unstick_advance_m,
        unstick_radius_mult=cfg.closed_loop_unstick_radius_mult,
        unstick_teleport_after=cfg.closed_loop_unstick_teleport_after,
        draw_every=draw_every,
        draw_workers=cfg.closed_loop_draw_workers,
        replan_interval=cfg.closed_loop_replan_interval,
        tracker_mode=cfg.closed_loop_tracker_mode,
        neighbor_history_mode=cfg.closed_loop_neighbor_history_mode,
        yaw_gate=cfg.closed_loop_yaw_gate,
        strong_brake_mps2=cfg.closed_loop_strong_brake_mps2,
        abort_deviation_m=cfg.closed_loop_abort_deviation_m,
        abort_after=cfg.closed_loop_abort_after,
        abort_max_snaps=cfg.closed_loop_abort_max_snaps,
        drop_objects=drop_objects,
        goal_mode=cfg.closed_loop_goal_mode,
        title_prefix=cfg.closed_loop_title_prefix,
        distance_label_offset_m=cfg.closed_loop_distance_label_offset_m,
        view_half_m=cfg.closed_loop_view_half_m,
        max_stuck_steps=cfg.closed_loop_max_stuck_steps,
        goal_reach_m=cfg.closed_loop_goal_reach_m,
        interpolate=cfg.closed_loop_interpolate,
        color_by_uuid=cfg.closed_loop_color_by_uuid,
        window=cfg.closed_loop_window,
        max_steps=cfg.closed_loop_max_steps,
        timeline_progress_mode=cfg.closed_loop_timeline_progress_mode,
        deviation_collision_thresh_m=cfg.closed_loop_deviation_collision_thresh_m,
        # Native-H5 evaluation has no separate media toggle.  Keep its rendering
        # output consistent with the regular closed-loop evaluator.
        colormap_metrics=tuple(cfg.closed_loop_colormap_metrics),
    )
