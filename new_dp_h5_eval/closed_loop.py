"""Adapters that let the existing reproducer and aggregators consume native H5."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import argparse
import json

import numpy as np
import torch

from scenario_generation.route_timeline import RouteTimeline

from .dataset import H5FrameIndex
from .model import NewDpOnnxRunner
from .schema import MODEL_INPUT_NAMES


class NativeH5RouteTimeline(RouteTimeline):
    """A RouteTimeline whose scene payload is native H5, while poses use exact sidecars."""

    native_h5 = True

    def __init__(self, index_path: str | Path, source_npz_paths: list[Path],
                 sidecar_dir: str | Path | None = None, timers=None) -> None:
        self.h5_index = H5FrameIndex(index_path)
        super().__init__(source_npz_paths, Path(sidecar_dir) if sidecar_dir else None, timers)
        # RouteTimeline owns the canonical frame-index ordering.
        self._h5_rows = [self.h5_index.index_for_source(path) for path in self.npz_paths]

    def npz(self, idx: int) -> dict[str, np.ndarray]:
        return self.h5_index.frame(self._h5_rows[idx])

    def neighbor_last(self, idx: int) -> np.ndarray:
        frame = self.npz(idx)
        if frame["agent_shape"].shape[0] != frame["neighbor_agents_past"].shape[0]:
            raise ValueError("native H5 agent_shape/neighbor_agents_past slot mismatch")
        out = np.zeros((frame["neighbor_agents_past"].shape[0], 11), dtype=np.float32)
        out[:, :4] = frame["neighbor_agents_past"][:, -1]
        out[:, 6:8] = frame["agent_shape"]
        out[:, 8:11] = frame["agent_label"]
        return out


class TorchNativeNormalizer:
    """Torch equivalent of new DP's PlannerDataNormalizer."""

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(data)
        for key in ("ego_agent_past", "neighbor_agents_past", "goal_pose"):
            if key in out:
                out[key] = out[key].clone(); out[key][..., :2] /= 50.0
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
    return SimpleNamespace(new_dp_h5=True, observation_normalizer=TorchNativeNormalizer(),
                           predicted_neighbor_num=320, future_len=80)


class ReproducerOnnxModel:
    """Expose the old model call protocol while executing the new ONNX graph."""

    def __init__(self, model_path: str | Path, providers: list[str] | None = None,
                 seed: int = 0) -> None:
        self.runner = NewDpOnnxRunner(str(model_path), providers)
        self.seed = int(seed)
        self.calls = 0

    def __call__(self, data: dict[str, torch.Tensor]):
        device = data["ego_agent_past"].device
        feed = {key: data[key].detach().cpu().numpy() for key in MODEL_INPUT_NAMES}
        batch = feed["ego_agent_past"].shape[0]
        noise = np.stack([
            torch.randn(
                (321, 80, 4), generator=torch.Generator().manual_seed(self.seed + self.calls + i)
            ).numpy()
            for i in range(batch)
        ])
        self.calls += batch
        feed["initial_noise"] = noise
        trajectory, logits3 = self.runner.session.run(None, feed)
        if np.asarray(trajectory).shape != (batch, 321, 80, 4) or np.asarray(logits3).shape != (batch, 3):
            raise ValueError("new-DP ONNX output contract changed")
        trajectory[..., :2] *= 50.0
        yaw = trajectory[..., 2:4]
        trajectory[..., 2:4] = yaw / np.maximum(np.linalg.norm(yaw, axis=-1, keepdims=True), 1e-6)
        # New classes [DISABLE, LEFT, RIGHT] correspond to old feedback ids [1,2,3].
        logits5 = np.full((batch, 5), -1e9, dtype=np.float32)
        logits5[:, 1:4] = logits3
        outputs = {
            "prediction": torch.from_numpy(trajectory).to(device),
            "turn_indicator_logit": torch.from_numpy(logits5).to(device),
        }
        return None, outputs


def main() -> None:
    """Run the unchanged old reproducer metrics over routes backed by native H5."""
    from scenario_generation.closed_loop_eval import (
        aggregate, enumerate_multi_root_routes, format_summary_lines, segment_row_for_json,
    )
    from scenario_generation.reproducer_rollout import render_segment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="native-H5 Parquet index")
    parser.add_argument("onnx", type=Path)
    parser.add_argument("npz_root", type=Path,
                        help="old route root/path-list, used only for ordering and pose sidecars")
    parser.add_argument("output", type=Path)
    parser.add_argument("--segment-length", type=int, default=600)
    parser.add_argument("--max-routes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--tracker-mode", choices=("perfect", "mpc"), default="mpc")
    parser.add_argument("--neighbor-history-mode", choices=("recorded", "sim"), default="sim")
    parser.add_argument("--near-miss-thresh", type=float, default=0.5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    routes, sidecars = enumerate_multi_root_routes(args.npz_root)
    selected = sorted(routes.items())[:args.max_routes]
    # Check completeness before writing partial evaluation artifacts.
    with H5FrameIndex(args.index) as check:
        missing = [str(path) for _, paths in selected for path in paths
                   if str(Path(path).resolve()) not in check._by_source]
    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"native-H5 index is missing {len(missing)} requested frames:\n{preview}")

    model = ReproducerOnnxModel(args.onnx, args.providers, args.seed)
    rows = []
    for route_key, paths in selected:
        timeline = NativeH5RouteTimeline(args.index, paths, sidecars[route_key])
        for start, end in timeline.iter_segments(args.segment_length):
            metrics = render_segment(
                model, model_args(), timeline, start, end,
                args.output / f"{route_key}_{start}_{end}", device="cpu",
                near_miss_thresh=args.near_miss_thresh, search_radius=1.5,
                warmup_steps=0, unstick_after=300, unstick_advance_m=5.0,
                unstick_radius_mult=3.0, unstick_teleport_after=300,
                draw_every=None, replan_interval=1, tracker_mode=args.tracker_mode,
                neighbor_history_mode=args.neighbor_history_mode, yaw_gate=True,
                strong_brake_mps2=-2.5, abort_deviation_m=0.0, abort_after=1,
                abort_max_snaps=0, drop_objects=False, goal_mode="segment",
                title_prefix=None, distance_label_offset_m=0.0, view_half_m=60.0,
                max_stuck_steps=0, goal_reach_m=5.0, interpolate=False,
                color_by_uuid=False, window=None, max_steps=args.max_steps,
                timeline_progress_mode="pose", draw_pool=None,
            )
            rows.append({"route": route_key, **metrics})
    (args.output / "segments.jsonl").write_text(
        "".join(json.dumps(segment_row_for_json(row), default=float) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = aggregate(rows, args.near_miss_thresh, strong_brake_mps2=-2.5)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                               encoding="utf-8")
    print("\n".join(format_summary_lines(summary)))


if __name__ == "__main__":
    main()
