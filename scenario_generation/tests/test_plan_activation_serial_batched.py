import json
from types import SimpleNamespace

import numpy as np
import torch

from scenario_generation import reproducer_rollout as rollout
from scenario_generation.route_timeline import RouteTimeline


class _StraightModel:
    def __init__(self, future_len: int):
        self.future_len = future_len

    def __call__(self, data):
        batch = data["ego_current_state"].shape[0]
        pred = torch.zeros((batch, 1, self.future_len, 4), dtype=torch.float32)
        pred[:, 0, :, 0] = torch.arange(1, self.future_len + 1) * 0.2
        pred[:, 0, :, 2] = 1.0
        return None, {
            "prediction": pred,
            "turn_indicator_logit": torch.zeros((batch, 5), dtype=torch.float32),
        }


def _timeline(tmp_path, n_frames=6):
    paths = []
    for frame in range(n_frames):
        neighbor = np.zeros((320, 31, 11), dtype=np.float32)
        arrays = {
            "ego_agent_past": np.zeros((31, 3), dtype=np.float32),
            "ego_current_state": np.array(
                [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            "neighbor_agents_past": neighbor,
            "lanes": np.zeros((1, 20, 33), dtype=np.float32),
            "lanes_speed_limit": np.zeros((1, 1), dtype=np.float32),
            "lanes_has_speed_limit": np.zeros((1, 1), dtype=bool),
            "route_lanes": np.zeros((1, 20, 33), dtype=np.float32),
            "route_lanes_speed_limit": np.zeros((1, 1), dtype=np.float32),
            "route_lanes_has_speed_limit": np.zeros((1, 1), dtype=bool),
            "polygons": np.zeros((1, 40, 3), dtype=np.float32),
            "line_strings": np.zeros((1, 20, 4), dtype=np.float32),
            "static_objects": np.zeros((5, 10), dtype=np.float32),
            "ego_shape": np.array([4.76, 7.24, 2.29], dtype=np.float32),
            "turn_indicators": np.zeros(31, dtype=np.int64),
            "goal_pose": np.array([10.0, 0.0, 0.0], dtype=np.float32),
        }
        path = tmp_path / f"route_{frame:010d}.npz"
        np.savez_compressed(path, **arrays)
        (tmp_path / f"route_{frame:010d}.json").write_text(
            json.dumps(
                {
                    "x": frame * 0.2,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                    "neighbor_ids": [""] * 320,
                }
            )
        )
        paths.append(path)
    return RouteTimeline(paths)


def _model_args(future_len=8):
    return SimpleNamespace(
        predicted_neighbor_num=0,
        future_len=future_len,
        use_velocity_representation=False,
        observation_normalizer=lambda data: data,
        state_normalizer=SimpleNamespace(
            mean=np.zeros((1, 1, 4), dtype=np.float32),
            std=np.ones((1, 1, 4), dtype=np.float32),
        ),
    )


def test_delay_plan_activation_serial_and_batched_match_one_segment(tmp_path, monkeypatch):
    tl = _timeline(tmp_path)
    args = _model_args()
    model = _StraightModel(args.future_len)
    serial_poses = []
    batched_poses = []
    sink = serial_poses
    original_advance = rollout._advance_step

    def capture_advance(*call_args, **call_kwargs):
        original_advance(*call_args, **call_kwargs)
        sink.append(call_args[0].live_pose.copy())

    monkeypatch.setattr(rollout, "_advance_step", capture_advance)
    rollout.render_segment(
        model=model,
        model_args=args,
        tl=tl,
        start=0,
        end=4,
        out_dir=tmp_path / "serial",
        device="cpu",
        near_miss_thresh=0.5,
        search_radius=1.5,
        warmup_steps=0,
        unstick_after=0,
        unstick_advance_m=5.0,
        unstick_radius_mult=1.0,
        unstick_teleport_after=0,
        draw_every=None,
        replan_interval=1,
        tracker_mode="mpc",
        neighbor_history_mode="recorded",
        yaw_gate=True,
        strong_brake_mps2=-2.5,
        abort_deviation_m=0.0,
        abort_after=1,
        abort_max_snaps=0,
        drop_objects=False,
        goal_mode="segment",
        title_prefix=None,
        distance_label_offset_m=1.2,
        view_half_m=50.0,
        max_stuck_steps=0,
        goal_reach_m=0.0,
        interpolate=False,
        color_by_uuid=False,
        window=None,
        max_steps=4,
        timeline_progress_mode="clock",
        draw_pool=None,
        delay_step=2,
    )

    sink = batched_poses
    rollout.run_segments_batched(
        model,
        args,
        [(tl, 0, 4)],
        device="cpu",
        batch_size=1,
        near_miss_thresh=0.5,
        search_radius=1.5,
        warmup_steps=0,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        unstick_after=0,
        max_steps_mult=1,
        n_build_threads=1,
        prefetch_ahead=0,
        tracker_mode="mpc",
        timeline_progress_mode="clock",
        delay_step=2,
    )

    assert len(serial_poses) == len(batched_poses) == 4
    np.testing.assert_allclose(batched_poses, serial_poses, rtol=0.0, atol=1e-6)
