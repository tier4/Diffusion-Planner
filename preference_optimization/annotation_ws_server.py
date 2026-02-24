"""WebSocket server for Lichtblick-based trajectory annotation UI.

This server mirrors the logic in preference_optimization/annotation_gui.py without
modifying it. It wraps PreferenceAnnotator and exposes a WebSocket API that
Lichtblick panels can use to drive the same annotation workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import websockets
from matplotlib.backends.backend_agg import FigureCanvasAgg
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
import sys

# Ensure parent directory is in path for diffusion_planner imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from preference_optimization.annotation_gui import PreferenceAnnotator
from preference_optimization.model_utils import load_model


@dataclass
class AnnotationParams:
    """Parameters that control trajectory generation and visualization."""

    noise_scale: float = 2.5
    fde_threshold: float = 2.0
    ade_threshold: float = 1.0
    max_retries: int = 50
    zoom_level: int = 5
    time_step: int = 40
    gt_similarity_mode: bool = True
    enable_initial_pruning: bool = True
    initial_pos_threshold: float = 0.055
    initial_yaw_threshold_deg: float = 0.55
    n_fixed_points: int = 0
    enable_guidance: bool = False
    use_collision: bool = True
    use_route_following: bool = False
    use_lane_keeping: bool = False
    use_centerline_following: bool = False
    guidance_scale: float = 0.5


@dataclass
class AnnotationState:
    """Cached state for last-rendered outputs."""

    metric_text: str = ""
    progress_text: str = ""
    metrics_text: str = ""
    sidebar_status: str = ""
    history_display: str = ""
    plots: dict[str, str | None] = field(default_factory=dict)


class AnnotationWsServer:
    """WebSocket server for annotation UI."""

    def __init__(
        self,
        model_path: Path,
        npz_list: Path,
        target_count: int | None,
        device: str,
        host: str,
        port: int,
    ) -> None:
        self.host = host
        self.port = port
        self.clients: set[websockets.WebSocketServerProtocol] = set()
        self._action_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._server_ready = threading.Event()
        self._server_thread: threading.Thread | None = None
        self.training_status: dict[str, Any] = {
            "phase": "annotation",
            "message": "Ready for annotation",
            "epoch": 0,
            "total_epochs": 0,
            "batch": 0,
            "total_batches": 0,
        }

        torch_device = torch.device(device)
        policy_model, model_args = load_model(model_path, torch_device)
        # Match annotation_gui.collect_preferences behavior: inference-only path.
        # Training mode expects inputs like diffusion_time and will fail here.
        policy_model.eval()

        with open(npz_list, "r") as f:
            npz_paths = json.load(f)

        if target_count is None:
            target_count = len(npz_paths)

        self.policy_model = policy_model
        self.model_args = model_args
        self.npz_paths = npz_paths
        self.annotator = PreferenceAnnotator(policy_model, model_args, npz_paths, target_count)
        self.params = AnnotationParams()
        self.state = AnnotationState()
        self.started_at = time.time()

        # Load initial sample to populate state.
        self._load_sample()

    def _fig_to_base64(self, fig) -> str | None:
        if fig is None:
            return None
        buffer = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buffer)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _extract_metrics_tables(self, metrics_text: str) -> dict[str, str]:
        if not metrics_text:
            return {"full": "", "ade_fde": ""}

        lines = metrics_text.splitlines()
        table_lines = [line for line in lines if line.startswith("|")]
        if not table_lines:
            return {"full": metrics_text, "ade_fde": ""}

        ade_fde_lines = [line for line in table_lines if "ADE" in line or "FDE" in line]
        ade_fde_table = "\n".join([table_lines[0], table_lines[1], *ade_fde_lines]) if len(table_lines) >= 2 else ""

        return {"full": "\n".join(table_lines), "ade_fde": ade_fde_table}

    def _build_state_payload(
        self,
        plots: tuple[Any, Any, Any] | None,
        metric_text: str,
        progress_text: str,
        metrics_text: str,
        sidebar_status: str,
        history_display: str,
    ) -> dict:
        if plots is None:
            plot_payload = self.state.plots
        else:
            plot_payload = {
                "trajectory": self._fig_to_base64(plots[0]),
                "velocity": self._fig_to_base64(plots[1]),
                "lateral": self._fig_to_base64(plots[2]),
            }

        metrics_tables = self._extract_metrics_tables(metrics_text)

        self.state = AnnotationState(
            metric_text=metric_text or "",
            progress_text=progress_text or "",
            metrics_text=metrics_text or "",
            sidebar_status=sidebar_status or "",
            history_display=history_display or "",
            plots=plot_payload,
        )

        return {
            "type": "state_update",
            "payload": {
                "texts": {
                    "metric": self.state.metric_text,
                    "progress": self.state.progress_text,
                    "metrics": self.state.metrics_text,
                    "metrics_full_table": metrics_tables["full"],
                    "metrics_ade_fde_table": metrics_tables["ade_fde"],
                    "sidebar": self.state.sidebar_status,
                    "history": self.state.history_display,
                },
                "plots": self.state.plots,
                "trajectory_messages": self._build_trajectory_messages(),
                "params": {
                    "noise_scale": self.params.noise_scale,
                    "fde_threshold": self.params.fde_threshold,
                    "ade_threshold": self.params.ade_threshold,
                    "max_retries": self.params.max_retries,
                    "zoom_level": self.params.zoom_level,
                    "time_step": self.params.time_step,
                    "gt_similarity_mode": self.params.gt_similarity_mode,
                    "enable_initial_pruning": self.params.enable_initial_pruning,
                    "initial_pos_threshold": self.params.initial_pos_threshold,
                    "initial_yaw_threshold_deg": self.params.initial_yaw_threshold_deg,
                    "n_fixed_points": self.params.n_fixed_points,
                    "enable_guidance": self.params.enable_guidance,
                    "use_collision": self.params.use_collision,
                    "use_route_following": self.params.use_route_following,
                    "use_lane_keeping": self.params.use_lane_keeping,
                    "use_centerline_following": self.params.use_centerline_following,
                    "guidance_scale": self.params.guidance_scale,
                },
                "status": {
                    "current_index": self.annotator.current_index,
                    "total_samples": len(self.annotator.npz_paths),
                    "total_preferences": len(self.annotator.preferences),
                    "target_count": self.annotator.target_count,
                    "annotation_complete": self.annotator.annotation_complete,
                    "current_filter": self.annotator.current_filter,
                    "auto_skip_labeled": self.annotator.auto_skip_labeled,
                    "current_jump_size": self.annotator.current_jump_size,
                    "is_pruned": self.annotator.is_pruned,
                    "initial_displacement": self.annotator.initial_displacement,
                    "initial_yaw_diff": self.annotator.initial_yaw_diff,
                    "gt_available": self.annotator.gt_available,
                },
                "server": {
                    "protocol": "annotation.websocket.v1",
                    "host": self.host,
                    "port": self.port,
                    "uptime_sec": max(0.0, time.time() - self.started_at),
                },
                "training": self.training_status,
            },
        }

    @staticmethod
    def _quat_from_heading(heading: float) -> dict[str, float]:
        return {
            "x": 0.0,
            "y": 0.0,
            "z": math.sin(heading / 2.0),
            "w": math.cos(heading / 2.0),
        }

    def _build_predicted_trajectory_message(self, trajectory: Any, frame_id: str = "map") -> dict | None:
        if trajectory is None:
            return None
        traj_np = torch.tensor(trajectory).cpu().numpy()
        points: list[dict[str, Any]] = []
        for i, row in enumerate(traj_np):
            x, y, cos_h, sin_h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            heading = math.atan2(sin_h, cos_h)
            points.append(
                {
                    "time_from_start": {"sec": 0, "nsec": int(i * 0.1 * 1e9)},
                    "pose": {
                        "position": {"x": x, "y": y, "z": 0.0},
                        "orientation": self._quat_from_heading(heading),
                    },
                    "longitudinal_velocity_mps": 0.0,
                    "lateral_velocity_mps": 0.0,
                    "acceleration_mps2": 0.0,
                    "heading_rate_rps": 0.0,
                    "front_wheel_angle_rad": 0.0,
                    "rear_wheel_angle_rad": 0.0,
                }
            )
        return {
            "header": {"stamp": {"sec": int(time.time()), "nsec": 0}, "frame_id": frame_id},
            "points": points,
        }

    def _build_gt_trajectory_message(self, frame_id: str = "map") -> dict | None:
        if (
            self.annotator.current_data is None
            or "ego_agent_future" not in self.annotator.current_data
        ):
            return None
        gt_np = self.annotator.current_data["ego_agent_future"][0].cpu().numpy()
        points: list[dict[str, Any]] = []
        for i, row in enumerate(gt_np):
            x, y, heading = float(row[0]), float(row[1]), float(row[2])
            points.append(
                {
                    "time_from_start": {"sec": 0, "nsec": int(i * 0.1 * 1e9)},
                    "pose": {
                        "position": {"x": x, "y": y, "z": 0.0},
                        "orientation": self._quat_from_heading(heading),
                    },
                    "longitudinal_velocity_mps": 0.0,
                    "lateral_velocity_mps": 0.0,
                    "acceleration_mps2": 0.0,
                    "heading_rate_rps": 0.0,
                    "front_wheel_angle_rad": 0.0,
                    "rear_wheel_angle_rad": 0.0,
                }
            )
        return {
            "header": {"stamp": {"sec": int(time.time()), "nsec": 0}, "frame_id": frame_id},
            "points": points,
        }

    def _build_trajectory_messages(self) -> dict[str, Any]:
        return {
            "deterministic": self._build_predicted_trajectory_message(self.annotator.trajectory_1),
            "stochastic": self._build_predicted_trajectory_message(self.annotator.trajectory_2),
            "ground_truth": self._build_gt_trajectory_message(),
        }

    def _load_sample(self) -> dict:
        result = self.annotator.load_sample(
            self.params.noise_scale,
            self.params.fde_threshold,
            self.params.ade_threshold,
            self.params.max_retries,
            self.params.zoom_level,
            self.params.gt_similarity_mode,
            self.params.enable_initial_pruning,
            self.params.initial_pos_threshold,
            self.params.initial_yaw_threshold_deg,
        )
        return self._refresh_from_tuple(result)

    def _refresh_from_tuple(self, result_tuple: tuple[Any, ...]) -> dict:
        # Keep visualization in sync with current UI time_step/zoom after any regeneration/navigation.
        plots = self.annotator.update_time_display(self.params.time_step, self.params.zoom_level)
        metric_text, progress_text, metrics_text, sidebar_status, history_display = result_tuple[3:8]
        return self._build_state_payload(plots, metric_text, progress_text, metrics_text, sidebar_status, history_display)

    async def _broadcast(self, message: dict) -> None:
        if not self.clients:
            return
        payload = json.dumps(message)
        await asyncio.gather(
            *[client.send(payload) for client in self.clients if client.open],
            return_exceptions=True,
        )

    async def _handle_action(self, action: str, payload: dict) -> dict:
        if action == "ping":
            return {
                "type": "pong",
                "payload": {
                    "protocol": "annotation.websocket.v1",
                    "time": time.time(),
                },
            }

        if action == "hello":
            return {
                "type": "hello_ack",
                "payload": {
                    "protocol": "annotation.websocket.v1",
                    "server": "annotation_ws_server",
                },
            }

        if action == "get_state":
            return self._build_state_payload(
                None,
                self.state.metric_text,
                self.state.progress_text,
                self.state.metrics_text,
                self.state.sidebar_status,
                self.state.history_display,
            )

        if action == "set_params":
            for key, value in payload.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
            return self._build_state_payload(
                None,
                self.state.metric_text,
                self.state.progress_text,
                self.state.metrics_text,
                self.state.sidebar_status,
                self.state.history_display,
            )

        if action == "load_sample":
            return self._refresh_from_tuple(
                self.annotator.load_sample(
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                    self.params.enable_initial_pruning,
                    self.params.initial_pos_threshold,
                    self.params.initial_yaw_threshold_deg,
                    self.params.n_fixed_points,
                    self.params.enable_guidance,
                    self.params.use_collision,
                    self.params.use_route_following,
                    self.params.use_lane_keeping,
                    self.params.use_centerline_following,
                    self.params.guidance_scale,
                )
            )

        if action == "regenerate":
            return self._refresh_from_tuple(
                self.annotator.regenerate(
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                    self.params.enable_initial_pruning,
                    self.params.initial_pos_threshold,
                    self.params.initial_yaw_threshold_deg,
                    self.params.n_fixed_points,
                    self.params.enable_guidance,
                    self.params.use_collision,
                    self.params.use_route_following,
                    self.params.use_lane_keeping,
                    self.params.use_centerline_following,
                    self.params.guidance_scale,
                )
            )

        if action == "select_winner":
            winner = payload.get("winner", "trajectory_2")
            if winner == "orange":
                winner = "trajectory_2"
            elif winner == "green":
                winner = "trajectory_1"
            return self._refresh_from_tuple(
                self.annotator.select_winner(
                    winner,
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                    self.params.enable_initial_pruning,
                    self.params.initial_pos_threshold,
                    self.params.initial_yaw_threshold_deg,
                    self.params.n_fixed_points,
                    self.params.enable_guidance,
                    self.params.use_collision,
                    self.params.use_route_following,
                    self.params.use_lane_keeping,
                    self.params.use_centerline_following,
                    self.params.guidance_scale,
                )
            )

        if action == "select_gt_as_winner":
            return self._refresh_from_tuple(
                self.annotator.select_gt_as_winner(
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                    self.params.enable_initial_pruning,
                    self.params.initial_pos_threshold,
                    self.params.initial_yaw_threshold_deg,
                    self.params.n_fixed_points,
                    self.params.enable_guidance,
                    self.params.use_collision,
                    self.params.use_route_following,
                    self.params.use_lane_keeping,
                    self.params.use_centerline_following,
                    self.params.guidance_scale,
                )
            )

        if action == "jump":
            delta = int(payload.get("delta", 0))
            self.annotator.update_jump_size(delta)
            return self._refresh_from_tuple(
                self.annotator.jump(
                    delta,
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                )
            )

        if action == "jump_to_index":
            target_index = int(payload.get("target_index", 1))
            return self._refresh_from_tuple(
                self.annotator.jump_to_index(
                    target_index,
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                )
            )

        if action == "jump_to_next_unlabeled":
            return self._refresh_from_tuple(
                self.annotator.jump_to_next_unlabeled(
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                )
            )

        if action == "toggle_filter":
            filter_mode = payload.get("filter_mode", "All")
            return self._refresh_from_tuple(
                self.annotator.toggle_filter(
                    filter_mode,
                    self.params.noise_scale,
                    self.params.fde_threshold,
                    self.params.ade_threshold,
                    self.params.max_retries,
                    self.params.zoom_level,
                    self.params.gt_similarity_mode,
                )
            )

        if action == "set_auto_skip":
            self.annotator.auto_skip_labeled = bool(payload.get("enabled", False))
            return self._build_state_payload(
                None,
                self.state.metric_text,
                self.state.progress_text,
                self.state.metrics_text,
                self.state.sidebar_status,
                self.state.history_display,
            )

        if action == "update_time":
            self.params.time_step = int(payload.get("time_step", self.params.time_step))
            plots = self.annotator.update_time_display(
                self.params.time_step,
                self.params.zoom_level,
            )
            return self._build_state_payload(
                plots,
                self.state.metric_text,
                self.state.progress_text,
                self.state.metrics_text,
                self.state.sidebar_status,
                self.state.history_display,
            )

        if action == "update_zoom":
            self.params.zoom_level = int(payload.get("zoom_level", self.params.zoom_level))
            plots = self.annotator.update_time_display(
                self.params.time_step,
                self.params.zoom_level,
            )
            return self._build_state_payload(
                plots,
                self.state.metric_text,
                self.state.progress_text,
                self.state.metrics_text,
                self.state.sidebar_status,
                self.state.history_display,
            )

        if action == "launch_training":
            return self._refresh_from_tuple(self.annotator.launch_training())

        raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _with_request_id(message: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        if request_id is None:
            return message
        output = dict(message)
        payload = dict(output.get("payload", {}))
        payload["request_id"] = request_id
        output["payload"] = payload
        return output

    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol) -> None:
        self.clients.add(websocket)
        peer = f"{websocket.remote_address}"
        try:
            await websocket.send(json.dumps(self._build_state_payload(
                None,
                self.state.metric_text,
                self.state.progress_text,
                self.state.metrics_text,
                self.state.sidebar_status,
                self.state.history_display,
            )))
        except ConnectionClosed:
            # Client disconnected before initial state delivery.
            self.clients.discard(websocket)
            return

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get("type", "") or data.get("action", "")
                    payload = data.get("payload", {})
                    request_id = data.get("request_id")
                    async with self._action_lock:
                        response = await self._handle_action(action, payload)
                    response = self._with_request_id(response, request_id)
                    await self._broadcast(response)
                except Exception as exc:  # noqa: BLE001
                    error_payload = {"type": "error", "payload": {"message": str(exc), "protocol": "annotation.websocket.v1"}}
                    await websocket.send(json.dumps(error_payload))
        except ConnectionClosedOK:
            # Normal close handshake.
            pass
        except ConnectionClosedError as exc:
            # Abrupt close (e.g., browser/tab killed) should not be treated as server error.
           pass
        except ConnectionClosed as exc:
            # Any remaining close-related condition.
            print(f"WebSocket client disconnected: {peer} ({exc})")
        finally:
            self.clients.discard(websocket)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()
        async with websockets.serve(self._handle_client, self.host, self.port):
            self._server_ready.set()
            print(f"Annotation WS server running at ws://{self.host}:{self.port}")
            await self._shutdown_event.wait()

    def start_background(self) -> None:
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        self._server_ready.clear()

        def _runner() -> None:
            asyncio.run(self.run())

        self._server_thread = threading.Thread(target=_runner, daemon=True)
        self._server_thread.start()
        self._server_ready.wait(timeout=30)

    def stop_background(self) -> None:
        if self._loop is None or self._shutdown_event is None:
            return
        self._loop.call_soon_threadsafe(self._shutdown_event.set)
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)

    def wait_for_annotation_complete(self, poll_interval_sec: float = 0.5) -> list[dict]:
        while not self.annotator.annotation_complete:
            time.sleep(poll_interval_sec)
        return list(self.annotator.preferences)

    def reset_annotation_round(self, target_count: int | None = None) -> None:
        if target_count is None:
            target_count = len(self.npz_paths)
        self.annotator = PreferenceAnnotator(self.policy_model, self.model_args, self.npz_paths, target_count)
        self.params.time_step = 40
        self.training_status = {
            "phase": "annotation",
            "message": "Ready for annotation",
            "epoch": 0,
            "total_epochs": 0,
            "batch": 0,
            "total_batches": 0,
        }
        self._load_sample()

    def update_training_status(
        self,
        *,
        phase: str,
        message: str,
        epoch: int = 0,
        total_epochs: int = 0,
        batch: int = 0,
        total_batches: int = 0,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.training_status = {
            "phase": phase,
            "message": message,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch": batch,
            "total_batches": total_batches,
            "metrics": metrics or {},
        }

        if self._loop is None:
            return
        payload = self._build_state_payload(
            None,
            self.state.metric_text,
            self.state.progress_text,
            self.state.metrics_text,
            self.state.sidebar_status,
            self.state.history_display,
        )
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotation WebSocket server for Lichtblick UI.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--npz-list", type=Path, required=True, help="Path to JSON list of NPZ files")
    parser.add_argument("--target-count", type=int, default=None, help="Target preference count (default: len(npz list))")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g., cuda:0 or cpu)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = AnnotationWsServer(
        model_path=args.model_path,
        npz_list=args.npz_list,
        target_count=args.target_count,
        device=args.device,
        host=args.host,
        port=args.port,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
