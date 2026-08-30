"""Visualize sampled trajectories from a checkpoint or sampler ONNX."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch

from diffusion_planner.data import (
    PlannerQuinticHermiteAugmentation,
    PlannerSpeedAugmentation,
    fill_unknown_traffic_light_futures,
)
from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    LoadedOnnxPlanner,
    LoadedPlanner,
    load_frame_index,
    load_planner,
    run_inference,
    run_onnx_inference,
    run_turn_indicator_inference,
)
from diffusion_planner_dashboard.ui.metadata import (
    render_index_summary,
    render_row_metadata,
)
from diffusion_planner_dashboard.ui.settings import (
    render_frame_selector,
    render_plot_options,
)
from diffusion_planner_dashboard.ui.tensor_inspector import render_tensor_inspector


@st.cache_data(show_spinner=False)
def _cached_index(path: str, modification_time_ns: int) -> FrameIndex:
    del modification_time_ns
    return load_frame_index(path)


@st.cache_resource
def _frame_loader() -> FrameLoader:
    return FrameLoader()


@st.cache_data(max_entries=64, show_spinner="Reading frame data from H5...")
def _cached_frame(
    h5_path: str,
    frame_index: int,
    frame_time_ns: int,
    modification_time_ns: int,
):
    del modification_time_ns
    row = FrameIndexRow(0, h5_path, frame_index, frame_time_ns, {})
    return _frame_loader().load(row)


@st.cache_resource(show_spinner="Loading planner model...")
def _cached_planner(
    model_path: str,
    modification_time_ns: int,
    device: str,
) -> LoadedPlanner | LoadedOnnxPlanner:
    del modification_time_ns
    return load_planner(model_path, device)


@st.cache_data(max_entries=32, show_spinner="Sampling trajectories...")
def _cached_prediction(
    model_path: str,
    model_modification_time_ns: int,
    h5_path: str,
    frame_index: int,
    frame_time_ns: int,
    h5_modification_time_ns: int,
    device: str,
    num_steps: int,
    time_epsilon: float,
    noise_scale: float,
    seed: int,
    apply_augmentation: bool,
    longitudinal_offset: float,
    lateral_offset: float,
    yaw_offset: float,
    ego_speed_scale: float,
    remove_neighbor_agents: bool,
    infer_future_traffic_lights: bool,
):
    planner = _cached_planner(model_path, model_modification_time_ns, device)
    frame_data = _cached_frame(
        h5_path,
        frame_index,
        frame_time_ns,
        h5_modification_time_ns,
    )
    if remove_neighbor_agents:
        frame_data = _remove_neighbor_agents(frame_data)
    if infer_future_traffic_lights:
        frame_data = _infer_future_traffic_lights(frame_data)
    frame_data = _fill_unknown_traffic_lights(frame_data)
    if apply_augmentation:
        frame_data = _augment_frame(
            frame_data,
            longitudinal_offset,
            lateral_offset,
            yaw_offset,
            ego_speed_scale,
        )
    if isinstance(planner, LoadedOnnxPlanner):
        return run_onnx_inference(
            planner.session,
            frame_data,
            noise_scale=noise_scale,
            seed=seed,
        )
    return run_inference(
        planner.model,
        frame_data,
        device=device,
        num_steps=num_steps,
        time_epsilon=time_epsilon,
        noise_scale=noise_scale,
        seed=seed,
    )


@st.cache_data(max_entries=32, show_spinner="Predicting turn indicator...")
def _cached_turn_indicator_prediction(
    model_path: str,
    model_modification_time_ns: int,
    h5_path: str,
    frame_index: int,
    frame_time_ns: int,
    h5_modification_time_ns: int,
    device: str,
    apply_augmentation: bool,
    longitudinal_offset: float,
    lateral_offset: float,
    yaw_offset: float,
    ego_speed_scale: float,
    remove_neighbor_agents: bool,
    infer_future_traffic_lights: bool,
):
    loaded = _cached_planner(model_path, model_modification_time_ns, device)
    if isinstance(loaded, LoadedOnnxPlanner):
        return None
    frame_data = _cached_frame(
        h5_path,
        frame_index,
        frame_time_ns,
        h5_modification_time_ns,
    )
    if remove_neighbor_agents:
        frame_data = _remove_neighbor_agents(frame_data)
    if infer_future_traffic_lights:
        frame_data = _infer_future_traffic_lights(frame_data)
    frame_data = _fill_unknown_traffic_lights(frame_data)
    if apply_augmentation:
        frame_data = _augment_frame(
            frame_data,
            longitudinal_offset,
            lateral_offset,
            yaw_offset,
            ego_speed_scale,
        )
    return run_turn_indicator_inference(loaded.model, frame_data, device=device)


def _augment_frame(
    frame_data: dict[str, Any],
    longitudinal_offset: float,
    lateral_offset: float,
    yaw_offset: float,
    ego_speed_scale: float,
) -> dict[str, Any]:
    """Apply a deterministic training augmentation to one frame."""
    speed_augmentation = PlannerSpeedAugmentation(
        speed_scale_range=(ego_speed_scale, ego_speed_scale),
        probability=1.0,
    )
    pose_augmentation = PlannerQuinticHermiteAugmentation(
        longitudinal_offset_range=(longitudinal_offset, longitudinal_offset),
        lateral_offset_range=(lateral_offset, lateral_offset),
        yaw_offset_range=(yaw_offset, yaw_offset),
        pose_probability=1.0,
    )
    return pose_augmentation(speed_augmentation(frame_data))


def _remove_neighbor_agents(frame_data: dict[str, Any]) -> dict[str, Any]:
    """Return a frame with every neighbor-agent tensor zeroed."""
    result = dict(frame_data)
    for key in (
        "neighbor_agents_past",
        "neighbor_agents_future",
        "agent_shape",
        "agent_label",
    ):
        if key in result:
            result[key] = np.zeros_like(np.asarray(result[key]))
    return result


def _infer_traffic_light_future(past: np.ndarray, future_length: int) -> np.ndarray:
    """Apply the Autoware inference rule to one traffic-light history tensor."""
    current = past[..., -1, :]
    future = np.repeat(current[..., None, :], future_length, axis=-2)
    flattened_past = past.reshape(-1, past.shape[-2], past.shape[-1])
    flattened_future = future.reshape(-1, future_length, future.shape[-1])
    amber_index = 1
    red_index = 2
    amber_duration_steps = 30
    for index, history in enumerate(flattened_past):
        if history[-1, amber_index] <= 0.5:
            continue
        elapsed_steps = 0
        for state in history[::-1]:
            if state[amber_index] <= 0.5:
                break
            elapsed_steps += 1
        remaining_steps = max(amber_duration_steps - elapsed_steps, 0)
        flattened_future[index, remaining_steps:, :] = 0
        flattened_future[index, remaining_steps:, red_index] = 1
    return future


def _infer_future_traffic_lights(frame_data: dict[str, Any]) -> dict[str, Any]:
    """Replace recorded traffic-light futures with inference-time estimates."""
    result = dict(frame_data)
    for past_key, future_key in (
        ("lane_traffic_light_past", "lane_traffic_light_future"),
        ("route_traffic_light_past", "route_traffic_light_future"),
    ):
        if past_key not in result or future_key not in result:
            continue
        past = np.asarray(result[past_key])
        future_length = np.asarray(result[future_key]).shape[-2]
        result[future_key] = _infer_traffic_light_future(past, future_length)
    return result


def _fill_unknown_traffic_lights(frame_data: dict[str, Any]) -> dict[str, Any]:
    """Forward-fill Unknown future states for lane and route traffic lights."""
    return fill_unknown_traffic_light_futures(frame_data)


def _render_source_settings() -> tuple[str | None, str | None]:
    """Apply the frame source and checkpoint together."""
    st.sidebar.subheader("Sources")
    with st.sidebar.form("training-result-source-settings"):
        source_candidate = st.text_input(
            "H5 or Parquet file",
            value=st.session_state.get("configured_frame_source_path", ""),
            placeholder="/path/to/frames.h5 or /path/to/train.parquet",
        )
        checkpoint_candidate = st.text_input(
            "Planner checkpoint or sampler ONNX",
            value=st.session_state.get("configured_checkpoint_path", ""),
            placeholder="/path/to/epoch_0001.pth or diffusion_planner_sampler.onnx",
        )
        assert source_candidate is not None
        assert checkpoint_candidate is not None
        applied = st.form_submit_button("Apply sources", use_container_width=True)
    if applied:
        st.session_state["configured_frame_source_path"] = source_candidate.strip()
        st.session_state["configured_checkpoint_path"] = checkpoint_candidate.strip()
    return (
        st.session_state.get("configured_frame_source_path") or None,
        st.session_state.get("configured_checkpoint_path") or None,
    )


def _render_inference_settings(onnx_model: bool) -> tuple[str, int, float, float, int]:
    st.sidebar.subheader("Inference")
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device = st.sidebar.selectbox("Device", devices)
    num_steps = int(
        st.sidebar.number_input(
            "Sampling steps",
            min_value=1,
            value=10 if onnx_model else 20,
            step=1,
            disabled=onnx_model,
            help="The sampler ONNX contains a fixed 10-step sampling loop."
            if onnx_model
            else None,
        )
    )
    time_epsilon = float(
        st.sidebar.number_input(
            "Time epsilon",
            min_value=1e-8,
            value=1e-5,
            format="%.1e",
            disabled=onnx_model,
        )
    )
    noise_scale = float(
        st.sidebar.number_input("Noise scale", min_value=0.0, value=1.0, step=0.1)
    )
    seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
    return device, num_steps, time_epsilon, noise_scale, seed


def _render_augmentation_settings() -> tuple[bool, float, float, float, float]:
    """Render deterministic augmentation controls for checkpoint inference."""
    st.sidebar.subheader("Data augmentation")
    enabled = st.sidebar.checkbox("Apply augmentation", value=False)
    longitudinal_offset = float(
        st.sidebar.slider(
            "Longitudinal offset [m]",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            disabled=not enabled,
        )
    )
    lateral_offset = float(
        st.sidebar.slider(
            "Lateral offset [m]",
            min_value=-5.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            disabled=not enabled,
        )
    )
    yaw_offset_degrees = float(
        st.sidebar.slider(
            "Yaw offset [deg]",
            min_value=-30.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            disabled=not enabled,
        )
    )
    ego_speed_scale = float(
        st.sidebar.slider(
            "Ego history speed scale",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.01,
            disabled=not enabled,
        )
    )
    return (
        enabled,
        longitudinal_offset,
        lateral_offset,
        math.radians(yaw_offset_degrees),
        ego_speed_scale,
    )


def _render_input_options() -> tuple[bool, bool]:
    """Render optional input transformations."""
    st.sidebar.subheader("Input options")
    remove_neighbor_agents = st.sidebar.checkbox(
        "Remove all neighbor agents", value=False
    )
    infer_future_traffic_lights = st.sidebar.checkbox(
        "Infer traffic light future from past", value=False
    )
    return (
        remove_neighbor_agents,
        infer_future_traffic_lights,
    )


def render_training_results() -> None:
    """Render checkpoint inference alongside ground-truth trajectories."""
    st.title("Training Results")
    source_path_text, checkpoint_path_text = _render_source_settings()
    device, num_steps, time_epsilon, noise_scale, seed = _render_inference_settings(
        checkpoint_path_text is not None
        and Path(checkpoint_path_text).suffix.lower() == ".onnx"
    )
    (
        apply_augmentation,
        longitudinal_offset,
        lateral_offset,
        yaw_offset,
        ego_speed_scale,
    ) = _render_augmentation_settings()
    (
        remove_neighbor_agents,
        infer_future_traffic_lights,
    ) = _render_input_options()
    if source_path_text is None or checkpoint_path_text is None:
        st.info("Configure both a frame source and a planner model from the sidebar.")
        return

    source_path = Path(source_path_text).expanduser()
    checkpoint_path = Path(checkpoint_path_text).expanduser()
    try:
        source_modification_time_ns = source_path.stat().st_mtime_ns
        checkpoint_modification_time_ns = checkpoint_path.stat().st_mtime_ns
        index = _cached_index(str(source_path), source_modification_time_ns)
        planner = _cached_planner(
            str(checkpoint_path), checkpoint_modification_time_ns, device
        )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return
    except Exception as error:
        st.exception(error)
        return

    render_index_summary(index)
    row = render_frame_selector(index)
    render_row_metadata(row)
    options = render_plot_options()

    metadata_columns = st.columns(6)
    metadata_columns[0].metric(
        "Checkpoint epoch",
        "-" if isinstance(planner, LoadedOnnxPlanner) else planner.epoch,
    )
    metadata_columns[1].metric(
        "Global step",
        "-" if isinstance(planner, LoadedOnnxPlanner) else planner.global_step,
    )
    metadata_columns[2].metric(
        "Sampling steps",
        planner.sampling_steps if isinstance(planner, LoadedOnnxPlanner) else num_steps,
    )
    metadata_columns[3].metric("Time epsilon", f"{time_epsilon:.1e}")
    metadata_columns[4].metric("Noise scale", noise_scale)
    metadata_columns[5].metric(
        "Device",
        planner.provider if isinstance(planner, LoadedOnnxPlanner) else device,
    )
    try:
        h5_modification_time_ns = Path(row.h5_path).stat().st_mtime_ns
        frame_data = _cached_frame(
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
        )
        visualized_frame = (
            _remove_neighbor_agents(frame_data)
            if remove_neighbor_agents
            else frame_data
        )
        if infer_future_traffic_lights:
            visualized_frame = _infer_future_traffic_lights(visualized_frame)
        visualized_frame = _fill_unknown_traffic_lights(visualized_frame)
        if apply_augmentation:
            visualized_frame = _augment_frame(
                visualized_frame,
                longitudinal_offset,
                lateral_offset,
                yaw_offset,
                ego_speed_scale,
            )
        prediction, inference_seconds = _cached_prediction(
            str(checkpoint_path),
            checkpoint_modification_time_ns,
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
            device,
            num_steps,
            time_epsilon,
            noise_scale,
            seed,
            apply_augmentation,
            longitudinal_offset,
            lateral_offset,
            yaw_offset,
            ego_speed_scale,
            remove_neighbor_agents,
            infer_future_traffic_lights,
        )
        turn_indicator_result = None
        if isinstance(planner, LoadedPlanner):
            turn_indicator_result = _cached_turn_indicator_prediction(
                str(checkpoint_path),
                checkpoint_modification_time_ns,
                row.h5_path,
                row.frame_index,
                row.frame_time_ns,
                h5_modification_time_ns,
                device,
                apply_augmentation,
                longitudinal_offset,
                lateral_offset,
                yaw_offset,
                ego_speed_scale,
                remove_neighbor_agents,
                infer_future_traffic_lights,
            )
    except Exception as error:
        st.exception(error)
        return

    st.caption(
        f"Inference: {inference_seconds:.3f} s · seed: {seed} · "
        f"model: `{checkpoint_path}`"
    )
    if apply_augmentation:
        st.caption(
            f"Augmentation: longitudinal offset {longitudinal_offset:.2f} m · "
            f"lateral offset {lateral_offset:.2f} m · "
            f"yaw offset {math.degrees(yaw_offset):.2f} deg · "
            f"ego history speed scale {ego_speed_scale:.2f}"
        )
    if remove_neighbor_agents:
        st.caption("Input option: all neighbor agents removed")
    if infer_future_traffic_lights:
        st.caption("Input option: traffic light future inferred from past")
    if turn_indicator_result is not None and isinstance(planner, LoadedPlanner):
        probabilities, predicted_report, turn_indicator_seconds = turn_indicator_result
        indicator_names = {0: "Missing", 1: "Disabled", 2: "Left", 3: "Right"}
        target_values = np.asarray(visualized_frame["turn_indicators_future"])
        history_values = np.asarray(visualized_frame["turn_indicators"])
        target_report = int(target_values.reshape(-1)[0])
        current_report = int(history_values.reshape(-1)[-1])
        st.subheader("Turn Indicator Prediction")
        indicator_columns = st.columns(6)
        indicator_columns[0].metric(
            "Prediction", indicator_names.get(predicted_report, str(predicted_report))
        )
        indicator_columns[1].metric(
            "Ground truth", indicator_names.get(target_report, str(target_report))
        )
        indicator_columns[2].metric(
            "Current", indicator_names.get(current_report, str(current_report))
        )
        indicator_columns[3].metric(
            "Confidence", f"{float(probabilities[predicted_report - 1]):.1%}"
        )
        indicator_columns[4].metric("Checkpoint epoch", planner.epoch)
        indicator_columns[5].metric("Inference", f"{turn_indicator_seconds:.3f} s")
        probability_columns = st.columns(3)
        for column, name, probability in zip(
            probability_columns,
            ("Disabled", "Left", "Right"),
            probabilities,
            strict=True,
        ):
            column.metric(f"P({name})", f"{float(probability):.1%}")
    figure = plot_frame(
        visualized_frame,
        options=options,
        predicted_trajectory=prediction,
    )
    chart_key = (
        f"training-result::{checkpoint_path}::{checkpoint_modification_time_ns}::"
        f"{row.h5_path}::{row.frame_index}::{num_steps}::{time_epsilon}::"
        f"{noise_scale}::{seed}::{apply_augmentation}::{longitudinal_offset}::"
        f"{lateral_offset}::{yaw_offset}::{ego_speed_scale}::"
        f"{remove_neighbor_agents}::"
        f"{infer_future_traffic_lights}"
    )
    figure.update_layout(autosize=True, uirevision=chart_key)
    st.plotly_chart(
        figure,
        width="stretch",
        height=900,
        key=chart_key,
        config={"responsive": True, "scrollZoom": True},
    )
    inspected_tensors = dict(visualized_frame)
    inspected_tensors["predicted_trajectory"] = prediction
    render_tensor_inspector(
        inspected_tensors,
        key_prefix="training-result-tensor-inspector",
    )
