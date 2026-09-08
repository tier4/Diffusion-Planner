"""Inspect training data augmentation on one H5 frame."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from diffusion_planner.data import (
    PlannerFixStopPoint,
    PlannerILQRRefinement,
    PlannerPoseAugmentation,
    PlannerSpeedAugmentation,
    PlannerStartDecisionAugmentation,
)
from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    inspect_augmentation,
    load_frame_index,
)
from diffusion_planner_dashboard.ui.metadata import (
    render_index_summary,
    render_row_metadata,
)
from diffusion_planner_dashboard.ui.settings import (
    render_data_source_settings,
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


@dataclass(frozen=True)
class ILQRSettings:
    """User-adjustable iLQR settings for the augmentation inspector."""

    num_refine: int
    wheelbase_m: float
    state_weights: tuple[float, float, float]
    terminal_weight_scale: float
    velocity_weight: float
    steering_weight: float
    velocity_rate_weight: float
    steering_rate_weight: float
    velocity_max: float
    steering_limit_rad: float
    max_iterations: int
    speed_threshold: float


@dataclass(frozen=True)
class AugmentationPipelineSettings:
    """Controls for the training-order augmentation preview pipeline."""

    apply_start_decision: bool
    start_stop_speed_threshold: float
    start_max_shift_steps: int
    apply_fix_stop_point: bool
    fix_stop_speed_threshold: float
    apply_pose: bool
    apply_ilqr: bool
    apply_speed: bool
    longitudinal_offset: float
    lateral_offset: float
    yaw_offset: float
    ego_speed_scale: float
    ilqr: ILQRSettings


def _render_augmentation_settings() -> AugmentationPipelineSettings:
    st.sidebar.subheader("Augmentation")
    st.sidebar.caption("Training order")
    apply_start_decision = st.sidebar.checkbox("1. Start Decision", value=False)
    apply_fix_stop_point = st.sidebar.checkbox("2. Fix Stop Point", value=True)
    apply_pose = st.sidebar.checkbox("3. Pose Augmentation", value=True)
    apply_ilqr = st.sidebar.checkbox("4. iLQR Refinement", value=True)
    apply_speed = st.sidebar.checkbox("5. Speed Augmentation", value=True)
    with st.sidebar.expander("Start Decision parameters", expanded=False):
        start_stop_speed_threshold = float(
            st.number_input(
                "Start stop-speed threshold [m/s]", 0.0, value=0.1, step=0.05
            )
        )
        start_max_shift_steps = int(
            st.number_input("Start maximum shift steps", 1, 30, 10, step=1)
        )
    longitudinal_offset = float(
        st.sidebar.slider(
            "Longitudinal offset [m]",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )
    )
    lateral_offset = float(
        st.sidebar.slider(
            "Lateral offset [m]",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )
    )
    yaw_offset_degrees = float(
        st.sidebar.slider(
            "Yaw offset [deg]",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=0.5,
        )
    )
    ego_speed_scale = float(
        st.sidebar.slider(
            "Ego history speed scale",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.01,
        )
    )
    with st.sidebar.expander("iLQR parameters", expanded=False):
        num_refine = int(
            st.number_input(
                "Refine / speed-check endpoint index",
                1,
                79,
                30,
                step=1,
                help="iLQR refines through this index and checks speed at this exact point.",
            )
        )
        wheelbase_m = float(
            st.number_input("Wheelbase [m]", 0.1, 10.0, 2.79, step=0.01)
        )
        st.caption("Trajectory tracking weights")
        q_x = float(st.number_input("Position X weight", 0.0, value=1.0, step=0.1))
        q_y = float(st.number_input("Position Y weight", 0.0, value=1.0, step=0.1))
        q_yaw = float(st.number_input("Yaw weight", 0.0, value=0.5, step=0.1))
        terminal_weight_scale = float(
            st.number_input("Terminal weight scale", 0.0, value=10.0, step=1.0)
        )
        st.caption("Control weights")
        velocity_weight = float(
            st.number_input("Velocity tracking weight", 0.0, value=0.2, step=0.1)
        )
        steering_weight = float(
            st.number_input("Steering weight", 0.0, value=0.1, step=0.1)
        )
        velocity_rate_weight = float(
            st.number_input("Velocity change weight", 0.0, value=1.0, step=0.1)
        )
        steering_rate_weight = float(
            st.number_input("Steering change weight", 0.0, value=10.0, step=1.0)
        )
        st.caption("Constraints and solver")
        velocity_max = float(
            st.number_input("Maximum velocity [m/s]", 0.1, value=30.0, step=1.0)
        )
        steering_limit_rad = math.radians(
            float(
                st.number_input(
                    "Steering limit [deg]", 0.1, 89.0, math.degrees(0.7), step=1.0
                )
            )
        )
        max_iterations = int(st.number_input("Maximum iterations", 1, 100, 15, step=1))
        speed_threshold = float(
            st.number_input(
                "Endpoint speed skip threshold [m/s]",
                0.0,
                value=1.0,
                step=0.1,
                help="Skip when endpoint speed is at or below this value.",
            )
        )
    with st.sidebar.expander("Fix stop point", expanded=False):
        fix_stop_speed_threshold = float(
            st.number_input(
                "Stop-point speed threshold [m/s]",
                0.0,
                value=0.1,
                step=0.05,
                help="Hold the ego pose and zero its motion from the first future point at or below this speed.",
            )
        )
    return AugmentationPipelineSettings(
        apply_start_decision=apply_start_decision,
        start_stop_speed_threshold=start_stop_speed_threshold,
        start_max_shift_steps=start_max_shift_steps,
        apply_fix_stop_point=apply_fix_stop_point,
        fix_stop_speed_threshold=fix_stop_speed_threshold,
        apply_pose=apply_pose,
        apply_ilqr=apply_ilqr,
        apply_speed=apply_speed,
        longitudinal_offset=longitudinal_offset,
        lateral_offset=lateral_offset,
        yaw_offset=math.radians(yaw_offset_degrees),
        ego_speed_scale=ego_speed_scale,
        ilqr=ILQRSettings(
            num_refine=num_refine,
            wheelbase_m=wheelbase_m,
            state_weights=(q_x, q_y, q_yaw),
            terminal_weight_scale=terminal_weight_scale,
            velocity_weight=velocity_weight,
            steering_weight=steering_weight,
            velocity_rate_weight=velocity_rate_weight,
            steering_rate_weight=steering_rate_weight,
            velocity_max=velocity_max,
            steering_limit_rad=steering_limit_rad,
            max_iterations=max_iterations,
            speed_threshold=speed_threshold,
        ),
    )


def _augment_frame(
    frame_data: dict[str, Any],
    settings: AugmentationPipelineSettings,
) -> dict[str, Any]:
    ilqr = settings.ilqr
    start_decision = PlannerStartDecisionAugmentation(
        probability=1.0,
        stop_speed_threshold=settings.start_stop_speed_threshold,
        max_shift_steps=settings.start_max_shift_steps,
    )
    fix_stop_point = PlannerFixStopPoint(
        stop_speed_threshold=settings.fix_stop_speed_threshold
    )
    speed_augmentation = PlannerSpeedAugmentation(
        speed_scale_range=(settings.ego_speed_scale, settings.ego_speed_scale),
        speed_noise_range=(0.0, 0.0),
        probability=1.0,
    )
    pose_augmentation = PlannerPoseAugmentation(
        normal_case={
            "probability": 1.0,
            "longitudinal_offset_range": (settings.longitudinal_offset,) * 2,
            "lateral_offset_range": (settings.lateral_offset,) * 2,
            "yaw_offset_range": (settings.yaw_offset,) * 2,
            "pose_augmentation_endpoint_speed_threshold": ilqr.speed_threshold,
            "pose_augmentation_speed_check_endpoint_index": ilqr.num_refine,
        },
        stopped_in_intersection={
            "probability": 1.0,
            "stopped_speed_threshold": 0.1,
            "longitudinal_offset_range": (settings.longitudinal_offset,) * 2,
            "lateral_offset_range": (settings.lateral_offset,) * 2,
            "yaw_offset_range": (settings.yaw_offset,) * 2,
        },
    )
    ilqr_refinement = PlannerILQRRefinement(
        num_refine=ilqr.num_refine,
        wheelbase_m=ilqr.wheelbase_m,
        state_weights=ilqr.state_weights,
        terminal_weight_scale=ilqr.terminal_weight_scale,
        velocity_weight=ilqr.velocity_weight,
        steering_weight=ilqr.steering_weight,
        velocity_rate_weight=ilqr.velocity_rate_weight,
        steering_rate_weight=ilqr.steering_rate_weight,
        velocity_bounds=(0.0, ilqr.velocity_max),
        steering_limit_rad=ilqr.steering_limit_rad,
        max_iterations=ilqr.max_iterations,
    )
    output = dict(frame_data)
    if settings.apply_start_decision:
        output = start_decision(output)
    if settings.apply_fix_stop_point:
        output = fix_stop_point(output)
    if settings.apply_pose:
        output = pose_augmentation(output)
    if settings.apply_ilqr:
        output = ilqr_refinement(output)
    if settings.apply_speed:
        output = speed_augmentation(output)
    return output


def _difference_frame(
    original: dict[str, Any], augmented: dict[str, Any]
) -> dict[str, np.ndarray]:
    difference: dict[str, np.ndarray] = {}
    for key, original_value in original.items():
        original_array = np.asarray(original_value)
        augmented_array = np.asarray(augmented[key])
        if np.issubdtype(original_array.dtype, np.number):
            difference[key] = augmented_array.astype(np.float64) - original_array
        else:
            difference[key] = np.zeros(original_array.shape, dtype=np.float64)
    return difference


def render_data_augmentation() -> None:
    """Render original and deterministically augmented frame data."""
    st.title("Data Augmentation")
    source_path_text = render_data_source_settings()
    settings = _render_augmentation_settings()
    if source_path_text is None:
        st.info("Configure an H5 file or frame-index Parquet from the sidebar.")
        return

    source_path = Path(source_path_text).expanduser()
    try:
        source_modification_time_ns = source_path.stat().st_mtime_ns
        index = _cached_index(str(source_path), source_modification_time_ns)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return

    render_index_summary(index)
    row = render_frame_selector(index)
    render_row_metadata(row)
    options = render_plot_options()

    try:
        h5_modification_time_ns = Path(row.h5_path).stat().st_mtime_ns
        original = _cached_frame(
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
        )
        augmented = _augment_frame(original, settings)
    except Exception as error:
        st.exception(error)
        return

    enabled = [
        name
        for name, applied in (
            ("Start Decision", settings.apply_start_decision),
            ("Fix Stop Point", settings.apply_fix_stop_point),
            ("Pose", settings.apply_pose),
            ("iLQR", settings.apply_ilqr),
            ("Speed", settings.apply_speed),
        )
        if applied
    ]
    st.caption(
        "Training-order pipeline: Start Decision → Fix Stop Point → Pose → "
        f"iLQR → Speed. Enabled: {', '.join(enabled) if enabled else 'none'}."
    )
    original_column, augmented_column = st.columns(2)
    chart_identity = f"{index.path}::{row.index}::{settings}"
    with original_column:
        st.subheader("Original")
        original_figure = plot_frame(
            original, options=replace(options, title="Original frame")
        )
        original_figure.update_layout(uirevision=f"original::{chart_identity}")
        st.plotly_chart(
            original_figure,
            width="stretch",
            height=700,
            key=f"augmentation-original::{chart_identity}",
            config={"responsive": True, "scrollZoom": True},
        )
    with augmented_column:
        st.subheader("Augmented")
        augmented_figure = plot_frame(
            augmented, options=replace(options, title="Augmented frame")
        )
        augmented_figure.update_layout(uirevision=f"augmented::{chart_identity}")
        st.plotly_chart(
            augmented_figure,
            width="stretch",
            height=700,
            key=f"augmentation-augmented::{chart_identity}",
            config={"responsive": True, "scrollZoom": True},
        )

    inspection = inspect_augmentation(
        original,
        augmented,
        nonrigid_keys={"ego_agent_past"} if settings.apply_start_decision else (),
    )
    failures = sum(not bool(row_data["valid"]) for row_data in inspection)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Checked tensors", len(inspection))
    metric_columns[1].metric("Failed checks", failures)
    metric_columns[2].metric(
        "Augmented ego current",
        np.array2string(np.asarray(augmented["ego_agent_past"])[-1, :4], precision=3),
    )
    st.subheader("Validation")
    if failures:
        st.error(f"{failures} augmentation invariant check(s) failed.")
    else:
        st.success("All augmentation invariant checks passed.")
    st.dataframe(inspection, width="stretch", hide_index=True)

    difference = _difference_frame(original, augmented)
    original_tab, augmented_tab, difference_tab = st.tabs(
        ("Original tensors", "Augmented tensors", "Difference")
    )
    with original_tab:
        render_tensor_inspector(original, key_prefix="augmentation-original-inspector")
    with augmented_tab:
        render_tensor_inspector(
            augmented, key_prefix="augmentation-augmented-inspector"
        )
    with difference_tab:
        render_tensor_inspector(
            difference, key_prefix="augmentation-difference-inspector"
        )
