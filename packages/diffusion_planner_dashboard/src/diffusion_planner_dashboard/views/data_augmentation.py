"""Inspect training data augmentation on one H5 frame."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from diffusion_planner.data import (
    PlannerQuinticHermiteAugmentation,
    PlannerSpeedAugmentation,
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


def _render_augmentation_settings() -> tuple[float, float, float, float]:
    st.sidebar.subheader("Augmentation")
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
    return (
        longitudinal_offset,
        lateral_offset,
        math.radians(yaw_offset_degrees),
        ego_speed_scale,
    )


def _augment_frame(
    frame_data: dict[str, Any],
    longitudinal_offset: float,
    lateral_offset: float,
    yaw_offset: float,
    ego_speed_scale: float,
) -> dict[str, Any]:
    speed_augmentation = PlannerSpeedAugmentation(
        speed_scale_range=(ego_speed_scale, ego_speed_scale),
        speed_noise_range=(0.0, 0.0),
        probability=1.0,
    )
    pose_augmentation = PlannerQuinticHermiteAugmentation(
        longitudinal_offset_range=(longitudinal_offset, longitudinal_offset),
        lateral_offset_range=(lateral_offset, lateral_offset),
        yaw_offset_range=(yaw_offset, yaw_offset),
        pose_probability=1.0,
    )
    return pose_augmentation(speed_augmentation(frame_data))


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
    longitudinal_offset, lateral_offset, yaw_offset, ego_speed_scale = (
        _render_augmentation_settings()
    )
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
        augmented = _augment_frame(
            original,
            longitudinal_offset,
            lateral_offset,
            yaw_offset,
            ego_speed_scale,
        )
    except Exception as error:
        st.exception(error)
        return

    yaw_offset_degrees = math.degrees(yaw_offset)
    st.caption(
        f"Applied fixed longitudinal offset {longitudinal_offset:.2f} m, "
        f"lateral offset {lateral_offset:.2f} m, and yaw offset "
        f"{yaw_offset_degrees:.2f} deg, and scaled ego history speed by "
        f"{ego_speed_scale:.2f} with probability 1.0."
    )
    original_column, augmented_column = st.columns(2)
    chart_identity = (
        f"{index.path}::{row.index}::{longitudinal_offset}::{lateral_offset}::"
        f"{yaw_offset_degrees}::{ego_speed_scale}"
    )
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

    inspection = inspect_augmentation(original, augmented)
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
