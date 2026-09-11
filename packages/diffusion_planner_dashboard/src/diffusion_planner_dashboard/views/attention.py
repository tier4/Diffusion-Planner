"""Scene editor and fusion-attention overlay.

Place an agent the recording never contained, then read what the model attends
to with it present. The two belong on one page: the reason to insert a genuinely
unknown-labelled object is to find out whether the model looks at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from diffusion_planner.analysis import (
    AGENT_CLASS_NAMES,
    UNLABELED_NAME,
    fusion_dimensions,
)
from diffusion_planner.analysis.scene_edit import (
    PlacedAgent,
    edited_slots,
    free_slots,
    insert_agent,
    occupied_slots,
    read_agent,
)
from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    LoadedPlanner,
    load_frame_index,
    load_planner_checkpoint,
)
from diffusion_planner_dashboard.services.attention import (
    AttentionReading,
    run_attention,
)
from diffusion_planner_dashboard.ui.settings import render_frame_selector

_PLACED_KEY = "attention_placed_agents"
_LAYER_MEAN = "mean"
_LAYER_LAST = "last"


@st.cache_data(show_spinner=False)
def _cached_index(path: str, modification_time_ns: int) -> FrameIndex:
    del modification_time_ns  # Part of the cache key so a changed file reloads.
    return load_frame_index(path)


@st.cache_resource
def _frame_loader() -> FrameLoader:
    return FrameLoader()


@st.cache_data(max_entries=64, show_spinner="Reading frame data from H5...")
def _cached_frame(
    h5_path: str, frame_index: int, frame_time_ns: int, modification_time_ns: int
) -> dict[str, Any]:
    del modification_time_ns
    row = FrameIndexRow(0, h5_path, frame_index, frame_time_ns, {})
    return _frame_loader().load(row)


@st.cache_resource(show_spinner="Loading checkpoint...")
def _cached_planner(
    model_path: str, modification_time_ns: int, device: str
) -> LoadedPlanner:
    del modification_time_ns
    return load_planner_checkpoint(model_path, device)


def _render_sources() -> tuple[str | None, str | None]:
    """Frame source and checkpoint, shared with the other views' session keys."""
    st.sidebar.subheader("Sources")
    with st.sidebar.form("attention-source-settings"):
        source_candidate = st.text_input(
            "H5 or Parquet file",
            value=st.session_state.get("configured_frame_source_path", ""),
            placeholder="/path/to/train.parquet",
        )
        checkpoint_candidate = st.text_input(
            "Planner checkpoint",
            value=st.session_state.get("configured_checkpoint_path", ""),
            placeholder="/path/to/epoch_0001.pth",
        )
        applied = st.form_submit_button("Apply sources", use_container_width=True)
    if applied:
        assert source_candidate is not None
        assert checkpoint_candidate is not None
        st.session_state["configured_frame_source_path"] = source_candidate.strip()
        st.session_state["configured_checkpoint_path"] = checkpoint_candidate.strip()
    return (
        st.session_state.get("configured_frame_source_path") or None,
        st.session_state.get("configured_checkpoint_path") or None,
    )


def _render_attention_settings(
    layer_count: int, head_count: int
) -> tuple[int | str, int | None, float]:
    """Layer, head and threshold controls."""
    st.sidebar.subheader("Attention")
    layer_options: list[int | str] = [_LAYER_MEAN, _LAYER_LAST, *range(layer_count)]
    layer = st.sidebar.selectbox(
        "Fusion layer",
        layer_options,
        help="Average every layer, take the last, or inspect one.",
    )
    head_labels = ["mean", *(str(index) for index in range(head_count))]
    head_label = st.sidebar.selectbox(
        "Attention head",
        head_labels,
        help="Heads are captured separately, so a single head can be isolated.",
    )
    head = None if head_label == "mean" else int(head_label)
    # Expressed against the frame's top token rather than as an absolute share.
    # A scene spreads attention over hundreds of valid tokens, so absolute shares
    # sit well under one percent and a fixed 0-20% scale would be dead travel.
    threshold = float(
        st.sidebar.slider(
            "Hide tokens below (% of the top token)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )
    )
    return layer, head, threshold


def _render_editor(frame: dict[str, Any]) -> dict[str, Any]:
    """Add or remove agents, returning the edited frame."""
    placed: list[dict[str, Any]] = st.session_state.setdefault(_PLACED_KEY, [])

    st.sidebar.subheader("Place an agent")
    with st.sidebar.form("attention-place-agent"):
        agent_class = st.selectbox(
            "Class",
            [*AGENT_CLASS_NAMES, UNLABELED_NAME],
            index=len(AGENT_CLASS_NAMES) - 1,
        )
        x_m = st.number_input("x (m, ahead)", value=12.0, step=1.0)
        y_m = st.number_input("y (m, left)", value=2.0, step=0.5)
        heading_deg = st.number_input("heading (deg)", value=0.0, step=5.0)
        speed_mps = st.number_input("speed (m/s)", value=1.4, min_value=0.0, step=0.5)
        width_m = st.number_input("width (m)", value=0.8, min_value=0.1, step=0.1)
        length_m = st.number_input("length (m)", value=0.8, min_value=0.1, step=0.1)
        add = st.form_submit_button("Add to scene", use_container_width=True)
    if add:
        placed.append(
            {
                "agent_class": str(agent_class),
                "x_m": float(x_m),
                "y_m": float(y_m),
                "heading_rad": float(np.deg2rad(float(heading_deg))),
                "speed_mps": float(speed_mps),
                "width_m": float(width_m),
                "length_m": float(length_m),
            }
        )

    if placed:
        st.sidebar.caption(f"{len(placed)} placed agent(s)")
        if st.sidebar.button("Remove all placed agents", use_container_width=True):
            placed.clear()

    edited = frame
    for specification in placed:
        try:
            edited, _ = insert_agent(edited, PlacedAgent(**specification))
        except ValueError as error:
            st.sidebar.error(str(error))
            break
    return edited


def _relative_cutoff(reading: AttentionReading, threshold_pct: float) -> float:
    """Absolute share below which a token is hidden, from a relative threshold."""
    if not reading.records:
        return 0.0
    top = max(record["attention_pct"] for record in reading.records)
    return top * threshold_pct / 100.0


def _attention_overlay(
    reading: AttentionReading, threshold: float, blocks: list[str], classes: list[str]
) -> go.Scatter | None:
    """Markers sized and coloured by attention share."""
    cutoff = _relative_cutoff(reading, threshold)
    selected = [
        record
        for record in reading.records
        if record["x_m"] is not None
        and record["attention_pct"] >= cutoff
        and record["block"] in blocks
        and (record["block"] != "neighbors" or record.get("agent_class", "") in classes)
    ]
    if not selected:
        return None
    shares = np.array([record["attention_pct"] for record in selected], dtype=float)
    largest = float(shares.max()) or 1.0
    labels = [
        f"{record['block']}[{record['block_index']}]"
        + (f" · {record['agent_class']}" if record["block"] == "neighbors" else "")
        + f"<br>{record['attention_pct']:.2f}% · {record['distance_m']:.1f} m"
        for record in selected
    ]
    return go.Scatter(
        x=[record["x_m"] for record in selected],
        y=[record["y_m"] for record in selected],
        mode="markers",
        marker={
            "size": 8.0 + 32.0 * shares / largest,
            "color": shares,
            "colorscale": "Inferno",
            "opacity": 0.75,
            "line": {"width": 1, "color": "rgba(20,20,20,0.6)"},
            "colorbar": {"title": "attention %"},
        },
        text=labels,
        hoverinfo="text",
        name="attention",
    )


def _render_summaries(reading: AttentionReading) -> None:
    """Per-block and per-agent-class tables."""
    left, right = st.columns(2)
    with left:
        st.caption("By token block")
        st.dataframe(
            [
                {
                    "block": name,
                    "tokens": int(values["tokens"]),
                    "share %": round(values["share"] * 100.0, 2),
                    "selectivity": round(values["selectivity"], 2),
                }
                for name, values in sorted(
                    reading.by_block().items(), key=lambda item: -item[1]["share"]
                )
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.caption("By agent class — selectivity of 1.0 is count-proportional")
        st.dataframe(
            [
                {
                    "class": name,
                    "tokens": int(values["tokens"]),
                    "share %": round(values["share"] * 100.0, 2),
                    "selectivity": round(values["selectivity"], 2),
                }
                for name, values in sorted(
                    reading.by_agent_class().items(), key=lambda item: -item[1]["share"]
                )
            ],
            width="stretch",
            hide_index=True,
        )


def render_attention() -> None:
    """Render the scene editor and the fusion-attention overlay."""
    st.title("Attention")
    source_path_text, checkpoint_text = _render_sources()
    if source_path_text is None or checkpoint_text is None:
        st.info("Configure a frame source and a planner checkpoint in the sidebar.")
        return

    source_path = Path(source_path_text).expanduser()
    try:
        index = _cached_index(str(source_path), source_path.stat().st_mtime_ns)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    row = render_frame_selector(index)
    try:
        frame = _cached_frame(
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            Path(row.h5_path).stat().st_mtime_ns,
        )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return

    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device = st.sidebar.selectbox("Device", devices)
    checkpoint_path = Path(checkpoint_text).expanduser()
    try:
        planner = _cached_planner(
            str(checkpoint_path), checkpoint_path.stat().st_mtime_ns, str(device)
        )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return

    edited = _render_editor(frame)
    added = edited_slots(frame, edited)
    layer_count, head_count = fusion_dimensions(planner.model)
    layer, head, threshold = _render_attention_settings(layer_count, head_count)
    try:
        reading = run_attention(
            planner.model, edited, device=str(device), layer=layer, head=head
        )
    except RuntimeError as error:
        st.error(str(error))
        return

    occupied = occupied_slots(edited)
    st.caption(
        f"epoch {planner.epoch} · step {planner.global_step:,} · "
        f"{len(occupied)} agents ({len(added)} placed) · "
        f"{len(free_slots(edited))} free slots · {reading.layer_count} fusion layers"
    )

    blocks = st.multiselect(
        "Token blocks",
        list(reading.layout.names),
        default=list(reading.layout.names),
    )
    classes = st.multiselect(
        "Agent classes",
        [*AGENT_CLASS_NAMES, UNLABELED_NAME],
        default=[*AGENT_CLASS_NAMES, UNLABELED_NAME],
    )

    figure = plot_frame(edited)
    overlay = _attention_overlay(reading, threshold, blocks, classes)
    if overlay is not None:
        figure.add_trace(overlay)
    chart_key = f"attention::{index.path}::{row.index}::{len(added)}"
    figure.update_layout(autosize=True, uirevision=chart_key)
    st.plotly_chart(
        figure,
        width="stretch",
        height=800,
        key=chart_key,
        config={"responsive": True, "scrollZoom": True},
    )

    _render_summaries(reading)

    if added.size:
        st.subheader("Placed agents")
        st.dataframe(
            [
                {
                    "slot": int(slot),
                    "class": getattr(read_agent(edited, int(slot)), "agent_class", "?"),
                    "attention %": round(
                        next(
                            (
                                record["attention_pct"]
                                for record in reading.records
                                if record["block"] == "neighbors"
                                and record["block_index"] == int(slot)
                            ),
                            float("nan"),
                        ),
                        3,
                    ),
                }
                for slot in added
            ],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Most-attended tokens")
    cutoff = _relative_cutoff(reading, threshold)
    visible = [
        record
        for record in reading.records
        if record["attention_pct"] >= cutoff and record["block"] in blocks
    ]
    st.dataframe(
        [
            {
                "token": record["token_index"],
                "block": record["block"],
                "index": record["block_index"],
                "class": record.get("agent_class", ""),
                "attention %": round(record["attention_pct"], 3),
                "distance m": (
                    None
                    if record["distance_m"] is None
                    else round(record["distance_m"], 1)
                ),
            }
            for record in visible[:100]
        ],
        width="stretch",
        hide_index=True,
    )


__all__ = ["render_attention"]
