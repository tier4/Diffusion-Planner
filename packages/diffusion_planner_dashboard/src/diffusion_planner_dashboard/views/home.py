"""Dashboard feature-selection home view."""

from __future__ import annotations

import streamlit as st


def render_home() -> None:
    """Render the dashboard landing page."""
    st.title("Diffusion Planner Dashboard")
    st.write("Select a feature from the sidebar.")

    with st.container(border=True):
        st.subheader("Frame Browser")
        st.write("Browse Parquet frame indices and visualize model-ready frame data.")
        st.caption("Open Frame Browser from the Features section in the sidebar.")

    with st.container(border=True):
        st.subheader("Training Results")
        st.write(
            "Run checkpoint or ONNX inference and compare predicted trajectories "
            "with the dataset labels."
        )
        st.caption("Open Training Results from the Features section in the sidebar.")

    with st.container(border=True):
        st.subheader("Data Augmentation")
        st.write(
            "Inspect pose and speed augmentation effects on model-ready frame data."
        )
        st.caption("Open Data Augmentation from the Features section in the sidebar.")
