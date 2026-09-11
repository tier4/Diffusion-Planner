"""Application shell for the diffusion planner dashboard."""

from __future__ import annotations

import streamlit as st

from diffusion_planner_dashboard.views import (
    render_attention,
    render_data_augmentation,
    render_frame_browser,
    render_home,
    render_training_results,
)


def main() -> None:
    """Configure Streamlit and run the selected dashboard feature."""
    st.set_page_config(page_title="Diffusion Planner Dashboard", layout="wide")
    navigation = st.navigation(
        {
            "Dashboard": [
                st.Page(
                    render_home, title="Home", icon=":material/home:", default=True
                ),
            ],
            "Features": [
                st.Page(
                    render_frame_browser,
                    title="Frame Browser",
                    icon=":material/animation:",
                ),
                st.Page(
                    render_training_results,
                    title="Training Results",
                    icon=":material/model_training:",
                ),
                st.Page(
                    render_data_augmentation,
                    title="Data Augmentation",
                    icon=":material/compare:",
                ),
                st.Page(
                    render_attention,
                    title="Attention",
                    icon=":material/visibility:",
                ),
            ],
        }
    )
    navigation.run()


main()
