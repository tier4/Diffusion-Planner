"""Scenario Generation GUI -- interactive scene creation on Lanelet2 maps.

Launch:
    source /opt/ros/humble/setup.bash
    source ~/autoware/install/setup.bash
    source .venv/bin/activate
    python -m scenario_generation.gui \
        --map_path /path/to/lanelet2_map.osm \
        [--port 7860]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.gui.scene_renderer import get_agent_info, render_scene_figure
from scene_search.map_canvas_js import build_map_canvas_js
from scene_search.map_renderer import MapRenderer, Viewport

MAP_CANVAS_JS = build_map_canvas_js(tool="rectangle")

def build_interface(
    renderer: MapRenderer,
    builder: LaneletSceneBuilder,
):
    """Build the complete Gradio interface."""

    full_vp = renderer.initial_viewport(canvas_w=4000, canvas_h=3200)
    print("Pre-rendering full map image...")
    full_map_b64 = renderer.render_viewport_base64(full_vp, dpi=100)
    full_bounds = full_vp.to_json()
    print(f"  Map image: {len(full_map_b64) // 1024}KB base64")

    def render_map(viewport_json: str) -> str:
        vp = Viewport.from_json(json.loads(viewport_json))
        return renderer.render_viewport_base64(vp, dpi=100)

    with gr.Blocks(title="Scenario Generation") as demo:

        scene_state = gr.State(value=None)  # pickled SceneContext
        rect_state = gr.State(value=None)   # (x1, y1, x2, y2)

        gr.Markdown("# Scenario Generation")

        with gr.Row():
            # ===================== LEFT SIDEBAR =====================
            with gr.Column(scale=1):
                gr.Markdown("### Map Selection")
                with gr.Row():
                    rect_x1 = gr.Number(label="X1", value=0, interactive=False)
                    rect_y1 = gr.Number(label="Y1", value=0, interactive=False)
                with gr.Row():
                    rect_x2 = gr.Number(label="X2", value=0, interactive=False)
                    rect_y2 = gr.Number(label="Y2", value=0, interactive=False)
                rect_info = gr.Markdown("Ctrl+drag on map to select area")

                gr.Markdown("### Generation Parameters")
                n_neighbors = gr.Slider(0, 10, value=3, step=1, label="Neighbors")
                min_speed = gr.Slider(0, 20, value=3, step=0.5, label="Min speed (m/s)")
                max_speed = gr.Slider(1, 25, value=12, step=0.5, label="Max speed (m/s)")
                min_sep = gr.Slider(3, 30, value=8, step=1, label="Min separation (m)")
                route_len = gr.Slider(20, 300, value=120, step=10, label="Route length (m)")
                gen_btn = gr.Button("Generate", variant="primary")
                gen_status = gr.Markdown("")

                gr.Markdown("### Focus Mode")
                focus_dropdown = gr.Dropdown(
                    choices=["All Agents"], value="All Agents",
                    label="Focus agent", interactive=True,
                )
                agent_info = gr.Markdown("Generate a scene to see agent info")

                gr.Markdown("### Export")
                save_path = gr.Textbox(
                    label="Save path",
                    value=str(Path.cwd() / "generated_scene.pkl"),
                )
                save_btn = gr.Button("Save SceneContext", variant="secondary")
                save_status = gr.Markdown("")

            # ===================== MAIN CONTENT =====================
            with gr.Column(scale=3):
                gr.Markdown("### Map (Ctrl+drag to select rectangle)")
                map_canvas = gr.HTML(
                    value="",
                    js_on_load=MAP_CANVAS_JS,
                    server_functions=[render_map],
                    min_height=750,
                    map_b64=full_map_b64,
                    map_bounds=json.dumps(full_bounds),
                )

                gr.Markdown("### Generated Scene")
                scene_plot = gr.Plot(label="Scene View", elem_classes=["scene-plot"])

        # ===================== EVENT HANDLERS =====================

        def on_rect_placed(evt: gr.EventData):
            x1, y1 = evt.x1, evt.y1
            x2, y2 = evt.x2, evt.y2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            info = f"Selected: {w:.0f}m x {h:.0f}m"
            rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            return (
                round(rect[0], 1), round(rect[1], 1),
                round(rect[2], 1), round(rect[3], 1),
                info, rect,
            )

        map_canvas.click(
            on_rect_placed,
            outputs=[rect_x1, rect_y1, rect_x2, rect_y2, rect_info, rect_state],
        )

        # Generate scene
        def on_generate(rect, nn, ms, mxs, sep, rlen):
            if rect is None:
                return (
                    None,
                    "Select an area first (Ctrl+drag on map)",
                    gr.update(),
                    gr.update(),
                    "",
                )

            try:
                scene = builder.build_scene_context(
                    rect=rect,
                    n_neighbors=int(nn),
                    min_separation_m=sep,
                    min_speed=ms,
                    max_speed=mxs,
                    route_length_m=rlen,
                )
            except ValueError as e:
                return (
                    None,
                    f"Generation failed: {e}",
                    gr.update(),
                    gr.update(),
                    "",
                )

            scene_pkl = pickle.dumps(scene)
            agent_ids = ["All Agents"] + [a.id for a in scene.agents]
            n_lanes = int((np.abs(scene.map_data.lanes[:, :, :2]).sum(axis=(1, 2)) > 1e-6).sum())
            n_placed = len(scene.agents) - 1
            rw = abs(rect[2] - rect[0])
            rh = abs(rect[3] - rect[1])
            status = f"Generated ego + {n_placed} neighbors, {n_lanes} lanes ({rw:.0f}x{rh:.0f}m area)"

            fig = render_scene_figure(scene, focus_agent_id=None)
            info = get_agent_info(scene, "ego")

            return (
                scene_pkl,
                status,
                gr.update(choices=agent_ids, value="All Agents"),
                fig,
                info,
            )

        gen_btn.click(
            on_generate,
            inputs=[rect_state, n_neighbors, min_speed, max_speed, min_sep, route_len],
            outputs=[scene_state, gen_status, focus_dropdown, scene_plot, agent_info],
        )

        # Focus mode change
        def on_focus_change(focus_id, scene_pkl):
            if scene_pkl is None:
                return gr.update(), "No scene generated"
            scene = pickle.loads(scene_pkl)
            fid = None if focus_id == "All Agents" else focus_id
            fig = render_scene_figure(scene, focus_agent_id=fid)
            try:
                info = get_agent_info(scene, fid or "ego")
            except KeyError:
                info = get_agent_info(scene, "ego")
            return fig, info

        focus_dropdown.change(
            on_focus_change,
            inputs=[focus_dropdown, scene_state],
            outputs=[scene_plot, agent_info],
        )

        # Save
        def on_save(scene_pkl, path):
            if scene_pkl is None:
                return "No scene to save"
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                f.write(scene_pkl)
            return f"Saved to {p}"

        save_btn.click(on_save, inputs=[scene_state, save_path], outputs=[save_status])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Scenario Generation GUI")
    parser.add_argument("--map_path", type=str, required=True, help="Path to lanelet2_map.osm")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Loading map...")
    map_renderer = MapRenderer(args.map_path)
    print("Building scene builder...")
    scene_builder = LaneletSceneBuilder(args.map_path)
    print("Starting GUI...")

    demo = build_interface(map_renderer, scene_builder)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
