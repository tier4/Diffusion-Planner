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
import math
import pickle
from pathlib import Path

import gradio as gr
import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.gui.scene_renderer import get_agent_info, render_scene_figure
from scene_search.map_canvas_js import build_map_canvas_js
from scene_search.map_renderer import MapRenderer, Viewport

MAP_CANVAS_JS = build_map_canvas_js(tool="rectangle_and_arrow")

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
        rotation_state = gr.State(value=0.0)  # map rotation angle (radians)
        ego_pose_state = gr.State(value=None)  # (x, y, heading_rad) or None

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
                gr.Markdown("### Ego Pose")
                with gr.Row():
                    ego_x = gr.Number(label="X", value=0, interactive=False)
                    ego_y = gr.Number(label="Y", value=0, interactive=False)
                ego_heading = gr.Number(label="Heading (deg)", value=0, interactive=False)
                ego_pose_info = gr.Markdown("Shift+drag to set ego pose (optional)")

                gr.Markdown("### Generation Parameters")
                n_neighbors = gr.Slider(0, 32, value=3, step=1, label="Neighbors")
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
                zoom_slider = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Scene zoom")
                agent_info = gr.Markdown("Generate a scene to see agent info")

                gr.Markdown("### Export")
                save_path = gr.Textbox(
                    label="Save path",
                    value=str(Path.cwd() / "generated_scene.pkl"),
                )
                save_btn = gr.Button("Save SceneContext", variant="secondary")
                save_status = gr.Markdown("")

                gr.Markdown("### Save Map Snippet")
                selection_name = gr.Textbox(label="Snippet name", value="my_snippet")
                snippets_dir = gr.Textbox(
                    label="Snippets directory",
                    value=str(Path.cwd() / ".map_snippets"),
                )
                save_sel_btn = gr.Button("Save Lanelet Snippet", variant="secondary")
                save_sel_status = gr.Markdown("")

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

        def on_canvas_click(evt: gr.EventData):
            evt_type = getattr(evt, "type", "rect")
            # All outputs: rect coords (4) + rect_info + rect_state + rotation +
            #              ego coords (3) + ego_pose_info + ego_pose_state
            # Use gr.update() for fields we don't change
            no_change = gr.update()

            if evt_type == "ego_pose":
                x, y = evt.x, evt.y
                heading_deg = evt.heading
                heading_rad = heading_deg * math.pi / 180
                info = f"Ego: ({x:.0f}, {y:.0f}) heading={heading_deg:.0f} deg"
                return (
                    no_change, no_change, no_change, no_change, no_change, no_change, no_change,
                    round(x, 1), round(y, 1), round(heading_deg, 1),
                    info, (x, y, heading_rad),
                )
            else:
                x1, y1 = evt.x1, evt.y1
                x2, y2 = evt.x2, evt.y2
                rot = getattr(evt, "rotation", 0.0) or 0.0
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                rot_deg = float(rot) * 180 / math.pi
                info = f"Selected: {w:.0f}m x {h:.0f}m (rot: {rot_deg:.0f} deg)"
                rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                return (
                    round(rect[0], 1), round(rect[1], 1),
                    round(rect[2], 1), round(rect[3], 1),
                    info, rect, float(rot),
                    no_change, no_change, no_change, no_change, no_change,
                )

        map_canvas.click(
            on_canvas_click,
            outputs=[
                rect_x1, rect_y1, rect_x2, rect_y2, rect_info, rect_state, rotation_state,
                ego_x, ego_y, ego_heading, ego_pose_info, ego_pose_state,
            ],
        )

        # Generate scene
        def on_generate(rect, nn, ms, mxs, sep, rlen, rot, zoom, ego_pose):
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
                    ego_pose=ego_pose,
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

            fig = render_scene_figure(scene, focus_agent_id=None, rotation=rot, zoom=zoom)
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
            inputs=[rect_state, n_neighbors, min_speed, max_speed, min_sep, route_len,
                    rotation_state, zoom_slider, ego_pose_state],
            outputs=[scene_state, gen_status, focus_dropdown, scene_plot, agent_info],
        )

        # Focus mode change
        def on_focus_change(focus_id, scene_pkl, rot, zoom):
            if scene_pkl is None:
                return gr.update(), "No scene generated"
            scene = pickle.loads(scene_pkl)
            fid = None if focus_id == "All Agents" else focus_id
            fig = render_scene_figure(scene, focus_agent_id=fid, rotation=rot, zoom=zoom)
            try:
                info = get_agent_info(scene, fid or "ego")
            except KeyError:
                info = get_agent_info(scene, "ego")
            return fig, info

        focus_dropdown.change(
            on_focus_change,
            inputs=[focus_dropdown, scene_state, rotation_state, zoom_slider],
            outputs=[scene_plot, agent_info],
        )

        # Re-render on zoom change
        zoom_slider.release(
            on_focus_change,
            inputs=[focus_dropdown, scene_state, rotation_state, zoom_slider],
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

        # Save map snippet
        def on_save_snippet(rect, ego_pose, name, snip_dir):
            if rect is None:
                return "Select an area first"

            ll_ids = builder.lanelets_in_rect(*rect)
            if not ll_ids:
                return "No lanelets in selection"

            out_dir = Path(snip_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            snippet = {
                "lanelet_ids": ll_ids,
                "rect": list(rect),
                "ego_pose": list(ego_pose) if ego_pose else None,
            }
            out_file = out_dir / f"{name}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(snippet, f)

            return f"Saved '{name}' ({len(ll_ids)} lanelets) to {out_file}"

        save_sel_btn.click(
            on_save_snippet,
            inputs=[rect_state, ego_pose_state, selection_name, snippets_dir],
            outputs=[save_sel_status],
        )

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
