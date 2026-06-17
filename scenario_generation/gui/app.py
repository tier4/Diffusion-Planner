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
from scenario_generation.route import Route
from scene_search.map_canvas_js import build_map_canvas_js
from scene_search.map_renderer import MapRenderer, Viewport


def _build_route_polyline_json(
    builder: LaneletSceneBuilder,
    route_ll_ids: list[int] | None,
) -> str:
    """Serialise a lanelet-id sequence to the JSON format consumed by
    ``window.__setRoute`` in the canvas JS: ``[[[x,y], ...], ...]`` — one
    centerline polyline per resolved lanelet."""
    if not route_ll_ids:
        return "[]"
    polylines = []
    for ll_id in route_ll_ids:
        if ll_id not in builder._cache:
            continue
        cl = builder._cache[ll_id].raw_centerline
        polylines.append(cl[:, :2].tolist())
    return json.dumps(polylines)


def _recompute_route(
    builder: LaneletSceneBuilder,
    ego_pose,
    goal_pose,
    waypoints,
) -> tuple[str, list[int] | None, str]:
    """Resolve the route for the current start / goal / waypoints.

    Goal and waypoints are snapped with ``reachable_from=start_id`` so the
    click never lands on a geometrically-close but topologically-disconnected
    sub-network (common at overpass crossings on Shinagawa where two road
    grids overlap but don't connect).

    Returns ``(polyline_json, route_ll_ids, status_message)``. Returns an empty
    polyline and ``None`` for ``route_ll_ids`` whenever the route cannot be
    computed (missing start/goal, unreachable, off-map).
    """
    if ego_pose is None or goal_pose is None:
        return "[]", None, ""
    start_id = builder.snap_to_nearest_ll(
        np.asarray(ego_pose[:2], dtype=np.float32),
        heading_rad=float(ego_pose[2]),
    )
    if start_id is None:
        return "[]", None, "Could not snap start to a routable lanelet"
    goal_id = builder.snap_to_nearest_ll(
        np.asarray(goal_pose[:2], dtype=np.float32),
        reachable_from=start_id,
        heading_rad=float(goal_pose[2]),
    )
    if goal_id is None:
        return (
            "[]",
            None,
            (
                f"No routable lanelet near the goal is reachable from start "
                f"({start_id}). Try clicking somewhere else or add a waypoint."
            ),
        )
    via_ids: list[int] = []
    prev_id = start_id
    for i, wp in enumerate(waypoints or []):
        vid = builder.snap_to_nearest_ll(
            np.asarray(wp[:2], dtype=np.float32),
            reachable_from=prev_id,
            heading_rad=float(wp[2]),
        )
        if vid is None:
            return (
                "[]",
                None,
                (f"Waypoint #{i + 1} has no routable lanelet reachable from the previous point."),
            )
        via_ids.append(vid)
        prev_id = vid
    route_ids = builder.route_with_waypoints(start_id, via_ids, goal_id)
    if route_ids is None:
        return (
            "[]",
            None,
            (
                f"Routing failed: no path from start ({start_id}) "
                f"through {via_ids} to goal ({goal_id})"
            ),
        )
    poly_json = _build_route_polyline_json(builder, route_ids)

    # Diagnostic: how far did each click move to reach its snapped lanelet?
    # Large snap distances indicate the click landed on an orphan / wrong-
    # direction / non-routable lanelet and the snap had to walk away.
    def _snap_dist(xy, ll_id):
        cl = builder._cache[ll_id].raw_centerline
        return float(np.linalg.norm(cl - np.asarray(xy, dtype=np.float32), axis=1).min())

    ds = _snap_dist(ego_pose[:2], start_id)
    dg = _snap_dist(goal_pose[:2], goal_id)
    snap_note = ""
    if ds > 10.0 or dg > 10.0:
        snap_note = (
            f" (snap: start {ds:.0f} m, goal {dg:.0f} m — click may be on "
            f"an orphan / wrong-direction lanelet)"
        )
    return poly_json, route_ids, f"Route: {len(route_ids)} lanelets{snap_note}"


_ARROW_LEN_M = 6.0


def _snapped_arrow_json(
    builder: LaneletSceneBuilder,
    pose: tuple | None,
    reachable_from: int | None = None,
) -> tuple[str, int | None]:
    """Snap a pose to a lanelet and return ``(arrow_json, snapped_ll_id)``.

    The returned JSON matches the JS ``window.__setStartArrow`` /
    ``__setGoalArrow`` format: ``{"start": [x, y], "end": [x + L*cos(h),
    y + L*sin(h)]}`` with the start placed at the closest point on the
    snapped lanelet's centerline (not the raw click xy) and the tip offset
    ``_ARROW_LEN_M`` metres along the provided heading. This closes the
    visual gap between the click position and where the route polyline
    actually starts / ends.

    Returns ``("null", None)`` when ``pose`` is ``None`` or no routable
    lanelet is available. The JS setters interpret ``"null"`` as "erase".
    """
    if pose is None:
        return "null", None
    x, y, h = float(pose[0]), float(pose[1]), float(pose[2])
    ll_id = builder.snap_to_nearest_ll(
        np.asarray([x, y], dtype=np.float32),
        reachable_from=reachable_from,
        heading_rad=h,
    )
    if ll_id is None:
        return "null", None
    cl = builder._cache[ll_id].raw_centerline
    dists = np.linalg.norm(cl - np.array([x, y], dtype=np.float32), axis=1)
    closest = int(np.argmin(dists))
    sx, sy = float(cl[closest, 0]), float(cl[closest, 1])
    ex = sx + _ARROW_LEN_M * math.cos(h)
    ey = sy + _ARROW_LEN_M * math.sin(h)
    return json.dumps({"start": [sx, sy], "end": [ex, ey]}), ll_id


def _waypoint_arrows_json(
    builder: LaneletSceneBuilder,
    waypoints: list,
    start_id: int | None,
) -> str:
    """Return the ``window.__setWaypointArrows`` JSON for the current
    waypoints list, each snapped to its reachable lanelet (sequentially
    reachable from the previous point, starting with ``start_id``)."""
    arrows = []
    prev = start_id
    for wp in waypoints or []:
        x, y, h = float(wp[0]), float(wp[1]), float(wp[2])
        ll_id = builder.snap_to_nearest_ll(
            np.asarray([x, y], dtype=np.float32),
            reachable_from=prev,
            heading_rad=h,
        )
        if ll_id is None:
            # Fall back to the raw click pose so the user still sees something.
            sx, sy = x, y
        else:
            cl = builder._cache[ll_id].raw_centerline
            dists = np.linalg.norm(cl - np.array([x, y], dtype=np.float32), axis=1)
            closest = int(np.argmin(dists))
            sx, sy = float(cl[closest, 0]), float(cl[closest, 1])
            prev = ll_id
        ex = sx + _ARROW_LEN_M * math.cos(h)
        ey = sy + _ARROW_LEN_M * math.sin(h)
        arrows.append({"start": [sx, sy], "end": [ex, ey]})
    return json.dumps(arrows)


def _format_waypoints_md(waypoints: list, builder: LaneletSceneBuilder) -> str:
    """Render the waypoints list as a compact markdown block with snapped IDs."""
    if not waypoints:
        return "*No waypoints. Set Mode = **Add Waypoint** and drag on the map to add one.*"
    lines = ["| # | lanelet | x | y | heading |", "|---|---|---|---|---|"]
    for i, wp in enumerate(waypoints, 1):
        x, y, h = wp
        ll = builder.snap_to_nearest_ll(
            np.asarray([x, y], dtype=np.float32),
            heading_rad=float(h),
        )
        ll_s = str(ll) if ll is not None else "?"
        lines.append(f"| {i} | {ll_s} | {x:.1f} | {y:.1f} | {math.degrees(h):.0f}° |")
    return "\n".join(lines)


MAP_CANVAS_JS = build_map_canvas_js(tool="rectangle_and_arrow")


def build_interface(
    renderer: MapRenderer,
    builder: LaneletSceneBuilder,
    map_path_arg: str,
):
    """Build the complete Gradio interface.

    Args:
        renderer: Map tile renderer (base PNGs for the canvas).
        builder: Lanelet scene builder used for routing + snapping.
        map_path_arg: Absolute path to the lanelet2 ``.osm`` file. Persisted
            into saved ``Route`` pickles so replay can re-open the same map.
    """

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
        rect_state = gr.State(value=None)  # (x1, y1, x2, y2)
        rotation_state = gr.State(value=0.0)  # map rotation angle (radians)
        ego_pose_state = gr.State(value=None)  # (x, y, heading_rad) or None
        goal_pose_state = gr.State(value=None)  # (x, y, heading_rad) or None
        waypoints_state = gr.State(value=[])  # list[(x, y, heading_rad)] — user order
        route_ids_state = gr.State(value=None)  # list[int] resolved lanelet path

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
                gr.Markdown("### Mode")
                mode_radio = gr.Radio(
                    choices=[
                        ("Pan", "pan"),
                        ("Set Start", "start"),
                        ("Set Goal", "goal"),
                        ("Add Waypoint", "waypoint"),
                    ],
                    value="pan",
                    label="Click mode",
                    info=(
                        "Pan = drag scrolls the map. Pick a mode, then plain "
                        "drag on the map to place it. Ctrl+drag always = "
                        "rectangle select (for map snippets). Shift+drag is "
                        "a shortcut for Set Start. 'Add Waypoint' keeps "
                        "appending waypoints until you switch mode."
                    ),
                )

                gr.Markdown("### Ego Start Pose")
                with gr.Row():
                    ego_x = gr.Number(label="X", value=0, interactive=False)
                    ego_y = gr.Number(label="Y", value=0, interactive=False)
                ego_heading = gr.Number(label="Heading (deg)", value=0, interactive=False)
                ego_pose_info = gr.Markdown(
                    "Select mode **Set Start** (or Shift+drag), then drag on the map."
                )
                clear_start_btn = gr.Button("Clear Start", variant="secondary", size="sm")

                gr.Markdown("### Ego Goal Pose")
                with gr.Row():
                    goal_x = gr.Number(label="X", value=0, interactive=False)
                    goal_y = gr.Number(label="Y", value=0, interactive=False)
                goal_heading = gr.Number(label="Heading (deg)", value=0, interactive=False)
                goal_pose_info = gr.Markdown("Select mode **Set Goal**, then drag on the map.")
                clear_goal_btn = gr.Button("Clear Goal", variant="secondary", size="sm")

                gr.Markdown("### Waypoints")
                gr.Markdown(
                    "Select mode **Add Waypoint**, then drag on the map to "
                    "append one. The resolved route is forced through them "
                    "in drop order. Stays in waypoint mode until you switch."
                )
                waypoints_display = gr.Markdown(
                    "*No waypoints. Pick 'Add Waypoint' mode and drag on the map.*"
                )
                clear_waypoints_btn = gr.Button("Clear Waypoints", variant="secondary", size="sm")

                gr.Markdown("### Route")
                route_status = gr.Markdown("*Set start + goal to resolve a route.*")
                # Hidden textboxes carrying JSON state for the JS canvas
                # (route polyline + arrow positions after snapping).
                route_polyline_json = gr.Textbox(value="[]", visible=False)
                start_arrow_json = gr.Textbox(value="null", visible=False)
                goal_arrow_json = gr.Textbox(value="null", visible=False)
                waypoint_arrows_json = gr.Textbox(value="[]", visible=False)

                gr.Markdown("### Save Route")
                route_save_path = gr.Textbox(
                    label="Save path",
                    value=str(Path.cwd() / "my_route.pkl"),
                )
                save_route_btn = gr.Button("Save Route", variant="primary")
                save_route_status = gr.Markdown("")

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
                    choices=["All Agents"],
                    value="All Agents",
                    label="Focus agent",
                    interactive=True,
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

        def on_canvas_click(evt: gr.EventData, current_waypoints):
            """Handle a canvas click. Dispatches on evt.type:

            * ``rect``                 → Ctrl+drag rectangle selection
            * ``ego_pose``             → Set-Start mode or Shift+drag
            * ``goal_pose``            → Set-Goal mode
            * ``waypoint_append``      → Add-Waypoint mode

            After placing a start or goal the mode is reset to ``pan``.
            Waypoint mode stays active so multiple waypoints can be placed
            in sequence. For rect events the mode is left untouched.
            """
            evt_type = getattr(evt, "type", "rect")
            no_change = gr.update()
            reset_mode = gr.update(value="pan")
            # Output order:
            #   rect_x1, rect_y1, rect_x2, rect_y2, rect_info, rect_state, rotation_state,
            #   ego_x, ego_y, ego_heading, ego_pose_info, ego_pose_state,
            #   goal_x, goal_y, goal_heading, goal_pose_info, goal_pose_state,
            #   waypoints_state, waypoints_display,
            #   mode_radio

            if evt_type == "ego_pose":
                x, y = evt.x, evt.y
                heading_deg = evt.heading
                heading_rad = heading_deg * math.pi / 180
                info = f"Start: ({x:.0f}, {y:.0f}) heading={heading_deg:.0f}°"
                return (
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    round(x, 1),
                    round(y, 1),
                    round(heading_deg, 1),
                    info,
                    (x, y, heading_rad),
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    reset_mode,
                )
            elif evt_type == "goal_pose":
                x, y = evt.x, evt.y
                heading_deg = evt.heading
                heading_rad = heading_deg * math.pi / 180
                info = f"Goal: ({x:.0f}, {y:.0f}) heading={heading_deg:.0f}°"
                return (
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    round(x, 1),
                    round(y, 1),
                    round(heading_deg, 1),
                    info,
                    (x, y, heading_rad),
                    no_change,
                    no_change,
                    reset_mode,
                )
            elif evt_type == "waypoint_append":
                x, y = evt.x, evt.y
                heading_rad = evt.heading * math.pi / 180
                new_waypoints = list(current_waypoints or []) + [(x, y, heading_rad)]
                return (
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    new_waypoints,
                    _format_waypoints_md(new_waypoints, builder),
                    no_change,
                )
            else:  # rect
                x1, y1 = evt.x1, evt.y1
                x2, y2 = evt.x2, evt.y2
                rot = getattr(evt, "rotation", 0.0) or 0.0
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                rot_deg = float(rot) * 180 / math.pi
                info = f"Selected: {w:.0f}m x {h:.0f}m (rot: {rot_deg:.0f}°)"
                rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                return (
                    round(rect[0], 1),
                    round(rect[1], 1),
                    round(rect[2], 1),
                    round(rect[3], 1),
                    info,
                    rect,
                    float(rot),
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                    no_change,
                )

        def refresh_route_and_arrows(ego_pose, goal_pose, waypoints):
            """Recompute the route polyline + snapped arrow JSONs for the
            current start / goal / waypoints triple.

            Returns ``(route_polyline_json, start_arrow_json, goal_arrow_json,
            waypoint_arrows_json, route_ids, status_message)``. The arrow
            JSONs carry the *snapped* positions so the canvas viz sits on
            the routable lanelet centerline, matching where the green route
            polyline begins and ends — no visual gap to the raw click xy.
            """
            poly_json, route_ids, msg = _recompute_route(
                builder,
                ego_pose,
                goal_pose,
                waypoints,
            )
            # Snap the three arrow sets. Start has no reachability constraint;
            # goal and waypoints use reachable_from for consistency with the
            # same snap logic used inside _recompute_route.
            start_json, start_id = _snapped_arrow_json(builder, ego_pose)
            goal_json, _ = _snapped_arrow_json(
                builder,
                goal_pose,
                reachable_from=start_id,
            )
            wp_json = _waypoint_arrows_json(builder, waypoints, start_id)
            return (
                poly_json,
                start_json,
                goal_json,
                wp_json,
                route_ids,
                msg or "*Set start + goal to resolve a route.*",
            )

        # JS hooks that read the hidden textboxes and call the canvas setters.
        PUSH_ROUTE_JS = "(j) => { if (window.__setRoute) { window.__setRoute(j); } return []; }"
        PUSH_START_JS = (
            "(j) => { if (window.__setStartArrow) { window.__setStartArrow(j); } return []; }"
        )
        PUSH_GOAL_JS = (
            "(j) => { if (window.__setGoalArrow) { window.__setGoalArrow(j); } return []; }"
        )
        PUSH_WAYPOINTS_JS = "(j) => { if (window.__setWaypointArrows) { window.__setWaypointArrows(j); } return []; }"
        PUSH_MODE_JS = "(m) => { if (window.__setMode) { window.__setMode(m); } return []; }"

        # Push the mode to the canvas JS whenever the radio changes.
        mode_radio.change(None, inputs=[mode_radio], outputs=[], js=PUSH_MODE_JS)

        map_canvas.click(
            on_canvas_click,
            inputs=[waypoints_state],
            outputs=[
                rect_x1,
                rect_y1,
                rect_x2,
                rect_y2,
                rect_info,
                rect_state,
                rotation_state,
                ego_x,
                ego_y,
                ego_heading,
                ego_pose_info,
                ego_pose_state,
                goal_x,
                goal_y,
                goal_heading,
                goal_pose_info,
                goal_pose_state,
                waypoints_state,
                waypoints_display,
                mode_radio,
            ],
        ).then(
            refresh_route_and_arrows,
            inputs=[ego_pose_state, goal_pose_state, waypoints_state],
            outputs=[
                route_polyline_json,
                start_arrow_json,
                goal_arrow_json,
                waypoint_arrows_json,
                route_ids_state,
                route_status,
            ],
        ).then(None, inputs=[route_polyline_json], outputs=[], js=PUSH_ROUTE_JS).then(
            None, inputs=[start_arrow_json], outputs=[], js=PUSH_START_JS
        ).then(None, inputs=[goal_arrow_json], outputs=[], js=PUSH_GOAL_JS).then(
            None, inputs=[waypoint_arrows_json], outputs=[], js=PUSH_WAYPOINTS_JS
        )

        def on_clear_waypoints():
            return [], _format_waypoints_md([], builder)

        clear_waypoints_btn.click(
            on_clear_waypoints,
            outputs=[waypoints_state, waypoints_display],
        ).then(
            refresh_route_and_arrows,
            inputs=[ego_pose_state, goal_pose_state, waypoints_state],
            outputs=[
                route_polyline_json,
                start_arrow_json,
                goal_arrow_json,
                waypoint_arrows_json,
                route_ids_state,
                route_status,
            ],
        ).then(None, inputs=[route_polyline_json], outputs=[], js=PUSH_ROUTE_JS).then(
            None, inputs=[start_arrow_json], outputs=[], js=PUSH_START_JS
        ).then(None, inputs=[goal_arrow_json], outputs=[], js=PUSH_GOAL_JS).then(
            None, inputs=[waypoint_arrows_json], outputs=[], js=PUSH_WAYPOINTS_JS
        )

        def on_clear_start():
            return None, 0, 0, 0, "Select mode **Set Start** (or Shift+drag), then drag on the map."

        clear_start_btn.click(
            on_clear_start,
            outputs=[ego_pose_state, ego_x, ego_y, ego_heading, ego_pose_info],
        ).then(
            refresh_route_and_arrows,
            inputs=[ego_pose_state, goal_pose_state, waypoints_state],
            outputs=[
                route_polyline_json,
                start_arrow_json,
                goal_arrow_json,
                waypoint_arrows_json,
                route_ids_state,
                route_status,
            ],
        ).then(None, inputs=[route_polyline_json], outputs=[], js=PUSH_ROUTE_JS).then(
            None, inputs=[start_arrow_json], outputs=[], js=PUSH_START_JS
        ).then(None, inputs=[goal_arrow_json], outputs=[], js=PUSH_GOAL_JS).then(
            None, inputs=[waypoint_arrows_json], outputs=[], js=PUSH_WAYPOINTS_JS
        )

        def on_clear_goal():
            return None, 0, 0, 0, "Select mode **Set Goal**, then drag on the map."

        clear_goal_btn.click(
            on_clear_goal,
            outputs=[goal_pose_state, goal_x, goal_y, goal_heading, goal_pose_info],
        ).then(
            refresh_route_and_arrows,
            inputs=[ego_pose_state, goal_pose_state, waypoints_state],
            outputs=[
                route_polyline_json,
                start_arrow_json,
                goal_arrow_json,
                waypoint_arrows_json,
                route_ids_state,
                route_status,
            ],
        ).then(None, inputs=[route_polyline_json], outputs=[], js=PUSH_ROUTE_JS).then(
            None, inputs=[start_arrow_json], outputs=[], js=PUSH_START_JS
        ).then(None, inputs=[goal_arrow_json], outputs=[], js=PUSH_GOAL_JS).then(
            None, inputs=[waypoint_arrows_json], outputs=[], js=PUSH_WAYPOINTS_JS
        )

        def on_save_route(ego_pose, goal_pose, waypoints, path):
            """Persist the current route spec to ``path`` as a pickled ``Route``.

            Computes a fresh resolved path via ``shortestPathWithVia`` so the
            saved Route carries ``route_lanelet_ids`` without needing a re-query
            at replay time.
            """
            if ego_pose is None or goal_pose is None:
                return "Both start and goal are required"

            start_pose = np.array(ego_pose, dtype=np.float32)
            goal_pose_arr = np.array(goal_pose, dtype=np.float32)

            start_id = builder.snap_to_nearest_ll(
                start_pose[:2],
                heading_rad=float(start_pose[2]),
            )
            if start_id is None:
                return "Could not snap start to a routable lanelet"
            goal_id = builder.snap_to_nearest_ll(
                goal_pose_arr[:2],
                reachable_from=start_id,
                heading_rad=float(goal_pose_arr[2]),
            )
            if goal_id is None:
                return (
                    f"No routable lanelet near the goal is reachable from "
                    f"start ({start_id}). Try a different goal or add a waypoint."
                )

            waypoint_poses = [np.array(w, dtype=np.float32) for w in (waypoints or [])]
            waypoint_ids: list[int] = []
            prev_id = start_id
            for i, wp in enumerate(waypoint_poses):
                wid = builder.snap_to_nearest_ll(
                    wp[:2],
                    reachable_from=prev_id,
                    heading_rad=float(wp[2]),
                )
                if wid is None:
                    return (
                        f"Waypoint #{i + 1} has no routable lanelet "
                        f"reachable from the previous point."
                    )
                waypoint_ids.append(wid)
                prev_id = wid

            route_ids = builder.route_with_waypoints(start_id, waypoint_ids, goal_id)
            if route_ids is None:
                return (
                    f"Routing failed: no path from start ({start_id}) "
                    f"through {waypoint_ids} to goal ({goal_id})"
                )
            if route_ids[-1] != goal_id:
                # shortestPathWithVia can land on a lane-change-adjacent lanelet;
                # surface this loudly so users notice before running replay.
                return (
                    f"Routing warning: resolved path ends at {route_ids[-1]}, "
                    f"not goal lanelet {goal_id}. Adjust start/waypoints/goal."
                )

            route = Route(
                map_path=str(map_path_arg),
                start_pose=start_pose,
                goal_pose=goal_pose_arr,
                start_lanelet_id=start_id,
                goal_lanelet_id=goal_id,
                waypoint_poses=waypoint_poses,
                waypoint_lanelet_ids=waypoint_ids,
                route_lanelet_ids=route_ids,
            )
            route.save(path)
            return f"Saved to {path} — {len(route_ids)} lanelets, {len(waypoint_ids)} waypoints"

        save_route_btn.click(
            on_save_route,
            inputs=[ego_pose_state, goal_pose_state, waypoints_state, route_save_path],
            outputs=[save_route_status],
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
            status = (
                f"Generated ego + {n_placed} neighbors, {n_lanes} lanes ({rw:.0f}x{rh:.0f}m area)"
            )

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
            inputs=[
                rect_state,
                n_neighbors,
                min_speed,
                max_speed,
                min_sep,
                route_len,
                rotation_state,
                zoom_slider,
                ego_pose_state,
            ],
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

    demo = build_interface(map_renderer, scene_builder, args.map_path)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
