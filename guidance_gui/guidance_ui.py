"""Shared guidance UI panel for Diffusion-Planner Gradio applications.

Provides a single source of truth for the guidance controls (checkboxes, sliders,
prototype gallery) that are shared across the DPO annotation GUI, the Guidance
Playground, and any future RLVR/GRPO training interfaces.

Usage
-----
Within a ``gr.Blocks`` context, call ``build_guidance_panel()`` to create all
components and receive a ``GuidancePanelComponents`` dataclass.  The caller
positions the components however it likes; the canonical ``panel.inputs`` list
is passed directly to Gradio ``inputs=`` arguments.  Convert UI values to a
``GuidanceSetConfig`` with the pure ``make_guidance_set_config()`` function.

Adding a new guidance function
-------------------------------
1. Add the ``MyGuidance`` class to ``diffusion_planner/model/guidance/``.
2. Add one ``gr.Checkbox`` + ``gr.Slider`` pair to ``build_guidance_panel()``.
3. Add the corresponding ``GuidanceConfig(...)`` line to
   ``make_guidance_set_config()``.
4. Append the new components to the ``inputs`` property of
   ``GuidancePanelComponents``.

All apps that import this module update automatically with no further changes.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import gradio as gr
import numpy as np

from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from guidance_gui.visualization import render_prototype_gallery

_DEFAULT_PROTOTYPES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prototypes_k16.npy")


@functools.lru_cache(maxsize=4)
def _get_prototype_k(path: str) -> int:
    """Return K (number of prototypes) from a .npy file, cached by path."""
    if not path or not os.path.exists(path):
        return 64
    return int(np.load(path).shape[0])


@dataclass
class GuidancePanelComponents:
    """References to every Gradio component in the shared guidance panel.

    The ``inputs`` property returns components in the canonical fixed order
    expected by ``make_guidance_set_config()``.  Callers must not reorder
    this list.
    """

    enable_cb: gr.Checkbox
    global_scale: gr.Slider
    collision_cb: gr.Checkbox
    collision_scale: gr.Slider
    route_cb: gr.Checkbox
    route_scale: gr.Slider
    lane_cb: gr.Checkbox
    lane_scale: gr.Slider
    centerline_cb: gr.Checkbox
    centerline_scale: gr.Slider
    anchor_cb: gr.Checkbox
    anchor_scale: gr.Slider
    speed_cb: gr.Checkbox
    speed_scale: gr.Slider
    speed_limit: gr.Slider
    anchor_index: gr.Slider
    anchor_path: gr.Textbox
    gallery: gr.Gallery
    reload_btn: gr.Button

    @property
    def inputs(self) -> list:
        """Canonical ordered list of components for Gradio ``inputs=``.

        Order must match the positional parameters of
        ``make_guidance_set_config()``:
          enable_cb,
          collision_cb, collision_scale,
          route_cb, route_scale,
          lane_cb, lane_scale,
          centerline_cb, centerline_scale,
          anchor_cb, anchor_scale,
          speed_cb, speed_scale, speed_limit,
          anchor_index, anchor_path,
          global_scale
        """
        return [
            self.enable_cb,
            self.collision_cb,   self.collision_scale,
            self.route_cb,       self.route_scale,
            self.lane_cb,        self.lane_scale,
            self.centerline_cb,  self.centerline_scale,
            self.anchor_cb,      self.anchor_scale,
            self.speed_cb,       self.speed_scale,    self.speed_limit,
            self.anchor_index,   self.anchor_path,
            self.global_scale,
        ]


def build_guidance_panel(
    default_prototypes_path: str = _DEFAULT_PROTOTYPES_PATH,
) -> GuidancePanelComponents:
    """Create all guidance-related Gradio components inside the caller's ``gr.Blocks``.

    Must be called inside an active ``gr.Blocks`` context.  The caller is
    responsible for positioning the returned components within its own layout.

    Gallery event wiring (gallery click → anchor index, reload button → gallery)
    is handled inside this function via ``panel.wire_gallery()``, which is called
    automatically before returning.

    Args:
        default_prototypes_path: Default filesystem path to the prototypes
            ``.npy`` file shown in the gallery on startup.

    Returns:
        ``GuidancePanelComponents`` with references to every created component.
    """
    enable_cb = gr.Checkbox(
        value=False,
        label="Enable Guidance",
        info="When enabled, guidance shapes the stochastic trajectory via score correction",
    )

    global_scale = gr.Slider(
        minimum=0.0,
        maximum=5.0,
        value=0.5,
        step=0.1,
        label="Global Guidance Scale",
        info="Multiplies the total gradient correction from all active guidance functions",
    )

    with gr.Row():
        with gr.Column():
            collision_cb = gr.Checkbox(
                value=True,
                label="Collision Avoidance",
                info="Penalise trajectories that collide with neighbouring agents",
            )
            collision_scale = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                label="Collision Scale",
            )
        with gr.Column():
            route_cb = gr.Checkbox(
                value=False,
                label="Route Following",
                info="Penalise trajectories that stray from the planned route",
            )
            route_scale = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                label="Route Following Scale",
            )
        with gr.Column():
            lane_cb = gr.Checkbox(
                value=False,
                label="Lane Keeping",
                info="Penalise trajectories where the vehicle protrudes beyond lane boundaries",
            )
            lane_scale = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                label="Lane Keeping Scale",
            )
        with gr.Column():
            centerline_cb = gr.Checkbox(
                value=False,
                label="Centerline Following",
                info="Continuously attract the trajectory toward the nearest lane centerline (quadratic cost)",
            )
            centerline_scale = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                label="Centerline Scale",
            )
        with gr.Column():
            anchor_cb = gr.Checkbox(
                value=False,
                label="Anchor Following",
                info="Guide trajectory toward a prototype motion mode",
            )
            anchor_scale = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                label="Anchor Scale",
            )
            anchor_index = gr.Slider(
                minimum=0,
                maximum=max(0, _get_prototype_k(default_prototypes_path) - 1),
                value=0,
                step=1,
                label="Anchor Index",
            )
            anchor_path = gr.Textbox(
                value=default_prototypes_path,
                label="Prototypes Path",
                info="Path to prototypes .npy file (K, 80, 2)",
            )

    with gr.Row():
        with gr.Column():
            speed_cb = gr.Checkbox(
                value=False,
                label="Target path speed",
                info="Penalise path speed outside [0, v_high] m/s (squared hinge)",
            )
            speed_scale = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                label="Speed guidance scale",
            )
            speed_limit = gr.Slider(
                minimum=2.0, maximum=40.0, value=14.0, step=0.5,
                label="Speed upper bound v_high (m/s)",
                info="Lower bound v_low is fixed at 0 m/s",
            )

    with gr.Accordion("Prototype Gallery — click to select anchor", open=False):
        _default_gallery = render_prototype_gallery(default_prototypes_path) or []
        gallery = gr.Gallery(
            value=_default_gallery,
            columns=8,
            rows=2,
            height=260,
            allow_preview=False,
            selected_index=0 if _default_gallery else None,
            label="Motion Mode Prototypes",
        )
        reload_btn = gr.Button("↺ Reload Gallery from Path", size="sm")

    panel = GuidancePanelComponents(
        enable_cb=enable_cb,
        global_scale=global_scale,
        collision_cb=collision_cb,
        collision_scale=collision_scale,
        route_cb=route_cb,
        route_scale=route_scale,
        lane_cb=lane_cb,
        lane_scale=lane_scale,
        centerline_cb=centerline_cb,
        centerline_scale=centerline_scale,
        anchor_cb=anchor_cb,
        anchor_scale=anchor_scale,
        speed_cb=speed_cb,
        speed_scale=speed_scale,
        speed_limit=speed_limit,
        anchor_index=anchor_index,
        anchor_path=anchor_path,
        gallery=gallery,
        reload_btn=reload_btn,
    )

    # Wire gallery events once here so callers never have to repeat this boilerplate.
    # Gradio 6.x requires gr.SelectData type annotation for SelectData to be injected;
    # bare lambdas receive None without it.
    def _on_gallery_select(evt: gr.SelectData, path: str) -> tuple:
        updated = render_prototype_gallery(path, selected_index=evt.index) or []
        k = len(updated) if updated else _get_prototype_k(path)
        clamped = min(evt.index, max(0, k - 1))
        return (
            gr.update(value=clamped, maximum=max(0, k - 1)),
            gr.update(value=updated),
        )

    def _on_reload(path: str) -> tuple:
        updated = render_prototype_gallery(path) or []
        k = len(updated) if updated else _get_prototype_k(path)
        return (
            gr.update(value=updated),
            gr.update(maximum=max(0, k - 1), value=0),
        )

    panel.gallery.select(
        fn=_on_gallery_select,
        inputs=[panel.anchor_path],
        outputs=[panel.anchor_index, panel.gallery],
    )
    panel.reload_btn.click(
        fn=_on_reload,
        inputs=[panel.anchor_path],
        outputs=[panel.gallery, panel.anchor_index],
    )

    return panel


def make_guidance_set_config(
    eg: bool,
    uc: bool,   ucs: float,
    urf: bool,  urfs: float,
    ulk: bool,  ulks: float,
    ucf: bool,  ucfs: float,
    ua: bool,   uas: float,
    u_speed: bool, u_speed_scale: float, u_speed_limit: float,
    ai: int,    ap: str,
    gs: float,
) -> GuidanceSetConfig | None:
    """Convert flat Gradio component values into a ``GuidanceSetConfig``.

    This is the single canonical implementation shared by all apps.  Parameter
    order matches ``GuidancePanelComponents.inputs``.

    Args:
        eg:   Enable guidance master switch.
        uc:   Enable collision avoidance.
        ucs:  Collision scale.
        urf:  Enable route following.
        urfs: Route following scale.
        ulk:  Enable lane keeping.
        ulks: Lane keeping scale.
        ucf:  Enable centerline following.
        ucfs: Centerline scale.
        ua:   Enable anchor following.
        uas:  Anchor following scale.
        u_speed: Enable path-speed guidance.
        u_speed_scale: Speed guidance per-function scale.
        u_speed_limit: Upper speed bound v_high (m/s); v_low is 0.
        ai:   Anchor prototype index.
        ap:   Path to prototypes ``.npy`` file.
        gs:   Global guidance scale.

    Returns:
        ``GuidanceSetConfig`` when guidance is enabled and at least one function
        is potentially active, or ``None`` when the master switch is off.
    """
    if not eg:
        return None

    fns = [
        GuidanceConfig("collision",            enabled=bool(uc),  scale=float(ucs)),
        GuidanceConfig("route_following",      enabled=bool(urf), scale=float(urfs)),
        GuidanceConfig("lane_keeping",         enabled=bool(ulk), scale=float(ulks)),
        GuidanceConfig("centerline_following", enabled=bool(ucf), scale=float(ucfs)),
        GuidanceConfig("speed", enabled=bool(u_speed), scale=float(u_speed_scale), 
                params={"v_low": 0.0, "v_high": u_speed_limit, "dt": 0.1})
    ]
    if ua and ap and os.path.exists(str(ap)):
        k = _get_prototype_k(str(ap))
        fns.append(GuidanceConfig(
            "anchor_following", enabled=True, scale=float(uas),
            params={"prototypes_path": str(ap), "anchor_index": max(0, min(int(ai), k - 1))},
        ))
    return GuidanceSetConfig(global_scale=float(gs), functions=fns)
