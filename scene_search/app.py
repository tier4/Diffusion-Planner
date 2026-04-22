"""Scene Search GUI — visual search for NPZ driving scenes on a lanelet2 map.

Launch:
    source /opt/ros/humble/setup.bash
    source /home/danielsanchez/autoware/install/setup.bash
    source .venv/bin/activate
    python -m scene_search.app \
        --map_path /home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm \
        --npz_list /path/to/path_list.json \
        [--index /path/to/cached_index.parquet] \
        [--port 7860]
"""

import argparse
import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image as PILImage

from scene_search.batch_search import (
    Batch,
    build_index,
    find_batches,
    load_index_parquet,
    save_index_parquet,
)
from scene_search.constraints import list_available as list_constraints
from scene_search.constraints.registry import build as build_constraint
from scene_search.map_canvas_js import build_map_canvas_js
from scene_search.map_renderer import MapRenderer, Viewport
from scene_search.replay_index import load_replay_runs
from scene_search.scene_previewer import (
    render_batch_thumbnails,
    render_single_thumbnail,
    thumbnails_to_pil_images,
)

## Heatmap metric auto-discovery
#
# The replay writes whatever the RewardBreakdown dataclass exposes — adding
# a new reward component on the training side should "just work" here.
# We derive the dropdown choices from the union of numeric fields across
# loaded heatmap points, and transmit each metric's observed min/max range
# to the canvas so the JS colour ramp can auto-scale.
#
# Polarity (is higher = good or bad?) can't be inferred from data alone,
# so we fall back to a naming heuristic: metric names containing any of
# these substrings are treated as "higher = worse". Everything else is
# assumed "higher = better" (distance-, reward-, and score-style fields).
_HIGHER_IS_WORSE_SUBSTRINGS = (
    "penalty", "crossing", "collision", "off_road", "red_light",
    "near_frac", "wide_frac",
)


def _metric_polarity(name: str) -> int:
    """Return +1 if higher is better (safe), -1 if higher is worse (drift)."""
    lo = name.lower()
    for s in _HIGHER_IS_WORSE_SUBSTRINGS:
        if s in lo:
            return -1
    return +1

MAX_VISIBLE_BATCHES = 10

MAP_CANVAS_JS = build_map_canvas_js(tool="arrow")

def _heatmap_points(index: list[dict]) -> list[dict]:
    """Pick entries that carry per-step metrics (replay runs) and project to
    the compact form the JS canvas consumes. Every numeric/bool field from
    ``metrics`` is copied verbatim — no per-field allowlist. Sidecar-backed
    entries have no "metrics" key and are silently skipped."""
    out = []
    for e in index:
        m = e.get("metrics") or {}
        if not m:
            continue
        point = {"x": e["x"], "y": e["y"]}
        for k, v in m.items():
            if isinstance(v, bool):
                point[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                point[k] = float(v)
            # Drop None / str / list — not colourable.
        out.append(point)
    return out


def _heatmap_metadata(points: list[dict]) -> dict:
    """Derive per-metric display metadata from the loaded heatmap points.

    Returns::

        {
            "metrics": ["total", "rb_min_dist", ...],   # sorted field names
            "ranges":  {"total": [min, max], ...},       # observed range
            "polarity": {"total": 1, "rb_near_penalty": -1, ...},
        }

    The canvas uses this to auto-scale a colour ramp without any metric-
    specific JS. Polarity comes from ``_metric_polarity`` (naming heuristic).
    """
    if not points:
        return {"metrics": [], "ranges": {}, "polarity": {}}
    field_names: set[str] = set()
    for p in points:
        for k in p.keys():
            if k in ("x", "y"):
                continue
            field_names.add(k)
    ranges: dict[str, list[float]] = {}
    for k in field_names:
        vals = [p[k] for p in points if k in p and p[k] is not None]
        if not vals:
            continue
        ranges[k] = [float(min(vals)), float(max(vals))]
    polarity = {k: _metric_polarity(k) for k in ranges}
    # cl_score is a signed rear-axle offset; the magnitude reading is what
    # users usually want, so offer a synthetic "abs_cl_score" entry on the
    # dropdown whenever cl_score is present. The canvas handles the |·|
    # and polarity flip for this one derived field.
    metrics = sorted(ranges.keys())
    if "cl_score" in ranges:
        metrics.insert(metrics.index("cl_score") + 1, "abs_cl_score")
    return {
        "metrics": metrics,
        "ranges": ranges,
        "polarity": polarity,
    }


def build_interface(renderer: MapRenderer, index: list[dict], index_path: str | None = None):
    """Build the complete Gradio interface."""

    # Pre-render full map at high resolution (sent once to client)
    full_vp = renderer.initial_viewport(canvas_w=4000, canvas_h=3200)
    print("Pre-rendering full map image...")
    full_map_b64 = renderer.render_viewport_base64(full_vp, dpi=100)
    full_bounds = full_vp.to_json()
    print(f"  Map image: {len(full_map_b64)//1024}KB base64")

    heatmap_points = _heatmap_points(index)
    heatmap_meta = _heatmap_metadata(heatmap_points)
    heatmap_json = json.dumps(heatmap_points)
    heatmap_meta_json = json.dumps(heatmap_meta)
    if heatmap_points:
        print(f"  Heatmap: {len(heatmap_points)} scored points across "
              f"{len(heatmap_meta['metrics'])} metric(s) "
              f"({len(heatmap_json)//1024}KB + {len(heatmap_meta_json)//1024}KB JSON)")

    def render_map(viewport_json: str) -> str:
        """Server function for hi-res tile at current zoom. Called from JS (debounced)."""
        vp = Viewport.from_json(json.loads(viewport_json))
        return renderer.render_viewport_base64(vp, dpi=100)

    with gr.Blocks(title="Scene Search") as demo:

        search_results_state = gr.State(value=[])
        kept_batches_state = gr.State(value=[])
        index_state = gr.State(value=index)

        gr.Markdown("# Scene Search")

        with gr.Row():
            # ===================== LEFT SIDEBAR =====================
            with gr.Column(scale=1):
                gr.Markdown("### Search Parameters")
                with gr.Row():
                    arrow_x = gr.Number(label="X (MGRS)", value=89130, interactive=True)
                    arrow_y = gr.Number(label="Y (MGRS)", value=42440, interactive=True)
                arrow_heading = gr.Number(label="Heading (deg)", value=106, interactive=True)
                radius_slider = gr.Slider(1, 200, value=50, step=1, label="Search radius (m)")
                heading_tol_slider = gr.Slider(5, 180, value=30, step=5, label="Heading tolerance (deg)")
                n_before_slider = gr.Slider(0, 100, value=30, step=5, label="Frames before")
                n_after_slider = gr.Slider(0, 200, value=80, step=5, label="Frames after")
                search_btn = gr.Button("Search", variant="primary")

                if heatmap_points:
                    gr.Markdown("### Heatmap")
                    heatmap_metric_dd = gr.Dropdown(
                        choices=["off"] + heatmap_meta["metrics"],
                        value="off",
                        label="Overlay metric",
                        interactive=True,
                    )
                else:
                    heatmap_metric_dd = None

                gr.Markdown("### Constraints")
                # Build toggle panels for each registered constraint
                constraint_components = {}  # name → {enable: Checkbox, params: {name: Component}}
                available = list_constraints()
                for cname in available:
                    c = build_constraint(cname)
                    spec = c.get_params_spec()
                    with gr.Accordion(c.name, open=False):
                        enable_cb = gr.Checkbox(label=f"Enable {c.name}", value=False)
                        param_components = {}
                        for pname, pspec in spec.items():
                            if pspec["type"] == "int":
                                param_components[pname] = gr.Number(
                                    label=pspec["label"], value=pspec["default"],
                                    minimum=pspec.get("min"), maximum=pspec.get("max"),
                                    precision=0, interactive=True,
                                )
                            else:
                                param_components[pname] = gr.Number(
                                    label=pspec["label"], value=pspec["default"],
                                    minimum=pspec.get("min"), maximum=pspec.get("max"),
                                    interactive=True,
                                )
                        constraint_components[cname] = {"enable": enable_cb, "params": param_components}

                gr.Markdown("### Kept Batches")
                kept_summary = gr.Markdown("No batches kept yet")
                save_btn = gr.Button("Save All Kept → JSON", variant="secondary")
                _default_save = str(Path.cwd() / "kept_scenes")
                save_path_input = gr.Textbox(label="Save base name", value=_default_save)
                downsample_n = gr.Number(label="Downsample to N (0=all)", value=0, precision=0)
                save_status = gr.Markdown("")
                clear_kept_btn = gr.Button("Clear All Kept", variant="stop")

            # ===================== MAIN CONTENT =====================
            with gr.Column(scale=3):
                map_canvas = gr.HTML(
                    value="",
                    js_on_load=MAP_CANVAS_JS,
                    server_functions=[render_map],
                    min_height=750,
                    map_b64=full_map_b64,
                    map_bounds=json.dumps(full_bounds),
                    radius=50,
                    heading_tol=30,
                    heatmap_json=heatmap_json,
                    heatmap_meta_json=heatmap_meta_json,
                    heatmap_metric="off",
                )

                gr.Markdown("### Search Results")
                with gr.Row():
                    results_info = gr.Markdown("Shift+drag on the map to place an arrow, then click Search")
                    keep_all_btn = gr.Button("Keep All Batches", size="sm", variant="primary", visible=False)

                batch_groups = []
                batch_labels = []
                batch_galleries = []
                batch_keep_btns = []
                for i in range(MAX_VISIBLE_BATCHES):
                    with gr.Group(visible=False) as grp:
                        lbl = gr.Markdown(f"Batch {i+1}")
                        gal = gr.Gallery(label=f"Batch {i+1}", columns=6, rows=2,
                                         height=220, object_fit="contain",
                                         preview=True)
                        with gr.Row():
                            keep_b = gr.Button(f"Keep Batch {i+1}", size="sm", variant="primary")
                    batch_groups.append(grp)
                    batch_labels.append(lbl)
                    batch_galleries.append(gal)
                    batch_keep_btns.append(keep_b)

                gr.Markdown("### Kept Batches")
                kept_display = gr.Markdown("No batches kept")

        # ===================== EVENT HANDLERS =====================

        # Arrow placed on canvas
        def on_arrow_placed(evt: gr.EventData):
            return round(evt.x, 1), round(evt.y, 1), round(evt.heading, 1)

        map_canvas.click(on_arrow_placed, outputs=[arrow_x, arrow_y, arrow_heading])

        # Update canvas props when sliders change
        def on_radius_change(val):
            return gr.update(radius=val)
        radius_slider.release(on_radius_change, inputs=[radius_slider], outputs=[map_canvas])

        def on_heading_tol_change(val):
            return gr.update(heading_tol=val)
        heading_tol_slider.release(on_heading_tol_change, inputs=[heading_tol_slider], outputs=[map_canvas])

        if heatmap_metric_dd is not None:
            def on_heatmap_metric_change(val):
                return gr.update(heatmap_metric=val)
            heatmap_metric_dd.change(on_heatmap_metric_change,
                                     inputs=[heatmap_metric_dd],
                                     outputs=[map_canvas])

        # --- Build constraint input list for search ---
        # Order: [enable_1, param_1a, param_1b, ..., enable_2, param_2a, ...]
        constraint_input_list = []
        constraint_input_meta = []  # [(name, n_params)] to unpack in on_search
        for cname in available:
            cc = constraint_components[cname]
            constraint_input_list.append(cc["enable"])
            param_names = list(cc["params"].keys())
            for pn in param_names:
                constraint_input_list.append(cc["params"][pn])
            constraint_input_meta.append((cname, param_names))

        # --- Search ---
        def on_search(x, y, heading, radius, heading_tol, n_before, n_after, idx, *constraint_vals):
            if x == 0 and y == 0:
                outputs = ["Enter coordinates and click Search", gr.update(visible=False)]
                for _ in range(MAX_VISIBLE_BATCHES):
                    outputs.extend([gr.update(visible=False), gr.update(), gr.update(value=None)])
                outputs.append([])
                return outputs

            active_filters = []
            val_idx = 0
            for cname, param_names in constraint_input_meta:
                enabled = constraint_vals[val_idx]
                val_idx += 1
                params = {}
                for pn in param_names:
                    params[pn] = constraint_vals[val_idx]
                    val_idx += 1
                if enabled:
                    c = build_constraint(cname)
                    active_filters.append((c, params))

            batches = find_batches(
                index=idx, center_x=x, center_y=y, heading_deg=heading,
                radius=radius, heading_tolerance=heading_tol,
                n_before=int(n_before), n_after=int(n_after),
                constraint_filters=active_filters if active_filters else None,
            )
            batch_dicts = [
                {"bag_prefix": b.bag_prefix, "scenes": b.scenes,
                 "central_indices": b.central_indices, "metadata": b.metadata}
                for b in batches
            ]
            total = sum(b.n_scenes for b in batches)
            n_constraints = len(active_filters)
            constraint_info = f" ({n_constraints} constraint{'s' if n_constraints != 1 else ''} active)" if active_filters else ""

            # Phase 1: show batch info instantly, no thumbnail rendering
            outputs = [
                f"Found **{len(batches)} batches** ({total} total scenes){constraint_info} — rendering thumbnails...",
                gr.update(visible=len(batches) > 0),
            ]
            for i in range(MAX_VISIBLE_BATCHES):
                if i < len(batches):
                    outputs.extend([gr.update(visible=True), f"**Batch {i+1}**: {batches[i].summary()}", gr.update(value=None)])
                else:
                    outputs.extend([gr.update(visible=False), gr.update(), gr.update(value=None)])
            outputs.append(batch_dicts)
            return outputs

        search_outputs = [results_info, keep_all_btn]
        for i in range(MAX_VISIBLE_BATCHES):
            search_outputs.extend([batch_groups[i], batch_labels[i], batch_galleries[i]])
        search_outputs.append(search_results_state)

        def on_search_fill(batch_dicts):
            """Phase 2: fill in all thumbnails for each batch."""
            if not batch_dicts:
                return [gr.update()] * (1 + MAX_VISIBLE_BATCHES)

            from concurrent.futures import ThreadPoolExecutor

            from scene_search.batch_search import Batch

            def _render_one(bd):
                b = Batch(bag_prefix=bd["bag_prefix"], scenes=bd["scenes"],
                          central_indices=bd["central_indices"], metadata=bd["metadata"])
                thumbs = render_batch_thumbnails(b, every_nth=10, max_workers=4)
                return thumbnails_to_pil_images(thumbs)

            from concurrent.futures import as_completed as _as_completed
            all_pils = [None] * len(batch_dicts)
            with ThreadPoolExecutor(max_workers=min(len(batch_dicts), 6)) as tex:
                futures = {tex.submit(_render_one, bd): i for i, bd in enumerate(batch_dicts)}
                for fut in _as_completed(futures):
                    all_pils[futures[fut]] = fut.result()

            total = sum(len(bd["scenes"]) for bd in batch_dicts)
            outputs = [f"Found **{len(batch_dicts)} batches** ({total} total scenes)"]
            for i in range(MAX_VISIBLE_BATCHES):
                if i < len(batch_dicts):
                    outputs.append(all_pils[i])
                else:
                    outputs.append(gr.update())
            return outputs

        # Phase 2 outputs: results_info + one gallery per slot
        fill_outputs = [results_info]
        for i in range(MAX_VISIBLE_BATCHES):
            fill_outputs.append(batch_galleries[i])

        search_btn.click(
            on_search,
            inputs=[arrow_x, arrow_y, arrow_heading, radius_slider, heading_tol_slider,
                    n_before_slider, n_after_slider, index_state] + constraint_input_list,
            outputs=search_outputs,
        ).then(
            on_search_fill,
            inputs=[search_results_state],
            outputs=fill_outputs,
        )

        # --- Keep ---
        def _batch_key(b):
            """Unique key for a batch: central scene path."""
            ci = b["central_indices"]
            return b["scenes"][ci[0]] if ci else b["scenes"][0]

        def _kept_summary(kept):
            total = sum(len(k["scenes"]) for k in kept)
            summary = f"**{len(kept)} batches** kept ({total} scenes)"
            detail = "\n".join(f"- {k['bag_prefix'].split('/')[-1][:25]}... ({len(k['scenes'])} scenes)" for k in kept)
            return summary, detail

        def make_keep_handler(idx):
            def fn(results, kept):
                if idx >= len(results):
                    return kept, gr.update(), gr.update()
                b = results[idx]
                existing_keys = {_batch_key(k) for k in kept}
                if _batch_key(b) not in existing_keys:
                    kept = kept + [b]
                summary, detail = _kept_summary(kept)
                return kept, summary, detail
            return fn

        # --- Keep All ---
        def on_keep_all(search_results, kept_batches):
            existing_keys = {_batch_key(k) for k in kept_batches}
            new_kept = kept_batches[:]
            for b in search_results:
                if _batch_key(b) not in existing_keys:
                    new_kept.append(b)
                    existing_keys.add(_batch_key(b))
            summary, detail = _kept_summary(new_kept)
            return new_kept, summary, detail

        keep_all_btn.click(
            on_keep_all,
            inputs=[search_results_state, kept_batches_state],
            outputs=[kept_batches_state, kept_summary, kept_display],
        )

        for i in range(MAX_VISIBLE_BATCHES):
            batch_keep_btns[i].click(
                make_keep_handler(i),
                inputs=[search_results_state, kept_batches_state],
                outputs=[kept_batches_state, kept_summary, kept_display],
            )

        # --- Clear ---
        def on_clear():
            return [], "No batches kept yet", "No batches kept", ""
        clear_kept_btn.click(on_clear, outputs=[kept_batches_state, kept_summary, kept_display, save_status])

        # --- Save ---
        def _next_available_path(base_path: str) -> str:
            """Auto-increment: kept_scenes → kept_scenes_0.json, kept_scenes_1.json, ..."""
            p = Path(base_path)
            stem = p.stem
            suffix = p.suffix or ".json"
            parent = p.parent
            i = 0
            while True:
                candidate = parent / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    return str(candidate.resolve())
                i += 1

        def on_save(kept, path, ds):
            if not kept:
                return "No batches to save"
            scenes = []
            seen = set()
            for k in kept:
                for s in k["scenes"]:
                    if s not in seen:
                        seen.add(s); scenes.append(s)
            if ds and int(ds) > 0 and int(ds) < len(scenes):
                scenes = sorted(random.sample(scenes, int(ds)))
            actual_path = _next_available_path(path)
            Path(actual_path).parent.mkdir(parents=True, exist_ok=True)
            with open(actual_path, "w") as f:
                json.dump(scenes, f, indent=4)
            return f"Saved **{len(scenes)}** scenes to `{actual_path}`"

        save_btn.click(on_save, inputs=[kept_batches_state, save_path_input, downsample_n],
                       outputs=[save_status])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Scene Search GUI")
    parser.add_argument("--map_path", type=Path, required=True, help="Path to lanelet2 map (.osm)")
    parser.add_argument("--npz_list", type=str, default=None,
                        help="path_list.json or NPZ directory (sidecar-backed scenes)")
    parser.add_argument("--replay_runs", type=str, nargs="+", default=None,
                        help="One or more scenario_generation.replay output "
                             "directories. Uses trajectory_log.json + "
                             "metrics_log.json instead of per-NPZ sidecars; "
                             "enables the drift heatmap overlay.")
    parser.add_argument("--index", type=str, default=None, help="Cached parquet index (requires pyarrow)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not args.npz_list and not args.replay_runs:
        parser.error("supply --npz_list and/or --replay_runs")

    print("Loading lanelet2 map...")
    renderer = MapRenderer(str(args.map_path))

    index: list[dict] = []
    if args.npz_list:
        if args.index and Path(args.index).exists():
            print(f"Loading cached index from {args.index}")
            index.extend(load_index_parquet(args.index))
        else:
            print("Building spatial index from NPZ sidecars...")
            p = Path(args.npz_list)
            if p.is_file() and p.suffix == ".json":
                with open(p) as f: npz_paths = json.load(f)
            elif p.is_dir():
                npz_paths = sorted(str(f) for f in p.rglob("*.npz"))
            else:
                raise ValueError(f"--npz_list must be .json or directory: {args.npz_list}")
            sidecar_index = build_index(npz_paths, workers=8)
            index.extend(sidecar_index)
            if args.index:
                save_index_parquet(sidecar_index, args.index)
                print(f"Saved sidecar index to {args.index}")

    if args.replay_runs:
        print(f"Loading {len(args.replay_runs)} replay run(s)...")
        replay_entries = load_replay_runs(args.replay_runs)
        index.extend(replay_entries)
        print(f"  Replay entries: {len(replay_entries)}")

    print(f"Index: {len(index)} scenes")
    demo = build_interface(renderer, index, args.index)
    try:
        demo.launch(server_port=args.port, share=args.share, inbrowser=True)
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()


if __name__ == "__main__":
    main()
