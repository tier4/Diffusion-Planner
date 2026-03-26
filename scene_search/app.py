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

from scene_search.batch_search import Batch, build_index, find_batches, load_index_parquet, save_index_parquet
from scene_search.constraints import list_available as list_constraints
from scene_search.constraints.registry import build as build_constraint
from scene_search.map_renderer import MapRenderer, Viewport
from scene_search.scene_previewer import render_batch_thumbnails, render_single_thumbnail, thumbnails_to_pil_images
from PIL import Image as PILImage

MAX_VISIBLE_BATCHES = 10

# ── JS for the interactive map canvas ──────────────────────────────────────
# All pan/zoom is client-side. The full-map image is loaded once via props.map_b64.
# Only calls server.render_map() on deep zoom (debounced).

MAP_CANVAS_JS = r"""
(function() {
    const W = 900, H = 700;
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    canvas.style.cssText = 'display:block; margin:auto; border:1px solid #ccc; cursor:grab;';
    element.innerHTML = '';
    element.appendChild(canvas);
    const help = document.createElement('div');
    help.style.cssText = 'text-align:center; font-size:12px; color:#666; margin-top:4px;';
    help.innerHTML = 'Drag=pan | Scroll=zoom | <b>Shift+drag=arrow</b>';
    element.appendChild(help);
    const ctx = canvas.getContext('2d');

    // World bounds of the full map image
    const bounds = JSON.parse(props.map_bounds);
    // Current view in world coords
    let vx0 = bounds.xmin, vy0 = bounds.ymin, vx1 = bounds.xmax, vy1 = bounds.ymax;

    // Full map image (loaded once)
    let fullImg = null;
    // Hi-res tile for current zoom (loaded on demand)
    let tileImg = null;
    let tileBounds = null;

    // Arrow state — stored in WORLD coords so it survives pan/zoom
    let arrowStartPx = null, arrowEndPx = null;
    let arrowStartWorld = null, arrowEndWorld = null;
    let isDrawing = false;
    // Pan state
    let isPanning = false, panPrev = null;
    // Debounce timer for tile re-render
    let tileTimer = null;

    function worldToCanvas(wx, wy) {
        return {
            x: (wx - vx0) / (vx1 - vx0) * W,
            y: (vy1 - wy) / (vy1 - vy0) * H
        };
    }
    function canvasToWorld(cx, cy) {
        return {
            x: vx0 + (cx / W) * (vx1 - vx0),
            y: vy1 - (cy / H) * (vy1 - vy0)
        };
    }
    function m2px(m) { return m / (vx1 - vx0) * W; }

    function drawMapImage() {
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, W, H);

        // Draw the tile (hi-res) if available and covers viewport, else draw full image
        let img = null, ib = null;
        if (tileImg && tileImg.complete && tileBounds) {
            img = tileImg; ib = tileBounds;
        } else if (fullImg && fullImg.complete) {
            img = fullImg; ib = bounds;
        }
        if (img && ib) {
            // Source rect in image pixels
            const imgW = img.naturalWidth, imgH = img.naturalHeight;
            const sx = (vx0 - ib.xmin) / (ib.xmax - ib.xmin) * imgW;
            const sy = (ib.ymax - vy1) / (ib.ymax - ib.ymin) * imgH;
            const sw = (vx1 - vx0) / (ib.xmax - ib.xmin) * imgW;
            const sh = (vy1 - vy0) / (ib.ymax - ib.ymin) * imgH;
            ctx.drawImage(img, sx, sy, sw, sh, 0, 0, W, H);
        }
    }

    function redraw() {
        drawMapImage();
        // Recompute pixel positions from world coords (so arrow follows pan/zoom)
        if (arrowStartWorld && !isDrawing) {
            arrowStartPx = worldToCanvas(arrowStartWorld.x, arrowStartWorld.y);
            if (arrowEndWorld) {
                arrowEndPx = worldToCanvas(arrowEndWorld.x, arrowEndWorld.y);
            }
        }
        const radius = parseFloat(props.radius) || 50;
        // Radius circle + arrow
        if (arrowStartPx) {
            const rPx = m2px(radius);
            ctx.strokeStyle = 'rgba(255,68,0,0.5)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.arc(arrowStartPx.x, arrowStartPx.y, rPx, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
            // Heading tolerance wedge
            const htol = parseFloat(props.heading_tol) || 30;
            if (arrowEndPx) {
                const dx = arrowEndPx.x - arrowStartPx.x;
                const dy = arrowEndPx.y - arrowStartPx.y;
                const ang = Math.atan2(dy, dx);
                const halfTol = htol * Math.PI / 180;
                ctx.fillStyle = 'rgba(255,68,0,0.08)';
                ctx.beginPath();
                ctx.moveTo(arrowStartPx.x, arrowStartPx.y);
                ctx.arc(arrowStartPx.x, arrowStartPx.y, rPx, ang - halfTol, ang + halfTol);
                ctx.closePath();
                ctx.fill();
            }
        }
        if (arrowStartPx && arrowEndPx) {
            const dx = arrowEndPx.x - arrowStartPx.x;
            const dy = arrowEndPx.y - arrowStartPx.y;
            const len = Math.sqrt(dx*dx + dy*dy);
            if (len > 3) {
                const ang = Math.atan2(dy, dx);
                ctx.strokeStyle = '#ff4400'; ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(arrowStartPx.x, arrowStartPx.y);
                ctx.lineTo(arrowEndPx.x, arrowEndPx.y);
                ctx.stroke();
                const hl = Math.min(18, len * 0.3);
                ctx.fillStyle = '#ff4400';
                ctx.beginPath();
                ctx.moveTo(arrowEndPx.x, arrowEndPx.y);
                ctx.lineTo(arrowEndPx.x - hl*Math.cos(ang-0.4), arrowEndPx.y - hl*Math.sin(ang-0.4));
                ctx.lineTo(arrowEndPx.x - hl*Math.cos(ang+0.4), arrowEndPx.y - hl*Math.sin(ang+0.4));
                ctx.closePath(); ctx.fill();
            }
            ctx.fillStyle = '#ff4400';
            ctx.beginPath();
            ctx.arc(arrowStartPx.x, arrowStartPx.y, 5, 0, Math.PI * 2);
            ctx.fill();
        }
        // Scale bar
        const scaleM = Math.pow(10, Math.floor(Math.log10((vx1-vx0)*0.2)));
        const scalePx = m2px(scaleM);
        ctx.fillStyle = '#333'; ctx.font = '11px monospace';
        ctx.fillText(scaleM >= 1000 ? (scaleM/1000)+'km' : scaleM+'m', 15, H-20);
        ctx.strokeStyle = '#333'; ctx.lineWidth = 2; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(15, H-12); ctx.lineTo(15+scalePx, H-12); ctx.stroke();
    }

    function scheduleTileRefresh() {
        clearTimeout(tileTimer);
        tileTimer = setTimeout(async () => {
            const vpJson = JSON.stringify({xmin:vx0, ymin:vy0, xmax:vx1, ymax:vy1,
                                           canvas_w: 2000, canvas_h: 1600});
            try {
                const b64 = await server.render_map(vpJson);
                tileImg = new Image();
                tileBounds = {xmin:vx0, ymin:vy0, xmax:vx1, ymax:vy1};
                tileImg.onload = redraw;
                tileImg.src = 'data:image/png;base64,' + b64;
            } catch(e) { console.warn('tile render failed', e); }
        }, 400);
    }

    // Mouse handlers
    canvas.addEventListener('mousedown', (e) => {
        const r = canvas.getBoundingClientRect();
        const cx = e.clientX - r.left, cy = e.clientY - r.top;
        if (e.shiftKey) {
            isDrawing = true;
            arrowStartPx = {x:cx, y:cy};
            arrowEndPx = {x:cx, y:cy};
            canvas.style.cursor = 'crosshair';
        } else {
            isPanning = true;
            panPrev = {x:cx, y:cy};
            canvas.style.cursor = 'grabbing';
        }
    });
    canvas.addEventListener('mousemove', (e) => {
        const r = canvas.getBoundingClientRect();
        const cx = e.clientX - r.left, cy = e.clientY - r.top;
        if (isDrawing) {
            arrowEndPx = {x:cx, y:cy};
            redraw();
        } else if (isPanning && panPrev) {
            const dx = cx - panPrev.x, dy = cy - panPrev.y;
            panPrev = {x:cx, y:cy};
            const dwx = (dx / W) * (vx1 - vx0);
            const dwy = (dy / H) * (vy1 - vy0);
            vx0 -= dwx; vx1 -= dwx;
            vy0 += dwy; vy1 += dwy;
            redraw();
            scheduleTileRefresh();
        }
    });
    canvas.addEventListener('mouseup', (e) => {
        if (isDrawing && arrowStartPx && arrowEndPx) {
            isDrawing = false;
            canvas.style.cursor = 'grab';
            // Store in world coords so arrow survives pan/zoom
            arrowStartWorld = canvasToWorld(arrowStartPx.x, arrowStartPx.y);
            arrowEndWorld = canvasToWorld(arrowEndPx.x, arrowEndPx.y);
            const hdeg = Math.atan2(arrowEndWorld.y - arrowStartWorld.y,
                                     arrowEndWorld.x - arrowStartWorld.x) * 180 / Math.PI;
            trigger('click', {x: arrowStartWorld.x, y: arrowStartWorld.y, heading: hdeg});
            redraw();
        }
        if (isPanning) { isPanning = false; panPrev = null; canvas.style.cursor = 'grab'; }
    });
    canvas.addEventListener('mouseleave', () => {
        if (isPanning) { isPanning = false; panPrev = null; canvas.style.cursor = 'grab'; }
    });
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const r = canvas.getBoundingClientRect();
        const cx = e.clientX - r.left, cy = e.clientY - r.top;
        const factor = e.deltaY > 0 ? 1.25 : 0.8;
        const cw = canvasToWorld(cx, cy);
        const nw = (vx1 - vx0) * factor, nh = (vy1 - vy0) * factor;
        // Zoom centered on cursor
        const rx = cx / W, ry = cy / H;
        vx0 = cw.x - rx * nw; vx1 = cw.x + (1-rx) * nw;
        vy0 = cw.y - (1-ry) * nh; vy1 = cw.y + ry * nh;
        redraw();
        scheduleTileRefresh();
    }, {passive: false});

    // Load full map image
    const b64 = props.map_b64;
    if (b64) {
        fullImg = new Image();
        fullImg.onload = redraw;
        fullImg.src = 'data:image/png;base64,' + b64;
    }
})();
"""


def build_interface(renderer: MapRenderer, index: list[dict], index_path: str | None = None):
    """Build the complete Gradio interface."""

    # Pre-render full map at high resolution (sent once to client)
    full_vp = renderer.initial_viewport(canvas_w=4000, canvas_h=3200)
    print("Pre-rendering full map image...")
    full_map_b64 = renderer.render_viewport_base64(full_vp, dpi=100)
    full_bounds = full_vp.to_json()
    print(f"  Map image: {len(full_map_b64)//1024}KB base64")

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

            all_pils = [None] * len(batch_dicts)
            with ThreadPoolExecutor(max_workers=min(len(batch_dicts), 6)) as tex:
                futures = {tex.submit(_render_one, bd): i for i, bd in enumerate(batch_dicts)}
                for fut in futures:
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
    parser.add_argument("--npz_list", type=str, required=True, help="path_list.json or NPZ directory")
    parser.add_argument("--index", type=str, default=None, help="Cached parquet index (requires pyarrow)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Loading lanelet2 map...")
    renderer = MapRenderer(str(args.map_path))

    if args.index and Path(args.index).exists():
        print(f"Loading cached index from {args.index}")
        index = load_index_parquet(args.index)
    else:
        print("Building spatial index from NPZ sidecars...")
        p = Path(args.npz_list)
        if p.is_file() and p.suffix == ".json":
            with open(p) as f: npz_paths = json.load(f)
        elif p.is_dir():
            npz_paths = sorted(str(f) for f in p.rglob("*.npz"))
        else:
            raise ValueError(f"--npz_list must be .json or directory: {args.npz_list}")
        index = build_index(npz_paths, workers=8)
        if args.index:
            save_index_parquet(index, args.index)
            print(f"Saved index to {args.index}")

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
