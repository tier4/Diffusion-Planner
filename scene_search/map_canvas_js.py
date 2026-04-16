"""Composable JS canvas for Gradio-based map GUIs.

The base canvas provides pan (drag), zoom (scroll), and rotate (Alt+drag).
Each GUI injects tool-specific JS for overlays and interactions.
"""

# ── Base canvas JS ───────────────────────────────────────────────────────────
# Handles: pan, zoom, rotate, tile refresh, scale bar, coordinate transforms.
# Extension points (injected per-GUI):
#   __TOOL_STATE__     - tool-specific state variables
#   __TOOL_DRAW__      - called in redraw() after map + before scale bar
#   __TOOL_MOUSEDOWN__ - called when no base modifier is active
#   __TOOL_MOUSEMOVE__ - called when tool is active
#   __TOOL_MOUSEUP__   - called when tool finishes
#   __TOOL_HELP__      - appended to help text
#   __EXTRA_PROPS__    - additional props read from Gradio

_BASE_JS = r"""
(function() {
    const W = 900, H = 700;
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    canvas.style.cssText = 'display:block; margin:auto; border:1px solid #ccc; cursor:grab;';
    element.innerHTML = '';
    element.appendChild(canvas);
    const help = document.createElement('div');
    help.style.cssText = 'text-align:center; font-size:12px; color:#666; margin-top:4px;';
    help.innerHTML = 'Drag=pan | Scroll=zoom | <b>Alt+drag=rotate</b>__TOOL_HELP__';
    element.appendChild(help);
    const ctx = canvas.getContext('2d');

    const bounds = JSON.parse(props.map_bounds);
    let vx0 = bounds.xmin, vy0 = bounds.ymin, vx1 = bounds.xmax, vy1 = bounds.ymax;
    let viewRotation = 0;

    let fullImg = null;
    let tileImg = null, tileBounds = null;

    let isPanning = false, panPrev = null;
    let isRotating = false, rotateStartX = 0;
    let tileTimer = null;

    // Tool state
    __TOOL_STATE__

    // ── Coordinate transforms (with rotation) ──
    function worldToCanvas(wx, wy) {
        let px = (wx - vx0) / (vx1 - vx0) * W;
        let py = (vy1 - wy) / (vy1 - vy0) * H;
        if (Math.abs(viewRotation) > 0.001) {
            const c = Math.cos(viewRotation), s = Math.sin(viewRotation);
            const dx = px - W/2, dy = py - H/2;
            px = W/2 + dx*c - dy*s;
            py = H/2 + dx*s + dy*c;
        }
        return {x: px, y: py};
    }
    function canvasToWorld(cx, cy) {
        let px = cx, py = cy;
        if (Math.abs(viewRotation) > 0.001) {
            const c = Math.cos(-viewRotation), s = Math.sin(-viewRotation);
            const dx = px - W/2, dy = py - H/2;
            px = W/2 + dx*c - dy*s;
            py = H/2 + dx*s + dy*c;
        }
        return {
            x: vx0 + (px / W) * (vx1 - vx0),
            y: vy1 - (py / H) * (vy1 - vy0)
        };
    }
    function m2px(m) { return m / (vx1 - vx0) * W; }

    // Expansion factor for rotation (covers rotated corners)
    function rotMargin() {
        return Math.abs(Math.cos(viewRotation)) + Math.abs(Math.sin(viewRotation));
    }

    // ── Drawing ──
    function drawMapImage() {
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, W, H);
        let img = null, ib = null;
        if (tileImg && tileImg.complete && tileBounds) { img = tileImg; ib = tileBounds; }
        else if (fullImg && fullImg.complete) { img = fullImg; ib = bounds; }
        if (!img || !ib) return;

        ctx.save();
        if (Math.abs(viewRotation) > 0.001) {
            ctx.translate(W/2, H/2);
            ctx.rotate(viewRotation);
            ctx.translate(-W/2, -H/2);
        }
        // Draw expanded area to fill rotated canvas corners
        const m = rotMargin();
        const dw = W * m, dh = H * m;
        const dx = (W - dw) / 2, dy = (H - dh) / 2;
        const ecx = (vx0+vx1)/2, ecy = (vy0+vy1)/2;
        const ehw = (vx1-vx0)/2*m, ehh = (vy1-vy0)/2*m;
        const evx0 = ecx-ehw, evy0 = ecy-ehh, evx1 = ecx+ehw, evy1 = ecy+ehh;

        const imgW = img.naturalWidth, imgH = img.naturalHeight;
        const sx = (evx0 - ib.xmin) / (ib.xmax - ib.xmin) * imgW;
        const sy = (ib.ymax - evy1) / (ib.ymax - ib.ymin) * imgH;
        const sw = (evx1 - evx0) / (ib.xmax - ib.xmin) * imgW;
        const sh = (evy1 - evy0) / (ib.ymax - ib.ymin) * imgH;
        ctx.drawImage(img, sx, sy, sw, sh, dx, dy, dw, dh);
        ctx.restore();
    }

    function redraw() {
        drawMapImage();
        // Tool overlay (uses rotated worldToCanvas)
        __TOOL_DRAW__
        // Scale bar (always screen-aligned)
        const scaleM = Math.pow(10, Math.floor(Math.log10((vx1-vx0)*0.2)));
        const scalePx = m2px(scaleM);
        ctx.fillStyle = '#333'; ctx.font = '11px monospace';
        ctx.fillText(scaleM >= 1000 ? (scaleM/1000)+'km' : scaleM+'m', 15, H-20);
        ctx.strokeStyle = '#333'; ctx.lineWidth = 2; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(15, H-12); ctx.lineTo(15+scalePx, H-12); ctx.stroke();
        // Rotation indicator
        if (Math.abs(viewRotation) > 0.01) {
            const deg = (viewRotation * 180 / Math.PI).toFixed(0);
            ctx.fillStyle = '#666'; ctx.font = '11px monospace';
            ctx.fillText(deg + '\u00B0', W - 40, H - 20);
        }
    }

    function scheduleTileRefresh() {
        clearTimeout(tileTimer);
        tileTimer = setTimeout(async () => {
            const m = rotMargin() * 1.1;
            const ecx = (vx0+vx1)/2, ecy = (vy0+vy1)/2;
            const ehw = (vx1-vx0)/2*m, ehh = (vy1-vy0)/2*m;
            const vpJson = JSON.stringify({
                xmin: ecx-ehw, ymin: ecy-ehh, xmax: ecx+ehw, ymax: ecy+ehh,
                canvas_w: 2000, canvas_h: 1600
            });
            try {
                const b64 = await server.render_map(vpJson);
                tileImg = new Image();
                tileBounds = {xmin: ecx-ehw, ymin: ecy-ehh, xmax: ecx+ehw, ymax: ecy+ehh};
                tileImg.onload = redraw;
                tileImg.src = 'data:image/png;base64,' + b64;
            } catch(e) { console.warn('tile render failed', e); }
        }, 400);
    }

    // ── Mouse handlers ──
    canvas.addEventListener('mousedown', (e) => {
        const r = canvas.getBoundingClientRect();
        const cx = e.clientX - r.left, cy = e.clientY - r.top;
        if (e.altKey) {
            isRotating = true;
            rotateStartX = cx;
            canvas.style.cursor = 'crosshair';
        } else {
            // Check tool first, then fall back to pan
            let toolHandled = false;
            __TOOL_MOUSEDOWN__
            if (!toolHandled) {
                isPanning = true;
                panPrev = {x: cx, y: cy};
                canvas.style.cursor = 'grabbing';
            }
        }
    });
    canvas.addEventListener('mousemove', (e) => {
        const r = canvas.getBoundingClientRect();
        const cx = e.clientX - r.left, cy = e.clientY - r.top;
        if (isRotating) {
            const dx = cx - rotateStartX;
            viewRotation += dx * 0.005;
            rotateStartX = cx;
            redraw();
            scheduleTileRefresh();
        } else if (isPanning && panPrev) {
            const dx = cx - panPrev.x, dy = cy - panPrev.y;
            panPrev = {x: cx, y: cy};
            // Rotate screen delta by -viewRotation so pan follows cursor when rotated
            const c = Math.cos(-viewRotation), s = Math.sin(-viewRotation);
            const rdx = dx * c - dy * s, rdy = dx * s + dy * c;
            const dwx = (rdx / W) * (vx1 - vx0);
            const dwy = (rdy / H) * (vy1 - vy0);
            vx0 -= dwx; vx1 -= dwx;
            vy0 += dwy; vy1 += dwy;
            redraw();
            scheduleTileRefresh();
        } else {
            __TOOL_MOUSEMOVE__
        }
    });
    canvas.addEventListener('mouseup', (e) => {
        const r = canvas.getBoundingClientRect();
        const cx = e.clientX - r.left, cy = e.clientY - r.top;
        if (isRotating) {
            isRotating = false;
            canvas.style.cursor = 'grab';
        } else if (isPanning) {
            isPanning = false;
            panPrev = null;
            canvas.style.cursor = 'grab';
        } else {
            __TOOL_MOUSEUP__
        }
    });
    canvas.addEventListener('mouseleave', () => {
        isPanning = false; panPrev = null;
        isRotating = false;
        canvas.style.cursor = 'grab';
    });
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const r = canvas.getBoundingClientRect();
        const cx = e.clientX - r.left, cy = e.clientY - r.top;
        const factor = e.deltaY > 0 ? 1.25 : 0.8;
        // Zoom centered on cursor (in world coords, accounting for rotation)
        const cw = canvasToWorld(cx, cy);
        const nw = (vx1 - vx0) * factor, nh = (vy1 - vy0) * factor;
        const ecx = (vx0+vx1)/2, ecy = (vy0+vy1)/2;
        // Keep cursor world point fixed
        vx0 = cw.x - (cw.x - vx0) / (vx1 - vx0) * nw;
        vx1 = vx0 + nw;
        vy0 = cw.y - (cw.y - vy0) / (vy1 - vy0) * nh;
        vy1 = vy0 + nh;
        redraw();
        scheduleTileRefresh();
    }, {passive: false});

    // Double-click to reset rotation
    canvas.addEventListener('dblclick', () => {
        viewRotation = 0;
        redraw();
        scheduleTileRefresh();
    });

    __EXTRA_PROPS__

    // Load full map image
    const b64 = props.map_b64;
    if (b64) {
        fullImg = new Image();
        fullImg.onload = redraw;
        fullImg.src = 'data:image/png;base64,' + b64;
    }
})();
"""

# ── Tool JS snippets ─────────────────────────────────────────────────────────

ARROW_TOOL = {
    "help": " | <b>Shift+drag=arrow</b>",
    "state": """
    let arrowStartWorld = null, arrowEndWorld = null;
    let isDrawingArrow = false, arrowStartPx = null, arrowEndPx = null;
    """,
    "draw": """
    // Recompute arrow pixels from world coords
    if (arrowStartWorld && arrowEndWorld && !isDrawingArrow) {
        arrowStartPx = worldToCanvas(arrowStartWorld.x, arrowStartWorld.y);
        arrowEndPx = worldToCanvas(arrowEndWorld.x, arrowEndWorld.y);
    }
    const radius = parseFloat(props.radius) || 50;
    if (arrowStartPx) {
        const rPx = m2px(radius);
        ctx.strokeStyle = 'rgba(255,68,0,0.5)'; ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.arc(arrowStartPx.x, arrowStartPx.y, rPx, 0, Math.PI*2); ctx.stroke();
        ctx.setLineDash([]);
        const htol = parseFloat(props.heading_tol) || 30;
        if (arrowEndPx) {
            const adx = arrowEndPx.x - arrowStartPx.x, ady = arrowEndPx.y - arrowStartPx.y;
            const ang = Math.atan2(ady, adx), halfTol = htol * Math.PI / 180;
            ctx.fillStyle = 'rgba(255,68,0,0.08)';
            ctx.beginPath(); ctx.moveTo(arrowStartPx.x, arrowStartPx.y);
            ctx.arc(arrowStartPx.x, arrowStartPx.y, rPx, ang-halfTol, ang+halfTol);
            ctx.closePath(); ctx.fill();
        }
    }
    if (arrowStartPx && arrowEndPx) {
        const adx = arrowEndPx.x - arrowStartPx.x, ady = arrowEndPx.y - arrowStartPx.y;
        const len = Math.sqrt(adx*adx + ady*ady);
        if (len > 3) {
            const ang = Math.atan2(ady, adx);
            ctx.strokeStyle = '#ff4400'; ctx.lineWidth = 3;
            ctx.beginPath(); ctx.moveTo(arrowStartPx.x, arrowStartPx.y);
            ctx.lineTo(arrowEndPx.x, arrowEndPx.y); ctx.stroke();
            const hl = Math.min(18, len*0.3);
            ctx.fillStyle = '#ff4400'; ctx.beginPath();
            ctx.moveTo(arrowEndPx.x, arrowEndPx.y);
            ctx.lineTo(arrowEndPx.x - hl*Math.cos(ang-0.4), arrowEndPx.y - hl*Math.sin(ang-0.4));
            ctx.lineTo(arrowEndPx.x - hl*Math.cos(ang+0.4), arrowEndPx.y - hl*Math.sin(ang+0.4));
            ctx.closePath(); ctx.fill();
        }
        ctx.fillStyle = '#ff4400'; ctx.beginPath();
        ctx.arc(arrowStartPx.x, arrowStartPx.y, 5, 0, Math.PI*2); ctx.fill();
    }
    """,
    "mousedown": """
    if (e.shiftKey) {
        isDrawingArrow = true;
        arrowStartPx = {x: cx, y: cy};
        arrowEndPx = {x: cx, y: cy};
        canvas.style.cursor = 'crosshair';
        toolHandled = true;
    }
    """,
    "mousemove": """
    if (isDrawingArrow) {
        arrowEndPx = {x: cx, y: cy};
        redraw();
    }
    """,
    "mouseup": """
    if (isDrawingArrow && arrowStartPx && arrowEndPx) {
        isDrawingArrow = false;
        canvas.style.cursor = 'grab';
        arrowStartWorld = canvasToWorld(arrowStartPx.x, arrowStartPx.y);
        arrowEndWorld = canvasToWorld(arrowEndPx.x, arrowEndPx.y);
        const hdeg = Math.atan2(arrowEndWorld.y - arrowStartWorld.y,
                                 arrowEndWorld.x - arrowStartWorld.x) * 180 / Math.PI;
        trigger('click', {x: arrowStartWorld.x, y: arrowStartWorld.y, heading: hdeg});
        redraw();
    }
    """,
}

RECTANGLE_TOOL = {
    "help": " | <b>Ctrl+drag=rectangle</b>",
    "state": """
    let rectWorld = null;
    let isDrawingRect = false;
    let rectStartPx = null, rectEndPx = null;
    """,
    "draw": """
    if (rectStartPx && rectEndPx) {
        const rx = Math.min(rectStartPx.x, rectEndPx.x);
        const ry = Math.min(rectStartPx.y, rectEndPx.y);
        const rw = Math.abs(rectEndPx.x - rectStartPx.x);
        const rh = Math.abs(rectEndPx.y - rectStartPx.y);
        ctx.strokeStyle = 'rgba(51, 102, 204, 0.8)'; ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]); ctx.strokeRect(rx, ry, rw, rh); ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(51, 102, 204, 0.08)'; ctx.fillRect(rx, ry, rw, rh);
        if (rectWorld) {
            const wm = Math.abs(rectWorld.x2 - rectWorld.x1);
            const hm = Math.abs(rectWorld.y2 - rectWorld.y1);
            ctx.fillStyle = '#3366cc'; ctx.font = '11px monospace';
            ctx.fillText(wm.toFixed(0) + 'm x ' + hm.toFixed(0) + 'm', rx + 4, ry + 14);
        }
    }
    """,
    "mousedown": """
    if (e.ctrlKey || e.metaKey) {
        isDrawingRect = true;
        rectStartPx = {x: cx, y: cy};
        rectEndPx = {x: cx, y: cy};
        canvas.style.cursor = 'crosshair';
        toolHandled = true;
    }
    """,
    "mousemove": """
    if (isDrawingRect) {
        rectEndPx = {x: cx, y: cy};
        const c1 = canvasToWorld(rectStartPx.x, rectStartPx.y);
        const c2 = canvasToWorld(rectEndPx.x, rectStartPx.y);
        const c3 = canvasToWorld(rectEndPx.x, rectEndPx.y);
        const c4 = canvasToWorld(rectStartPx.x, rectEndPx.y);
        const xs = [c1.x, c2.x, c3.x, c4.x], ys = [c1.y, c2.y, c3.y, c4.y];
        rectWorld = {
            x1: Math.min(...xs), y1: Math.min(...ys),
            x2: Math.max(...xs), y2: Math.max(...ys)
        };
        redraw();
    }
    """,
    "mouseup": """
    if (isDrawingRect && rectStartPx && rectEndPx) {
        isDrawingRect = false;
        canvas.style.cursor = 'grab';
        const c1 = canvasToWorld(rectStartPx.x, rectStartPx.y);
        const c2 = canvasToWorld(rectEndPx.x, rectStartPx.y);
        const c3 = canvasToWorld(rectEndPx.x, rectEndPx.y);
        const c4 = canvasToWorld(rectStartPx.x, rectEndPx.y);
        const xs = [c1.x, c2.x, c3.x, c4.x], ys = [c1.y, c2.y, c3.y, c4.y];
        rectWorld = {
            x1: Math.min(...xs), y1: Math.min(...ys),
            x2: Math.max(...xs), y2: Math.max(...ys)
        };
        trigger('click', {
            type: 'rect',
            x1: rectWorld.x1, y1: rectWorld.y1,
            x2: rectWorld.x2, y2: rectWorld.y2,
            rotation: viewRotation
        });
        redraw();
    }
    """,
}

RECTANGLE_AND_ARROW_TOOL = {
    "help": (
        " | <b>Ctrl+drag=rectangle</b> | Use the <b>Mode</b> buttons "
        "to pick what a plain drag places (Pan / Start / Goal / Waypoint)"
    ),
    "state": """
    let rectWorld = null;
    let isDrawingRect = false;
    let rectStartPx = null, rectEndPx = null;
    // Start (blue) arrow — placed by mode="start" or Shift+drag (legacy).
    let egoArrowStartWorld = null, egoArrowEndWorld = null;
    let isDrawingEgoArrow = false, egoArrowStartPx = null, egoArrowEndPx = null;
    // Goal (red) arrow — placed by mode="goal".
    let goalArrowStartWorld = null, goalArrowEndWorld = null;
    let isDrawingGoalArrow = false, goalArrowStartPx = null, goalArrowEndPx = null;
    // Waypoint (yellow) arrows — appended by mode="waypoint".
    let waypointArrows = [];  // array of {start: {x,y}, end: {x,y}}
    let isDrawingWaypoint = false, waypointDragStartPx = null, waypointDragEndPx = null;
    // Route polyline for visualization — set via window.__setRoute(json).
    // Format: [[[x0,y0],[x1,y1],...], ...] one polyline per resolved lanelet.
    let routePolylines = [];
    // Mode-selector state. Valid values: "pan" (default) | "start" | "goal" |
    // "waypoint". Python pushes updates via window.__setMode(mode).
    let currentMode = "pan";
    window.__setMode = function(mode) {
        currentMode = mode || "pan";
        // Cursor hint so the user sees the mode is active.
        if (currentMode === "pan") {
            canvas.style.cursor = 'grab';
        } else {
            canvas.style.cursor = 'crosshair';
        }
    };
    window.__setRoute = function(json) {
        try {
            routePolylines = typeof json === 'string' ? JSON.parse(json) : (json || []);
        } catch (e) { routePolylines = []; console.warn('bad route json', e); }
        redraw();
    };
    // Python is the source of truth for all arrow positions — these setters
    // overwrite the JS-local state after every change so the viz is always
    // consistent with the Gradio state (fixes the stale-arrow bug where
    // clearing waypoints didn't wipe their viz).
    window.__setWaypointArrows = function(json) {
        try {
            const parsed = typeof json === 'string' ? JSON.parse(json) : (json || []);
            waypointArrows = Array.isArray(parsed) ? parsed : [];
        } catch (e) { waypointArrows = []; console.warn('bad waypoints json', e); }
        redraw();
    };
    window.__setStartArrow = function(json) {
        try {
            const d = typeof json === 'string' ? JSON.parse(json) : json;
            if (d && d.start && d.end) {
                egoArrowStartWorld = {x: d.start[0], y: d.start[1]};
                egoArrowEndWorld = {x: d.end[0], y: d.end[1]};
                egoArrowStartPx = null; egoArrowEndPx = null; // force recompute from world
            } else {
                egoArrowStartWorld = null; egoArrowEndWorld = null;
                egoArrowStartPx = null; egoArrowEndPx = null;
            }
        } catch (e) { console.warn('bad start-arrow json', e); }
        redraw();
    };
    window.__setGoalArrow = function(json) {
        try {
            const d = typeof json === 'string' ? JSON.parse(json) : json;
            if (d && d.start && d.end) {
                goalArrowStartWorld = {x: d.start[0], y: d.start[1]};
                goalArrowEndWorld = {x: d.end[0], y: d.end[1]};
                goalArrowStartPx = null; goalArrowEndPx = null;
            } else {
                goalArrowStartWorld = null; goalArrowEndWorld = null;
                goalArrowStartPx = null; goalArrowEndPx = null;
            }
        } catch (e) { console.warn('bad goal-arrow json', e); }
        redraw();
    };
    // Back-compat aliases — kept so old handlers still work.
    window.__clearWaypoints = function() { window.__setWaypointArrows('[]'); };
    window.__clearStart = function() { window.__setStartArrow('null'); };
    window.__clearGoal = function() { window.__setGoalArrow('null'); };
    """,
    "draw": """
    // Reusable arrow renderer: colour, start, end (all in canvas pixels), label.
    function drawArrow(color, sx, sy, ex, ey, label) {
        const adx = ex - sx, ady = ey - sy;
        const len = Math.sqrt(adx*adx + ady*ady);
        ctx.strokeStyle = color; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
        if (len > 3) {
            const ang = Math.atan2(ady, adx);
            const hl = Math.min(18, len*0.3);
            ctx.fillStyle = color; ctx.beginPath();
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - hl*Math.cos(ang-0.4), ey - hl*Math.sin(ang-0.4));
            ctx.lineTo(ex - hl*Math.cos(ang+0.4), ey - hl*Math.sin(ang+0.4));
            ctx.closePath(); ctx.fill();
        }
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI*2); ctx.fill();
        if (label) {
            ctx.fillStyle = color; ctx.font = 'bold 11px monospace';
            ctx.fillText(label, sx + 8, sy - 8);
        }
    }

    // Route polyline (drawn underneath arrows and rectangle so arrows stay visible)
    if (routePolylines && routePolylines.length > 0) {
        ctx.strokeStyle = 'rgba(0, 170, 68, 0.85)';  // green
        ctx.lineWidth = 4;
        ctx.setLineDash([]);
        for (const line of routePolylines) {
            if (!line || line.length < 2) continue;
            ctx.beginPath();
            const p0 = worldToCanvas(line[0][0], line[0][1]);
            ctx.moveTo(p0.x, p0.y);
            for (let i = 1; i < line.length; i++) {
                const p = worldToCanvas(line[i][0], line[i][1]);
                ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();
        }
    }

    // Rectangle
    if (rectStartPx && rectEndPx) {
        const rx = Math.min(rectStartPx.x, rectEndPx.x);
        const ry = Math.min(rectStartPx.y, rectEndPx.y);
        const rw = Math.abs(rectEndPx.x - rectStartPx.x);
        const rh = Math.abs(rectEndPx.y - rectStartPx.y);
        ctx.strokeStyle = 'rgba(51, 102, 204, 0.8)'; ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]); ctx.strokeRect(rx, ry, rw, rh); ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(51, 102, 204, 0.08)'; ctx.fillRect(rx, ry, rw, rh);
        if (rectWorld) {
            const wm = Math.abs(rectWorld.x2 - rectWorld.x1);
            const hm = Math.abs(rectWorld.y2 - rectWorld.y1);
            ctx.fillStyle = '#3366cc'; ctx.font = '11px monospace';
            ctx.fillText(wm.toFixed(0) + 'm x ' + hm.toFixed(0) + 'm', rx + 4, ry + 14);
        }
    }

    // Ego start arrow (blue)
    if (egoArrowStartWorld && egoArrowEndWorld && !isDrawingEgoArrow) {
        egoArrowStartPx = worldToCanvas(egoArrowStartWorld.x, egoArrowStartWorld.y);
        egoArrowEndPx = worldToCanvas(egoArrowEndWorld.x, egoArrowEndWorld.y);
    }
    if (egoArrowStartPx && egoArrowEndPx) {
        drawArrow('#3366cc', egoArrowStartPx.x, egoArrowStartPx.y,
                  egoArrowEndPx.x, egoArrowEndPx.y, 'start');
    }

    // Goal arrow (red)
    if (goalArrowStartWorld && goalArrowEndWorld && !isDrawingGoalArrow) {
        goalArrowStartPx = worldToCanvas(goalArrowStartWorld.x, goalArrowStartWorld.y);
        goalArrowEndPx = worldToCanvas(goalArrowEndWorld.x, goalArrowEndWorld.y);
    }
    if (goalArrowStartPx && goalArrowEndPx) {
        drawArrow('#cc3333', goalArrowStartPx.x, goalArrowStartPx.y,
                  goalArrowEndPx.x, goalArrowEndPx.y, 'goal');
    }

    // Waypoint arrows (yellow, numbered)
    for (let i = 0; i < waypointArrows.length; i++) {
        const w = waypointArrows[i];
        const sp = worldToCanvas(w.start.x, w.start.y);
        const ep = worldToCanvas(w.end.x, w.end.y);
        drawArrow('#e6a400', sp.x, sp.y, ep.x, ep.y, (i + 1).toString());
    }
    // Waypoint in-progress drag
    if (isDrawingWaypoint && waypointDragStartPx && waypointDragEndPx) {
        drawArrow('#e6a400', waypointDragStartPx.x, waypointDragStartPx.y,
                  waypointDragEndPx.x, waypointDragEndPx.y,
                  (waypointArrows.length + 1).toString());
    }
    """,
    "mousedown": """
    // Ctrl+drag always = rectangle (legacy snippet flow). Everything else
    // routes through the Mode radio (Pan / Start / Goal / Waypoint).
    // Shift+drag stays as a power-user shortcut for start, regardless of mode.
    if (e.ctrlKey || e.metaKey) {
        isDrawingRect = true;
        rectStartPx = {x: cx, y: cy};
        rectEndPx = {x: cx, y: cy};
        canvas.style.cursor = 'crosshair';
        toolHandled = true;
    } else if (e.shiftKey || currentMode === "start") {
        isDrawingEgoArrow = true;
        egoArrowStartPx = {x: cx, y: cy};
        egoArrowEndPx = {x: cx, y: cy};
        canvas.style.cursor = 'crosshair';
        toolHandled = true;
    } else if (currentMode === "goal") {
        isDrawingGoalArrow = true;
        goalArrowStartPx = {x: cx, y: cy};
        goalArrowEndPx = {x: cx, y: cy};
        canvas.style.cursor = 'crosshair';
        toolHandled = true;
    } else if (currentMode === "waypoint") {
        isDrawingWaypoint = true;
        waypointDragStartPx = {x: cx, y: cy};
        waypointDragEndPx = {x: cx, y: cy};
        canvas.style.cursor = 'crosshair';
        toolHandled = true;
    }
    // Else (currentMode === "pan"): fall through to base pan handler.
    """,
    "mousemove": """
    if (isDrawingRect) {
        rectEndPx = {x: cx, y: cy};
        const c1 = canvasToWorld(rectStartPx.x, rectStartPx.y);
        const c2 = canvasToWorld(rectEndPx.x, rectStartPx.y);
        const c3 = canvasToWorld(rectEndPx.x, rectEndPx.y);
        const c4 = canvasToWorld(rectStartPx.x, rectEndPx.y);
        const xs = [c1.x, c2.x, c3.x, c4.x], ys = [c1.y, c2.y, c3.y, c4.y];
        rectWorld = {
            x1: Math.min(...xs), y1: Math.min(...ys),
            x2: Math.max(...xs), y2: Math.max(...ys)
        };
        redraw();
    } else if (isDrawingEgoArrow) {
        egoArrowEndPx = {x: cx, y: cy};
        redraw();
    } else if (isDrawingGoalArrow) {
        goalArrowEndPx = {x: cx, y: cy};
        redraw();
    } else if (isDrawingWaypoint) {
        waypointDragEndPx = {x: cx, y: cy};
        redraw();
    }
    """,
    "mouseup": """
    if (isDrawingRect && rectStartPx && rectEndPx) {
        isDrawingRect = false;
        canvas.style.cursor = 'grab';
        const c1 = canvasToWorld(rectStartPx.x, rectStartPx.y);
        const c2 = canvasToWorld(rectEndPx.x, rectStartPx.y);
        const c3 = canvasToWorld(rectEndPx.x, rectEndPx.y);
        const c4 = canvasToWorld(rectStartPx.x, rectEndPx.y);
        const xs = [c1.x, c2.x, c3.x, c4.x], ys = [c1.y, c2.y, c3.y, c4.y];
        rectWorld = {
            x1: Math.min(...xs), y1: Math.min(...ys),
            x2: Math.max(...xs), y2: Math.max(...ys)
        };
        trigger('click', {
            type: 'rect',
            x1: rectWorld.x1, y1: rectWorld.y1,
            x2: rectWorld.x2, y2: rectWorld.y2,
            rotation: viewRotation
        });
        redraw();
    } else if (isDrawingEgoArrow && egoArrowStartPx && egoArrowEndPx) {
        isDrawingEgoArrow = false;
        canvas.style.cursor = 'grab';
        egoArrowStartWorld = canvasToWorld(egoArrowStartPx.x, egoArrowStartPx.y);
        egoArrowEndWorld = canvasToWorld(egoArrowEndPx.x, egoArrowEndPx.y);
        const hdeg = Math.atan2(egoArrowEndWorld.y - egoArrowStartWorld.y,
                                 egoArrowEndWorld.x - egoArrowStartWorld.x) * 180 / Math.PI;
        trigger('click', {
            type: 'ego_pose',
            x: egoArrowStartWorld.x, y: egoArrowStartWorld.y, heading: hdeg
        });
        redraw();
    } else if (isDrawingGoalArrow && goalArrowStartPx && goalArrowEndPx) {
        isDrawingGoalArrow = false;
        canvas.style.cursor = 'grab';
        goalArrowStartWorld = canvasToWorld(goalArrowStartPx.x, goalArrowStartPx.y);
        goalArrowEndWorld = canvasToWorld(goalArrowEndPx.x, goalArrowEndPx.y);
        const hdeg = Math.atan2(goalArrowEndWorld.y - goalArrowStartWorld.y,
                                 goalArrowEndWorld.x - goalArrowStartWorld.x) * 180 / Math.PI;
        trigger('click', {
            type: 'goal_pose',
            x: goalArrowStartWorld.x, y: goalArrowStartWorld.y, heading: hdeg
        });
        redraw();
    } else if (isDrawingWaypoint && waypointDragStartPx && waypointDragEndPx) {
        isDrawingWaypoint = false;
        canvas.style.cursor = 'grab';
        const ws = canvasToWorld(waypointDragStartPx.x, waypointDragStartPx.y);
        const we = canvasToWorld(waypointDragEndPx.x, waypointDragEndPx.y);
        const hdeg = Math.atan2(we.y - ws.y, we.x - ws.x) * 180 / Math.PI;
        waypointArrows.push({start: ws, end: we});
        trigger('click', {
            type: 'waypoint_append',
            x: ws.x, y: ws.y, heading: hdeg
        });
        redraw();
    }
    """,
}


def build_map_canvas_js(
    tool: str | dict | None = None,
    extra_props_js: str = "",
) -> str:
    """Build the complete JS for a map canvas with optional tool overlay.

    Args:
        tool: "arrow", "rectangle", a custom dict with keys
              {help, state, draw, mousedown, mousemove, mouseup}, or None.
        extra_props_js: Additional JS to run after canvas init (e.g., read extra props).

    Returns:
        Complete JS string for ``gr.HTML(js_on_load=...)``.
    """
    if isinstance(tool, str):
        tool = {
            "arrow": ARROW_TOOL,
            "rectangle": RECTANGLE_TOOL,
            "rectangle_and_arrow": RECTANGLE_AND_ARROW_TOOL,
        }[tool]

    if tool is None:
        tool = {k: "" for k in ("help", "state", "draw", "mousedown", "mousemove", "mouseup")}

    js = _BASE_JS
    js = js.replace("__TOOL_HELP__", tool.get("help", ""))
    js = js.replace("__TOOL_STATE__", tool.get("state", ""))
    js = js.replace("__TOOL_DRAW__", tool.get("draw", ""))
    js = js.replace("__TOOL_MOUSEDOWN__", tool.get("mousedown", ""))
    js = js.replace("__TOOL_MOUSEMOVE__", tool.get("mousemove", ""))
    js = js.replace("__TOOL_MOUSEUP__", tool.get("mouseup", ""))
    js = js.replace("__EXTRA_PROPS__", extra_props_js)
    return js
