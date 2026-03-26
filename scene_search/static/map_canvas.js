/**
 * Interactive map canvas for scene search GUI.
 *
 * Supports: drag-to-pan, scroll-to-zoom, shift+drag to draw arrow.
 * Communicates with Gradio via hidden textbox elements.
 *
 * Expected HTML structure (created by Python):
 *   <div id="map-canvas-container">
 *     <canvas id="map-canvas" width="900" height="700"></canvas>
 *   </div>
 *
 * Hidden Gradio textboxes (elem_id):
 *   - arrow-data: JSON {x, y, heading_deg} in world coords
 *   - viewport-action: JSON {action, ...params} for pan/zoom requests
 */

(function () {
  "use strict";

  let canvas, ctx;
  let mapImage = null; // Current map background image
  let viewport = null; // {xmin, ymin, xmax, ymax, canvas_w, canvas_h}

  // Arrow state
  let arrowStart = null; // {px, py} in canvas pixels
  let arrowEnd = null;
  let arrowWorldStart = null; // {x, y} in MGRS
  let arrowHeadingDeg = null;
  let radiusMeters = 50;
  let isDrawingArrow = false;

  // Pan state
  let isPanning = false;
  let panStartPx = null;

  // Scene dots
  let sceneDots = []; // [{px, py}] in canvas pixels

  function init() {
    canvas = document.getElementById("map-canvas");
    if (!canvas) {
      setTimeout(init, 200);
      return;
    }
    ctx = canvas.getContext("2d");

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    // Prevent context menu on canvas
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    console.log("[map_canvas] initialized");
  }

  function pixelToWorld(px, py) {
    if (!viewport) return { x: 0, y: 0 };
    const w = viewport.xmax - viewport.xmin;
    const h = viewport.ymax - viewport.ymin;
    return {
      x: viewport.xmin + (px / viewport.canvas_w) * w,
      y: viewport.ymax - (py / viewport.canvas_h) * h,
    };
  }

  function worldToPixel(wx, wy) {
    if (!viewport) return { px: 0, py: 0 };
    const w = viewport.xmax - viewport.xmin;
    const h = viewport.ymax - viewport.ymin;
    return {
      px: ((wx - viewport.xmin) / w) * viewport.canvas_w,
      py: ((viewport.ymax - wy) / h) * viewport.canvas_h,
    };
  }

  function metersToPixels(meters) {
    if (!viewport) return 0;
    const w = viewport.xmax - viewport.xmin;
    return (meters / w) * viewport.canvas_w;
  }

  // --- Drawing ---

  function redraw() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background
    if (mapImage) {
      ctx.drawImage(mapImage, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#f0f0f0";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#888";
      ctx.font = "16px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Loading map...", canvas.width / 2, canvas.height / 2);
    }

    // Scene dots
    if (sceneDots.length > 0) {
      ctx.fillStyle = "rgba(0, 100, 255, 0.5)";
      for (const dot of sceneDots) {
        ctx.beginPath();
        ctx.arc(dot.px, dot.py, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Radius circle
    if (arrowStart) {
      const radiusPx = metersToPixels(radiusMeters);
      ctx.strokeStyle = "rgba(255, 100, 0, 0.5)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.arc(arrowStart.px, arrowStart.py, radiusPx, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Arrow
    if (arrowStart && arrowEnd) {
      drawArrow(arrowStart.px, arrowStart.py, arrowEnd.px, arrowEnd.py);
    }

    // Heading tolerance wedge
    if (arrowStart && arrowHeadingDeg !== null) {
      drawHeadingWedge();
    }
  }

  function drawArrow(x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 5) return;

    const angle = Math.atan2(dy, dx);
    const headLen = Math.min(20, len * 0.3);

    // Shaft
    ctx.strokeStyle = "#ff4400";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Arrowhead
    ctx.fillStyle = "#ff4400";
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - headLen * Math.cos(angle - Math.PI / 6),
      y2 - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      x2 - headLen * Math.cos(angle + Math.PI / 6),
      y2 - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();

    // Start dot
    ctx.fillStyle = "#ff4400";
    ctx.beginPath();
    ctx.arc(x1, y1, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawHeadingWedge() {
    // Draw a transparent wedge showing heading tolerance
    const toleranceEl = document.querySelector("#heading-tolerance-value");
    const tolerance = toleranceEl ? parseFloat(toleranceEl.textContent) || 30 : 30;

    const radiusPx = metersToPixels(radiusMeters);
    // Canvas angles: 0 = right, positive = clockwise (Y inverted)
    // World heading: degrees CCW from +X → canvas angle = -heading in radians
    const centerAngle = -(arrowHeadingDeg * Math.PI) / 180;
    const halfTol = (tolerance * Math.PI) / 180;

    ctx.fillStyle = "rgba(255, 100, 0, 0.1)";
    ctx.beginPath();
    ctx.moveTo(arrowStart.px, arrowStart.py);
    ctx.arc(
      arrowStart.px,
      arrowStart.py,
      radiusPx,
      centerAngle - halfTol,
      centerAngle + halfTol
    );
    ctx.closePath();
    ctx.fill();
  }

  // --- Event Handlers ---

  function onMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    if (e.shiftKey) {
      // Shift+drag = draw arrow
      isDrawingArrow = true;
      arrowStart = { px, py };
      arrowEnd = { px, py };
      arrowWorldStart = pixelToWorld(px, py);
    } else {
      // Normal drag = pan
      isPanning = true;
      panStartPx = { px, py };
    }
  }

  function onMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    if (isDrawingArrow) {
      arrowEnd = { px, py };
      // Compute heading from start→end (world coords, Y up)
      const endWorld = pixelToWorld(px, py);
      const dx = endWorld.x - arrowWorldStart.x;
      const dy = endWorld.y - arrowWorldStart.y;
      arrowHeadingDeg = (Math.atan2(dy, dx) * 180) / Math.PI;
      redraw();
    } else if (isPanning) {
      const dx = px - panStartPx.px;
      const dy = py - panStartPx.py;
      panStartPx = { px, py };
      sendViewportAction({ action: "pan", dx: dx, dy: dy });
    }
  }

  function onMouseUp(e) {
    if (isDrawingArrow) {
      isDrawingArrow = false;
      if (arrowStart && arrowEnd && arrowWorldStart && arrowHeadingDeg !== null) {
        // Send arrow data to Gradio
        sendArrowData({
          x: arrowWorldStart.x,
          y: arrowWorldStart.y,
          heading_deg: arrowHeadingDeg,
        });
      }
    }
    if (isPanning) {
      isPanning = false;
      panStartPx = null;
    }
  }

  function onWheel(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const factor = e.deltaY > 0 ? 1.3 : 1 / 1.3; // Scroll down = zoom out
    sendViewportAction({ action: "zoom", factor: factor, px: px, py: py });
  }

  // --- Gradio Communication ---

  function setGradioTextbox(elemId, value) {
    // Find the Gradio textbox by elem_id and update its value
    const container = document.querySelector(`#${elemId}`);
    if (!container) return;
    const input =
      container.querySelector("textarea") || container.querySelector("input");
    if (!input) return;

    const nativeSet = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype || window.HTMLInputElement.prototype,
      "value"
    );
    if (nativeSet && nativeSet.set) {
      nativeSet.set.call(input, value);
    } else {
      input.value = value;
    }
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function sendArrowData(data) {
    setGradioTextbox("arrow-data", JSON.stringify(data));
  }

  function sendViewportAction(action) {
    setGradioTextbox("viewport-action", JSON.stringify(action));
  }

  // --- Public API (called from Python via gr.HTML js updates) ---

  window.mapCanvas = {
    setMapImage: function (base64Png) {
      const img = new Image();
      img.onload = function () {
        mapImage = img;
        redraw();
      };
      img.src = "data:image/png;base64," + base64Png;
    },

    setViewport: function (vp) {
      viewport = vp;
    },

    setRadius: function (r) {
      radiusMeters = r;
      redraw();
    },

    setSceneDots: function (dots) {
      // dots = [{x, y}] in world coords
      sceneDots = dots.map((d) => worldToPixel(d.x, d.y));
      redraw();
    },

    clearArrow: function () {
      arrowStart = null;
      arrowEnd = null;
      arrowWorldStart = null;
      arrowHeadingDeg = null;
      sceneDots = [];
      redraw();
    },

    redraw: redraw,
  };

  // Auto-init when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
