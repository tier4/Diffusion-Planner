"""Lanelet2 map loading, viewport management, and PNG tile rendering.

Loads the lanelet2 map once at startup, caches centerline and boundary geometry
as numpy arrays, and renders cropped views to PNG bytes on demand. Provides
pixel-to-world coordinate transforms for the JS canvas overlay.

Requires ROS + Autoware environment sourced for lanelet2 / MGRSProjector imports.
"""

import base64
import io
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

# lanelet2 requires ROS/Autoware Python paths (set by sourcing setup.bash).
_ROS_FALLBACK_PATHS = ["/opt/ros/humble/lib/python3.10/site-packages"]
_AUTOWARE_DIR = os.environ.get("AUTOWARE_INSTALL", os.path.expanduser("~/autoware/install"))
_ROS_FALLBACK_PATHS.append(
    f"{_AUTOWARE_DIR}/autoware_lanelet2_extension_python/local/lib/python3.10/dist-packages"
)
for _p in _ROS_FALLBACK_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class Viewport:
    """Axis-aligned rectangular region in MGRS world coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    canvas_w: int = 900
    canvas_h: int = 700

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        wx = self.xmin + (px / self.canvas_w) * self.width
        wy = self.ymax - (py / self.canvas_h) * self.height
        return wx, wy

    def world_to_pixel(self, wx: float, wy: float) -> tuple[float, float]:
        px = (wx - self.xmin) / self.width * self.canvas_w
        py = (self.ymax - wy) / self.height * self.canvas_h
        return px, py

    def zoom(self, factor: float, center_px: float = None, center_py: float = None):
        if center_px is not None and center_py is not None:
            cx, cy = self.pixel_to_world(center_px, center_py)
        else:
            cx, cy = self.center
        new_w = self.width * factor
        new_h = self.height * factor
        return Viewport(
            xmin=cx - new_w / 2,
            ymin=cy - new_h / 2,
            xmax=cx + new_w / 2,
            ymax=cy + new_h / 2,
            canvas_w=self.canvas_w,
            canvas_h=self.canvas_h,
        )

    def pan(self, dx_px: float, dy_px: float):
        dx_world = (dx_px / self.canvas_w) * self.width
        dy_world = -(dy_px / self.canvas_h) * self.height
        return Viewport(
            xmin=self.xmin - dx_world,
            ymin=self.ymin - dy_world,
            xmax=self.xmax - dx_world,
            ymax=self.ymax - dy_world,
            canvas_w=self.canvas_w,
            canvas_h=self.canvas_h,
        )

    def to_json(self) -> dict:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "canvas_w": self.canvas_w,
            "canvas_h": self.canvas_h,
        }

    @staticmethod
    def from_json(d: dict) -> "Viewport":
        return Viewport(
            **{k: d[k] for k in ("xmin", "ymin", "xmax", "ymax", "canvas_w", "canvas_h") if k in d}
        )


class MapRenderer:
    """Loads a lanelet2 map and renders cropped viewport images."""

    def __init__(self, lanelet_path: str):
        import lanelet2
        from autoware_lanelet2_extension_python.projection import MGRSProjector

        projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        lanelet_map = lanelet2.io.load(str(lanelet_path), projection)

        self.vehicle_segments: list[np.ndarray] = []
        self.pedestrian_segments: list[np.ndarray] = []

        for ll in lanelet_map.laneletLayer:
            subtype = ll.attributes["subtype"] if "subtype" in ll.attributes else ""
            pts = np.array([(p.x, p.y) for p in ll.centerline])
            if len(pts) < 2:
                continue
            if subtype == "pedestrian_lane":
                self.pedestrian_segments.append(pts)
            else:
                self.vehicle_segments.append(pts)

        self.boundary_segments: list[np.ndarray] = []
        for ll in lanelet_map.laneletLayer:
            subtype = ll.attributes["subtype"] if "subtype" in ll.attributes else ""
            if subtype == "pedestrian_lane":
                continue
            for bound in [ll.leftBound, ll.rightBound]:
                pts = np.array([(p.x, p.y) for p in bound])
                if len(pts) >= 2:
                    self.boundary_segments.append(pts)

        all_pts = np.vstack(self.vehicle_segments + self.pedestrian_segments)
        margin = 50.0
        self.full_bounds = Viewport(
            xmin=float(all_pts[:, 0].min()) - margin,
            ymin=float(all_pts[:, 1].min()) - margin,
            xmax=float(all_pts[:, 0].max()) + margin,
            ymax=float(all_pts[:, 1].max()) + margin,
        )

        print(
            f"MapRenderer: loaded {len(self.vehicle_segments)} vehicle lanes, "
            f"{len(self.pedestrian_segments)} pedestrian lanes, "
            f"{len(self.boundary_segments)} boundary segments"
        )
        print(
            f"  Map bounds: x=[{self.full_bounds.xmin:.0f}, {self.full_bounds.xmax:.0f}] "
            f"y=[{self.full_bounds.ymin:.0f}, {self.full_bounds.ymax:.0f}]"
        )

    def _crop_segments(self, segments: list[np.ndarray], vp: Viewport) -> list[np.ndarray]:
        margin = max(vp.width, vp.height) * 0.05
        xmin, ymin = vp.xmin - margin, vp.ymin - margin
        xmax, ymax = vp.xmax + margin, vp.ymax + margin
        result = []
        for seg in segments:
            if seg.size == 0:
                continue
            if (
                seg[:, 0].max() < xmin
                or seg[:, 0].min() > xmax
                or seg[:, 1].max() < ymin
                or seg[:, 1].min() > ymax
            ):
                continue
            result.append(seg)
        return result

    def render_viewport(self, viewport: Viewport, dpi: int = 100) -> bytes:
        fig_w = viewport.canvas_w / dpi
        fig_h = viewport.canvas_h / dpi
        fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])

        cropped_boundaries = self._crop_segments(self.boundary_segments, viewport)
        if cropped_boundaries:
            ax.add_collection(LineCollection(cropped_boundaries, colors="#404040", linewidths=0.5))

        cropped_vehicle = self._crop_segments(self.vehicle_segments, viewport)
        if cropped_vehicle:
            ax.add_collection(LineCollection(cropped_vehicle, colors="#000000", linewidths=0.8))

        cropped_ped = self._crop_segments(self.pedestrian_segments, viewport)
        if cropped_ped:
            ax.add_collection(LineCollection(cropped_ped, colors="#ffaaaa", linewidths=0.6))

        ax.set_xlim(viewport.xmin, viewport.xmax)
        ax.set_ylim(viewport.ymin, viewport.ymax)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("#f8f8f8")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def render_viewport_base64(self, viewport: Viewport, dpi: int = 100) -> str:
        return base64.b64encode(self.render_viewport(viewport, dpi=dpi)).decode("ascii")

    def initial_viewport(self, canvas_w: int = 900, canvas_h: int = 700) -> Viewport:
        bounds = self.full_bounds
        map_aspect = bounds.width / bounds.height
        canvas_aspect = canvas_w / canvas_h
        if map_aspect > canvas_aspect:
            w = bounds.width
            h = w / canvas_aspect
        else:
            h = bounds.height
            w = h * canvas_aspect
        cx, cy = bounds.center
        return Viewport(
            xmin=cx - w / 2,
            ymin=cy - h / 2,
            xmax=cx + w / 2,
            ymax=cy + h / 2,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
