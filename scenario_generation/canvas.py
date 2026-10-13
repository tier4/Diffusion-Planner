"""PIL figures for scenes and charts: panels in data coordinates, layered drawing, supersampled output.

A :class:`Figure` holds one or more :class:`Panel` boxes. Every panel primitive takes data
coordinates and a ``z`` layer; panels record the calls and draw them in ascending ``z`` (call
order within a layer) when the figure is composed, clipped to the panel box. Limits that are not
set are autoscaled from the recorded data with a 5 % margin, and ticks that are not set are
chosen on round numbers. Sizes are in points, as on a figure of ``size_in`` inches at ``dpi``.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from scenario_generation.colormaps import map_values

SUPERSAMPLE = 2
DASHES = {"dash": (3.7, 1.6), "dot": (1.0, 1.65), "dashdot": (6.4, 1.6, 1.0, 1.6)}
TEXT_COLOR = "#000000"
GRID_COLOR = "#b0b0b0"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_FILES = {
    ("sans", "normal"): "DejaVuSans.ttf",
    ("sans", "bold"): "DejaVuSans-Bold.ttf",
    ("mono", "normal"): "DejaVuSansMono.ttf",
    ("mono", "bold"): "DejaVuSansMono-Bold.ttf",
}
MARGIN = 0.05

Color = str | tuple[float, ...] | np.ndarray


def rgba(color: Color, alpha: float = 1.0) -> tuple[int, int, int, int]:
    """A colour name, ``#rgb``/``#rrggbb``/``#rrggbbaa`` or an RGB(A) tuple (floats in [0, 1] or
    uint8), times ``alpha``."""
    if isinstance(color, str):
        r, g, b, a = ImageColor.getcolor(color, "RGBA")
    else:
        values = np.asarray(color, dtype=np.float64)
        scale = 1.0 if np.issubdtype(np.asarray(color).dtype, np.integer) else 255.0
        r, g, b = (int(round(v * scale)) for v in values[:3])
        a = int(round(values[3] * scale)) if len(values) > 3 else 255
    return r, g, b, int(round(a * min(max(alpha, 0.0), 1.0)))


def is_single_color(color: Color | Sequence[Color]) -> bool:
    """Whether ``color`` is one colour (a string or a flat numeric RGB(A)) rather than a list."""
    return isinstance(color, str) or (
        np.ndim(color) == 1 and np.issubdtype(np.asarray(color).dtype, np.number)
    )


_FONTS: dict[tuple[int, str, str], ImageFont.FreeTypeFont] = {}


def font(size_px: float, weight: str = "normal", family: str = "sans") -> ImageFont.FreeTypeFont:
    """DejaVu Sans (or Sans Mono for ``family="mono"``) at ``size_px`` pixels, cached per size."""
    key = (max(1, round(size_px)), weight, family)
    if key not in _FONTS:
        path = FONT_DIR / FONT_FILES[(family, weight)]
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; install the fonts-dejavu-core package")
        _FONTS[key] = ImageFont.truetype(str(path), key[0])
    return _FONTS[key]


def finite_runs(points: np.ndarray) -> list[np.ndarray]:
    """Split a ``(N, 2)`` polyline at its non-finite points into the finite runs between them."""
    finite = np.isfinite(points).all(axis=1)
    runs = np.split(points, np.flatnonzero(~finite))
    return [run[np.isfinite(run).all(axis=1)] for run in runs]


def dash_pieces(points: np.ndarray, pattern: Sequence[float]) -> list[np.ndarray]:
    """Split a pixel polyline into the drawn pieces of an on/off ``pattern`` in pixels."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    offsets = np.cumsum([0.0, *pattern])
    pieces = []
    for cycle in np.arange(0.0, arc[-1], float(sum(pattern))):
        for start, end in zip(offsets[0::2], offsets[1::2]):
            a, b = cycle + start, min(cycle + end, arc[-1])
            if a >= b:
                continue
            s = np.concatenate([[a], arc[(arc > a) & (arc < b)], [b]])
            pieces.append(
                np.stack([np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])], 1)
            )
    return pieces


def oriented_box(
    x: float, y: float, heading: float, length: float, width: float, rear: float
) -> np.ndarray:
    """``(4, 2)`` corners of a box ``rear`` metres behind and ``length - rear`` ahead of ``(x, y)``."""
    c, s = math.cos(heading), math.sin(heading)
    local = np.array(
        [
            [-rear, -width / 2],
            [length - rear, -width / 2],
            [length - rear, width / 2],
            [-rear, width / 2],
        ]
    )
    return np.stack(
        [x + local[:, 0] * c - local[:, 1] * s, y + local[:, 0] * s + local[:, 1] * c], 1
    )


def nice_ticks(low: float, high: float, target: int = 6) -> np.ndarray:
    """Round tick values covering ``[low, high]``, about ``target`` of them."""
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return np.array([low]) if math.isfinite(low) else np.array([])
    raw = (high - low) / target
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * magnitude >= raw)
    first = math.ceil(low / step - 1e-9) * step
    ticks = np.arange(first, high + step * 1e-9, step)
    return np.round(ticks / step) * step


def tick_label(value: float) -> str:
    """A compact tick label: integers without decimals, others with up to 6 significant digits."""
    if abs(value) < 1e-12:
        return "0"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value)}".replace("-", "−")
    return f"{value:.6g}".replace("-", "−")


def _anchor(ha: str, va: str) -> str:
    """A PIL text anchor from matplotlib-style horizontal and vertical alignment names."""
    horizontal = {"left": "l", "center": "m", "right": "r"}[ha]
    vertical = {"top": "a", "center": "m", "bottom": "d", "baseline": "s"}[va]
    return horizontal + vertical


def _filled_cross(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    """The 12 corners of a filled diagonal cross reaching ``radius`` pixels from ``(x, y)``."""
    t = 0.3 * radius
    plus = np.array(
        [
            (t, radius),
            (t, t),
            (radius, t),
            (radius, -t),
            (t, -t),
            (t, -radius),
            (-t, -radius),
            (-t, -t),
            (-radius, -t),
            (-radius, t),
            (-t, t),
            (-t, radius),
        ]
    ) / math.sqrt(1.0 + (t / radius) ** 2)
    c = s = math.sqrt(0.5)
    return [(x + px * c - py * s, y + px * s + py * c) for px, py in plus]


def draw_marker(
    d: ImageDraw.ImageDraw,
    x: float,
    y: float,
    symbol: str,
    radius: float,
    face: tuple[int, int, int, int],
    edge: tuple[int, int, int, int] | None,
    edge_width: int,
) -> None:
    """One marker of ``radius`` pixels at pixel ``(x, y)``: ``o s ^ v D * x +`` or ``.``."""
    if symbol in ("x", "+"):
        offsets = (
            [(-1, -1, 1, 1), (-1, 1, 1, -1)] if symbol == "x" else [(-1, 0, 1, 0), (0, -1, 0, 1)]
        )
        for ax0, ay0, ax1, ay1 in offsets:
            d.line(
                [(x + ax0 * radius, y + ay0 * radius), (x + ax1 * radius, y + ay1 * radius)],
                fill=face,
                width=max(edge_width, 2),
            )
        return
    if symbol in ("o", "."):
        r = radius if symbol == "o" else radius * 0.5
        d.ellipse([x - r, y - r, x + r, y + r], fill=face, outline=edge, width=edge_width)
        return
    if symbol == "s":
        d.rectangle(
            [x - radius, y - radius, x + radius, y + radius],
            fill=face,
            outline=edge,
            width=edge_width,
        )
        return
    if symbol in ("^", "v"):
        sign = -1 if symbol == "^" else 1
        corners = [
            (x, y + sign * radius),
            (x + radius, y - sign * radius),
            (x - radius, y - sign * radius),
        ]
    elif symbol == "D":
        corners = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
    elif symbol == "X":
        corners = _filled_cross(x, y, radius)
    else:
        angles = -math.pi / 2 + np.arange(10) * math.pi / 5
        radii = np.where(np.arange(10) % 2 == 0, radius, 0.382 * radius)
        corners = list(zip(x + radii * np.cos(angles), y + radii * np.sin(angles)))
    d.polygon(corners, fill=face, outline=edge, width=edge_width)


def draw_text_box(
    d: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    face: ImageFont.FreeTypeFont,
    anchor: str,
    box: dict,
    size_px: float,
) -> None:
    """A rounded box behind text: ``face``, ``edge``, ``alpha`` and ``pad`` (in font sizes)."""
    left, top, right, bottom = d.multiline_textbbox(xy, text, font=face, anchor=anchor)
    pad = box.get("pad", 0.3) * size_px
    alpha = box.get("alpha", 1.0)
    edge = box.get("edge")
    d.rounded_rectangle(
        [left - pad, top - pad, right + pad, bottom + pad],
        radius=pad,
        fill=rgba(box.get("face", "#ffffff"), alpha),
        outline=None if edge is None else rgba(edge, alpha),
        width=SUPERSAMPLE,
    )


@dataclass
class LegendEntry:
    label: str
    kind: str
    color: Color
    width: float = 1.5
    dash: str | None = None
    symbol: str = "o"


@dataclass
class View:
    """A composed panel's mapping from data to page pixels."""

    box: tuple[float, float, float, float]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    dpi: float

    def px(self, xy: np.ndarray) -> np.ndarray:
        """Data ``(..., 2)`` to page pixels."""
        xy = np.asarray(xy, dtype=np.float64)
        left, top, right, bottom = self.box
        sx = (right - left) / (self.xlim[1] - self.xlim[0])
        sy = (bottom - top) / (self.ylim[1] - self.ylim[0])
        return np.stack(
            [left + (xy[..., 0] - self.xlim[0]) * sx, top + (self.ylim[1] - xy[..., 1]) * sy], -1
        )

    def pt(self, points: float) -> float:
        """Points as supersampled pixels."""
        return points * self.dpi / 72.0 * SUPERSAMPLE


@dataclass
class _Op:
    z: float
    draw: Callable[[ImageDraw.ImageDraw, View, Image.Image], None]


@dataclass
class Panel:
    """One plot box of a figure. ``rect`` is ``(left, bottom, width, height)`` in figure fractions."""

    figure: Figure
    rect: tuple[float, float, float, float]
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    equal: bool = False
    facecolor: Color = "#ffffff"
    frame: bool = True
    _ops: list[_Op] = field(default_factory=list)
    _text_ops: list[_Op] = field(default_factory=list)
    _extent: list[np.ndarray] = field(default_factory=list)
    _sticky_y0: bool = False
    _legend: list[LegendEntry] = field(default_factory=list)
    _legend_loc: str | None = None
    _legend_ncol: int = 1
    _legend_size: float = 9.0
    _xticks: tuple[np.ndarray, list[str] | None] | None = None
    _yticks: tuple[np.ndarray, list[str] | None] | None = None
    _tick_size: float = 10.0
    _xtick_rotation: float = 0.0
    _grid_alpha: float = 0.0
    _title: tuple[str, float, str, Color, str] | None = None
    _xlabel: str = ""
    _ylabel: str = ""
    _colorbar: tuple[str, float, float, str] | None = None
    _visible: bool = True

    def _add(self, z: float, draw: Callable, points: np.ndarray | None = None) -> None:
        self._ops.append(_Op(z, draw))
        if points is not None and np.size(points):
            self._extent.append(np.asarray(points, dtype=np.float64).reshape(-1, 2))

    def set_xlim(self, low: float, high: float) -> None:
        self.xlim = (float(low), float(high))

    def set_ylim(self, low: float, high: float) -> None:
        self.ylim = (float(low), float(high))

    def lines(
        self,
        lines: Sequence[np.ndarray],
        color: Color,
        width: float = 1.5,
        alpha: float = 1.0,
        dash: str | None = None,
        z: float = 2.0,
        label: str | None = None,
    ) -> None:
        """Open polylines ``(N, 2)``; ``dash`` is ``dash``, ``dot`` or ``dashdot``; ``width`` in points."""
        if label is not None:
            self._legend.append(LegendEntry(label, "line", color, width, dash))
        lines = [np.asarray(line, dtype=np.float64).reshape(-1, 2) for line in lines]
        fill = rgba(color, alpha)

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            line_width = max(1, round(view.pt(width)))
            pattern = None if dash is None else [view.pt(v * max(width, 1.0)) for v in DASHES[dash]]
            for line in lines:
                for pixels in finite_runs(view.px(line)):
                    if len(pixels) < 2:
                        continue
                    for piece in [pixels] if pattern is None else dash_pieces(pixels, pattern):
                        d.line(piece.ravel().tolist(), fill=fill, width=line_width, joint="curve")

        self._add(z, draw, np.concatenate(lines) if lines else None)

    def polygons(
        self,
        polygons: Sequence[np.ndarray],
        fill: Color | None = None,
        edge: Color | None = None,
        width: float = 1.0,
        alpha: float = 1.0,
        z: float = 1.0,
        label: str | None = None,
        edge_alpha: float | None = None,
    ) -> None:
        """Closed polygons ``(N, 2)``, filled and/or outlined; ``edge_alpha`` defaults to ``alpha``."""
        if label is not None:
            self._legend.append(LegendEntry(label, "patch", fill if fill is not None else edge))
        polygons = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons]
        polygons = [p for p in polygons if len(p) >= 3 and np.isfinite(p).all()]
        fill_rgba = None if fill is None else rgba(fill, alpha)
        edge_rgba = (
            None if edge is None else rgba(edge, alpha if edge_alpha is None else edge_alpha)
        )

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            edge_width = max(1, round(view.pt(width)))
            for polygon in polygons:
                corners = [tuple(p) for p in view.px(polygon)]
                if fill_rgba is not None:
                    d.polygon(corners, fill=fill_rgba)
                if edge_rgba is not None:
                    d.line([*corners, corners[0]], fill=edge_rgba, width=edge_width, joint="curve")

        self._add(z, draw, np.concatenate(polygons) if polygons else None)

    def markers(
        self,
        xy: np.ndarray,
        symbol: str = "o",
        size: float = 6.0,
        color: Color | Sequence[Color] = "#1f77b4",
        edge: Color | None = None,
        edge_width: float = 1.0,
        alpha: float = 1.0,
        z: float = 2.0,
        label: str | None = None,
        values: np.ndarray | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Markers of ``size`` points. ``values`` colours each point through ``cmap``; ``color`` may be one or one per point."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if values is not None:
            faces = [rgba(c, alpha) for c in map_values(values, cmap, vmin, vmax)]
        elif is_single_color(color):
            faces = [rgba(color, alpha)] * len(points)
        else:
            faces = [rgba(c, alpha) for c in color]
        if label is not None:
            self._legend.append(
                LegendEntry(
                    label, "marker", color if isinstance(color, str) else "#1f77b4", symbol=symbol
                )
            )
        outline = None if edge is None else rgba(edge, alpha)

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            radius = view.pt(size) / 2.0
            outline_width = max(1, round(view.pt(edge_width)))
            for (x, y), face in zip(view.px(points), faces):
                if math.isfinite(x) and math.isfinite(y):
                    draw_marker(d, x, y, symbol, radius, face, outline, outline_width)

        self._add(z, draw, points[np.isfinite(points).all(axis=1)])

    def arrows(
        self,
        tails: np.ndarray,
        heads: np.ndarray,
        color: Color,
        width: float = 1.5,
        head: float = 5.0,
        alpha: float = 1.0,
        z: float = 3.0,
    ) -> None:
        """Arrows from ``tails`` to ``heads`` ``(N, 2)`` with a filled head of ``head`` points."""
        tails = np.asarray(tails, dtype=np.float64).reshape(-1, 2)
        heads = np.asarray(heads, dtype=np.float64).reshape(-1, 2)
        fill = rgba(color, alpha)

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            line_width, head_px = max(1, round(view.pt(width))), view.pt(head)
            for start, end in zip(view.px(tails), view.px(heads)):
                delta = end - start
                length = float(np.hypot(*delta))
                if not math.isfinite(length) or length < 1e-6:
                    continue
                unit = delta / length
                side = np.array([-unit[1], unit[0]])
                size = min(head_px, 0.6 * length)
                base = end - unit * size
                d.line([tuple(start), tuple(base)], fill=fill, width=line_width)
                d.polygon(
                    [tuple(end), tuple(base + side * size * 0.4), tuple(base - side * size * 0.4)],
                    fill=fill,
                )

        self._add(z, draw, np.concatenate([tails, heads]))

    def bars(
        self,
        positions: np.ndarray,
        lengths: np.ndarray,
        width: float | np.ndarray = 0.8,
        base: float | np.ndarray = 0.0,
        color: Color | Sequence[Color] = "#1f77b4",
        edge: Color | None = None,
        alpha: float = 1.0,
        horizontal: bool = False,
        z: float = 1.0,
        label: str | None = None,
    ) -> None:
        """Bars centred on ``positions`` from ``base`` to ``base + lengths``; ``horizontal`` swaps the axes."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1)
        lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
        widths = np.broadcast_to(np.asarray(width, dtype=np.float64), positions.shape)
        bases = np.broadcast_to(np.asarray(base, dtype=np.float64), positions.shape)
        colors = [color] * len(positions) if is_single_color(color) else list(color)
        if label is not None:
            self._legend.append(LegendEntry(label, "patch", colors[0] if colors else color))
        rects = []
        for p, length, w, b in zip(positions, lengths, widths, bases):
            corners = np.array(
                [[p - w / 2, b], [p + w / 2, b], [p + w / 2, b + length], [p - w / 2, b + length]]
            )
            rects.append(corners[:, ::-1] if horizontal else corners)
        if not horizontal and np.all(bases == 0.0):
            self._sticky_y0 = True
        edge_rgba = None if edge is None else rgba(edge, alpha)

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            for rect, c in zip(rects, colors):
                pixels = view.px(rect)
                if not np.isfinite(pixels).all():
                    continue
                left, top = pixels.min(axis=0)
                right, bottom = pixels.max(axis=0)
                d.rectangle(
                    [left, top, right, bottom],
                    fill=rgba(c, alpha),
                    outline=edge_rgba,
                    width=SUPERSAMPLE,
                )

        self._add(z, draw, np.concatenate(rects) if rects else None)

    def hist(
        self,
        values: np.ndarray,
        bins: int | np.ndarray = 10,
        range: tuple[float, float] | None = None,
        color: Color = "#1f77b4",
        alpha: float = 1.0,
        density: bool = False,
        edge: Color | None = None,
        label: str | None = None,
        z: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """A histogram of the finite ``values`` as bars; returns ``(counts, edges)`` like NumPy."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        counts, edges = np.histogram(
            values[np.isfinite(values)], bins=bins, range=range, density=density
        )
        self.bars(
            (edges[:-1] + edges[1:]) / 2,
            counts,
            np.diff(edges),
            0.0,
            color,
            edge,
            alpha,
            z=z,
            label=label,
        )
        return counts, edges

    def fill_between(
        self,
        x: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray | float = 0.0,
        color: Color = "#1f77b4",
        alpha: float = 0.3,
        z: float = 1.0,
        label: str | None = None,
    ) -> None:
        """The area between ``y1`` and ``y2`` over ``x``."""
        x = np.asarray(x, dtype=np.float64)
        y1 = np.broadcast_to(np.asarray(y1, dtype=np.float64), x.shape)
        y2 = np.broadcast_to(np.asarray(y2, dtype=np.float64), x.shape)
        polygon = np.concatenate([np.stack([x, y1], 1), np.stack([x[::-1], y2[::-1]], 1)])
        self.polygons([polygon], fill=color, alpha=alpha, z=z, label=label)

    def hline(
        self,
        y: float,
        color: Color = "#000000",
        width: float = 1.0,
        dash: str | None = None,
        alpha: float = 1.0,
        z: float = 1.5,
        label: str | None = None,
    ) -> None:
        """A horizontal line across the panel at ``y``."""
        self._span_line(y, True, color, width, dash, alpha, z, label)

    def vline(
        self,
        x: float,
        color: Color = "#000000",
        width: float = 1.0,
        dash: str | None = None,
        alpha: float = 1.0,
        z: float = 1.5,
        label: str | None = None,
    ) -> None:
        """A vertical line across the panel at ``x``."""
        self._span_line(x, False, color, width, dash, alpha, z, label)

    def _span_line(
        self,
        value: float,
        horizontal: bool,
        color: Color,
        width: float,
        dash: str | None,
        alpha: float,
        z: float,
        label: str | None,
    ) -> None:
        """A line across the composed limits, at a fixed ``value`` of one axis."""
        if label is not None:
            self._legend.append(LegendEntry(label, "line", color, width, dash))
        fill = rgba(color, alpha)

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            ends = (
                np.array([[view.xlim[0], value], [view.xlim[1], value]])
                if horizontal
                else np.array([[value, view.ylim[0]], [value, view.ylim[1]]])
            )
            pixels = view.px(ends)
            pattern = None if dash is None else [view.pt(v * max(width, 1.0)) for v in DASHES[dash]]
            for piece in [pixels] if pattern is None else dash_pieces(pixels, pattern):
                d.line(piece.ravel().tolist(), fill=fill, width=max(1, round(view.pt(width))))

        self._ops.append(_Op(z, draw))

    def span(
        self,
        low: float,
        high: float,
        horizontal: bool = True,
        color: Color = "#1f77b4",
        alpha: float = 0.2,
        z: float = 0.5,
    ) -> None:
        """A band across the panel between ``low`` and ``high`` of y (``horizontal``) or x."""
        fill = rgba(color, alpha)

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            if horizontal:
                corners = np.array([[view.xlim[0], low], [view.xlim[1], high]])
            else:
                corners = np.array([[low, view.ylim[0]], [high, view.ylim[1]]])
            (x0, y0), (x1, y1) = view.px(corners)
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill)

        self._ops.append(_Op(z, draw))

    def image(
        self,
        array: np.ndarray,
        extent: tuple[float, float, float, float] | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        origin: str = "upper",
        alpha: float = 1.0,
        z: float = 0.0,
    ) -> None:
        """A ``(H, W)`` array through ``cmap`` or an RGB(A) ``(H, W, 3|4)`` image over ``extent``
        ``(left, right, bottom, top)``, row 0 at the top for ``origin="upper"``. Without ``extent`` the
        panel shows pixel indices, row 0 at the top, at equal aspect."""
        array = np.asarray(array)
        height, width = array.shape[:2]
        if extent is None:
            extent = (-0.5, width - 0.5, height - 0.5, -0.5)
            self.xlim = self.xlim or (extent[0], extent[1])
            self.ylim = self.ylim or (extent[2], extent[3])
            self.equal = True
        x0, x1, y0, y1 = extent
        if array.ndim == 2:
            colors = map_values(array, cmap, vmin, vmax)
            mask = np.isfinite(array)
            rgba_array = np.dstack([colors, np.where(mask, round(255 * alpha), 0).astype(np.uint8)])
        else:
            scale = 1.0 if np.issubdtype(array.dtype, np.integer) else 255.0
            rgb = np.clip(array[..., :3] * scale, 0, 255).astype(np.uint8)
            a = (
                np.clip(array[..., 3] * scale, 0, 255).astype(np.uint8)
                if array.shape[2] == 4
                else np.full((height, width), 255, np.uint8)
            )
            rgba_array = np.dstack([rgb, (a * alpha).astype(np.uint8)])
        tile = Image.fromarray(rgba_array if origin == "upper" else rgba_array[::-1], "RGBA")

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            (px0, py0), (px1, py1) = view.px(np.array([[x0, y0], [x1, y1]]))
            left, top, right, bottom = min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1)
            size = (max(1, round(right - left)), max(1, round(bottom - top)))
            scaled = tile.resize(size, Image.Resampling.NEAREST)
            page.paste(scaled, (round(left), round(top)), scaled)

        self._add(z, draw, np.array([[x0, y0], [x1, y1]]))

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        size: float = 10.0,
        color: Color = TEXT_COLOR,
        ha: str = "left",
        va: str = "baseline",
        offset: tuple[float, float] = (0.0, 0.0),
        box: dict | None = None,
        z: float = 4.0,
        coords: str = "data",
        weight: str = "normal",
        family: str = "sans",
    ) -> None:
        """Text at a data point (or panel fraction with ``coords="axes"``), shifted by ``offset`` points
        (+y up) and drawn over the clipped scene; skipped when the point is outside the panel.
        ``weight`` is ``normal`` or ``bold``, ``family`` is ``sans`` or ``mono``."""
        anchor = _anchor(ha, va)
        point = np.asarray(xy, dtype=np.float64)

        def draw(d: ImageDraw.ImageDraw, view: View, page: Image.Image) -> None:
            left, top, right, bottom = view.box
            if coords == "axes":
                x, y = left + point[0] * (right - left), bottom - point[1] * (bottom - top)
            else:
                x, y = view.px(point)
                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and left <= x <= right
                    and top <= y <= bottom
                ):
                    return
            x, y = x + view.pt(offset[0]), y - view.pt(offset[1])
            face = font(view.pt(size), weight, family)
            if box is not None:
                draw_text_box(d, (x, y), text, face, anchor, box, view.pt(size))
            d.multiline_text((x, y), text, fill=rgba(color), font=face, anchor=anchor, align=ha)

        self._text_ops.append(_Op(z, draw))

    def legend(self, loc: str = "upper right", ncol: int = 1, size: float = 9.0) -> None:
        """Draw the labelled elements' legend in ``upper/lower left/right`` of the panel, in
        ``ncol`` columns of ``size``-point text."""
        self._legend_loc, self._legend_ncol, self._legend_size = loc, ncol, size

    def title(
        self,
        text: str,
        size: float = 12.0,
        weight: str = "normal",
        color: Color = TEXT_COLOR,
        family: str = "sans",
    ) -> None:
        """Title lines centred above the panel."""
        self._title = (text, size, weight, color, family)

    def tick_style(self, size: float = 10.0, xrotation: float = 0.0) -> None:
        """Tick label ``size`` in points, and x tick labels rotated by ``xrotation`` degrees."""
        self._tick_size, self._xtick_rotation = size, xrotation

    def xlabel(self, text: str) -> None:
        self._xlabel = text

    def ylabel(self, text: str) -> None:
        self._ylabel = text

    def xticks(self, values: Sequence[float], labels: Sequence[str] | None = None) -> None:
        """Fixed x ticks, with ``labels`` in place of the values."""
        self._xticks = (
            np.asarray(values, dtype=np.float64),
            None if labels is None else list(labels),
        )

    def yticks(self, values: Sequence[float], labels: Sequence[str] | None = None) -> None:
        """Fixed y ticks, with ``labels`` in place of the values."""
        self._yticks = (
            np.asarray(values, dtype=np.float64),
            None if labels is None else list(labels),
        )

    def grid(self, alpha: float = 0.3) -> None:
        """Grid lines at the ticks."""
        self._grid_alpha = alpha

    def axis_off(self) -> None:
        """No frame, ticks or labels."""
        self.frame = False

    def hide(self) -> None:
        """Leave this panel empty on the figure."""
        self._visible = False

    def colorbar(
        self, cmap: str = "viridis", vmin: float = 0.0, vmax: float = 1.0, label: str = ""
    ) -> None:
        """A vertical colour scale beside the panel, taking part of its width."""
        self._colorbar = (cmap, float(vmin), float(vmax), label)

    def save(self, path: str | Path) -> None:
        """Write the panel's figure as a PNG."""
        self.figure.save(path)

    def _limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The set limits, or the recorded data extent with a margin."""
        points = np.concatenate(self._extent) if self._extent else np.zeros((0, 2))
        points = points[np.isfinite(points).all(axis=1)]

        def auto(column: int, sticky_zero: bool) -> tuple[float, float]:
            if not len(points):
                return 0.0, 1.0
            low, high = float(points[:, column].min()), float(points[:, column].max())
            if high - low < 1e-12:
                return low - 0.5, high + 0.5
            pad = (high - low) * MARGIN
            return (low if sticky_zero and low == 0.0 else low - pad), high + pad

        xlim = self.xlim if self.xlim is not None else auto(0, False)
        ylim = self.ylim if self.ylim is not None else auto(1, self._sticky_y0)
        return xlim, ylim

    def _view(self) -> View:
        """The page box and limits this panel composes with."""
        fig = self.figure
        left, bottom, width, height = self.rect
        if self._colorbar is not None:
            width *= 0.86
        box = [
            left * fig.width_px,
            (1.0 - bottom - height) * fig.height_px,
            (left + width) * fig.width_px,
            (1.0 - bottom) * fig.height_px,
        ]
        xlim, ylim = self._limits()
        if self.equal:
            box_w, box_h = box[2] - box[0], box[3] - box[1]
            span_x, span_y = abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])
            scale = min(box_w / span_x, box_h / span_y)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            box = [
                cx - span_x * scale / 2,
                cy - span_y * scale / 2,
                cx + span_x * scale / 2,
                cy + span_y * scale / 2,
            ]
        return View(tuple(box), xlim, ylim, fig.dpi)

    def _ticks(self, view: View) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
        """Tick values and labels on both axes, set or chosen on round numbers."""

        def axis(fixed, lim):
            if fixed is None:
                values = nice_ticks(*lim)
                return values, [tick_label(v) for v in values]
            values, labels = fixed
            labels = labels if labels is not None else [tick_label(v) for v in values]
            keep = (values >= min(lim) - 1e-9) & (values <= max(lim) + 1e-9)
            return values[keep], [label for label, k in zip(labels, keep) if k]

        xv, xl = axis(self._xticks, view.xlim)
        yv, yl = axis(self._yticks, view.ylim)
        return xv, xl, yv, yl

    def _compose(self, page: Image.Image) -> None:
        """Draw this panel onto the figure page."""
        if not self._visible:
            return
        view = self._view()
        left, top, right, bottom = (round(v) for v in view.box)
        xv, xl, yv, yl = self._ticks(view)
        plot = Image.new("RGB", page.size, rgba(self.facecolor)[:3])
        draw = ImageDraw.Draw(plot, "RGBA")
        if self._grid_alpha > 0.0 and self.frame:
            grid_color, grid_width = rgba(GRID_COLOR, self._grid_alpha), max(1, round(view.pt(0.8)))
            for x in view.px(np.stack([xv, np.full_like(xv, view.ylim[0])], 1))[:, 0]:
                draw.line([(x, top), (x, bottom)], fill=grid_color, width=grid_width)
            for y in view.px(np.stack([np.full_like(yv, view.xlim[0]), yv], 1))[:, 1]:
                draw.line([(left, y), (right, y)], fill=grid_color, width=grid_width)
        for op in sorted(self._ops, key=lambda op: op.z):
            op.draw(draw, view, plot)
        page.paste(plot.crop((left, top, right, bottom)), (left, top))
        d = ImageDraw.Draw(page, "RGBA")
        for op in sorted(self._text_ops, key=lambda op: op.z):
            op.draw(d, view, page)
        if self.frame:
            self._draw_axes(d, view, xv, xl, yv, yl)
        if self._legend_loc is not None and self._legend:
            self._draw_legend(d, view)
        if self._title is not None:
            text, size, weight, color, family = self._title
            d.multiline_text(
                ((left + right) / 2, top - view.pt(6)),
                text,
                fill=rgba(color),
                font=font(view.pt(size), weight, family),
                anchor="md",
                align="center",
            )
        if self._colorbar is not None:
            self._draw_colorbar(page, d, view)

    def _draw_axes(self, d: ImageDraw.ImageDraw, view: View, xv, xl, yv, yl) -> None:
        """Frame, ticks, tick labels and axis labels."""
        left, top, right, bottom = view.box
        black = rgba(TEXT_COLOR)
        d.rectangle([left, top, right, bottom], outline=black, width=max(1, round(view.pt(0.8))))
        tick, face = view.pt(3.5), font(view.pt(self._tick_size))
        label_bottom = bottom
        for x, label in zip(view.px(np.stack([xv, np.full_like(xv, view.ylim[0])], 1))[:, 0], xl):
            d.line([(x, bottom), (x, bottom + tick)], fill=black, width=SUPERSAMPLE)
            if self._xtick_rotation:
                height = self.figure._rotated_text(
                    d, (x, bottom + tick * 1.6), label, face, angle=self._xtick_rotation
                )
                label_bottom = max(label_bottom, bottom + tick * 1.6 + height)
                continue
            d.text((x, bottom + tick * 1.6), label, fill=black, font=face, anchor="ma")
            label_bottom = max(
                label_bottom, d.textbbox((x, bottom + tick * 1.6), label, font=face, anchor="ma")[3]
            )
        label_left = left
        for y, label in zip(view.px(np.stack([np.full_like(yv, view.xlim[0]), yv], 1))[:, 1], yl):
            d.line([(left - tick, y), (left, y)], fill=black, width=SUPERSAMPLE)
            d.text((left - tick * 1.6, y), label, fill=black, font=face, anchor="rm")
            label_left = min(
                label_left, d.textbbox((left - tick * 1.6, y), label, font=face, anchor="rm")[0]
            )
        if self._xlabel:
            d.text(
                ((left + right) / 2, label_bottom + view.pt(4)),
                self._xlabel,
                fill=black,
                font=face,
                anchor="ma",
            )
        if self._ylabel:
            self.figure._rotated_text(
                d,
                (label_left - view.pt(4), (top + bottom) / 2),
                self._ylabel,
                face,
                anchor_right=True,
            )

    def _draw_legend(self, d: ImageDraw.ImageDraw, view: View) -> None:
        """Legend rows in the corner named by the legend location, filled column by column."""
        face = font(view.pt(self._legend_size))
        row = round(1.5 * face.size)
        entries = list({entry.label: entry for entry in self._legend}.values())
        rows = math.ceil(len(entries) / self._legend_ncol)
        column = int(max(face.getlength(e.label) for e in entries) + view.pt(36))
        width = column * min(self._legend_ncol, len(entries))
        height = row * rows + int(view.pt(8))
        left, top, right, bottom = view.box
        pad = view.pt(6)
        x = right - pad - width if "right" in self._legend_loc else left + pad
        y = top + pad if "upper" in self._legend_loc else bottom - pad - height
        d.rounded_rectangle(
            [x, y, x + width, y + height],
            radius=view.pt(3),
            fill=(255, 255, 255, 217),
            outline=(204, 204, 204, 255),
            width=SUPERSAMPLE,
        )
        for index, entry in enumerate(entries):
            cx = x + (index // rows) * column
            cy = y + view.pt(4) + (index % rows) * row + row / 2
            a, b = cx + view.pt(5), cx + view.pt(25)
            if entry.kind == "line":
                piece = np.array([[a, cy], [b, cy]])
                pattern = (
                    None if entry.dash is None else [view.pt(v * 1.5) for v in DASHES[entry.dash]]
                )
                for segment in [piece] if pattern is None else dash_pieces(piece, pattern):
                    d.line(
                        segment.ravel().tolist(),
                        fill=rgba(entry.color),
                        width=max(1, round(view.pt(min(entry.width, 3.0)))),
                    )
            elif entry.kind == "patch":
                d.rectangle([a, cy - view.pt(4), b, cy + view.pt(4)], fill=rgba(entry.color, 0.6))
            else:
                draw_marker(
                    d, (a + b) / 2, cy, entry.symbol, view.pt(3), rgba(entry.color), None, 1
                )
            d.text(
                (cx + view.pt(30), cy), entry.label, fill=rgba(TEXT_COLOR), font=face, anchor="lm"
            )

    def _draw_colorbar(self, page: Image.Image, d: ImageDraw.ImageDraw, view: View) -> None:
        """The colour scale to the right of the panel box, with ticks and a label."""
        cmap, vmin, vmax, label = self._colorbar
        left, top, right, bottom = view.box
        bar_left, bar_right = right + view.pt(10), right + view.pt(22)
        height = max(1, round(bottom - top))
        ramp = map_values(
            np.linspace(vmax, vmin, height)[:, None].repeat(2, axis=1), cmap, vmin, vmax
        )
        page.paste(
            Image.fromarray(ramp, "RGB").resize((max(1, round(bar_right - bar_left)), height)),
            (round(bar_left), round(top)),
        )
        black, face = rgba(TEXT_COLOR), font(view.pt(9))
        d.rectangle([bar_left, top, bar_right, top + height], outline=black, width=SUPERSAMPLE)
        text_right = bar_right
        for value in nice_ticks(vmin, vmax, 5):
            y = bottom - (value - vmin) / (vmax - vmin) * (bottom - top) if vmax > vmin else bottom
            d.line([(bar_right, y), (bar_right + view.pt(3), y)], fill=black, width=SUPERSAMPLE)
            d.text(
                (bar_right + view.pt(5), y), tick_label(value), fill=black, font=face, anchor="lm"
            )
            text_right = max(
                text_right,
                d.textbbox((bar_right + view.pt(5), y), tick_label(value), font=face, anchor="lm")[
                    2
                ],
            )
        if label:
            self.figure._rotated_text(
                d, (text_right + view.pt(4), (top + bottom) / 2), label, face, anchor_right=False
            )


@dataclass
class Figure:
    """A page of ``size_in`` inches at ``dpi`` holding panels and page text."""

    size_in: tuple[float, float] = (10.0, 10.0)
    dpi: float = 100.0
    facecolor: Color = "#ffffff"
    panels: list[Panel] = field(default_factory=list)
    _page_text: list[tuple] = field(default_factory=list)
    _page: Image.Image | None = None

    def __post_init__(self) -> None:
        self.width_px = round(self.size_in[0] * self.dpi) * SUPERSAMPLE
        self.height_px = round(self.size_in[1] * self.dpi) * SUPERSAMPLE

    def panel(
        self, rect: tuple[float, float, float, float] = (0.125, 0.11, 0.775, 0.77), **kwargs
    ) -> Panel:
        """A panel at ``rect`` ``(left, bottom, width, height)`` in figure fractions."""
        panel = Panel(self, rect, **kwargs)
        self.panels.append(panel)
        return panel

    def grid(
        self,
        rows: int,
        cols: int,
        margins: tuple[float, float, float, float] = (0.08, 0.97, 0.08, 0.92),
        wspace: float = 0.25,
        hspace: float = 0.35,
        **kwargs,
    ) -> list[list[Panel]]:
        """``rows`` x ``cols`` panels between ``margins`` ``(left, right, bottom, top)``; spaces are fractions of a panel."""
        left, right, bottom, top = margins
        width = (right - left) / (cols + (cols - 1) * wspace)
        height = (top - bottom) / (rows + (rows - 1) * hspace)
        return [
            [
                self.panel(
                    (
                        left + c * width * (1 + wspace),
                        top - height - r * height * (1 + hspace),
                        width,
                        height,
                    ),
                    **kwargs,
                )
                for c in range(cols)
            ]
            for r in range(rows)
        ]

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        size: float = 10.0,
        ha: str = "center",
        va: str = "center",
        weight: str = "normal",
        color: Color = TEXT_COLOR,
    ) -> None:
        """Text at figure fractions ``xy`` (0, 0 bottom left)."""
        self._page_text.append((xy, text, size, ha, va, weight, color))

    def suptitle(
        self, text: str, size: float = 14.0, weight: str = "normal", color: Color = TEXT_COLOR
    ) -> None:
        """A title centred at the top of the page."""
        self.text((0.5, 0.985), text, size, va="top", weight=weight, color=color)

    def _rotated_text(
        self,
        d: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        face: ImageFont.FreeTypeFont,
        anchor_right: bool = False,
        angle: float = 90.0,
    ) -> float:
        """Text rotated by ``angle`` degrees; returns its height in pixels. At 90 degrees it is centred
        on ``xy`` vertically, ending at (``anchor_right``) or starting from ``xy``; at other angles its
        top right corner sits at ``xy``, as a tick label."""
        left, top, right, bottom = d.multiline_textbbox((0, 0), text, font=face, anchor="lt")
        label = Image.new("RGBA", (int(right - left) + 4, int(bottom - top) + 4), (0, 0, 0, 0))
        ImageDraw.Draw(label).multiline_text(
            (2, 2), text, fill=rgba(TEXT_COLOR), font=face, anchor="lt"
        )
        label = label.rotate(angle, expand=True)
        if angle == 90.0:
            x = xy[0] - label.width if anchor_right else xy[0]
            y = xy[1] - label.height / 2
        else:
            x, y = xy[0] - label.width, xy[1]
        self._page.paste(label, (int(x), int(y)), label)
        return float(label.height)

    def image(self) -> Image.Image:
        """Compose the page at the output size."""
        self._page = Image.new("RGB", (self.width_px, self.height_px), rgba(self.facecolor)[:3])
        for panel in self.panels:
            panel._compose(self._page)
        d = ImageDraw.Draw(self._page, "RGBA")
        for (fx, fy), text, size, ha, va, weight, color in self._page_text:
            d.multiline_text(
                (fx * self.width_px, (1.0 - fy) * self.height_px),
                text,
                fill=rgba(color),
                font=font(size * self.dpi / 72.0 * SUPERSAMPLE, weight),
                anchor=_anchor(ha, va),
                align=ha,
            )
        page, self._page = self._page, None
        return page.reduce(SUPERSAMPLE)

    def save(self, path: str | Path) -> None:
        """Write the composed page as a PNG, creating its directory."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.image().save(path, compress_level=1)

    def png_bytes(self) -> bytes:
        """The composed page as PNG bytes."""
        buffer = io.BytesIO()
        self.image().save(buffer, format="PNG", compress_level=1)
        return buffer.getvalue()

    def array(self) -> np.ndarray:
        """The composed page as an ``(H, W, 3)`` uint8 array."""
        return np.asarray(self.image())


def scene_panel(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    size_in: tuple[float, float] = (10.0, 10.0),
    dpi: float = 100.0,
    margins: tuple[float, float, float, float] = (0.10, 0.95, 0.08, 0.92),
    facecolor: Color = "#ffffff",
) -> Panel:
    """A one-panel figure with an equal-aspect box between ``margins`` ``(left, right, bottom, top)``."""
    left, right, bottom, top = margins
    figure = Figure(size_in, dpi, facecolor)
    return figure.panel(
        (left, bottom, right - left, top - bottom), xlim=xlim, ylim=ylim, equal=True
    )
