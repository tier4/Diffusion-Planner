"""PIL raster canvas for ego-frame scene images: supersampled, clipped to a square view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUPERSAMPLE = 2
POINT_PX = 120.0 / 72.0
GRID_COLOR = "#b0b0b0"
TEXT_COLOR = (0, 0, 0)


def rgba(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    """``#rrggbb`` plus an alpha in [0, 1] as an RGBA tuple."""
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), round(255 * alpha))


def dashes(points: np.ndarray, on: float, off: float) -> list[np.ndarray]:
    """Split a ``(N, 2)`` polyline into the dash pieces of an ``on``/``off`` pattern."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    starts = np.arange(0.0, arc[-1], on + off)
    pieces = []
    for start in starts:
        end = min(start + on, arc[-1])
        inner = (arc > start) & (arc < end)
        s = np.concatenate([[start], arc[inner], [end]])
        pieces.append(
            np.stack([np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])], 1)
        )
    return pieces


def finite_runs(points: np.ndarray) -> list[np.ndarray]:
    """Split a ``(N, 2)`` polyline at its non-finite points into the finite runs between them."""
    finite = np.isfinite(points).all(axis=1)
    runs = np.split(points, np.flatnonzero(~finite))
    return [run[np.isfinite(run).all(axis=1)] for run in runs]


def draw_marker(
    draw: ImageDraw.ImageDraw, x: float, y: float, symbol: str, color: str, radius: float
) -> None:
    """One ``^``, ``x`` or ``*`` marker of ``radius`` pixels at pixel ``(x, y)``."""
    if symbol == "x":
        width = max(1, round(1.5 * POINT_PX * SUPERSAMPLE))
        draw.line(
            [(x - radius, y - radius), (x + radius, y + radius)], fill=rgba(color), width=width
        )
        draw.line(
            [(x - radius, y + radius), (x + radius, y - radius)], fill=rgba(color), width=width
        )
        return
    if symbol == "^":
        corners = [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
    else:
        angles = -np.pi / 2 + np.arange(10) * np.pi / 5
        radii = np.where(np.arange(10) % 2 == 0, radius, 0.382 * radius)
        corners = list(zip(x + radii * np.cos(angles), y + radii * np.sin(angles)))
    draw.polygon(corners, fill=rgba(color))


class SceneCanvas:
    """A square ego-frame view of ``half_m`` metres around the origin, with grid, title and legend."""

    def __init__(self, size_px: int, half_m: float, grid_m: float, title: str) -> None:
        s = SUPERSAMPLE
        self.size = size_px * s
        self.half_m = half_m
        self.margin_left, self.margin_top = 62 * s, 44 * s
        self.plot_px = self.size - self.margin_left - 18 * s
        self.scale = self.plot_px / (2.0 * half_m)
        self.image = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        self.scene = Image.new("RGB", (self.plot_px, self.plot_px), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.scene, "RGBA")
        self.legend_entries: list[tuple[str, str, str]] = []
        self.title = title
        self.grid_m = grid_m
        self._grid()

    def px(self, points: np.ndarray) -> np.ndarray:
        """Ego-frame metres ``(..., 2)`` to scene pixels."""
        points = np.asarray(points, dtype=np.float64)
        return np.stack(
            [
                (points[..., 0] + self.half_m) * self.scale,
                (self.half_m - points[..., 1]) * self.scale,
            ],
            -1,
        )

    def width(self, points: float) -> int:
        """A matplotlib line width in points as supersampled pixels."""
        return max(1, round(points * POINT_PX * SUPERSAMPLE))

    def _grid(self) -> None:
        """Grid lines at every ``grid_m`` metres, under the scene."""
        color = rgba(GRID_COLOR, 0.3)
        for value in np.arange(-self.half_m, self.half_m + 1e-6, self.grid_m):
            x, y = self.px(np.array([value, value]))
            self.draw.line([(x, 0), (x, self.plot_px)], fill=color, width=self.width(0.8))
            self.draw.line([(0, y), (self.plot_px, y)], fill=color, width=self.width(0.8))

    def polylines(
        self,
        lines: list[np.ndarray],
        color: str,
        alpha: float,
        width: float,
        dash: tuple[float, float] | None = None,
        label: str | None = None,
    ) -> None:
        """Open polylines ``(N, 2)`` in metres, in the legend as ``label`` when there are any."""
        if label is not None and lines:
            self._add_legend(label, "line" if dash is None else "dash", color)
        fill, line_width = rgba(color, alpha), self.width(width)
        for pixels in (run for line in lines for run in finite_runs(self.px(line))):
            if len(pixels) < 2:
                continue
            pieces = [pixels]
            if dash is not None:
                on, off = (value * POINT_PX * SUPERSAMPLE * width for value in dash)
                pieces = dashes(pixels, on, off)
            for piece in pieces:
                self.draw.line(piece.ravel().tolist(), fill=fill, width=line_width, joint="curve")

    def polygons(self, polygons: list[np.ndarray], color: str, alpha: float) -> None:
        """Filled polygons ``(N, 2)`` in metres."""
        for polygon in polygons:
            self.draw.polygon(self.px(polygon).ravel().tolist(), fill=rgba(color, alpha))

    def marker(
        self, point: np.ndarray, symbol: str, color: str, label: str, size: float = 6.0
    ) -> None:
        """One ``^``, ``x`` or ``*`` marker of ``size`` points at a point in metres, if finite."""
        self._add_legend(label, symbol, color)
        x, y = self.px(point)
        if np.isfinite([x, y]).all():
            draw_marker(self.draw, x, y, symbol, color, size * POINT_PX * SUPERSAMPLE / 2.0)

    def _add_legend(self, label: str, kind: str, color: str) -> None:
        """Add a legend row once per label; ``kind`` is ``line``, ``dash`` or a marker symbol."""
        if all(entry[0] != label for entry in self.legend_entries):
            self.legend_entries.append((label, kind, color))

    def _font(self, points: float) -> ImageFont.FreeTypeFont:
        """Pillow's bundled FreeType font at ``points`` size, supersampled."""
        return ImageFont.load_default(size=round(points * POINT_PX * SUPERSAMPLE))

    def _frame(self) -> None:
        """Paste the scene, then draw the axes box, tick labels and title on the page."""
        s, x0, y0, plot = SUPERSAMPLE, self.margin_left, self.margin_top, self.plot_px
        self.image.paste(self.scene, (x0, y0))
        page = ImageDraw.Draw(self.image)
        page.rectangle([x0, y0, x0 + plot, y0 + plot], outline=TEXT_COLOR, width=self.width(0.8))
        font = self._font(10)
        for value in np.arange(-self.half_m, self.half_m + 1e-6, self.grid_m):
            x, y = self.px(np.array([value, value]))
            label = f"{value:.0f}"
            page.line([(x0 + x, y0 + plot), (x0 + x, y0 + plot + 6 * s)], fill=TEXT_COLOR, width=s)
            page.text((x0 + x, y0 + plot + 9 * s), label, fill=TEXT_COLOR, font=font, anchor="ma")
            page.line([(x0 - 6 * s, y0 + y), (x0, y0 + y)], fill=TEXT_COLOR, width=s)
            page.text((x0 - 9 * s, y0 + y), label, fill=TEXT_COLOR, font=font, anchor="rm")
        page.text(
            (x0 + plot / 2, y0 - 12 * s),
            self.title,
            fill=TEXT_COLOR,
            font=self._font(12),
            anchor="md",
        )

    def _legend(self) -> None:
        """Legend rows in the plot corner that covers the fewest drawn pixels."""
        if not self.legend_entries:
            return
        s = SUPERSAMPLE
        font = self._font(10)
        row = round(1.6 * font.size)
        text_width = max(font.getlength(label) for label, _, _ in self.legend_entries)
        width = int(text_width) + 46 * s
        height = row * len(self.legend_entries) + 8 * s
        left, top = self._emptiest_corner(width, height, 8 * s)
        right = left + width
        draw = ImageDraw.Draw(self.image, "RGBA")
        draw.rounded_rectangle(
            [left, top, right, top + height],
            radius=4 * s,
            fill=(255, 255, 255, 204),
            outline=(204, 204, 204, 255),
            width=s,
        )
        for index, (label, kind, color) in enumerate(self.legend_entries):
            y = top + 4 * s + index * row + row / 2
            if kind in ("line", "dash"):
                if kind == "line":
                    segments = [((left + 8 * s, y), (left + 34 * s, y))]
                else:
                    segments = [
                        ((left + 8 * s, y), (left + 17 * s, y)),
                        ((left + 23 * s, y), (left + 32 * s, y)),
                    ]
                for a, b in segments:
                    draw.line([a, b], fill=rgba(color), width=self.width(1.5))
            else:
                draw_marker(draw, left + 21 * s, y, kind, color, 6.0 * POINT_PX * s / 2.0)
            draw.text((left + 40 * s, y), label, fill=TEXT_COLOR, font=font, anchor="lm")

    def _emptiest_corner(self, width: int, height: int, pad: int) -> tuple[int, int]:
        """Page position of the plot corner box of this size that covers the fewest drawn pixels."""
        scene = np.asarray(self.scene.convert("L")) < 225
        far_x, far_y = self.plot_px - pad - width, self.plot_px - pad - height
        corners = [(far_x, pad), (pad, pad), (pad, far_y), (far_x, far_y)]
        covered = [scene[y : y + height, x : x + width].sum() for x, y in corners]
        x, y = corners[int(np.argmin(covered))]
        return self.margin_left + x, self.margin_top + y

    def save(self, path: Path) -> None:
        """Compose the page and write it as a PNG at the output size."""
        self._frame()
        self._legend()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.reduce(SUPERSAMPLE).save(path, compress_level=1)
