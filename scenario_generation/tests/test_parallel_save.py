"""Tests for parallel image saving in simulate.py."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from scenario_generation.canvas import Figure


def _line_figure(size: tuple[float, float] = (4, 4)) -> Figure:
    """A one-panel figure with a diagonal line."""
    fig = Figure(size, 100.0)
    fig.panel().lines([np.array([[0.0, 0.0], [1.0, 1.0]])], "#1f77b4")
    return fig


class TestSave:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "test.png"
        _line_figure().save(out)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_size_matches_dpi(self, tmp_path):
        out = tmp_path / "test.png"
        _line_figure((4, 3)).save(out)

        assert Image.open(out).size == (400, 300)


class TestParallelSaves:
    def test_no_corruption(self, tmp_path):
        """Save 10 figures in parallel, verify all PNGs are valid."""
        n = 10
        figs = []
        for i in range(n):
            fig = Figure((4, 4), 100.0)
            ax = fig.panel()
            ax.markers(np.random.randn(20, 2), "o")
            ax.title(f"Figure {i}")
            figs.append(fig)

        paths = [tmp_path / f"fig_{i:02d}.png" for i in range(n)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fig.save, path) for fig, path in zip(figs, paths)]
            for f in futures:
                f.result()

        for p in paths:
            assert p.exists(), f"{p} not created"
            assert p.stat().st_size > 100, f"{p} too small ({p.stat().st_size} bytes)"
            Image.open(p).verify()

    def test_different_sizes(self, tmp_path):
        """Figures with different sizes save correctly in parallel."""
        sizes = [(4, 4), (8, 6), (10, 10), (3, 3)]
        figs = [_line_figure(size) for size in sizes]

        paths = [tmp_path / f"fig_{i}.png" for i in range(len(sizes))]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fig.save, path) for fig, path in zip(figs, paths)]
            for f in futures:
                f.result()

        for p, (w, h) in zip(paths, sizes):
            assert p.exists()
            assert Image.open(p).size == (w * 100, h * 100)
