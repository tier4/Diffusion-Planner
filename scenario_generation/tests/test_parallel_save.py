"""Tests for parallel image saving in simulate.py."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scenario_generation.simulate import _save_and_close


class TestSaveAndClose:
    def test_creates_file(self, tmp_path):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "test.png"
        _save_and_close(fig, out)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_figure_closed(self, tmp_path):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1])
        fig_num = fig.number
        _save_and_close(fig, tmp_path / "test.png")

        assert fig_num not in plt.get_fignums()


class TestParallelSaves:
    def test_no_corruption(self, tmp_path):
        """Save 10 figures in parallel, verify all PNGs are valid."""
        n = 10
        figs = []
        for i in range(n):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.random.randn(20), np.random.randn(20), "o")
            ax.set_title(f"Figure {i}")
            fig.tight_layout()
            figs.append(fig)

        paths = [tmp_path / f"fig_{i:02d}.png" for i in range(n)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(_save_and_close, fig, path)
                for fig, path in zip(figs, paths)
            ]
            for f in futures:
                f.result()

        for p in paths:
            assert p.exists(), f"{p} not created"
            assert p.stat().st_size > 100, f"{p} too small ({p.stat().st_size} bytes)"

    def test_different_sizes(self, tmp_path):
        """Figures with different sizes save correctly in parallel."""
        sizes = [(4, 4), (8, 6), (10, 10), (3, 3)]
        figs = []
        for w, h in sizes:
            fig, ax = plt.subplots(1, 1, figsize=(w, h))
            ax.plot([0, 1], [0, 1])
            figs.append(fig)

        paths = [tmp_path / f"fig_{i}.png" for i in range(len(sizes))]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(_save_and_close, fig, path)
                for fig, path in zip(figs, paths)
            ]
            for f in futures:
                f.result()

        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
