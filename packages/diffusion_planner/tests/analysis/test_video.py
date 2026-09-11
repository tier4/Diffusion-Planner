"""Tests for rendering figure sequences to PNGs and MP4."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plotly.graph_objects as go

from diffusion_planner.analysis import video


def _figure(value: float) -> go.Figure:
    return go.Figure(go.Scatter(x=[0.0, 1.0], y=[0.0, value]))


class WriteFramesTest(unittest.TestCase):
    def test_writes_one_zero_padded_png_per_figure(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            directory = Path(workspace) / "frames"

            written = video.write_frames(
                [_figure(1.0), _figure(2.0), _figure(3.0)],
                directory,
                width=160,
                height=120,
            )

            self.assertEqual(
                [path.name for path in written],
                ["frame_000000.png", "frame_000001.png", "frame_000002.png"],
            )
            for path in written:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(path.read_bytes()[:4], b"\x89PNG")

    def test_creates_the_directory_and_accepts_a_generator(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            directory = Path(workspace) / "nested" / "frames"

            written = video.write_frames(
                (_figure(float(index)) for index in range(2)),
                directory,
                width=120,
                height=90,
            )

            self.assertTrue(directory.is_dir())
            self.assertEqual(len(written), 2)


class WriteVideoTest(unittest.TestCase):
    def setUp(self) -> None:
        if not video.ffmpeg_available():
            self.skipTest("ffmpeg is not installed")

    def test_encodes_an_mp4_and_discards_the_frames(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            output = Path(workspace) / "out.mp4"

            written = video.write_video(
                [_figure(float(index)) for index in range(3)],
                output,
                fps=4.0,
                width=160,
                height=120,
            )

            self.assertEqual(written, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(list(Path(workspace).glob("frame_*.png")), [])

    def test_keeps_frames_when_asked(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            output = Path(workspace) / "out.mp4"
            frames = Path(workspace) / "frames"

            video.write_video(
                [_figure(1.0), _figure(2.0)],
                output,
                fps=2.0,
                width=160,
                height=120,
                keep_frames=frames,
            )

            self.assertEqual(len(list(frames.glob("frame_*.png"))), 2)

    def test_rejects_an_empty_sequence(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            self.assertRaises(RuntimeError),
        ):
            video.write_video([], Path(workspace) / "out.mp4")

    def test_rejects_a_non_positive_frame_rate(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            self.assertRaises(ValueError),
        ):
            video.write_video([_figure(1.0)], Path(workspace) / "out.mp4", fps=0.0)


class MissingFfmpegTest(unittest.TestCase):
    def test_reports_a_useful_message_instead_of_a_stack_trace(self) -> None:
        with (
            mock.patch.object(video.shutil, "which", return_value=None),
            tempfile.TemporaryDirectory() as workspace,
        ):
            self.assertFalse(video.ffmpeg_available())
            with self.assertRaises(RuntimeError) as caught:
                video.write_video([_figure(1.0)], Path(workspace) / "out.mp4")

        self.assertIn("ffmpeg", str(caught.exception))
        self.assertIn("write_frames", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
