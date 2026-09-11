"""Render a sequence of plotly figures to an MP4.

Frames are written as PNGs into a temporary directory and stitched with
``ffmpeg``, which keeps the dependency surface to a binary that is either on the
PATH or clearly reported as missing. Plotly's static export needs ``kaleido``,
declared as a package dependency.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import plotly.graph_objects as go

__all__ = ["FFMPEG_MISSING_MESSAGE", "ffmpeg_available", "write_frames", "write_video"]

FFMPEG_MISSING_MESSAGE = (
    "ffmpeg was not found on the PATH; install it, or use write_frames() to "
    "keep the PNG sequence and encode it elsewhere"
)


def ffmpeg_available() -> bool:
    """Whether an ``ffmpeg`` binary can be found."""
    return shutil.which("ffmpeg") is not None


def write_frames(
    figures: Iterable[go.Figure],
    directory: Path,
    *,
    width: int = 1280,
    height: int = 800,
    scale: float = 1.0,
) -> list[Path]:
    """Write each figure as a zero-padded PNG and return the paths in order."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, figure in enumerate(figures):
        path = directory / f"frame_{index:06d}.png"
        path.write_bytes(
            figure.to_image(format="png", width=width, height=height, scale=scale)
        )
        written.append(path)
    return written


def _encode(directory: Path, output: Path, fps: float) -> None:
    """Stitch ``frame_%06d.png`` into ``output`` with ffmpeg."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(directory / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        # Both dimensions must be even for yuv420p; round down rather than fail.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}: "
            f"{result.stderr.strip()[:500]}"
        )


def write_video(
    figures: Iterable[go.Figure],
    output: Path,
    *,
    fps: float = 10.0,
    width: int = 1280,
    height: int = 800,
    scale: float = 1.0,
    keep_frames: Path | None = None,
) -> Path:
    """Render figures to an MP4.

    Args:
        figures: Figures in display order. Consumed lazily, so a generator can
            build each frame only when it is about to be encoded.
        output: MP4 path to write.
        fps: Output frame rate.
        width: Pixel width of each rendered frame.
        height: Pixel height of each rendered frame.
        scale: Plotly scale factor, multiplying the effective resolution.
        keep_frames: Directory to keep the PNG sequence in. When omitted the
            frames go to a temporary directory and are discarded.

    Returns:
        The written MP4 path.

    Raises:
        RuntimeError: If ``ffmpeg`` is missing, no figures were supplied, or the
            encode fails.
    """
    if not ffmpeg_available():
        raise RuntimeError(FFMPEG_MISSING_MESSAGE)
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    if keep_frames is not None:
        written = write_frames(
            figures, keep_frames, width=width, height=height, scale=scale
        )
        if not written:
            raise RuntimeError(
                "no figures were supplied, so there is nothing to encode"
            )
        _encode(keep_frames, output, fps)
        return output

    with tempfile.TemporaryDirectory(prefix="planner-video-") as workspace:
        directory = Path(workspace)
        written = write_frames(
            figures, directory, width=width, height=height, scale=scale
        )
        if not written:
            raise RuntimeError(
                "no figures were supplied, so there is nothing to encode"
            )
        _encode(directory, output, fps)
    return output
