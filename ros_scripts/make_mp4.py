"""Encode a directory of (non-sequentially named) PNGs into a single MP4.

Drop-in Python replacement for the former ``~/misc/ffmpeg_lib/make_mp4_from_unsequential_png.sh``:
recursively globs ``*.<suffix>`` under ``target_dir``, orders them with a natural sort (so
non-zero-padded names like ``2 < 10`` still play in order), and encodes them with ffmpeg's concat
demuxer to ``<target_dir>/../<target_dir name>.mp4``. Encode params match the old script (libx264,
yuv420p, even dimensions, 10 fps).

    python3 ros_scripts/make_mp4.py /path/to/visualize_result
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

from natsort import natsorted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target_dir", type=Path, help="dir of frame images (globbed recursively)")
    p.add_argument("--suffix", type=str, default="png", help="frame image suffix (default: png)")
    p.add_argument("--fps", type=int, default=10, help="frame rate (default: 10)")
    return p.parse_args()


def encode_mp4(frames: list[Path], mp4_path: Path, fps: int) -> None:
    """Encode an ORDERED list of image files into an MP4 via ffmpeg's concat demuxer."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for frame in frames:
            f.write(f"file '{frame.resolve()}'\n")
        list_path = f.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-r",
                str(fps),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-r",
                str(fps),
                str(mp4_path),
            ],
            check=True,
        )
    finally:
        Path(list_path).unlink()


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.resolve()
    frames = [Path(p) for p in natsorted(str(x) for x in target_dir.rglob(f"*.{args.suffix}"))]
    if not frames:
        raise SystemExit(f"No *.{args.suffix} files under {target_dir}")
    mp4_path = target_dir.parent / f"{target_dir.name}.mp4"
    encode_mp4(frames, mp4_path, args.fps)
    print(f"Wrote {mp4_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
