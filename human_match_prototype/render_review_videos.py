"""Render animated BEV clips (MP4 + GIF) for review-set scenes.

Stage 3's `render_report.py` emits a single BEV frame per scene. That shows the
planner's sample fan but not how the scene evolves, which is what you actually
need to judge whether a high Energy Score is a real disagreement or just a
legitimately multi-modal situation. This renders the full clip instead: the
scene plays forward while the planner fan and human trajectory stay overlaid,
with a marker tracking where the human actually is at each timestep.

GIFs are produced because Confluence renders them inline; MP4s are kept as the
higher-quality source.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from human_match_prototype.route_projection import stitch_route_lanes
from human_match_prototype.sampler import TrajectorySampler

DEFAULT_MODEL_DIR = Path("/home/chenglin/workspace/dp_exp_models/original_backup")

PLANNER_COLOR = "#E040FB"
HUMAN_COLOR = "#00E676"
ROUTE_COLOR = "#00BCD4"
NOW_COLOR = "#FFEB3B"


def render_scene_video(
    sampler: TrajectorySampler,
    npz_path: str,
    out_mp4: Path,
    out_gif: Path,
    num_samples: int = 64,
    seed: int = 0,
    temperature: float = 1.0,
    step: int = 2,
    fps: int = 10,
    gif_width: int = 520,
) -> None:
    """Render one scene as an animated BEV clip.

    Args:
        step: render every Nth timestep (2 halves the frame count).
        gif_width: GIF output width in pixels; keeps Confluence attachments small.
    """
    from src.visualization import PAST_FRAMES, TOTAL_FRAMES, precompute_static, render_frame

    data = dict(np.load(npz_path, allow_pickle=True))
    result = sampler.sample(
        str(npz_path), num_samples=num_samples, seed=seed, temperature=temperature
    )
    human = result.human_future[:, :2]

    static = precompute_static(data)
    static["view_half"] = static["view_half"] * 0.7

    route = stitch_route_lanes(np.asarray(data["route_lanes"]))
    has_route = route.qa.route_valid and len(route.centerline) > 1

    timesteps = list(range(0, TOTAL_FRAMES, step))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fig, ax = plt.subplots(figsize=(9, 9))

        for i, t in enumerate(timesteps):
            ax.clear()
            render_frame(fig, ax, data, static, t=t, filename=Path(npz_path).name)

            for s in result.ego_samples:
                ax.plot(s[:, 0], s[:, 1], color=PLANNER_COLOR, alpha=0.25, lw=0.5, zorder=40)
            ax.plot(human[:, 0], human[:, 1], color=HUMAN_COLOR, lw=2.0, zorder=45, label="Human")
            if has_route:
                ax.plot(
                    route.centerline[:, 0],
                    route.centerline[:, 1],
                    color=ROUTE_COLOR,
                    lw=1.4,
                    ls="--",
                    alpha=0.7,
                    zorder=38,
                    label="Route",
                )

            # Marker for where the human actually is at this timestep.
            fut = t - (PAST_FRAMES - 1)
            if 0 <= fut < len(human):
                ax.plot(
                    human[fut, 0],
                    human[fut, 1],
                    "o",
                    color=NOW_COLOR,
                    ms=9,
                    zorder=60,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )

            phase = "past" if t < PAST_FRAMES else f"+{(t - PAST_FRAMES + 1) * 0.1:.1f}s"
            ax.set_title(f"{Path(npz_path).stem}   t={t}/{TOTAL_FRAMES - 1}  ({phase})", fontsize=9)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            fig.savefig(
                tmp_path / f"f{i:04d}.png", dpi=90, facecolor="#1A1A1A", bbox_inches="tight"
            )

        plt.close(fig)

        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "f%04d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                str(out_mp4),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(out_mp4),
                "-vf",
                f"fps={max(1, fps // 2)},scale={gif_width}:-1:flags=lanczos,"
                "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                str(out_gif),
            ],
            check=True,
        )


def main():
    p = argparse.ArgumentParser(description="Render animated BEV clips for review scenes.")
    p.add_argument("--review_set", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = p.parse_args()

    sampler = TrajectorySampler(
        str(args.model_dir / "args.json"),
        str(args.model_dir / "diffusion_planner.onnx"),
        args.device,
    )

    review = pd.read_csv(args.review_set)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(review.iterrows(), 1):
        npz = row["npz_path"]
        stem = Path(npz).stem
        print(f"[{i}/{len(review)}] {stem}", flush=True)
        render_scene_video(
            sampler,
            npz,
            out_dir / f"{stem}.mp4",
            out_dir / f"{stem}.gif",
            num_samples=args.num_samples,
            seed=args.seed,
            temperature=args.temperature,
            step=args.step,
            fps=args.fps,
        )

    print(f"Wrote {len(review)} clips to {out_dir}")


if __name__ == "__main__":
    main()
