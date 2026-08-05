#!/usr/bin/env python3
"""Render similarity search results as side-by-side BEV videos.

Reuses the clip-review-tool's visualization module to produce animated
mp4 videos showing the query scene next to its top-K matches.

Usage:
    uv run python scripts/visualize_similar_video.py \
      --query data/or_scene_npz_v2/takanawa/frame.npz \
      --results data/demo_search_results_takanawa.json \
      --output data/demo_video_takanawa.mp4 \
      [--top_k 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

# Import clip-review-tool visualization
CLIP_REVIEW_SRC = Path(
    "/home/chenglin/workspace/at-team-tools-clip-review/lin/clip-review-tool/src"
)
sys.path.insert(0, str(CLIP_REVIEW_SRC))
from visualization import TOTAL_FRAMES, precompute_static, render_frame


def load_scene(npz_path: str) -> dict:
    return dict(np.load(npz_path, allow_pickle=True))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/demo_video.mp4"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    results = results[: args.top_k]

    # Load all scenes
    scenes = []
    labels = []

    print(f"Loading query: {args.query.name}")
    scenes.append(load_scene(str(args.query)))
    labels.append("QUERY (override)")

    for r in results:
        p = r["path"]
        if Path(p).exists():
            print(f"Loading match #{r['rank']}: d={r['distance']:.3f}  {Path(p).name}")
            scenes.append(load_scene(p))
            labels.append(f"Match #{r['rank']} (d={r['distance']:.3f})")
        else:
            print(f"MISSING: {p}")

    n = len(scenes)
    if n == 0:
        print("No scenes to render")
        return

    # Precompute static elements for each scene
    statics = [precompute_static(s) for s in scenes]

    # Create figure with side-by-side subplots
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), dpi=100)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1A1A1A")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05)

    def animate(t):
        for i, (ax, scene, static, label) in enumerate(zip(axes, scenes, statics, labels)):
            render_frame(fig, ax, scene, static, t, filename=label)

    print(f"Rendering {TOTAL_FRAMES} frames at {args.fps} fps...")
    anim = FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=1000 // args.fps)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output), writer=writer)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
