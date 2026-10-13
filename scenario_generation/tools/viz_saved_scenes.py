#!/usr/bin/env python3
"""Visualize + verify scenes saved by the Scene Branch Editor ("Save for RSFT").

Renders each saved scene with the EXACT same renderer the editor GUI uses
(`render_scene_at_step`) so the output matches what you see in the GUI: lane
network, road borders, route, traffic-light overlay, neighbor OBBs, ego box,
and the baked ego target trajectory (`ego_agent_future`) with rear-axle-correct
footprints + per-step and worst-case RB / neighbor clearance lines.

Each scene's baked target is also scored with `compute_reward_batch` (the same
gate the editor applies on save); the PASS/FAIL verdict + sc_min/rb_min are put
in the figure title. Per-scene PNGs + a collage are written.

Usage (needs the lanelet/ROS env — same recipe as launching the editor):
    python -m scenario_generation.tools.viz_saved_scenes \
        --scene_dir <dir_of_saved_scene_npzs> \
        --reward_config <reward.json> \
        --output_dir <out_dir> [--view_half 45] [--cols 4]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import compute_reward_batch
from scenario_generation.canvas import Figure, font, rgba
from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_render import (
    _ensure_neighbor_future_4col,
    render_scene_at_step,
)


def _score_saved_target(npz_path, reward_config):
    data = load_npz_data(str(npz_path), torch.device("cpu"))
    if "neighbor_agents_future" in data:
        data["neighbor_agents_future"] = _ensure_neighbor_future_4col(
            data["neighbor_agents_future"]
        )
    ego_fut = data["ego_agent_future"]
    if ego_fut.dim() == 3:
        ego_fut = ego_fut[0]
    traj = ego_fut[:, :4].unsqueeze(0)
    return compute_reward_batch(traj, data, reward_config)[0]


def _titled(fig, text, size, color):
    """The composed figure with ``text`` centred in a band above it, in ``color``."""
    page = fig.image()
    face = font(size * fig.dpi / 72.0)
    probe = ImageDraw.Draw(page)
    left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=face, align="center")
    pad = round(face.size * 0.4)
    band = bottom - top + 2 * pad
    out = Image.new("RGB", (page.width, page.height + band), rgba(fig.facecolor)[:3])
    out.paste(page, (0, band))
    ImageDraw.Draw(out).multiline_text(
        (page.width / 2, pad - top), text, fill=rgba(color), font=face, anchor="ma", align="center"
    )
    return out


def _verdict(r):
    flags = []
    if getattr(r, "rb_crossing", False):
        flags.append(f"RB CROSS ({r.rb_min_dist:.2f}m)")
    if getattr(r, "lane_crossing", False):
        flags.append("LANE DEP")
    if getattr(r, "kinematic_violated", False):
        flags.append("KINEMATIC")
    if getattr(r, "collision_step", None) is not None:
        flags.append(f"COLLISION@t{r.collision_step}")
    if getattr(r, "static_crossing", False):
        flags.append(f"SC CROSS ({r.sc_min_dist:.2f}m)")
    ok = not flags
    summary = (
        f"sc_min={getattr(r, 'sc_min_dist', float('nan')):.2f}m  "
        f"rb_min={getattr(r, 'rb_min_dist', float('nan')):.2f}m  "
        f"total={getattr(r, 'total', float('nan')):.1f}"
    )
    return ok, summary, flags


def _xyh_from_xycs(fut):
    """(T, >=4) [x,y,cos,sin] -> (T,3) [x,y,heading], valid steps only."""
    if fut.ndim == 3:
        fut = fut[0]
    heading = np.arctan2(fut[:, 3], fut[:, 2])
    traj = np.column_stack([fut[:, :2], heading]).astype(np.float32)
    valid = np.abs(fut[:, :2]).sum(axis=-1) > 1e-6
    return traj[valid] if valid.sum() > 1 else traj


def _target_xyh(npz_path):
    """Baked ego target as (T, 3) [x, y, heading]."""
    return _xyh_from_xycs(np.load(str(npz_path), allow_pickle=True)["ego_agent_future"])


def _model_det(npz_path, model, model_args, device, reward_config):
    """Run the model's deterministic prediction. Returns (xyh traj, reward breakdown)."""
    from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched

    data = load_npz_data(str(npz_path), device)
    if "neighbor_agents_future" in data:
        data["neighbor_agents_future"] = _ensure_neighbor_future_4col(
            data["neighbor_agents_future"]
        )
    traj4 = det_inference_batched(model, model_args, [data], device)  # (1, T, 4)
    r = compute_reward_batch(
        traj4[:, :, :4].to("cpu"),
        {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in data.items()},
        reward_config,
    )[0]
    return _xyh_from_xycs(traj4[0].cpu().numpy()), r


def _render_one(npz_path, reward_config, view_half, model=None, model_args=None, device=None):
    scene = from_npz(str(npz_path))
    target = _target_xyh(npz_path)
    r_t = _score_saved_target(npz_path, reward_config)
    ok_t, sum_t, flags_t = _verdict(r_t)

    if model is not None:
        # GREEN = baked target (what we train toward); BLUE = model's current
        # deterministic prediction (baseline, untrained on this avoidance).
        det, r_m = _model_det(npz_path, model, model_args, device, reward_config)
        ok_m, sum_m, _ = _verdict(r_m)
        fig = render_scene_at_step(
            scene,
            gt_traj=target,
            det_traj=det,
            view_half=view_half,
            show_rb_dist=True,
            show_nb_dist=True,
            show_traj_rb=True,
            show_traj_nb=True,
            dim_neighbors=False,
            figsize=(8, 8),
            dpi=110.0,
        )
        title = (
            f"{Path(npz_path).stem}\n"
            f"GT/target [{'PASS' if ok_t else 'FAIL'}] {sum_t}\n"
            f"model DET [{'PASS' if ok_m else 'FAIL'}] {sum_m}"
        )
        image = _titled(fig, title, 9, "#1f9e3a" if ok_t else "#cc2222")
        return image, (ok_t, sum_t, flags_t)

    fig = render_scene_at_step(
        scene,
        det_traj=target,
        view_half=view_half,
        show_rb_dist=True,
        show_nb_dist=True,
        show_traj_rb=True,
        show_traj_nb=True,
        dim_neighbors=False,
        figsize=(8, 8),
        dpi=110.0,
    )
    verdict = "PASS" if ok_t else "FAIL: " + ", ".join(flags_t)
    image = _titled(
        fig,
        f"{Path(npz_path).stem}   [{verdict}]   {sum_t}",
        11,
        "#1f9e3a" if ok_t else "#cc2222",
    )
    return image, (ok_t, sum_t, flags_t)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene_dir")
    g.add_argument("--scenes", help="Path to a JSON file containing a list of NPZ paths")
    p.add_argument("--reward_config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--view_half", type=float, default=45.0)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument(
        "--model_path",
        default=None,
        help="If set, overlay the model's DET prediction (blue) vs the baked target (green).",
    )
    args = p.parse_args()

    model = model_args = device = None
    if args.model_path:
        from rlvr.autoresearch.tools.eval_det_avoidance import load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_args = load_model(args.model_path, device)

    if args.scene_dir:
        paths = sorted(Path(args.scene_dir).glob("scene_*.npz"))
    else:
        import json

        with open(args.scenes) as f:
            raw = json.load(f)
        seen, paths = set(), []
        for x in raw:
            if x not in seen:
                seen.add(x)
                paths.append(Path(x))
    if not paths:
        raise SystemExit("No scene NPZs found.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reward_config = load_reward_config(args.reward_config)

    png_paths, n_pass = [], 0
    for pth in paths:
        image, (ok, summary, flags) = _render_one(
            pth, reward_config, args.view_half, model=model, model_args=model_args, device=device
        )
        png = out / f"{pth.stem}.png"
        image.save(png, compress_level=1)
        png_paths.append(png)
        n_pass += int(ok)
        print(f"{pth.stem}: {'PASS' if ok else 'FAIL(' + ','.join(flags) + ')'}  {summary}")

    # Collage from the rendered PNGs (preserves GUI-style panels)
    n = len(png_paths)
    cols = args.cols
    rows = math.ceil(n / cols)
    fig = Figure((cols * 5, rows * 5), 90.0)
    axes = [
        panel
        for row in fig.grid(rows, cols, margins=(0.01, 0.99, 0.01, 0.99), wspace=0.04, hspace=0.04)
        for panel in row
    ]
    for i, png in enumerate(png_paths):
        axes[i].image(np.asarray(Image.open(png).convert("RGB")))
        axes[i].axis_off()
    for j in range(n, len(axes)):
        axes[j].hide()
    fig.save(out / "collage.png")

    print(f"\n{n_pass}/{n} scenes PASS all gates.")
    print(f"Wrote per-scene PNGs + collage.png to {out}")


if __name__ == "__main__":
    main()
