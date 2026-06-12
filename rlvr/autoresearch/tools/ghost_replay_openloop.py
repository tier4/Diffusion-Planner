"""Open-loop perfect-tracking replay video, one per avoidance scene — DUAL MODEL.

For each scene: ONE deterministic inference at t0 per model (baseline + best)
gives each model's 80-step ego trajectory; each ego then PERFECT-TRACKS its own
plan (no closed-loop re-inference). The video is preceded by the 3-second
(30-step) recorded ego history that is in the model context (shared, both models
start from the same state). Stopped neighbors are HIDDEN during the history and
APPEAR at t=0 when the predictions begin.

Timeline (10 Hz): 30 history frames (t=-3.0..-0.1s, one grey ego, no neighbors)
+ 80 trajectory frames (t=0.0..7.9s, baseline + best egos, neighbors shown)
= 110 frames = 11.0 s @ 10 fps.

Reuses the ghost-sim drawing helpers (recovery_sim / ghost_sim_common). WebM (VP9).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model
from rlvr.autoresearch.tools.ghost_sim_common import (
    _NB_COLOR,
    extract_scene_polylines,
    extract_stopped_neighbors,
)
from rlvr.autoresearch.tools.recovery_sim import (
    _LANE_BORDER_COLOR,
    _LANE_COLOR,
    _ROAD_BORDER_COLOR,
    _ROUTE_COLOR,
    _draw_agent_box,
)

BASELINE_COLOR = "#1f77b4"  # blue
BEST_COLOR = "#d62728"      # red
HIST_COLOR = "#555555"      # grey (shared history)


def _heading(row):
    if row.shape[-1] >= 4:
        return math.atan2(float(row[3]), float(row[2]))
    return float(row[2])


def _scene_base(ax, polylines, cx, cy, view_half):
    centerlines, lefts, rights, border_polylines, route_polylines, _ = polylines
    if centerlines:
        ax.add_collection(LineCollection(centerlines, colors=_LANE_COLOR,
                                         linewidths=0.6, alpha=0.28, zorder=1))
    for grp in (lefts, rights):
        if grp:
            ax.add_collection(LineCollection(grp, colors=_LANE_BORDER_COLOR,
                                             linewidths=1.1, alpha=0.7, zorder=2))
    half = view_half * 1.5
    fb = [pl for pl in border_polylines if pl.shape[0] >= 2 and (
        (pl[:, 0] >= cx - half) & (pl[:, 0] <= cx + half)
        & (pl[:, 1] >= cy - half) & (pl[:, 1] <= cy + half)).any()]
    if fb:
        ax.add_collection(LineCollection(fb, colors=_ROAD_BORDER_COLOR,
                                         linewidths=2.0, alpha=0.9, zorder=5))
    for pl in route_polylines:
        if pl.shape[0] >= 2:
            ax.plot(pl[:, 0], pl[:, 1], "-", color=_ROUTE_COLOR, lw=2.5, alpha=0.55, zorder=3)


def _render_frame(out_png, egos, polylines, neighbor_boxes, show_nb, title,
                  view_half, ego_shape, extra_lines=None):
    # egos: list of dicts {pose:[x,y,h], trail:(N,2), color, label, lw}
    # extra_lines: optional list of (xy:(N,2), color, alpha, label|None) static lines
    cx = float(np.mean([e["pose"][0] for e in egos]))
    cy = float(np.mean([e["pose"][1] for e in egos]))
    fig = Figure(figsize=(11, 11))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("#f8f8f8")
    _scene_base(ax, polylines, cx, cy, view_half)
    if show_nb and neighbor_boxes:
        for nx, ny, nh, nl, nw in neighbor_boxes:
            _draw_agent_box(ax, nx, ny, nh, nl, nw, _NB_COLOR, alpha=0.8, lw=1.5, zorder=14)
    if extra_lines:
        labeled = False
        for xy, color, alpha, lab in extra_lines:
            ax.plot(xy[:, 0], xy[:, 1], "-", color=color, lw=1.0, alpha=alpha,
                    zorder=16, label=(lab if not labeled and lab else None))
            labeled = labeled or bool(lab)
    for e in egos:
        tr = e["trail"]
        if tr.shape[0] > 1:
            ax.plot(tr[:, 0], tr[:, 1], "-", color=e["color"], lw=1.3, alpha=0.5, zorder=18)
        ex, ey, eh = e["pose"]
        _draw_agent_box(ax, ex, ey, eh, ego_shape[1], ego_shape[2], e["color"],
                        alpha=0.78, lw=2, zorder=20, wheelbase=ego_shape[0])
        al = max(ego_shape[1], 2.5)
        ax.annotate("", xy=(ex + al * math.cos(eh), ey + al * math.sin(eh)),
                    xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color=e["color"], lw=1.2,
                                    mutation_scale=10), zorder=22)
        ax.plot([], [], "-", color=e["color"], lw=2, label=e["label"])
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(cx - view_half, cx + view_half)
    ax.set_ylim(cy - view_half, cy + view_half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    fig.clf()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_baseline", required=True)
    p.add_argument("--model_best", default=None,
                   help="second model .pth; omit when using --policy_dir")
    p.add_argument("--policy_dir", default=None,
                   help="exploration-policy dir: 'best' = baseline + guidance "
                        "(same frozen model, policy-chosen etas via composer)")
    p.add_argument("--label_baseline", default="baseline")
    p.add_argument("--label_best", default="best")
    p.add_argument("--scenes", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ego_shape", required=True, help="WB,L,W")
    p.add_argument("--view_half", type=float, default=28.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hist_steps", type=int, default=30)
    p.add_argument("--ahead_thresh", type=float, default=8.0,
                   help="stopped neighbors with t0 longitudinal x >= this are AHEAD "
                        "(ego approaching) -> appear at t=0; x < this are beside/behind "
                        "(post-avoidance/branch, were visible) -> shown during history too")
    # Guidance envelope (must match the policy's training labels)
    p.add_argument("--lambda_lat", type=float, default=5.0)
    p.add_argument("--lat_scale", type=float, default=2.0)
    p.add_argument("--col_scale", type=float, default=9.0)
    p.add_argument("--col_range", type=float, default=8.0)
    p.add_argument("--lambda_spd", type=float, default=0.2)
    p.add_argument("--stretch_scale", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=0.5)
    p.add_argument("--n_candidates", type=int, default=0,
                   help="additionally sample N etas from the policy distribution "
                        "and draw their guided trajectories as a faint candidate fan")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = [float(x) for x in args.ego_shape.split(",")]
    m_base, a_base = load_model(args.model_baseline, device)
    policy = None
    if args.policy_dir:
        from exploration_policy.utils import run_frozen_encoder
        from guidance_gui.generate_samples import generate_samples
        from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer
        policy, heads = load_policy(args.policy_dir, a_base, device)
        m_best, a_best = m_base, a_base
    elif args.model_best:
        m_best, a_best = load_model(args.model_best, device)
    else:
        raise SystemExit("pass either --model_best or --policy_dir")
    scenes = json.load(open(args.scenes))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for sp in scenes:
        name = Path(sp).stem
        data = load_npz_data(sp, device)
        traj_base = det_inference_batched(m_base, a_base, [data], device)[0].cpu().numpy()
        label_best = args.label_best
        cand_lines = []
        if policy is not None:
            norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in data.items()}
            norm_data = a_base.observation_normalizer(norm_data)
            x_ref = torch.from_numpy(np.ascontiguousarray(traj_base)).float()
            x_ref = x_ref.unsqueeze(0).to(device)
            norm_data["reference_trajectory"] = x_ref
            enc = run_frozen_encoder(m_base, norm_data)
            pout = policy(enc, x_ref, deterministic=True)
            etas = {h: (2.0 * pout.dists[h].mean - 1.0).reshape(1) for h in heads}
            composer = make_composer(etas, args)
            traj_best = generate_samples(
                model=m_base, model_args=a_base, data=norm_data,
                noise_scale=0.0, n_samples=1, composer=composer, device=device,
            )[0]
            eta_str = " ".join(f"{h[:3]}={float(v.item()):+.2f}" for h, v in etas.items())
            label_best = f"{args.label_best} ({eta_str})"
            cand_lines = []
            if args.n_candidates > 0:
                from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
                N = args.n_candidates
                cand = {h: (2.0 * pout.dists[h].rsample((N,)).reshape(-1) - 1.0)
                        for h in heads}
                N_data = {}
                for k, v in norm_data.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                        N_data[k] = v.expand(N, *v.shape[1:]).contiguous()
                    else:
                        N_data[k] = v
                cand_trajs = _batched_generate_varied_noise(
                    m_base, a_base, N_data, noise_min=0.0, noise_max=0.0,
                    first_deterministic=False, composer=make_composer(cand, args),
                    device=device,
                ).cpu().numpy()
                cand_lines = [
                    (cand_trajs[i, :, :2], BEST_COLOR, 0.22,
                     f"{N} candidates from policy distribution" if i == 0 else None)
                    for i in range(N)
                ]
        else:
            traj_best = det_inference_batched(m_best, a_best, [data], device)[0].cpu().numpy()
        past = np.load(sp, allow_pickle=True)["ego_agent_past"].astype(np.float32)
        nb_boxes = extract_stopped_neighbors(sp)
        polylines = extract_scene_polylines(data)

        # PER-NEIGHBOR: beside/behind (x < thresh) were visible -> show in history;
        # ahead (x >= thresh, ego approaching) -> appear at t=0.
        hist_nb = [b for b in nb_boxes if b[0] < args.ahead_thresh]
        n_ahead = len(nb_boxes) - len(hist_nb)
        hidden_note = f" — {n_ahead} neighbor(s) ahead appear at t=0" if n_ahead else ""

        H = min(args.hist_steps, past.shape[0])
        hist = past[past.shape[0] - H:]
        sc_dir = out_root / name
        sc_dir.mkdir(parents=True, exist_ok=True)

        fi = 0
        hist_trail = []
        # --- history: one shared grey ego; only already-visible (beside/behind) neighbors ---
        for i, row in enumerate(hist):
            pose = np.array([row[0], row[1], _heading(row)])
            hist_trail.append(pose[:2])
            t = -(H - i) * 0.1
            egos = [dict(pose=pose, trail=np.array(hist_trail), color=HIST_COLOR,
                         label=f"ego history  t={t:+.1f}s", lw=2)]
            _render_frame(sc_dir / f"f{fi:04d}.png", egos, polylines, hist_nb,
                          True, f"{name}   t={t:+.1f}s   HISTORY (model context){hidden_note}",
                          args.view_half, ego_shape)
            fi += 1
        # --- track: baseline + best egos perfect-track their plans, neighbors shown ---
        tb, tk = [], []
        for i in range(traj_base.shape[0]):
            pb = np.array([traj_base[i, 0], traj_base[i, 1], _heading(traj_base[i])])
            pk = np.array([traj_best[i, 0], traj_best[i, 1], _heading(traj_best[i])])
            tb.append(pb[:2]); tk.append(pk[:2])
            egos = [
                dict(pose=pb, trail=np.array(tb), color=BASELINE_COLOR, label=args.label_baseline, lw=2),
                dict(pose=pk, trail=np.array(tk), color=BEST_COLOR, label=label_best, lw=2),
            ]
            _render_frame(sc_dir / f"f{fi:04d}.png", egos, polylines, nb_boxes,
                          True, f"{name}   t={i*0.1:+.1f}s   PERFECT-TRACK (baseline vs best)",
                          args.view_half, ego_shape, extra_lines=cand_lines)
            fi += 1

        webm = out_root / f"{name}.webm"
        rc = subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(sc_dir / "f%04d.png"),
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1",
            "-pix_fmt", "yuv420p", str(webm)], capture_output=True)
        ok = "OK" if rc.returncode == 0 else f"FAIL {rc.stderr.decode()[-150:]}"
        print(f"{name}: {fi} frames ({H} hist + {traj_base.shape[0]} track), {len(nb_boxes)} nb -> {webm.name} [{ok}]")


if __name__ == "__main__":
    main()
