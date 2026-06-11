#!/usr/bin/env python3
"""Policy-guided avoidance eval: frozen model + exploration policy vs baseline.

Per scene: the exploration policy (deterministic=True) picks per-head etas,
ONE guided deterministic trajectory is generated through the guidance
composer, plus the unguided det trajectory; both are scored with the
canonical reward. Reports the full side-by-side table (guided vs baseline)
on avoidance scenes, and inertness metrics (|eta|, trajectory deviation) on
normal scenes.

Optionally renders per-scene baseline-vs-guided comparison PNGs + a collage.

Usage:
    python -m rlvr.autoresearch.tools.eval_policy_avoidance \
        --model_path <base.pth> --policy_dir <run_dir with exploration_policy.pth> \
        --scenes <avoidance_val.json> [--normal_scenes <normal_val.json>] \
        --config <reward_config.json> --ego_shape WB,L,W --output_dir <dir> \
        [--render] [--lambda_lat 4.0] [--col_scale 6.0] [--lat_scale 1.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401 -- registers batched guidance
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import aggregate_stats, load_model
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import compute_reward_batch


def load_policy(policy_dir: str, model_args, device) -> tuple[ExplorationPolicy, list[str]]:
    pdir = Path(policy_dir)
    cfg = ExplorationPolicyConfig.from_json(pdir / "exploration_policy_config.json")
    policy = ExplorationPolicy(cfg, ref_seq_len=model_args.future_len).to(device)
    state = torch.load(pdir / "exploration_policy.pth", map_location=device,
                       weights_only=False)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, cfg.heads


def make_composer(etas: dict[str, torch.Tensor], args) -> GuidanceComposer:
    # Thin wrapper over the shared head mapping; head_protect > 0 zeroes
    # guidance on the first N plan steps.
    from rlvr.guidance_batched import build_head_composer
    return build_head_composer(
        etas,
        lambda_lat=args.lambda_lat,
        lat_scale=args.lat_scale,
        col_scale=args.col_scale,
        col_range=args.col_range,
        lambda_spd=args.lambda_spd,
        stretch_scale=args.stretch_scale,
        guidance_scale=args.guidance_scale,
        head_protect=int(getattr(args, "head_protect", 0)),
        envelope=getattr(args, "envelope", "v1"),
        lambda_col=getattr(args, "lambda_col", 3.0),
    )


def _new_violation(g, d) -> bool:
    """Guided introduces a violation the baseline det didn't have."""
    return (
        (g.static_crossing and not d.static_crossing)
        or (g.rb_crossing and not d.rb_crossing)
        or (g.lane_crossing and not d.lane_crossing)
        or (g.collision_step is not None and d.collision_step is None)
    )


@torch.no_grad()
def eval_scene(model, model_args, policy, heads, npz_path, rcfg, args, device):
    data = load_npz_data(npz_path, device)
    norm_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    norm_data = model_args.observation_normalizer(norm_data)

    x_ref_np = generate_reference_trajectory(model, model_args, norm_data, device)
    x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(device)
    norm_data["reference_trajectory"] = x_ref
    enc = run_frozen_encoder(model, norm_data)

    out = policy(enc, x_ref, deterministic=True)
    mean_etas = {h: (2.0 * out.dists[h].mean - 1.0).reshape(1) for h in heads}

    K = max(1, args.refine_k)
    if K > 1:
        # Candidate 0 = the deterministic mean; the rest sampled from the
        # policy's Beta distributions (its own uncertainty = search width).
        cand = {
            h: torch.cat([
                mean_etas[h],
                (2.0 * out.dists[h].rsample((K - 1,)).reshape(-1) - 1.0),
            ])
            for h in heads
        }
        K_data = {}
        for k, v in norm_data.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                K_data[k] = v.expand(K, *v.shape[1:]).contiguous()
            else:
                K_data[k] = v
        composer = make_composer(cand, args)
        from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
        cands = _batched_generate_varied_noise(
            model, model_args, K_data, noise_min=0.0, noise_max=0.0,
            first_deterministic=False, composer=composer, device=device,
        ).cpu().numpy()
    else:
        cand = mean_etas
        composer = make_composer(cand, args)
        cands = generate_samples(model=model, model_args=model_args, data=norm_data,
                                 noise_scale=0.0, n_samples=1, composer=composer,
                                 device=device)[None, 0]

    det = x_ref_np  # x_ref IS the unguided det trajectory
    traj_batch = torch.tensor(
        np.concatenate([cands, det[None]]), device=device, dtype=torch.float32,
    )
    bds = compute_reward_batch(traj_batch, data, rcfg)
    cand_bds, d_bd = bds[:-1], bds[-1]

    # Pick the best candidate: no NEW violations, then max total reward.
    # Certify semantics: guidance exists ONLY for avoidance — when the
    # unguided det has no static crossing, keep it untouched (strict
    # inertness on scenes with no avoidance need). When det fails, pick the
    # best clean candidate; fall back to det if every candidate regresses.
    pick, g_bd = 0, cand_bds[0]
    if args.certify and not d_bd.static_crossing:
        pick, g_bd = -1, d_bd
    elif args.certify or K > 1:
        clean = [i for i, b in enumerate(cand_bds) if not _new_violation(b, d_bd)]
        if not clean:
            pick, g_bd = -1, d_bd  # fall back to baseline
        else:
            pick = max(clean, key=lambda i: cand_bds[i].total)
            g_bd = cand_bds[pick]

    guided = det if pick == -1 else cands[pick]
    etas = (
        {h: 0.0 for h in heads} if pick == -1
        else {h: float(cand[h][pick].item()) for h in heads}
    )
    etas = {h: torch.tensor([v], device=device) for h, v in etas.items()}
    deviation = float(np.linalg.norm(guided[:, :2] - det[:, :2], axis=-1).mean())

    def _row(r):
        return {
            "sc_min_dist": float(r.sc_min_dist), "rb_min_dist": float(getattr(r, "rb_min_dist", 99.0)),
            "cl": float(r.centerline), "total": float(r.total),
            "static_crossing": bool(r.static_crossing), "rb_cross": bool(r.rb_crossing),
            "lane_cross": bool(r.lane_crossing), "kin_violated": bool(r.kinematic_violated),
            "sc_n_stopped": int(r.sc_n_stopped),
        }

    return {
        "scene": Path(npz_path).name, "scene_path": str(npz_path),
        "etas": {h: float(v.item()) for h, v in etas.items()},
        "deviation": deviation,
        "guided": _row(g_bd), "baseline": _row(d_bd),
        "_trajs": (guided, det),
    }


def render_scene(row, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rlvr.autoresearch.tools.ghost_sim_common import extract_stopped_neighbors
    from rlvr.autoresearch.tools.viz_cl_recovery import draw_scene_base
    from scenario_generation.visualize import draw_agent_box

    guided, det = row["_trajs"]
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_scene_base(ax, row["scene_path"])
    for (x, y, h, length, w) in extract_stopped_neighbors(row["scene_path"]):
        draw_agent_box(ax, x, y, h, length, w, color="crimson", alpha=0.5)
    ax.plot(det[:, 0], det[:, 1], "-", color="black", lw=2.2,
            label=f"baseline det sc={row['baseline']['sc_min_dist']:+.2f}m")
    ax.plot(guided[:, 0], guided[:, 1], "-", color="lime", lw=2.2,
            label=f"policy-guided sc={row['guided']['sc_min_dist']:+.2f}m")
    eta_str = " ".join(f"{h}={v:+.2f}" for h, v in row["etas"].items())
    ax.set_title(f"{row['scene']}  η: {eta_str}  dev={row['deviation']:.2f}m")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    span = max(np.abs(det[..., :2]).max(), np.abs(guided[..., :2]).max()) + 12
    ax.set_xlim(-12, max(span, 40))
    ax.set_ylim(-span / 2, span / 2)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_collage(png_paths: list[Path], out_path: Path, cols: int = 4):
    from PIL import Image
    if not png_paths:
        return
    ims = [Image.open(p) for p in png_paths]
    w = min(im.width for im in ims)
    ims = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    h = max(im.height for im in ims)
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(ims):
        canvas.paste(im, ((i % cols) * w, (i // cols) * h))
    canvas.save(out_path)


def summarize(rows: list[dict], which: str) -> dict:
    sub = [dict(r[which], scene=r["scene"]) for r in rows]
    agg = aggregate_stats([
        {**s, "static_crossing": s["static_crossing"], "rb_cross": s["rb_cross"],
         "lane_cross": s["lane_cross"], "kin_violated": s["kin_violated"]}
        for s in sub
    ])
    return agg


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--scenes", required=True, help="avoidance eval scene list")
    parser.add_argument("--normal_scenes", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ego_shape", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--certify", action="store_true",
                        help="Fall back to the unguided det trajectory whenever "
                             "the guided one introduces a NEW violation (or "
                             "lowers total reward on an already-clean scene). "
                             "Guarantees no regressions by construction.")
    parser.add_argument("--refine_k", type=int, default=1,
                        help="Sample K-1 extra eta candidates from the policy "
                             "distribution, score all, pick the best clean one "
                             "(policy = prior, reward = judge). 1 = mean only.")
    # Guidance envelope (must match the sweep that produced the training labels)
    parser.add_argument("--lambda_lat", type=float, default=4.0)
    parser.add_argument("--lat_scale", type=float, default=1.5)
    parser.add_argument("--col_scale", type=float, default=6.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--envelope", choices=["v1", "v2"], default="v1")
    parser.add_argument("--lambda_col", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    rcfg = load_reward_config(args.config)
    model, model_args = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, model_args, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_set(scene_list_path, tag):
        with open(scene_list_path) as f:
            paths = json.load(f)
        rows = []
        render_dir = out_dir / f"render_{tag}"
        if args.render:
            render_dir.mkdir(exist_ok=True)
        for p in paths:
            try:
                with np.load(p, allow_pickle=True) as npz:
                    if "ego_shape" not in set(npz.keys()):
                        print(f"  [skip] {Path(p).name}: missing ego_shape")
                        continue
                    es = np.asarray(npz["ego_shape"]).reshape(-1)[:3]
                if not np.allclose(es, ego_shape, atol=1e-2):
                    print(f"  [skip] {Path(p).name}: ego_shape={es.tolist()}")
                    continue
                row = eval_scene(model, model_args, policy, heads, p, rcfg, args, device)
            except Exception as e:  # noqa: BLE001
                print(f"  [err ] {Path(p).name}: {e}")
                continue
            if args.render:
                render_scene(row, render_dir / f"{Path(p).stem}.png")
            row.pop("_trajs")
            rows.append(row)
            g, b = row["guided"], row["baseline"]
            flag = ("FIX " if b["static_crossing"] and not g["static_crossing"] else
                    "BRK " if g["static_crossing"] and not b["static_crossing"] else
                    "COL " if g["static_crossing"] else "    ")
            eta_str = " ".join(f"{h[0]}={v:+.2f}" for h, v in row["etas"].items())
            print(f"  [{tag}] {flag} sc {b['sc_min_dist']:+.2f}->{g['sc_min_dist']:+.2f} "
                  f"dev={row['deviation']:.2f} η[{eta_str}]  {row['scene']}")
        if args.render and rows:
            pngs = sorted(render_dir.glob("*.png"))
            make_collage(pngs, out_dir / f"collage_{tag}.png")
        return rows

    avoid_rows = run_set(args.scenes, "avoid")
    normal_rows = run_set(args.normal_scenes, "normal") if args.normal_scenes else []

    report = {
        "guidance_args": {k: getattr(args, k) for k in (
            "lambda_lat", "lat_scale", "col_scale", "col_range",
            "lambda_spd", "stretch_scale", "guidance_scale")},
        "avoid": {
            "n": len(avoid_rows),
            "guided": summarize(avoid_rows, "guided") if avoid_rows else {},
            "baseline": summarize(avoid_rows, "baseline") if avoid_rows else {},
            "deviation_mean": float(np.mean([r["deviation"] for r in avoid_rows])) if avoid_rows else 0.0,
            "eta_abs_mean": {
                h: float(np.mean([abs(r["etas"][h]) for r in avoid_rows]))
                for h in heads
            } if avoid_rows else {},
        },
        "normal": {
            "n": len(normal_rows),
            "guided": summarize(normal_rows, "guided") if normal_rows else {},
            "baseline": summarize(normal_rows, "baseline") if normal_rows else {},
            "deviation_mean": float(np.mean([r["deviation"] for r in normal_rows])) if normal_rows else 0.0,
            "deviation_max": float(np.max([r["deviation"] for r in normal_rows])) if normal_rows else 0.0,
            "eta_abs_mean": {
                h: float(np.mean([abs(r["etas"][h]) for r in normal_rows]))
                for h in heads
            } if normal_rows else {},
        },
        "scenes_avoid": avoid_rows,
        "scenes_normal": normal_rows,
    }
    with open(out_dir / "policy_eval.json", "w") as f:
        json.dump(report, f, indent=1)

    for tag, rows in (("avoid", avoid_rows), ("normal", normal_rows)):
        if not rows:
            continue
        g = report[tag]["guided"]
        b = report[tag]["baseline"]
        print(f"\n== {tag} ({len(rows)} scenes) ==")
        print(f"  static crossings: baseline {b['static_crossings']} -> guided {g['static_crossings']}")
        print(f"  rb crossings:     baseline {b['rb_crossings']} -> guided {g['rb_crossings']}")
        print(f"  lane crossings:   baseline {b['lane_crossings']} -> guided {g['lane_crossings']}")
        print(f"  sc_min_dist p5:   baseline {b['sc_min_dist']['p5']:+.3f} -> guided {g['sc_min_dist']['p5']:+.3f}")
        print(f"  deviation mean:   {report[tag]['deviation_mean']:.3f} m")
        print(f"  |eta| mean:       {report[tag]['eta_abs_mean']}")
    print(f"\nWrote {out_dir / 'policy_eval.json'}")


if __name__ == "__main__":
    main()
