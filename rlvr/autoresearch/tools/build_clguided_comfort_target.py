"""Build curated CL-guided targets that are CENTERED *and* COMFORTABLE.

Drop-in comfort-aware replacement for build_clguided_target.py. Same generation
(K trajectories under strong route-CL guidance), but the SELECTION is multi-factor
instead of pure most-centered:

  Among the K candidates, keep those that are
    (1) KINEMATICALLY FEASIBLE         — reward.compute_kinematic_gate == 1
    (2) CENTERED ENOUGH                — centerline_score >= max_cl - cl_margin
  then pick the one with the LOWEST comfort cost over the [past ⊕ future] path
    comfort = mean|lat_accel| + jerk_weight * mean|jerk|
  evaluated on positions of ego_agent_past PREPENDED to the candidate future, so a
  target that is smooth in isolation but discontinuous with the current/past ego
  state (a violent t=0 correction) is penalized and rejected.

Scenes where NO candidate is feasible+centered are DROPPED (logged, not written) —
no silent poison targets.

WHY: build_clguided_target picks argmax(centerline) with zero comfort/feasibility
consideration. Under strong CL guidance the most-centered slot on a curve is a
centered-but-SHARP trajectory; curated SFT on it teaches the model to plan sharp
curves (the open-loop lat-accel regression the real vehicle feels as "violent
curves" even though psim's controller smooths it). This keeps centering (CL/RB/bias)
while removing the sharpness, and guarantees continuity with the actual ego state.

Reuses ONLY existing fns: eval_det_avoidance.{load_model,load_npz_data},
grpo_trainer_batched.{_stack_scene_data,_normalize_batch,generate_all_scenes_batched},
reward.{compute_centerline_score_batch,compute_kinematic_gate,RewardConfig},
eval_driving_metrics.lat_accel_smoothed, reward._build_sg_diff_kernel.
"""
import argparse, json, os
import numpy as np
import torch

from rlvr.autoresearch.tools.eval_det_avoidance import load_model, load_npz_data
from rlvr.grpo_trainer_batched import _stack_scene_data, _normalize_batch, generate_all_scenes_batched
from rlvr.reward import compute_centerline_score_batch, compute_kinematic_gate, RewardConfig, _build_sg_diff_kernel
from rlvr.autoresearch.tools.eval_driving_metrics import lat_accel_smoothed

DT = 0.1


def _jerk_mag(xy: np.ndarray, window: int) -> np.ndarray:
    """|d3(x,y)/dt3| per step from positions (T,2) via SG deriv=3 (reuses reward kernel)."""
    if xy.shape[0] < window:
        return np.zeros(xy.shape[0])
    k = _build_sg_diff_kernel(window=window, poly=3, deriv=3, delta=DT)
    pad = window // 2
    t = torch.from_numpy(xy).float().permute(1, 0).unsqueeze(0)  # [1,2,T]
    t = torch.nn.functional.pad(t, (pad, pad), mode="replicate")
    j = torch.nn.functional.conv1d(t, k.view(1, 1, -1).expand(2, 1, -1), groups=2)[0].numpy()
    return np.sqrt(j[0] ** 2 + j[1] ** 2)


def _comfort_cost(past_xy: np.ndarray, fut_xy: np.ndarray, jerk_weight: float,
                  peak_weight: float, window: int, mean_weight: float = 1.0) -> float:
    """Comfort cost over past⊕future positions (continuity-aware).

    cost = mean_weight*mean|lat_accel| + peak_weight*max|lat_accel|
           + jerk_weight*(mean|jerk| + max|jerk|/10)

    'Violent curves' is dominated by the PEAK lateral accel and the JERK (sudden
    steering), NOT the mean — a centered curve can be taken with a gentle steering
    ramp (low jerk/peak) at the same mean lat_accel. Set mean_weight=0 to target
    ONLY peak+jerk: this keeps the trajectory's mean lat-accel GT-like (so it stays
    close to GT → low ego-L2 cost), unlike penalizing the mean which drifts ego off
    GT (the v1 +7.2% ego-L2 regression).
    """
    cat = np.concatenate([past_xy, fut_xy], axis=0)  # (Tp+T, 2)
    la = np.abs(lat_accel_smoothed(cat, window=window))
    jk = _jerk_mag(cat, window)
    return float(mean_weight * la.mean() + peak_weight * la.max()
                 + jerk_weight * (jk.mean() + jk.max() / 10.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="centered source (baseline ep60_80)")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--variant", default="rl_cl_soft_sweep_stretch")
    ap.add_argument("--gt_max_speed", type=float, default=9.0)
    ap.add_argument("--cl_margin", type=float, default=0.08,
                    help="keep candidates with centerline_score >= max_cl - cl_margin (centered enough)")
    ap.add_argument("--jerk_weight", type=float, default=0.1,
                    help="weight of jerk (mean+max/10) in comfort cost")
    ap.add_argument("--peak_weight", type=float, default=0.5,
                    help="weight of max|lat_accel| (the 'violence' peak) in comfort cost")
    ap.add_argument("--mean_weight", type=float, default=1.0,
                    help="weight of mean|lat_accel|; set 0 to target ONLY peak+jerk (keeps mean GT-like → low ego-L2 cost)")
    ap.add_argument("--comfort_window", type=int, default=11, help="SG window for lat_accel/jerk")
    ap.add_argument("--report", default=None, help="optional JSON report of per-scene selection")
    args = ap.parse_args()

    WB, L, W = [float(x) for x in args.ego_shape.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model, dev)
    cfg = RewardConfig()
    paths = json.load(open(args.scenes))
    os.makedirs(args.out_dir, exist_ok=True)

    written, dropped, report = [], [], []
    n_cen = 0
    for p in paths:
        d = load_npz_data(p, dev)
        es = d["ego_shape"]; es_one = es[0] if es.dim() > 1 else es
        nb = _normalize_batch(_stack_scene_data([d], dev), margs)
        trajs = generate_all_scenes_batched(
            model, margs, nb, K=args.K, noise_range=(0.5, 2.0), device=dev,
            gen_chunk_size=args.K, gt_max_speed=args.gt_max_speed,
            generation_variant=args.variant, use_route_cl_guidance=True)[0]  # (K,T,4)

        cl = compute_centerline_score_batch(trajs, es_one, d, usage_mode="baselink")  # (K,)
        gate = compute_kinematic_gate(trajs, cfg, es_one)  # (K,) 1/0
        cl_max = float(cl.max())
        feasible = (gate > 0.5) & (cl >= cl_max - args.cl_margin)

        raw = dict(np.load(p, allow_pickle=True))
        past_xy = np.asarray(raw["ego_agent_past"], dtype=np.float32)[:, :2]
        idxs = torch.nonzero(feasible, as_tuple=False).flatten().tolist()
        if not idxs:
            dropped.append(p)
            report.append({"scene": os.path.basename(p), "kept": 0, "cl_max": cl_max,
                           "n_feasible_gate": int((gate > 0.5).sum()), "reason": "no feasible+centered candidate"})
            continue

        costs = [(_comfort_cost(past_xy, trajs[i, :, :2].detach().cpu().numpy(),
                                args.jerk_weight, args.peak_weight, args.comfort_window,
                                args.mean_weight), i) for i in idxs]
        best_cost, best = min(costs)
        if float(cl[best]) > -0.05:
            n_cen += 1
        raw["ego_agent_future"] = trajs[best].detach().cpu().numpy().astype(np.float32)
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        np.savez(out_p, **raw); written.append(out_p)
        report.append({"scene": os.path.basename(p), "kept": len(idxs), "best_slot": int(best),
                       "best_cl": float(cl[best]), "cl_max": cl_max, "comfort_cost": round(best_cost, 4)})

    json.dump(written, open(args.out_list, "w"), indent=1)
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)
    print(f"wrote {len(written)} comfort+centered targets -> {args.out_dir} "
          f"(centerline>-0.05: {n_cen}/{len(written)}; dropped {len(dropped)}/{len(paths)} no-feasible)")


if __name__ == "__main__":
    main()
