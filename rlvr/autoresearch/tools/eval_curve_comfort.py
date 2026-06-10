"""Open-loop comfort on ROUTE-CURVE scenes (NOT the avoidance val set).

The 'violent curves' live on the route's big curve (cruise/wedge scenes), which the
avoidance val_v2 set does NOT represent — so comfort must be scored on curve scenes.
For each model, runs det inference and reports per-scene lat-accel peak distribution
and the speed in high-curvature steps. Compares any number of models side by side.

Reuses eval_det_avoidance.{load_model,load_npz_data} + eval_driving_metrics.generate_trajectory.

Usage:
  python -m rlvr.autoresearch.tools.eval_curve_comfort \
    --scenes <curve_scenes.json> --models name1=/path1.pth name2=/path2.pth [--n 60]
"""
import argparse, json
import numpy as np
import torch

from rlvr.autoresearch.tools.eval_det_avoidance import load_model, load_npz_data
from rlvr.autoresearch.tools.eval_driving_metrics import generate_trajectory

DT = 0.1


def _stats(model_path, scenes, n, dev):
    m, margs = load_model(model_path, dev)
    paths = json.load(open(scenes))[:n]
    lat_peak, curve_speed = [], []
    for p in paths:
        d = load_npz_data(p, dev)
        tr = generate_trajectory(m, margs, d, dev)
        tr = (tr.detach().cpu().numpy() if hasattr(tr, "detach") else np.asarray(tr))[:, :2].astype(np.float64)
        v = np.diff(tr, axis=0) / DT
        sp = np.linalg.norm(v, axis=1)
        th = np.arctan2(v[:, 1], v[:, 0])
        dth = np.diff(np.unwrap(th)) / DT
        la = np.abs(sp[:-1] * dth)
        lat_peak.append(la.max())
        if (la > 1.0).any():
            curve_speed.append(sp[:-1][la > 1.0].mean())
    return np.array(lat_peak), np.array(curve_speed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="curve-scene JSON list")
    ap.add_argument("--models", nargs="+", required=True, help="name=path.pth ...")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# curve comfort on {args.scenes} (n={args.n})")
    print(f"  {'model':<16} {'lat_peak mean':>13} {'p95':>6} {'max':>6} {'curve_speed':>11}")
    for spec in args.models:
        name, path = spec.split("=", 1)
        lp, cs = _stats(path, args.scenes, args.n, dev)
        print(f"  {name:<16} {lp.mean():>13.2f} {np.percentile(lp,95):>6.2f} {lp.max():>6.2f} "
              f"{(cs.mean() if len(cs) else float('nan')):>11.2f}")


if __name__ == "__main__":
    main()
