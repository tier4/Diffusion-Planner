"""Re-time curated targets to cap CURVE lateral acceleration WITHOUT de-centering.

The comfort regression on the big curve is sustained lateral accel = v²·κ (the
centered curve's radius). You cannot lower it by moving the path (that de-centers
and breaks RB). The one lever that keeps the centered path is SLOWING in the curve:
lat = v²·κ, so capping v where κ is high cuts lat-accel while the geometry (centering)
is byte-identical. Straights keep GT speed → minimal longitudinal-L2 / progress cost.

For each curated target NPZ this:
  1. takes ego_agent_future (x,y,cos,sin) as a fixed PATH (prepended with the
     current ego origin),
  2. computes per-node curvature κ and the original speed profile,
  3. caps speed v_cap = min(v_orig, sqrt(a_lat_max / κ)) in curves,
  4. applies forward/backward accel limits so it brakes BEFORE the curve and
     accelerates after (no jerk step),
  5. re-samples the SAME path at the new (slower-in-curve) arc positions over 80
     steps and recomputes headings from the path tangent,
  6. writes the re-timed trajectory back into ego_agent_future (all else verbatim).

Path geometry is unchanged → centering / RB / lane preserved by construction.
Only the speed/timing changes. No model, no GPU.
"""
import argparse, json, os
import numpy as np

DT = 0.1


def _retime(fut: np.ndarray, a_lat_max: float, a_long_max: float, v_min: float) -> np.ndarray:
    """fut: [T,4] (x,y,cos,sin). Returns re-timed [T,4] along the same path."""
    T = fut.shape[0]
    P = fut[:, :2].astype(np.float64)
    nodes = np.concatenate([np.zeros((1, 2)), P], axis=0)  # [T+1,2], origin + path
    seg = np.diff(nodes, axis=0)                            # [T,2]
    seg_len = np.linalg.norm(seg, axis=1)                   # [T]
    S = np.concatenate([[0.0], np.cumsum(seg_len)])         # [T+1] arc at each node
    L = float(S[-1])
    if L < 1e-3:
        return fut  # degenerate / stopped target — leave as-is

    # Heading per segment, curvature per interior node.
    theta = np.arctan2(seg[:, 1], seg[:, 0])               # [T]
    dtheta = np.diff(np.unwrap(theta))                     # [T-1]
    mid_len = 0.5 * (seg_len[:-1] + seg_len[1:]) + 1e-6
    kappa = np.abs(dtheta) / mid_len                       # [T-1] curvature at nodes 1..T-1
    kappa = np.concatenate([kappa[:1], kappa, kappa[-1:]]) # pad to [T+1] per node
    kappa = np.maximum(kappa[:T], 1e-4)                    # per-segment, [T]

    v_orig = seg_len / DT                                  # [T] original speed per segment
    v_cap = np.minimum(v_orig, np.sqrt(a_lat_max / kappa)) # comfort cap in curves
    v_cap = np.maximum(v_cap, v_min)
    v_cap = np.minimum(v_cap, v_orig)                      # never speed UP vs original

    # Accel/decel limits: forward (limit acceleration), backward (brake early).
    v = v_cap.copy()
    for k in range(1, T):
        v[k] = min(v[k], v[k - 1] + a_long_max * DT)
    for k in range(T - 2, -1, -1):
        v[k] = min(v[k], v[k + 1] + a_long_max * DT)
    v = np.maximum(v, 0.0)

    # New arc positions over T steps (start one step in, like the original future[0]).
    s_new = np.cumsum(v * DT)                              # [T] arc reached at each step
    s_new = np.clip(s_new, 0.0, L)

    # Re-sample the original path (nodes, S) at s_new → new xy.
    xs = np.interp(s_new, S, nodes[:, 0])
    ys = np.interp(s_new, S, nodes[:, 1])

    # Headings: interpolate the ORIGINAL smooth headings (cos/sin) at the new arc
    # positions, NOT from re-sampled point diffs (those jitter at low speed, which
    # aliased into lat-accel spikes). The original future point i sits at arc S[i+1].
    arc_fut = S[1:]                                        # [T] arc of each original future point
    cos_i = np.interp(s_new, arc_fut, fut[:, 2].astype(np.float64))
    sin_i = np.interp(s_new, arc_fut, fut[:, 3].astype(np.float64))
    nrm = np.sqrt(cos_i ** 2 + sin_i ** 2) + 1e-9
    out = fut.copy()
    out[:, 0] = xs
    out[:, 1] = ys
    out[:, 2] = cos_i / nrm
    out[:, 3] = sin_i / nrm
    return out.astype(np.float32)


def _lat_peak(fut: np.ndarray) -> float:
    P = fut[:, :2].astype(np.float64)
    v = np.diff(P, axis=0) / DT
    sp = np.linalg.norm(v, axis=1)
    th = np.arctan2(v[:, 1], v[:, 0])
    dth = np.diff(np.unwrap(th)) / DT
    return float(np.abs(sp[:-1] * dth).max()) if len(sp) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="JSON list of curated-target NPZs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--a_lat_max", type=float, default=1.5, help="comfort lateral-accel cap (m/s²)")
    ap.add_argument("--a_long_max", type=float, default=1.5, help="accel/decel limit for the speed profile (m/s²)")
    ap.add_argument("--v_min", type=float, default=2.0, help="floor speed so it never crawls to a stop")
    args = ap.parse_args()

    paths = json.load(open(args.scenes))
    os.makedirs(args.out_dir, exist_ok=True)
    written, before, after = [], [], []
    for p in paths:
        raw = dict(np.load(p, allow_pickle=True))
        fut = np.asarray(raw["ego_agent_future"]).astype(np.float32)
        before.append(_lat_peak(fut))
        fut2 = _retime(fut, args.a_lat_max, args.a_long_max, args.v_min)
        after.append(_lat_peak(fut2))
        raw["ego_agent_future"] = fut2
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        np.savez(out_p, **raw); written.append(out_p)
    json.dump(written, open(args.out_list, "w"), indent=1)
    before, after = np.array(before), np.array(after)
    print(f"wrote {len(written)} re-timed targets -> {args.out_dir}")
    print(f"  lat_peak BEFORE: mean={before.mean():.2f} p95={np.percentile(before,95):.2f} max={before.max():.2f}")
    print(f"  lat_peak AFTER : mean={after.mean():.2f} p95={np.percentile(after,95):.2f} max={after.max():.2f} "
          f"(cap {args.a_lat_max}); reduced {int((after<before-0.05).sum())}/{len(written)}")


if __name__ == "__main__":
    main()
