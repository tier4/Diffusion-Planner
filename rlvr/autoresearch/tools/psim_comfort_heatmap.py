"""Realized closed-loop ego COMFORT/DYNAMICS heatmap from psim bag(s).

The road-border / clearance heatmaps measure GEOMETRY. This one measures the
DYNAMICS the geometry hides: lateral acceleration, lateral jerk, yaw rate,
curvature, and centerline-offset oscillation of the ACTUAL realized ego
(from /localization/kinematic_state), binned along the route arc.

Why: a model can score great on mean centerline distance (cl) yet take a
dynamically violent / oscillating line through a turn — the comfort cost lives
in the derivatives, which no geometric metric in the suite ever measured.

★ Metrics use MEASURED twist (no position derivatives), and dt is derived from the
message timestamps per bag — NOT assumed. (An earlier version assumed 100 Hz / dt=0.1
and took 3rd-derivatives of position; psim localization is ~40 Hz, so that inflated
speed ~2.5x, lat-accel ~6x, jerk ~16x. Fixed.)
- kinematics (t, x, y, speed=twist.linear.x, yaw_rate=twist.angular.z) from
  /localization/kinematic_state via _extract_kinematics (base_link → twist.linear.y≈0).
- lat_accel (PRIMARY) = |accel.y| from /localization/acceleration (base_link, EKF) — the
  measured/felt lateral accel, interpolated onto the pose timestamps (first msg dropped as
  NaN/uninitialized). In psim there is no IMU so the EKF derives it as the kinematic
  centripetal v·ω → verified identical to |yaw_rate*speed| (corr 1.000 on a psim bag). On a
  REAL-vehicle bag it is the genuine felt accel and can diverge from v·ω (slip/bank/noise).
  lat_accel_kin = |yaw_rate*speed| is kept as a SECONDARY sanity column (the accel.y-vs-v·ω
  gap is ≈0 in psim, diagnostic on a real bag).
- jerk = |d(lat_accel)/dt| (lateral jerk: ONE derivative of the measured accel signal,
  reward._build_sg_diff_kernel deriv=1 with the REAL dt — NEVER a 3rd-derivative of position).
  curvature = |yaw_rate|/max(|speed|,0.5). speed = measured forward speed.
- route centerline polyline + arc + signed lateral: _heatmap_common
  (build_route_polyline / project_to_polyline / segments_from_polyline / plot_route_heatmap).

Run under the ROS env (lanelet2 + rosbag), e.g.:
  bash -c "source /opt/ros/humble/setup.bash && ROS_DOMAIN_ID=33 \
    .venv/bin/python -m rlvr.autoresearch.tools.psim_comfort_heatmap \
    --route <pkl> --bag <champion_bag> --baseline_bag <baseline_bag> \
    --ego_shape 4.76,7.24,2.29 --out_dir <dir> --label model --baseline_label baseline"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scenario_generation.tools._heatmap_common import (
    build_route_polyline,
    load_route,
    plot_route_heatmap,
    project_to_polyline,
    segments_from_polyline,
)
from rlvr.reward import _build_sg_diff_kernel

# SG window/poly for the lateral-jerk derivative. ★ dt is DERIVED from the message
# timestamps per bag (psim localization is ~40 Hz, NOT 100 Hz) — assuming a fixed dt
# was a real bug that inflated every position-derivative (speed 2.5x, lat-accel 6x,
# jerk 16x). All comfort scalars below use MEASURED twist (no position derivatives).
SG_WINDOW = 11
SG_POLY = 3


def _extract_kinematics(bag_dir):
    """(t[s], x, y, speed[m/s], yaw_rate[rad/s]) from /localization/kinematic_state.

    speed = twist.linear.x, yaw_rate = twist.angular.z — both MEASURED in base_link
    (twist.linear.y ≈ 0, so lateral motion is NOT in the y component; lateral accel =
    yaw_rate*speed). No position derivative; dt comes from the timestamps.
    """
    import sqlite3
    import glob as _glob
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    dbs = sorted(_glob.glob(str(Path(bag_dir) / "*.db3")))
    if not dbs:
        raise SystemExit(f"no .db3 in {bag_dir}")
    con = sqlite3.connect(dbs[0]); cur = con.cursor()
    row = cur.execute("SELECT id FROM topics WHERE name='/localization/kinematic_state'").fetchone()
    if row is None:
        con.close()
        raise SystemExit(f"no /localization/kinematic_state in {bag_dir}")
    msgs = cur.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (row[0],)).fetchall()
    con.close()
    t = np.array([m[0] for m in msgs], dtype=np.float64) / 1e9
    M = [deserialize_message(m[1], Odometry) for m in msgs]
    x = np.array([m.pose.pose.position.x for m in M])
    y = np.array([m.pose.pose.position.y for m in M])
    speed = np.array([m.twist.twist.linear.x for m in M])
    yaw_rate = np.array([m.twist.twist.angular.z for m in M])
    return t, x, y, speed, yaw_rate


def _extract_accel_y(bag_dir):
    """(t[s], accel_y[m/s²]) measured lateral accel from /localization/acceleration.

    accel.y in base_link is the felt lateral acceleration. The first message is dropped
    (NaN/uninitialized). Raises if the topic is absent — no silent fallback to v·ω; the
    caller decides. (In psim accel.y ≡ v·ω, verified corr 1.000, so dropping back would be
    lossless there, but on a real bag the measured signal is the one we actually want.)
    """
    import sqlite3
    import glob as _glob
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import AccelWithCovarianceStamped
    dbs = sorted(_glob.glob(str(Path(bag_dir) / "*.db3")))
    if not dbs:
        raise SystemExit(f"no .db3 in {bag_dir}")
    con = sqlite3.connect(dbs[0]); cur = con.cursor()
    row = cur.execute("SELECT id FROM topics WHERE name='/localization/acceleration'").fetchone()
    if row is None:
        con.close()
        raise SystemExit(f"no /localization/acceleration in {bag_dir} — needed for the measured "
                         f"lateral-accel (accel.y) signal")
    msgs = cur.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (row[0],)).fetchall()
    con.close()
    msgs = msgs[1:]   # drop first message (NaN / uninitialized EKF state)
    t = np.array([m[0] for m in msgs], dtype=np.float64) / 1e9
    M = [deserialize_message(m[1], AccelWithCovarianceStamped) for m in msgs]
    accel_y = np.array([m.accel.accel.linear.y for m in M])
    return t, accel_y


def _sg_deriv1(arr_1d, dt: float) -> np.ndarray:
    """SG 1st derivative of a 1-D signal (replicate-padded)."""
    kernel = _build_sg_diff_kernel(window=SG_WINDOW, poly=SG_POLY, deriv=1, delta=dt)
    x = torch.tensor(np.asarray(arr_1d)[None, None], dtype=torch.float32)
    pad = SG_WINDOW // 2
    xp = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(xp, kernel.view(1, 1, -1)).squeeze().numpy()


def compute_dynamics(speed, yaw_rate, dt: float, lat_accel_meas=None,
                     max_jump_speed: float = 30.0) -> tuple[dict, np.ndarray]:
    """Per-sample MEASURED comfort scalars (inputs already subsampled).

    lat_accel (PRIMARY) = |lat_accel_meas| when given (measured |accel.y|), else the
    kinematic |yaw_rate*speed|. lat_accel_kin = |yaw_rate*speed| is always emitted as a
    SECONDARY sanity column. jerk = |d(lat_accel)/dt| (lateral jerk: ONE derivative of the
    primary accel signal, with dt from timestamps — never a 3rd-derivative of position).
    curvature = |yaw_rate|/max(|speed|,0.5). Edge (SG pad) + glitch (|speed|>max_jump_speed)
    samples are NaN'd.
    """
    speed = np.asarray(speed, np.float64); yaw_rate = np.asarray(yaw_rate, np.float64)
    T = len(speed); pad = SG_WINDOW // 2
    lat_accel_kin = np.abs(yaw_rate * speed)
    lat_accel = np.abs(np.asarray(lat_accel_meas, np.float64)) if lat_accel_meas is not None else lat_accel_kin
    curvature = np.abs(yaw_rate) / np.clip(np.abs(speed), 0.5, None)
    jerk = np.abs(_sg_deriv1(lat_accel, dt)) if T > SG_WINDOW else np.full(T, np.nan)
    glitch = np.abs(speed) > max_jump_speed
    if glitch.any():
        for j in np.where(glitch)[0]:
            glitch[max(0, j - pad):min(T, j + pad + 1)] = True
    out = {"lat_accel": lat_accel, "lat_accel_kin": lat_accel_kin, "jerk": np.asarray(jerk, np.float64),
           "yaw_rate": np.abs(yaw_rate), "curvature": curvature, "speed": np.abs(speed)}
    for k in out:
        out[k] = out[k].astype(np.float64).copy()
        out[k][glitch] = np.nan
        if T > 2 * pad:
            out[k][:pad] = np.nan
            out[k][-pad:] = np.nan
    return out, glitch


def _bin_stats(arc_s, vals, s_max, bin_m):
    """Per-arc-bin (mean, p95, max) of a per-sample scalar, ignoring NaN.

    Same bin index convention as _heatmap_common.bin_scalar_by_arc; we add p95/max
    (the WORST-case comfort, which the mean hides) — pure binning bookkeeping.
    """
    import math
    n_bins = max(1, int(math.ceil(s_max / bin_m)))
    mid = (np.arange(n_bins) + 0.5) * bin_m
    mean = np.full(n_bins, np.nan)
    p95 = np.full(n_bins, np.nan)
    mx = np.full(n_bins, np.nan)
    idx = np.clip((arc_s // bin_m).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        m = (idx == b) & np.isfinite(vals)
        if m.any():
            v = vals[m]
            mean[b] = float(np.mean(v))
            p95[b] = float(np.percentile(v, 95))
            mx[b] = float(np.max(v))
    return mid, mean, p95, mx


def analyze_bag(bag, route_obj, pts, arc, args):
    """Realized comfort scalars binned by arc for one bag (MEASURED kinematics)."""
    t, x, y, speed, yaw_rate = _extract_kinematics(bag)
    s = slice(None, None, args.stride)
    t, x, y, speed, yaw_rate = t[s], x[s], y[s], speed[s], yaw_rate[s]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1   # REAL dt from timestamps
    # PRIMARY lateral signal: measured |accel.y|, interpolated onto the (strided) pose
    # timestamps (the accel topic and kinematic_state are not perfectly co-rate).
    t_acc, accel_y = _extract_accel_y(bag)
    lat_meas = np.interp(t, t_acc, np.abs(accel_y))
    dyn, glitch = compute_dynamics(speed, yaw_rate, dt, lat_accel_meas=lat_meas,
                                   max_jump_speed=args.max_jump_speed)
    # sanity: accel.y ≡ v·ω in psim (EKF-derived, no IMU); on a real bag the gap is real.
    fin = np.isfinite(dyn["lat_accel"]) & np.isfinite(dyn["lat_accel_kin"])
    if fin.sum() > 2:
        d = np.abs(dyn["lat_accel"][fin] - dyn["lat_accel_kin"][fin])
        cc = float(np.corrcoef(dyn["lat_accel"][fin], dyn["lat_accel_kin"][fin])[0, 1])
        print(f"  [{Path(bag).name}] accel.y vs v·ω: corr={cc:.3f} max|Δ|={d.max():.3f} "
              f"(≈0 in psim; >0 on a real bag = slip/noise)")

    arc_max = float(arc.max())
    proj = np.array([project_to_polyline(np.array([float(px), float(py)]), pts, arc)
                     for px, py in zip(x, y)])
    pose_arc = proj[:, 0]
    signed_lat = proj[:, 1].copy()
    abs_lat = proj[:, 2].copy()
    # Drop glitch samples from cl too (consistent with comfort scalars).
    signed_lat[glitch] = np.nan
    abs_lat[glitch] = np.nan
    keep = (pose_arc >= args.front_cut) & (pose_arc <= arc_max - args.tail_cut)
    if keep.sum() == 0:
        raise SystemExit(f"{bag}: no in-bounds poses after front/tail cut")

    n_glitch = int(glitch.sum())
    print(f"  [{Path(bag).name}] {len(x)} samples (dt={dt:.3f}s), "
          f"{int(keep.sum())} in-bounds, {n_glitch} glitch dropped (>{args.max_jump_speed:.0f} m/s)")

    # Probe: dump per-sample series for an arc window.
    if args.probe_lo is not None:
        pm = keep & (pose_arc >= args.probe_lo) & (pose_arc < args.probe_hi)
        print(f"  PROBE {Path(bag).name} arc [{args.probe_lo:.0f},{args.probe_hi:.0f}) "
              f"n={int(pm.sum())}: arc | lat_signed | lat_acc | jerk | yaw_rate | speed")
        for i in np.where(pm)[0]:
            print(f"    {pose_arc[i]:7.1f} | {proj[i,1]:+6.2f} | {dyn['lat_accel'][i]:6.2f} | "
                  f"{dyn['jerk'][i]:8.2f} | {dyn['yaw_rate'][i]:5.2f} | {dyn['speed'][i]:5.1f}")

    arc_k = pose_arc[keep]
    res = {"n_steps": int(keep.sum()), "n_glitch": n_glitch, "arc_max": arc_max,
           "dt": dt, "bins": {}}   # dt = the REAL per-bag dt actually used (from timestamps)
    # comfort scalars (speed + curvature included so lat_accel = v^2 * kappa can be decomposed)
    for name in ("lat_accel", "lat_accel_kin", "jerk", "yaw_rate", "curvature", "speed"):
        mid, mean, p95, mx = _bin_stats(arc_k, dyn[name][keep], arc_max, args.bin_m)
        res["bins"][name] = {"mid": mid.tolist(), "mean": mean.tolist(),
                             "p95": p95.tolist(), "max": mx.tolist()}
    # centerline offset: mean + std (oscillation) + max
    import math
    n_bins = max(1, int(math.ceil(arc_max / args.bin_m)))
    idx = np.clip((arc_k // args.bin_m).astype(int), 0, n_bins - 1)
    cl_mean = np.full(n_bins, np.nan); cl_std = np.full(n_bins, np.nan); cl_max = np.full(n_bins, np.nan)
    n_samp = np.zeros(n_bins, dtype=int)
    al_k = abs_lat[keep]; sl_k = signed_lat[keep]
    for b in range(n_bins):
        m = (idx == b) & np.isfinite(al_k)
        n_samp[b] = int(m.sum())
        if m.any():
            cl_mean[b] = float(np.mean(al_k[m]))
            cl_std[b] = float(np.std(sl_k[m]))   # signed std = lateral oscillation amplitude
            cl_max[b] = float(np.max(al_k[m]))
    res["bins"]["cl"] = {"mid": ((np.arange(n_bins) + 0.5) * args.bin_m).tolist(),
                         "mean": cl_mean.tolist(), "std": cl_std.tolist(),
                         "max": cl_max.tolist(), "n": n_samp.tolist()}
    return res


def _fmt(v, nd=2):
    return "—" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v:.{nd}f}"


def print_compare_table(route_name, base, model, args):
    """Markdown per-arc table: baseline vs model, all comfort metrics."""
    mids = base["bins"]["lat_accel"]["mid"]
    print(f"\n## {route_name} — realized comfort (baseline vs {args.label}), {args.bin_m:.0f}m bins")
    print("Columns: latacc p95/max (m/s²) | jerk p95 (m/s³) | yawrate max (rad/s) | cl mean/std (m). std = lateral oscillation.")
    hdr = ("| arc | b_latacc_p95 | m_latacc_p95 | b_latacc_max | m_latacc_max "
           "| b_jerk_p95 | m_jerk_p95 | b_yaw_max | m_yaw_max | b_cl_std | m_cl_std | m_n |")
    print(hdr)
    print("|" + "---|" * 12)
    for i, mid in enumerate(mids):
        lo = int(mid - args.bin_m / 2)
        def g(d, k, s):
            return d["bins"][k][s][i] if i < len(d["bins"][k][s]) else None
        mn = g(model, "cl", "n")
        row = [f"{lo}-{lo+int(args.bin_m)}",
               _fmt(g(base, "lat_accel", "p95")), _fmt(g(model, "lat_accel", "p95")),
               _fmt(g(base, "lat_accel", "max")), _fmt(g(model, "lat_accel", "max")),
               _fmt(g(base, "jerk", "p95")), _fmt(g(model, "jerk", "p95")),
               _fmt(g(base, "yaw_rate", "max")), _fmt(g(model, "yaw_rate", "max")),
               _fmt(g(base, "cl", "std")), _fmt(g(model, "cl", "std")),
               str(int(mn)) if mn is not None else "—"]
        print("| " + " | ".join(row) + " |")


def print_speed_decomp(route_name, base, model, args):
    """Decompose lat_accel = v^2 * kappa per arc to test 'too fast on the curve' vs 'sharper path'.

    speed = mean realized speed (m/s); curv = p95 curvature (1/m); latacc = p95 (m/s^2).
    """
    mids = base["bins"]["lat_accel"]["mid"]
    print(f"\n## {route_name} — lat_accel decomposition (v vs kappa), baseline vs {args.label}")
    print("speed=mean realized (m/s) | curv=p95 (1/m) | latacc=p95 (m/s²). lat_accel ≈ v²·κ.")
    print("| arc | b_speed | m_speed | b_curv | m_curv | b_latacc | m_latacc | dominant Δ |")
    print("|" + "---|" * 8)
    for i, mid in enumerate(mids):
        lo = int(mid - args.bin_m / 2)
        def g(d, k, s):
            v = d["bins"][k][s]
            return v[i] if i < len(v) else None
        bs, ms = g(base, "speed", "mean"), g(model, "speed", "mean")
        bc, mc = g(base, "curvature", "p95"), g(model, "curvature", "p95")
        bl, ml = g(base, "lat_accel", "p95"), g(model, "lat_accel", "p95")
        tag = ""
        if None not in (bs, ms, bc, mc, bl, ml) and (bl > 0.3 or ml > 0.3):
            sp_r = (ms / bs) if (bs and bs > 0.5) else float("nan")
            cu_r = (mc / bc) if (bc and bc > 1e-3) else float("nan")
            if np.isfinite(sp_r) and np.isfinite(cu_r):
                tag = "SPEED" if sp_r > 1.15 and sp_r > cu_r else ("CURV" if cu_r > 1.15 and cu_r > sp_r else "~")
        print("| " + " | ".join([f"{lo}-{lo+int(args.bin_m)}",
              _fmt(bs), _fmt(ms), _fmt(bc, 3), _fmt(mc, 3), _fmt(bl), _fmt(ml), tag]) + " |")


def plot_compare(route_obj, pts, arc, base, model, metric, stat, out_png, args):
    """Two-panel heatmap (baseline | model) of metric/stat, shared color scale."""
    import math
    arc_max = float(arc.max())
    n_bins = max(1, int(math.ceil(arc_max / args.bin_m)))
    segs = segments_from_polyline(pts, arc, args.bin_m, n_bins)
    bvals = np.array(base["bins"][metric][stat])
    mvals = np.array(model["bins"][metric][stat])
    allv = np.concatenate([bvals[np.isfinite(bvals)], mvals[np.isfinite(mvals)]])
    if allv.size == 0:
        return
    vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_route_heatmap(axes[0], pts, segs, bvals, f"baseline {metric} {stat}", vmin, vmax, "inferno")
    plot_route_heatmap(axes[1], pts, segs, mvals, f"{args.label} {metric} {stat}", vmin, vmax, "inferno")
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, fraction=0.04)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", required=True)
    ap.add_argument("--bag", required=True, help="contender/model psim bag dir")
    ap.add_argument("--baseline_bag", default=None, help="baseline psim bag dir (for comparison)")
    ap.add_argument("--ego_shape", required=True, help="WB,L,W (only printed; dynamics need no OBB)")
    ap.add_argument("--label", default="model")
    ap.add_argument("--baseline_label", default="baseline")
    ap.add_argument("--stride", type=int, default=10,
                    help="subsample realized poses (loc ~100Hz; stride 10 -> ~10Hz planning rate)")
    ap.add_argument("--dt", type=float, default=0.1,
                    help="DEPRECATED / ignored — dt is now derived per-bag from the message timestamps.")
    ap.add_argument("--front_cut", type=float, default=50.0)
    ap.add_argument("--tail_cut", type=float, default=50.0)
    ap.add_argument("--bin_m", type=float, default=100.0)
    ap.add_argument("--max_jump_speed", type=float, default=30.0,
                    help="reject poses implying a consecutive jump faster than this (m/s) as a "
                         "localization glitch; count is reported, not silently dropped")
    ap.add_argument("--probe_lo", type=float, default=None,
                    help="if set, dump per-sample series for arc window [probe_lo, probe_hi)")
    ap.add_argument("--probe_hi", type=float, default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    if (args.probe_lo is None) != (args.probe_hi is None):
        ap.error("--probe_lo and --probe_hi must be given together")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[comfort] route={Path(args.route).name} stride={args.stride} (dt derived from timestamps) "
          f"PRIMARY lat_accel=|accel.y| (measured) + v·ω sanity col, jerk=d(lat_accel)/dt; "
          f"bin={args.bin_m:.0f}m ego_shape={args.ego_shape}")

    route_obj = load_route(Path(args.route))
    pts, arc = build_route_polyline(route_obj)

    model = analyze_bag(args.bag, route_obj, pts, arc, args)
    # "dt" reflects the REAL per-bag dt derived from timestamps inside analyze_bag — NOT the
    # deprecated --dt flag (which is ignored). Record the model bag's dt at the top level.
    result = {"route": str(args.route), "stride": args.stride, "dt": model["dt"],
              "bin_m": args.bin_m, "model": {"label": args.label, "bag": args.bag, **model}}

    if args.baseline_bag:
        base = analyze_bag(args.baseline_bag, route_obj, pts, arc, args)
        result["baseline"] = {"label": args.baseline_label, "bag": args.baseline_bag, **base}
        print_compare_table(Path(args.route).stem, base, model, args)
        print_speed_decomp(Path(args.route).stem, base, model, args)
        for metric, stat in (("lat_accel", "p95"), ("jerk", "p95"), ("cl", "std")):
            plot_compare(route_obj, pts, arc, base, model, metric, stat,
                         out / f"comfort_{Path(args.route).stem}_{metric}_{stat}.png", args)

    json_path = out / f"comfort_{Path(args.route).stem}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"[comfort] wrote {json_path}")


if __name__ == "__main__":
    main()
