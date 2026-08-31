"""Frenet corridor augmentation (self-contained port of the frenet_aug prototype).

Pure-tensor Frenet corridor augmentation — quintic-speed, batched at B.

Everything runs as batched tensor ops on the training batch's device, one call
per batch exactly like the quintic StatePerturbation:

  1. corridor: road-border ray-cast along path normals = closed-form 2x2 solve,
     broadcast over (B, T, segments); neighbor cuts use TIME-ALIGNED recorded
     tracks (past for t<=0, GT future for t>0), broadcast over (B, N, T).
  2. draws: K joint (dy, dth, combo) tuples per scene (K=16 - we need ONE valid
     augmentation per scene, not a fan).
  3. feasibility: shared quintic basis GEMMs (B, K, T) per knob combo; comfort
     jerk (future), plausibility jerk (history), lat accel, Ackermann steering,
     corridor with per-candidate rotation margin.
  4. winner: first feasible draw per scene; its merge horizon is sampled among
     the feasible ones with P(M) ~ exp(-(M - Mmin) / 1 s) (biased to fast
     convergence, slower merges keep a chance; lowest peak future jerk breaks
     ties within the sampled M); rebuild = one gathered basis GEMM;
     headings/current-state re-derived from the augmented polyline. The ego
     history is rewritten to the perturbed polyline so past, current state and
     future stay kinematically consistent.

The corridor is a fast approximation; the WINNING candidate is re-verified with
the canonical exact signed-OBB clearance (planner_metrics) and vetoed back to
plain GT on a true neighbor overlap (~1% of winners). Scenes with no feasible
draw train on plain GT.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from diffusion_planner.utils.data_augmentation import StatePerturbation
from planner_metrics.geometry import build_road_border_segments
from planner_metrics.subscores import compute_ego_neighbor_signed_clearance

DT = 0.1
CORRIDOR_MARGIN = 0.10

# Feasibility limits: candidates violating these are REJECTED (never clipped).
LIMITS = {
    "lat_acc": 3.0,  # m/s^2
    "yaw_rate": 0.6,  # rad/s  (~34 deg/s)
    # Comfort jerk, applied to the FUTURE segment only - that is what the model
    # learns to output. (Peak lateral jerk of a quintic ~ 60*dy/T^3: a comfortable
    # recovery from dy=0.75 m needs T >= 2.8 s, so at M=2 s most realistic offsets
    # are correctly REJECTED. The production 2 s quintic bridge teaches ~6 m/s^3.)
    "jerk": 2.0,  # m/s^3
    # The HISTORY segment is context, not a target: it only has to be a plausible
    # (possibly uncomfortable) way to have arrived at the perturbed pose. The 3 s
    # past window would otherwise cap |dy| at 27*jerk/60 ~ 0.9 m.
    "jerk_history": 5.0,  # m/s^3
    # Ackermann / kinematic-bicycle feasibility (needs wheelbase from ego_shape):
    # steering angle delta = atan(WB * kappa), kappa = yaw_rate / speed.
    "steer": 0.61,  # rad (~35 deg), typical passenger-car lock
}

# the steering test runs in tan space (see _feasibility), so cache tan(limit)
_TAN_STEER = math.tan(LIMITS["steer"])


def _ddt(x: torch.Tensor, dim: int) -> torch.Tensor:
    """d/dt along `dim` on the fixed DT grid.

    Same arithmetic as torch.gradient(x, spacing=DT, dim=dim) — central
    differences inside, one-sided at the ends — written out because the library
    call costs ~7x more on the candidate tensors (3.5 ms vs 0.5 ms for the three
    derivatives at B=512). Verified bit-identical to torch.gradient.
    """
    lo = x.narrow(dim, 0, 1)
    lo2 = x.narrow(dim, 1, 1)
    hi2 = x.narrow(dim, x.shape[dim] - 2, 1)
    hi = x.narrow(dim, x.shape[dim] - 1, 1)
    mid = (x.narrow(dim, 2, x.shape[dim] - 2) - x.narrow(dim, 0, x.shape[dim] - 2)) / (2 * DT)
    return torch.cat([(lo2 - lo) / DT, mid, (hi - hi2) / DT], dim=dim)


def quintic_basis(t0: float, t1: float, t: np.ndarray):
    """Rows: response of l, l', l'', l''' to unit (pos, vel, acc) BCs at t1.

    BCs at t0 are zero (anchor / merge on the GT path), so the profile is linear
    in the three t1-side BCs alone. Returns 4 arrays shaped (3, len(t)).
    """
    M = []
    for tt in (t0, t1):
        M += [
            [1, tt, tt**2, tt**3, tt**4, tt**5],
            [0, 1, 2 * tt, 3 * tt**2, 4 * tt**3, 5 * tt**4],
            [0, 0, 2, 6 * tt, 12 * tt**2, 20 * tt**3],
        ]
    Minv = np.linalg.inv(np.array(M, float))  # coeffs = Minv @ (BCs at t0, BCs at t1)
    C = Minv[:, 3:]  # (6, 3) response to the t1-side BCs
    P = np.stack([np.ones_like(t), t, t**2, t**3, t**4, t**5])
    D1 = np.stack([np.zeros_like(t), np.ones_like(t), 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
    D2 = np.stack([np.zeros_like(t)] * 2 + [2 * np.ones_like(t), 6 * t, 12 * t**2, 20 * t**3])
    D3 = np.stack([np.zeros_like(t)] * 3 + [6 * np.ones_like(t), 24 * t, 60 * t**2])
    return (C.T @ P, C.T @ D1, C.T @ D2, C.T @ D3)


@dataclass(frozen=True)
class KnobGrid:
    merge_times: tuple = (2.0, 3.0, 4.0, 5.0)
    anchors: tuple = (2.0, 3.0)
    acc0_fracs: tuple = (0.0, -0.5, 0.5, -1.0, 1.0)


class FrenetStatePerturbationTensor(StatePerturbation):
    """Drop-in for StatePerturbation; fully batched, no per-sample python."""

    def centric_transform(self, inputs, ego_future, neighbors_future):
        out = super().centric_transform(inputs, ego_future, neighbors_future)
        # Dataset convention is base_link: velocity and acceleration are purely
        # longitudinal (raw NPZs carry vy = ay = 0.0 identically). The rebuilt
        # polyline's finite-diff velocity leaves a small tangent/heading residual
        # in vy (<= 0.08 m/s) after re-centering — project it out so augmented
        # states stay exactly on the data manifold. No-op for non-augmented rows.
        rows = self._aug_rows
        if rows is not None and bool(rows.any()):
            cur = inputs["ego_current_state"]
            vx, vy = cur[rows, 4], cur[rows, 5]
            cur[rows, 4] = torch.sqrt(vx * vx + vy * vy) * torch.where(vx < 0, -1.0, 1.0)
            cur[rows, 5] = 0.0
            cur[rows, 7] = 0.0  # ay
        return out

    def __init__(
        self,
        augment_prob: float,
        device,
        n_draws: int = 16,
        dy_max: float = 2.0,
        dth_max: float = 0.17,
        knobs: KnobGrid = KnobGrid(),
        seed: int = 0,
        ranked_temp_s: float = 1.0,
    ):
        super().__init__(
            augment_prob=augment_prob,
            num_refine=20,
            device=device,
            ego_past_noise_std=0.0,  # frenet rewrites the past kinematically
            use_smoothing_future_trajectory=False,
        )
        # The ego past is rewritten to the perturbed polyline, so it must also be
        # re-centered with the future in centric_transform (quintic/bridge keep the
        # GT past and skip that transform).
        self._transform_ego_past = True
        # rows augmented in the most recent __call__; centric_transform re-projects
        # velocity/accel to base_link (vy = ay = 0) for exactly these rows
        self._aug_rows = None
        self._nbr_near = None
        self.n_draws = n_draws
        self.dy_max = dy_max
        self.dth_max = dth_max
        self.knobs = knobs
        # offset the stream per DDP rank so ranks draw independent perturbations
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        self.gen = torch.Generator(device="cpu").manual_seed(seed + rank)
        self.ranked_temp_s = float(ranked_temp_s)
        self._basis_cache = {}

    # ---------- shared basis (depends only on the time grid + knobs) ----------
    def _bases(self, P, F, device, dtype):
        key = (P, F, device, dtype)
        if key in self._basis_cache:
            return self._basis_cache[key]
        t = np.concatenate([np.arange(-(P - 1), 1) * DT, np.arange(1, F + 1) * DT])
        combos = []
        for mt in self.knobs.merge_times:
            if int(mt / DT) > F:
                continue
            for anch in self.knobs.anchors:
                anch = min(anch, (P - 1) * DT)
                mh = (t >= -anch) & (t <= 0)
                mf = (t > 0) & (t <= mt)
                Bh = quintic_basis(-anch, 0.0, t[mh])
                Bf = quintic_basis(mt, 0.0, t[mf])
                T = len(t)
                Bmat = np.zeros((4, 3, T))
                for d in range(4):
                    Bmat[d][:, mh] = Bh[d]
                    Bmat[d][:, mf] = Bf[d]
                for frac in self.knobs.acc0_fracs:
                    combos.append(
                        {
                            "merge": mt,
                            "anchor": anch,
                            "frac": frac,
                            "B": torch.tensor(Bmat, dtype=dtype, device=device),
                            "m": torch.tensor(mh | mf, device=device),
                            "mh": torch.tensor(mh, device=device),
                            "mf": torch.tensor(mf, device=device),
                        }
                    )
        out = (torch.tensor(t, dtype=dtype, device=device), combos)
        self._basis_cache[key] = out
        return out

    # ---------- corridor, fully batched ----------
    def _corridor(self, inputs, xy, tan, nrm, half_w, half_l, wb):
        # reset per batch so a neighbor-less batch never sees a stale tensor
        self._nbr_st = self._nbr_valid = self._nbr_near = None
        B, T, _ = xy.shape
        dev, dtype = xy.device, xy.dtype
        # Unconstrained until a border or neighbor cuts them. NOTE: do not try to
        # cap these by dy_max — dy is the offset at t=0, not a bound on the
        # quintic, which overshoots it when the draw has an initial heading
        # slope (measured |L| = 3.39 m for dy = 1.98 m). A cap like that silently
        # rejects feasible perturbations.
        lo = torch.full((B, T), -20.0, device=dev, dtype=dtype)
        hi = torch.full((B, T), 20.0, device=dev, dtype=dtype)

        ls = inputs.get("line_strings")
        if ls is not None and ls.shape[-1] < 4:
            raise ValueError(
                f"line_strings has {ls.shape[-1]} channels; frenet augmentation needs "
                "channel 3 (road-border flag) to constrain the corridor"
            )
        if ls is not None:
            # lo/hi start at +-20 m, so a border further than that from every
            # path point can never tighten them; the shared builder drops those
            # before the ray math (~180 of 1140 slots survive on the pipeline
            # set) and pads to the batch maximum, so the bounds are unchanged.
            a, e, sv = build_road_border_segments(ls, xy, reach=25.0)
            d = e - a
            nx, ny = nrm[..., 0:1], nrm[..., 1:2]  # (B, T, 1)
            dx = d[:, None, :, 0]  # (B, 1, S)
            dy_ = d[:, None, :, 1]
            det = nx * (-dy_) - ny * (-dx)  # (B, T, S)
            rx = a[:, None, :, 0] - xy[..., 0:1]
            ry = a[:, None, :, 1] - xy[..., 1:2]
            s = (rx * (-dy_) - ry * (-dx)) / det
            u = (nx * ry - ny * rx) / det
            ok = torch.isfinite(s) & (u >= 0) & (u <= 1) & sv[:, None, :]
            pos = torch.where(ok & (s > 0), s, torch.full_like(s, torch.inf))
            neg = torch.where(ok & (s < 0), s, torch.full_like(s, -torch.inf))
            hi = torch.minimum(hi, pos.amin(-1) - half_w[:, None])
            lo = torch.maximum(lo, neg.amax(-1) + half_w[:, None])

        past = inputs.get("neighbor_agents_past")
        fut = inputs.get("neighbor_agents_future")
        if past is not None and fut is not None:
            # time-aligned neighbor states on the SAME grid as the ego polyline.
            # Futures come in two layouts: 4-col (x, y, cos, sin) or the canonical
            # 3-col (x, y, heading) — train_epoch only converts to cos/sin AFTER
            # augmentation, so handle both here.
            if fut.shape[-1] >= 4:
                fut4 = fut[..., :4]
            else:
                fut4 = torch.cat([fut[..., :2], fut[..., 2:3].cos(), fut[..., 2:3].sin()], dim=-1)
                # keep padded rows padded: an all-zero 3-col row must not become
                # (0, 0, 1, 0) after cos/sin
                fut4 = fut4 * (fut.abs().sum(-1, keepdim=True) > 0)
            st = torch.cat([past[..., :4], fut4], dim=2)  # (B, N, T, 4)
            valid = st.abs().sum(-1) > 0  # (B, N, T)
            self._nbr_st, self._nbr_valid = st, valid  # reused by the exact-OBB veto
            # Only slots holding a real track can cut the corridor, and they are
            # ~31 of the 320 padded slots. Flatten to the valid (scene, neighbor)
            # pairs so the projections below run over those alone, then reduce
            # back per scene; padding contributed +-inf, i.e. nothing.
            vb, vn = torch.nonzero(valid.any(-1), as_tuple=True)  # (M,)
            if vb.numel() == 0:
                self._nbr_near = valid
                return lo, hi
            valid_m = valid[vb, vn]  # (M, T)
            l_n = past[vb, vn, -1, 7][:, None]  # (M, 1)
            w_n = past[vb, vn, -1, 6][:, None]
            c = st[vb, vn, :, :2]  # (M, T, 2)
            axi = st[vb, vn, :, 2:4]
            per = torch.stack([-axi[..., 1], axi[..., 0]], dim=-1)
            tan_m, nrm_m = tan[vb], nrm[vb]  # (M, T, 2)
            half_l_m, half_w_m, wb_m = half_l[vb, None], half_w[vb, None], wb[vb, None]
            rel = c - xy[vb]  # (M, T, 2)
            # ego xy is base_link (rear axle); the footprint CENTER sits wb/2
            # ahead along the tangent, so shift the longitudinal window there —
            # otherwise a transient neighbor overlapping only the front bumper
            # imposes no lateral cut.
            lon = (rel * tan_m).sum(-1)  # rear-axle window (validated semantics)
            lat = (rel * nrm_m).sum(-1)
            ext_lon = ((tan_m * axi).sum(-1).abs() * l_n + (tan_m * per).sum(-1).abs() * w_n) / 2
            ext_lat = ((nrm_m * axi).sum(-1).abs() * l_n + (nrm_m * per).sum(-1).abs() * w_n) / 2
            near = valid_m & (lon.abs() <= half_l_m + ext_lon)
            # Same longitudinal test, padded, kept for the exact-OBB veto so it
            # only pairs neighbors that can actually reach the ego box. The
            # augmentation is a lateral offset, so a candidate's longitudinal
            # coordinate in this frame is the GT one; the pad covers the ego's
            # longitudinal half-extent growing under the heading change
            # (half_l*cos + half_w*sin <= half_l + half_w) plus the wb/2 centre
            # shift. Pairs outside it cannot overlap, so the verdicts are
            # unchanged — the canonical OBB check still decides every one.
            near_pad = valid_m & (lon.abs() <= half_l_m + ext_lon + half_w_m + wb_m / 2)
            self._nbr_near = torch.zeros_like(valid)
            self._nbr_near[vb, vn] = near_pad
            cut_hi = torch.where(
                near & (lat > 0), lat - ext_lat - half_w_m, torch.full_like(lat, torch.inf)
            )
            cut_lo = torch.where(
                near & (lat < 0), lat + ext_lat + half_w_m, torch.full_like(lat, -torch.inf)
            )
            idx = vb[:, None].expand_as(cut_hi)
            hi = hi.scatter_reduce(0, idx, cut_hi, reduce="amin")
            lo = lo.scatter_reduce(0, idx, cut_lo, reduce="amax")
        return lo, hi

    def _veto_true_overlaps(self, inputs, upd, aug_xy, heading):
        """Exact-OBB veto on the winning candidates.

        The corridor is a fast approximation (path-normal ray-cast with
        half-width margins around the rear-axle polyline) and is measured to
        accept ~1.4% of winners whose TRUE footprint overlaps a recorded
        neighbor box. Re-check each winner with the canonical signed OBB
        clearance (rear-axle ego convention, centroid neighbors) and train
        vetoed rows on plain GT. Borders are left to the corridor: the recorded
        drives themselves touch the mapped borders (GT corner distance reaches
        0.0), so an OBB border veto would reject GT-like data.
        """
        if self._nbr_st is None or not bool(upd.any()):
            return upd
        hd_cs = torch.stack([heading.cos(), heading.sin()], dim=-1)
        tr_aug = torch.cat([aug_xy, hd_cs], dim=-1)  # (B, T, 4)
        past_n = inputs["neighbor_agents_past"]
        # One paired call for the whole batch. Per-scene calls were one tiny
        # kernel each, so their launch overhead dominated training (228 ms/batch
        # at B=512, several times the rest of the augmenter) to reject the ~1.4%
        # of winners that truly overlap.
        pairs = upd[:, None] & self._nbr_near.any(-1)  # (B, N) scene x neighbor
        bi, ni = torch.nonzero(pairs, as_tuple=True)
        if bi.numel() == 0:
            return upd
        clr = compute_ego_neighbor_signed_clearance(
            tr_aug[bi],
            inputs["ego_shape"][bi],
            self._nbr_st[bi, ni],
            past_n[bi, ni, -1][:, [6, 7]],  # width, length
            self._nbr_valid[bi, ni],
            paired=True,
            overlap_only=True,  # the veto only asks whether they overlap
        )  # (M, T)
        overlapping = bi[clr.amin(-1) < 0.0]
        upd[overlapping] = False
        return upd

    # ---------- the augmentation ----------
    @torch.no_grad()
    def __call__(self, inputs, ego_future, neighbors_future):
        # ego_future may arrive as (x, y, heading) or (x, y, cos, sin) — same
        # branch heading_to_cos_sin uses upstream. Canonicalize to 3-col heading
        # here; centric_transform returns the rebuilt 3-col future either way,
        # and train_epoch re-converts to cos/sin after augmentation.
        if ego_future.shape[-1] >= 4:
            ego_future = torch.cat(
                [
                    ego_future[..., :2],
                    torch.atan2(ego_future[..., 3], ego_future[..., 2])[..., None],
                ],
                dim=-1,
            )
        past4 = inputs["ego_agent_past"]  # (B, P, 4) x, y, cos, sin
        B, P, _ = past4.shape
        F = ego_future.shape[1]
        dev, dtype = past4.device, past4.dtype
        t, combos = self._bases(P, F, dev, dtype)
        T = P + F

        fut_cs = torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1)
        xy = torch.cat([past4[..., :2], ego_future[..., :2]], dim=1)  # (B, T, 2)
        tan = torch.cat([past4[..., 2:4], fut_cs], dim=1)
        nrm = torch.stack([-tan[..., 1], tan[..., 0]], dim=-1)
        speed = _ddt(xy, 1).norm(dim=-1).clamp(min=0.5)  # (B, T)
        v0 = speed[:, P - 1]
        wb = inputs["ego_shape"][:, 0]
        half_w = inputs["ego_shape"][:, 2] / 2 + CORRIDOR_MARGIN
        half_l = inputs["ego_shape"][:, 1] / 2

        lo, hi = self._corridor(inputs, xy, tan, nrm, half_w, half_l, wb)

        K, C = self.n_draws, len(combos)
        r = torch.rand((B, 2 * K + 1), generator=self.gen).to(dev)
        do_aug = r[:, -1] < self._augment_prob
        # Low-speed / reverse gate (same threshold as the quintic augmenter): a
        # near-stationary ego makes every path-relative feasibility metric
        # degenerate (a pure sideways translation has zero lateral accel/jerk/yaw
        # rate), and a reversing ego would get motion-direction headings flipped
        # 180 deg by the polyline rewrite. Both train on plain GT instead.
        do_aug = do_aug & (inputs["ego_current_state"][:, 4] >= 2.0)
        dy = (r[:, :K] * 2 - 1) * self.dy_max
        dth = (r[:, K : 2 * K] * 2 - 1) * self.dth_max

        L, in_corr, merges = self._candidate_profiles(
            combos, dy, dth, v0, speed, half_l, lo, hi, dev, dtype
        )

        feasible, jerk_fut_peak = self._feasibility(xy, nrm, L, in_corr, wb, P)

        feasible &= do_aug[:, None, None]
        draw_ok = feasible.any(-1)  # (B, K): the drawn perturbation has >=1 valid merge
        has = draw_ok.any(-1)  # (B,)
        first = draw_ok.float().argmax(-1)  # first feasible PERTURBATION per scene

        if not bool(has.any()):
            # nothing feasible this batch: every row trains on plain GT
            self._aug_rows = has
            return self.centric_transform(inputs, ego_future, neighbors_future)

        aug_xy = self._select_candidate(
            feasible, jerk_fut_peak, first, merges, has, L, xy, nrm, B, dev, dtype
        )

        self._write_back(inputs, ego_future, aug_xy, xy, tan, wb, has, P)
        return self.centric_transform(inputs, ego_future, neighbors_future)

    def _candidate_profiles(self, combos, dy, dth, v0, speed, half_l, lo, hi, dev, dtype):
        """Lateral-offset profile of every (draw, knob-combo) pair + corridor test."""
        slope = v0[:, None] * torch.tan(dth)
        # NO combo loop: stack every combo's basis/mask once, then evaluate all
        # (draw, combo) pairs in one einsum per derivative. (C=40 tiny GEMMs in
        # a python loop cost ~240 kernel launches of pure overhead - the stacked
        # form is a single batched contraction.)
        Ball = torch.stack([cb["B"] for cb in combos])  # (C, 4, 3, T)
        m_all = torch.stack([cb["m"] for cb in combos])  # (C, T)
        merges = torch.tensor([cb["merge"] for cb in combos], device=dev, dtype=dtype)
        fracs = torch.tensor([cb["frac"] for cb in combos], device=dev, dtype=dtype)

        # Evaluate EVERY knob combo for every drawn (dy, dth): the merge choice
        # must be made FOR a perturbation, not across perturbations — selecting
        # min-merge over joint draws would systematically prefer small offsets
        # and shrink the dose.
        acc0 = fracs[None, None, :] * 5.77 * dy[..., None] / merges[None, None, :] ** 2
        bc = torch.stack(
            [dy[..., None].expand_as(acc0), slope[..., None].expand_as(acc0), acc0], dim=-1
        )  # (B, K, C, 3)
        prof = torch.einsum("bkci,cdit->bkcdt", bc, Ball)  # (B, K, C, 4, T)
        L, V, A, J = prof.unbind(dim=3)
        m = m_all[None, None]  # (1, 1, C, T)
        rot = half_l[:, None, None, None] * (V.abs() / speed[:, None, None]).clamp(max=0.35)
        in_corr = ((L - rot >= lo[:, None, None]) & (L + rot <= hi[:, None, None]) | ~m).all(
            -1
        )  # (B, K, C)
        return L, in_corr, merges

    def _feasibility(self, xy, nrm, L, in_corr, wb, P):
        """Physical-limit screen on the exact candidate polylines."""

        # Feasibility on the EXACT candidate polylines, not the lateral-delta
        # profile: build every candidate's xy and finite-difference it at DT.
        # The L(t)-only check misses the coupling between the lateral swing and
        # the GT path's own curvature (audited: ~1/3 of its accepts exceeded the
        # comfort jerk on the exact polyline, worst 2.6 vs 2.0 m/s^3). Limits are
        # GT-relative: where the recorded drive itself exceeds a limit (data
        # glitches), the candidate only has to stay within the GT's own maximum.
        def _exact_metrics(poly_xy):
            v = _ddt(poly_xy, -2)
            # speed is clamped away from zero, so one reciprocal serves the three
            # divisions below (a multiply is cheaper than a divide on 22M values)
            inv_sp = v.norm(dim=-1).clamp(min=0.5).reciprocal()
            a = _ddt(v, -2)
            lat_a = (v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]) * inv_sp
            jk = _ddt(a, -2)
            lat_j = (v[..., 0] * jk[..., 1] - v[..., 1] * jk[..., 0]) * inv_sp
            yaw_rate = lat_a * inv_sp
            return lat_a, lat_j, yaw_rate, inv_sp

        # Only candidates that already clear the corridor can end up feasible, so
        # build the exact polylines for those alone — the rest would be discarded
        # by the `in_corr &` below anyway, and they are ~40% of the grid.
        BIG = torch.tensor(1e9, device=L.device, dtype=L.dtype)
        feasible = torch.zeros_like(in_corr)
        jerk_fut_peak = torch.full(in_corr.shape, BIG.item(), device=L.device, dtype=L.dtype)
        b_i, k_i, c_i = torch.nonzero(in_corr, as_tuple=True)
        if b_i.numel() == 0:
            return feasible, jerk_fut_peak

        cand_xy = xy[b_i] + L[b_i, k_i, c_i][..., None] * nrm[b_i]  # (M, T, 2)
        laC, ljC, yrC, ivC = _exact_metrics(cand_xy)  # each (M, T)
        laG, ljG, yrG, ivG = _exact_metrics(xy)  # GT reference, (B, T)
        # Steering is compared in tan space: the limit test is
        # atan(x) <= max(atan(g), STEER), and atan is monotonic, so it is
        # equivalent to x <= max(g, tan(STEER)) — same verdict, no atan over the
        # candidate tensor.
        stC = (wb[b_i, None] * yrC * ivC).abs()
        stG = (wb[:, None] * yrG * ivG).abs()

        def _allow(gt_max, limit):
            return torch.clamp(gt_max, min=limit)[b_i]  # (M,)

        peak = ljC[:, P:].abs().amax(-1)  # (M,) — also the tiebreak
        a_ok = laC.abs().amax(-1) <= _allow(laG.abs().amax(-1), LIMITS["lat_acc"])
        j_ok = (peak <= _allow(ljG[:, P:].abs().amax(-1), LIMITS["jerk"])) & (
            ljC[:, :P].abs().amax(-1) <= _allow(ljG[:, :P].abs().amax(-1), LIMITS["jerk_history"])
        )
        y_ok = yrC.abs().amax(-1) <= _allow(yrG.abs().amax(-1), LIMITS["yaw_rate"])
        s_ok = stC.amax(-1) <= _allow(stG.amax(-1), _TAN_STEER)
        feasible[b_i, k_i, c_i] = a_ok & j_ok & y_ok & s_ok
        jerk_fut_peak[b_i, k_i, c_i] = peak
        return feasible, jerk_fut_peak

    def _select_candidate(
        self, feasible, jerk_fut_peak, first, merges, has, L, xy, nrm, B, dev, dtype
    ):
        """Sample a merge horizon per scene, take its lowest-jerk combo, build the polyline."""
        bi = torch.arange(B, device=dev)
        feas_k = feasible[bi, first]  # (B, C) — valid combos for the chosen draw
        jerk_k = jerk_fut_peak[bi, first]  # (B, C)
        BIG = torch.tensor(1e9, device=dev, dtype=dtype)
        # merge time sampled with per-combo weight exp(-(M-Mmin)/temp), i.e.
        # P(M) ∝ n_feasible_combos(M) * exp(-(M-Mmin)/temp) — biased to fast
        # convergence, slower merges keep a chance; within the sampled M,
        # lowest peak future jerk wins.
        m_feas = torch.where(feas_k, merges[None, :], BIG)
        m_min = m_feas.amin(-1, keepdim=True)
        w = torch.exp(-(merges[None, :] - m_min) / self.ranked_temp_s) * feas_k
        w = torch.where(has[:, None], w, torch.ones_like(w))  # avoid all-zero rows
        pick_m = torch.multinomial(w.cpu().double(), 1, generator=self.gen).to(dev)[:, 0]
        same_m = feas_k & (merges[None, :] == merges[pick_m][:, None])
        combo_idx = torch.where(same_m, jerk_k, BIG).argmin(-1)
        Lw = L[bi, first, combo_idx]  # (B, T)
        aug_xy = xy + Lw[..., None] * nrm
        return aug_xy

    def _write_back(self, inputs, ego_future, aug_xy, xy, tan, wb, has, P):
        """Veto true overlaps, then write history / future / current state in place."""
        g = _ddt(aug_xy, 1)
        gs = g.norm(dim=-1)
        hd_gt = torch.atan2(tan[..., 1], tan[..., 0])
        heading = torch.where(gs > 0.3, torch.atan2(g[..., 1], g[..., 0]), hd_gt)

        upd = self._veto_true_overlaps(inputs, has.clone(), aug_xy, heading)
        # future (x, y, heading) — canonical 3-col layout
        new_fut = torch.cat([aug_xy[:, P:], heading[:, P:, None]], dim=-1)
        ego_future[upd, :, :3] = new_fut[upd]
        # past (x, y, cos, sin) — rewritten so history, current state and
        # future all lie on the same perturbed polyline
        new_past = torch.cat(
            [aug_xy[:, :P], heading[:, :P, None].cos(), heading[:, :P, None].sin()],
            dim=-1,
        )
        inputs["ego_agent_past"][upd] = new_past[upd].to(inputs["ego_agent_past"].dtype)
        # current state at t=0, kinematically consistent with the new polyline
        i0 = P - 1
        cur = inputs["ego_current_state"]
        vel = g[:, i0]
        acc = _ddt(g, 1)[:, i0]
        yaw_rate = _ddt(torch.stack([heading.cos(), heading.sin()], -1), 1)[:, i0]
        yaw_rate = heading.cos()[:, i0] * yaw_rate[..., 1] - heading.sin()[:, i0] * yaw_rate[..., 0]
        steer = torch.atan(wb * yaw_rate / gs[:, i0].clamp(min=0.5))
        new_cur = torch.stack(
            [
                aug_xy[:, i0, 0],
                aug_xy[:, i0, 1],
                heading[:, i0].cos(),
                heading[:, i0].sin(),
                vel[..., 0],
                vel[..., 1],
                acc[..., 0],
                acc[..., 1],
                steer,
                yaw_rate,
            ],
            dim=-1,
        )
        n_cols = min(new_cur.shape[-1], cur.shape[-1])
        cur[upd, :n_cols] = new_cur[upd, :n_cols].to(cur.dtype)
        self._aug_rows = upd
