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

from dataclasses import dataclass

import numpy as np
import torch

from diffusion_planner.utils.augmentation_checks import (
    DT,
    border_lateral_bounds,
    ddt,
    kinematic_feasibility,
    neighbor_lateral_bounds,
    time_aligned_neighbor_tracks,
    unconstrained_bounds,
    veto_overlapping,
)
from diffusion_planner.utils.data_augmentation import StatePerturbation

# extra half-width the corridor keeps from borders and neighbors
CORRIDOR_MARGIN = 0.10


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
        """Lateral free space per timestep: where can this ego go sideways."""
        # reset per batch so a neighbor-less batch never sees a stale tensor
        self._nbr_st = self._nbr_valid = self._nbr_near = None
        B, T, _ = xy.shape
        lo, hi = unconstrained_bounds(B, T, xy.device, xy.dtype)

        ls = inputs.get("line_strings")
        if ls is not None:
            lo, hi = border_lateral_bounds(ls, xy, nrm, half_w, lo, hi)

        past = inputs.get("neighbor_agents_past")
        fut = inputs.get("neighbor_agents_future")
        if past is not None and fut is not None:
            st, valid = time_aligned_neighbor_tracks(past, fut)
            self._nbr_st, self._nbr_valid = st, valid  # reused by the exact-OBB veto
            shapes_wl = past[:, :, -1][..., [6, 7]]  # width, length
            lo, hi, self._nbr_near = neighbor_lateral_bounds(
                st, valid, shapes_wl, xy, tan, nrm, half_l, half_w, wb, lo, hi
            )
        return lo, hi

    def _veto_true_overlaps(self, inputs, upd, aug_xy, heading):
        """Drop winners whose footprint truly overlaps a recorded neighbor."""
        if self._nbr_st is None:
            return upd
        hd_cs = torch.stack([heading.cos(), heading.sin()], dim=-1)
        return veto_overlapping(
            upd,
            torch.cat([aug_xy, hd_cs], dim=-1),  # (B, T, 4)
            inputs["ego_shape"],
            self._nbr_st,
            self._nbr_valid,
            inputs["neighbor_agents_past"][:, :, -1][..., [6, 7]],
            self._nbr_near,
        )

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
        speed = ddt(xy, 1).norm(dim=-1).clamp(min=0.5)  # (B, T)
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
        """Screen corridor-passing candidates on the exact rebuilt polylines."""
        # Only candidates that already clear the corridor can end up feasible, so
        # build the exact polylines for those alone — the rest would be discarded
        # by the `in_corr &` anyway, and they are ~40% of the grid.
        BIG = 1e9
        feasible = torch.zeros_like(in_corr)
        jerk_fut_peak = torch.full(in_corr.shape, BIG, device=L.device, dtype=L.dtype)
        b_i, k_i, c_i = torch.nonzero(in_corr, as_tuple=True)
        if b_i.numel() == 0:
            return feasible, jerk_fut_peak

        cand_xy = xy[b_i] + L[b_i, k_i, c_i][..., None] * nrm[b_i]  # (M, T, 2)
        ok, peak = kinematic_feasibility(cand_xy, b_i, xy, wb, P)
        feasible[b_i, k_i, c_i] = ok
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
        g = ddt(aug_xy, 1)
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
        acc = ddt(g, 1)[:, i0]
        yaw_rate = ddt(torch.stack([heading.cos(), heading.sin()], -1), 1)[:, i0]
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


def frenet_augmenter_from_args(args) -> "FrenetStatePerturbationTensor":
    """Build the augmenter from a parsed config, for EVERY entrypoint that trains with it.

    One factory rather than a construction per entrypoint: the GRPO path used to build this
    with only ``augment_prob`` and ``device``, so it accepted every ``--frenet_*`` flag and
    then silently trained at the defaults. A run's recorded configuration has to be the
    configuration it actually used, and that only holds if there is a single place where the
    flags become an augmenter.

    ``argparse`` hands list fields back as strings, so the numeric coercion lives here too.
    """
    return FrenetStatePerturbationTensor(
        augment_prob=args.augment_prob,
        device=args.device,
        n_draws=int(args.frenet_n_draws),
        dy_max=float(args.frenet_dy_max),
        dth_max=float(args.frenet_dth_max),
        knobs=KnobGrid(
            merge_times=tuple(float(v) for v in args.frenet_merge_times),
            anchors=tuple(float(v) for v in args.frenet_anchors),
            acc0_fracs=tuple(float(v) for v in args.frenet_acc0_fracs),
        ),
        seed=int(args.frenet_seed),
        ranked_temp_s=float(args.frenet_ranked_temp_s),
    )
