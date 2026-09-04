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

from diffusion_planner.dimensions import EGOSTATE
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
from diffusion_planner.utils.data_augmentation import (
    StatePerturbation,
    cos_sin_to_heading,
    pose_to_cos_sin,
)

# extra half-width the corridor keeps from borders and neighbors
CORRIDOR_MARGIN = 0.10
# A neighbour counts as parked for the toward-parked nudge when its RECORDED velocity at
# the last history step is below this. The recorded (vx, vy) is used rather than a
# finite difference of the centroid: over one 0.1 s step a differenced position amplifies
# tracker jitter tenfold, so a stationary car wobbling 5 cm would read as 0.5 m/s.
PARKED_SPEED = 0.5  # m/s
# ...and it must also STAY put: a car stopped at a light is slow at t=0 and then departs,
# and the nudge would aim the ego at a lead vehicle that is no longer there. Every
# recorded future sample must stay within this distance of its t=0 position, which is
# what makes the "MOVING vehicles are excluded" claim true rather than aspirational.
PARKED_MAX_DISPLACEMENT = 1.0  # m


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
        # Restore "no history" rows to exact zero. Both the rewrite and the parent's
        # re-centering write real numbers into every history slot, and the encoder reads
        # ego_agent_past directly -- a transformed pad would be presented to the model as
        # history that never happened. Applied to the whole batch, not just augmented rows,
        # so a future non-identity transform on untouched rows cannot reintroduce it.
        pad = getattr(self, "_ego_past_pad", None)
        if pad is not None and bool(pad.any()):
            inputs["ego_agent_past"][pad] = 0.0
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
        # History noise runs LAST, after the re-centering rather than before it. Only
        # here is the history expressed about the augmented t=0 pose, so scaling it
        # about the origin leaves t=0 exactly at the origin and leaves the implied
        # speed history consistent with the current state it is scaled with. Scaling in
        # the pre-centering frame would instead drag the t=0 pose away from the current
        # state the whole batch is then centered on. It also runs after the pad
        # zeroing, so a padded row is exactly zero and no factor can resurrect it.
        if self.past_noise_std > 0.0 and rows is not None and bool(rows.any()):
            self._scale_history(inputs, rows)
        return out

    def _scale_history(self, inputs, rows):
        """Scale an augmented row's rewritten history by one factor per scene.

        The perturbation the quintic augmenter applies through ``ego_past_noise_std``.
        Frenet reads the same flag, but applies it to the history it rewrote rather than
        to the recorded one the base class would have scaled: one N(1, std) scalar per scene clamped to +-2 std, multiplying
        the history xy and the current velocity and acceleration together, so the
        implied speed history stays consistent with the state. Non-augmented rows are
        never touched -- they train on plain ground truth, unscaled.

        The draw comes from the augmenter's own generator, and is taken ONLY when the
        std is positive, so at the default the generator is left exactly where the
        unaugmented stream leaves it.
        """
        w = self.past_noise_std
        past, cur = inputs["ego_agent_past"], inputs["ego_current_state"]
        scale = torch.normal(1.0, w, size=(int(rows.sum()), 1, 1), generator=self.gen).clamp(
            1.0 - 2 * w, 1.0 + 2 * w
        )
        scale = scale.to(device=past.device, dtype=past.dtype)
        past[rows, :, :2] = past[rows, :, :2] * scale
        cur[rows, EGOSTATE.VX : EGOSTATE.AY + 1] = (
            cur[rows, EGOSTATE.VX : EGOSTATE.AY + 1] * scale[:, :, 0]
        )

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
        recovery_rounds: int = 0,
        toward_parked_prob: float = 0.0,
        min_clearance: float = 0.0,
        ego_past_noise_std: float = 0.0,
    ):
        super().__init__(
            augment_prob=augment_prob,
            num_refine=20,
            device=device,
            # The base class scales the RECORDED history, which frenet does not keep --
            # it rewrites the past from the perturbed polyline. The same perturbation is
            # applied to that rewrite instead, in _scale_history, so the base class is
            # disabled here and the caller's value is stored below.
            ego_past_noise_std=0.0,
            use_smoothing_future_trajectory=False,
        )
        # The ego past is rewritten to the perturbed polyline, so it must also be
        # re-centered with the future in centric_transform (quintic/bridge keep the
        # GT past and skip that transform).
        self._transform_ego_past = True
        # rows augmented in the most recent __call__; centric_transform re-projects
        # velocity/accel to base_link (vy = ay = 0) for exactly these rows
        self._aug_rows = None
        # corridor by-products, all reset at the top of every _corridor call
        self._nbr_near = None
        self._nbr_st = None
        self._nbr_valid = None
        self._nbr_lo = None
        self._nbr_hi = None
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
        # Rounds of draw-level re-selection allowed after an exact-OBB veto. 0 keeps
        # the original behaviour: a vetoed row falls back to plain GT.
        self.recovery_rounds = int(recovery_rounds)
        if self.recovery_rounds < 0:
            raise ValueError(f"recovery_rounds must be >= 0, got {recovery_rounds}")
        # Fraction of ELIGIBLE scenes (a PARKED vehicle bounding the corridor far enough
        # ahead that the avoidance is still in front of the ego) whose nudges are mirrored
        # to point at that vehicle. 0 reproduces the uniform draw exactly.
        self.toward_parked_prob = float(toward_parked_prob)
        if not 0.0 <= self.toward_parked_prob <= 1.0:
            raise ValueError(f"toward_parked_prob must be in [0, 1], got {toward_parked_prob}")
        # Exact footprint clearance EVERY accepted candidate must keep from every
        # recorded neighbour, on top of the corridor's own margin. Applies to all rows:
        # a trajectory that skims a parked car is not a valid target however it was
        # drawn. 0 keeps today's overlap-only veto and is bit-identical.
        self.min_clearance = float(min_clearance)
        if self.min_clearance < 0.0:
            raise ValueError(f"min_clearance must be >= 0, got {min_clearance}")
        # Std of the per-scene history-scale factor. The base class is told 0 above
        # because frenet rewrites the past kinematically and the quintic scaling would
        # be applied to the recorded history instead; this knob applies the same
        # perturbation to the REWRITTEN history. 0 draws nothing and is bit-identical.
        self.past_noise_std = float(ego_past_noise_std)
        if self.past_noise_std < 0.0:
            raise ValueError(f"ego_past_noise_std must be >= 0, got {ego_past_noise_std}")
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
    def _corridor(self, inputs, xy, tan, nrm, half_w, half_w_nbr, half_l, wb):
        """Lateral free space per timestep: where can this ego go sideways.

        Border and neighbour cuts are computed against separate copies of the
        unconstrained bounds and then intersected, rather than tightened in
        sequence. Both are max/min reductions so the intersection is identical,
        and keeping the neighbour-only pair is what lets the toward-parked nudge ask
        "which side is a VEHICLE on" without mistaking a kerb for a car.

        The two cuts take SEPARATE half-widths: ``min_clearance`` is a floor on the
        gap to a NEIGHBOUR, and widening the border cut with it as well would drop
        kerb-hugging scenes out of augmentation entirely (the recorded drives touch
        the mapped borders, which is exactly why borders are never vetoed either).
        ``half_w`` cuts against borders, ``half_w_nbr`` against neighbours; they are
        equal whenever ``min_clearance <= CORRIDOR_MARGIN``.
        """
        # reset per batch so a neighbor-less batch never sees a stale tensor
        self._nbr_st = self._nbr_valid = self._nbr_near = None
        self._nbr_lo = self._nbr_hi = None
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
            n_lo, n_hi = unconstrained_bounds(B, T, xy.device, xy.dtype)
            n_lo, n_hi, self._nbr_near = neighbor_lateral_bounds(
                st,
                valid,
                shapes_wl,
                xy,
                tan,
                nrm,
                half_l,
                half_w_nbr,
                wb,
                n_lo,
                n_hi,
                min_clearance=self.min_clearance,
            )
            lo, hi = torch.maximum(lo, n_lo), torch.minimum(hi, n_hi)
            if self.toward_parked_prob > 0.0:
                # bounds from PARKED neighbours alone: the toward-parked nudge points at
                # these and nothing else
                P = inputs["ego_agent_past"].shape[1]
                parked = self._parked_mask(past, st, valid, P)
                p_lo, p_hi = unconstrained_bounds(B, T, xy.device, xy.dtype)
                p_lo, p_hi, _ = neighbor_lateral_bounds(
                    st,
                    valid & parked[:, :, None],
                    shapes_wl,
                    xy,
                    tan,
                    nrm,
                    half_l,
                    half_w_nbr,
                    wb,
                    p_lo,
                    p_hi,
                )
                self._nbr_lo, self._nbr_hi = p_lo, p_hi
        return lo, hi

    @staticmethod
    def _parked_mask(past, st, valid, P):
        """Which recorded neighbours are parked for the whole recording.

        Two conditions, both necessary. STOPPED: the dataset's own velocity
        ``(vx, vy)`` at the last history step is below :data:`PARKED_SPEED` — the
        recorded velocity, not a one-step position difference, whose 1/DT factor turns
        centimetres of centroid jitter into half a metre per second. STAYS: no recorded
        future sample leaves a :data:`PARKED_MAX_DISPLACEMENT` ball around the t=0
        position, so a vehicle that is merely waiting to move is excluded rather than
        being treated as parked for the whole horizon.

        Args:
            past: (B, N, P, >=6) neighbour history; cols 4, 5 are vx, vy.
            st, valid: from :func:`time_aligned_neighbor_tracks`.
            P: number of history steps, so index ``P - 1`` is t=0.

        Returns:
            (B, N) bool.
        """
        stopped = valid[:, :, P - 1] & (past[:, :, P - 1, 4:6].norm(dim=-1) < PARKED_SPEED)
        # invalid future slots contribute 0 displacement: absence of a track is not
        # evidence of motion, and those slots cut no corridor anyway
        moved = (st[:, :, P:, :2] - st[:, :, P - 1, None, :2]).norm(dim=-1) * valid[:, :, P:]
        return stopped & (moved.amax(dim=-1) < PARKED_MAX_DISPLACEMENT)

    @staticmethod
    def _headings(aug_xy, tan):
        """Motion-direction heading of a polyline, falling back to the GT tangent.

        Below 0.3 m/step the finite-difference direction is noise, so the recorded
        heading is kept rather than derived from a near-zero displacement.
        """
        g = ddt(aug_xy, 1)
        hd_gt = torch.atan2(tan[..., 1], tan[..., 0])
        heading = torch.where(g.norm(dim=-1) > 0.3, torch.atan2(g[..., 1], g[..., 0]), hd_gt)
        return g, heading

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
            min_clearance=self.min_clearance,
        )

    # ---------- toward-parked nudge ----------
    def _parked_vehicle_ahead(self, P, half_w, merge_steps_min):
        """Which scenes can be made into a harder avoidance than the recording?

        Eligible when a PARKED vehicle (not a kerb, not a moving car) bounds the
        corridor within a nudge's reach, and it is reached late enough that the
        avoidance is still ahead of the ego. An object already beside the ego at t=0 leaves no approach
        to perturb, and no merge horizon could rejoin before it anyway -- so the
        floor is ``min(merge_times)``, derived rather than tuned.

        "Within reach" is a test in OFFSET space: ``neighbor_lateral_bounds`` returns
        bounds that already have the ego half-width subtracted, so the room it reports
        is room for the ego's centreline, which is what ``dy_max`` bounds.

        Args:
            P: number of history steps.
            half_w: (B,) ego half width; used for its shape/device/dtype only.
            merge_steps_min: shortest merge horizon in timesteps.

        Returns:
            eligible: (B,) bool.
            side: (B,) +1 / -1, the normal-direction sign the vehicle sits on.
            t_obs: (B,) index into the full time grid where the corridor is tightest.
        """
        if self._nbr_lo is None:
            z = torch.zeros(half_w.shape[0], dtype=torch.bool, device=half_w.device)
            return z, torch.ones_like(half_w), torch.zeros_like(z, dtype=torch.long)
        lo_f, hi_f = self._nbr_lo[:, P:], self._nbr_hi[:, P:]  # future only
        room_hi, room_lo = hi_f, -lo_f  # room toward +normal / -normal
        slack = torch.minimum(room_hi, room_lo)  # (B, F)
        s_min, t_rel = slack.min(dim=1)
        b = torch.arange(slack.shape[0], device=slack.device)
        # the vehicle sits on whichever side left less room at that timestep
        side = torch.where(room_hi[b, t_rel] <= room_lo[b, t_rel], 1.0, -1.0)
        eligible = (s_min <= self.dy_max) & (t_rel >= merge_steps_min)
        return eligible, side, t_rel + P

    # ---------- input canonicalisation ----------
    @staticmethod
    def _ego_past_4col(past):
        """Ego history as (x, y, cos, sin), from either recorded layout.

        A 3-col history cannot tell a padded row from the ego's own t=0 sample, which
        in the ego frame IS the origin with heading 0. Position resolves it: padding is
        a LEADING prefix, so the last sample is never padding, and the widened array
        then carries the same padded-rows-are-zero contract the 4-col layout has.
        """
        if past.shape[-1] >= 4:
            return past
        pad = torch.sum(torch.ne(past, 0), dim=-1) == 0
        pad[:, -1] = False
        return pose_to_cos_sin(past, pad=pad)

    def _history_polyline(self, past4):
        """History positions and tangents, with the zero-padded prefix held.

        Leading history can be zero-padded when a scene starts near the beginning of a
        recording. Those rows mean "no history", not "the ego was at the origin facing
        +x", so the pad mask is remembered on the instance and the rows are restored to
        exact zero after the rewrite and the re-centering (see ``centric_transform``).
        Same contract as the bridge path.

        Across the padding the first real pose is HELD rather than left at the origin: a
        (0, 0) position adjacent to a real one is a metres-wide phantom step, which would
        give the first REAL history sample a heading taken from that jump and a speed
        spike the feasibility screen would then judge. Replicating makes the differences
        zero across the padding boundary, so the first real sample keeps its own stored
        heading.
        """
        P = past4.shape[1]
        self._ego_past_pad = torch.sum(torch.ne(past4[..., :4], 0), dim=-1) == 0  # (B, P)
        past_xy, past_tan = past4[..., :2], past4[..., 2:4]
        if not bool(self._ego_past_pad.any()):
            return past_xy, past_tan
        first = torch.argmax((~self._ego_past_pad).to(torch.int8), dim=1)  # (B,)
        idx = torch.arange(P, device=past4.device)[None, :]
        src = torch.maximum(idx, first[:, None])  # pad -> first real, real -> itself
        return (
            torch.gather(past_xy, 1, src[..., None].expand(-1, -1, 2)),
            torch.gather(past_tan, 1, src[..., None].expand(-1, -1, 2)),
        )

    # ---------- toward-parked nudge: draws and selection ----------
    def _toward_parked_draws(self, r, K, P, half_w, do_aug, dy):
        """Point a fraction of the eligible scenes' drawn offsets AT a parked vehicle.

        On the scenes where a PARKED vehicle bounds the corridor further ahead than the
        shortest merge, every drawn offset is mirrored to point at that vehicle. Only
        the SIGN is mirrored, so |dy| keeps the distribution it already had -- the ego's
        t=0 state becomes a harder avoidance than the recording's, and nothing
        downstream (dth, merge sampling, shapes, corridor, kinematics, veto) is touched.
        Candidates too hard to drive are still rejected by the existing feasibility
        screen, so "harder" cannot become "impossible".

        Returns:
            toward: (B,) bool or None when the nudge is off.
            toward_any: ``bool(toward.any())``, computed once and reused.
            dy: the drawn offsets, mirrored on the toward rows.
            t_obs: (B,) tightest-corridor timestep, or None when the nudge is off.
        """
        if self.toward_parked_prob <= 0.0:
            return None, False, dy, None
        merge_steps_min = int(min(self.knobs.merge_times) / DT)
        eligible, side, t_obs = self._parked_vehicle_ahead(P, half_w, merge_steps_min)
        toward = eligible & (r[:, 2 * K + 1] < self.toward_parked_prob) & do_aug
        toward_any = bool(toward.any())
        if toward_any:
            dy = torch.where(toward[:, None], side[:, None] * dy.abs(), dy)
        return toward, toward_any, dy, t_obs

    @staticmethod
    def _largest_offset_draw(draw_ok, dy):
        """Index of the still-feasible draw with the biggest |dy|, per scene."""
        return torch.where(draw_ok, dy.abs(), torch.full_like(dy, -1.0)).argmax(-1)

    def _toward_parked_select(self, admissible, merges, dy, toward, toward_any, t_obs, P):
        """Merge gate and winner index; a no-op on every row that is not toward-parked.

        Two things happen to a toward row. The recovery has to FINISH while the vehicle
        still matters, so merge horizons reaching past the tightest-corridor timestep are
        struck out. And the winner is the LARGEST feasible offset rather than the first
        feasible draw: mirroring the sign only fixes direction; measured on 105k scenes,
        first-feasible left the baseline harder in 24.5% of rows, largest-feasible in
        9.5% -- the remainder being the merge gate refusing a horizon the baseline was
        allowed. "Harder" has to hold per scene, not on average.

        Returns:
            admissible (gated), draw_ok (B, K), first (B,) winning draw per scene.
        """
        if toward_any:
            merge_steps = (merges / DT).round().long()  # (C,)
            in_time = merge_steps[None, :] <= (t_obs - (P - 1))[:, None]  # (B, C)
            admissible = admissible & torch.where(
                toward[:, None, None], in_time[:, None, :], torch.ones_like(in_time[:, None, :])
            )
        draw_ok = admissible.any(-1)  # (B, K): the drawn perturbation has >=1 valid merge
        first = draw_ok.float().argmax(-1)  # first feasible PERTURBATION per scene
        if toward_any:
            first = torch.where(toward, self._largest_offset_draw(draw_ok, dy), first)
        return admissible, draw_ok, first

    # ---------- the augmentation ----------
    @torch.no_grad()
    def __call__(self, inputs, ego_future, neighbors_future):
        # Both fields arrive in either layout depending on the entrypoint. The future is
        # narrowed to a heading angle (centric_transform returns the rebuilt 3-col future
        # either way, and train_epoch re-converts afterwards); the history is widened,
        # because everything below indexes its cols 2:4 as a heading VECTOR. Both are
        # no-ops at the canonical widths the training loop already supplies.
        ego_future = cos_sin_to_heading(ego_future)
        inputs["ego_agent_past"] = self._ego_past_4col(inputs["ego_agent_past"])
        past4 = inputs["ego_agent_past"]  # (B, P, 4) x, y, cos, sin
        B, P, _ = past4.shape
        F = ego_future.shape[1]
        dev, dtype = past4.device, past4.dtype
        t, combos = self._bases(P, F, dev, dtype)

        fut_cs = torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1)
        past_xy, past_tan = self._history_polyline(past4)
        xy = torch.cat([past_xy, ego_future[..., :2]], dim=1)  # (B, T, 2)
        tan = torch.cat([past_tan, fut_cs], dim=1)
        nrm = torch.stack([-tan[..., 1], tan[..., 0]], dim=-1)
        speed = ddt(xy, 1).norm(dim=-1).clamp(min=0.5)  # (B, T)
        v0 = speed[:, P - 1]
        wb = inputs["ego_shape"][:, 0]
        # min_clearance is a floor on the gap to a NEIGHBOUR only; the border cut keeps
        # the plain corridor margin (see _corridor).
        half_w = inputs["ego_shape"][:, 2] / 2 + CORRIDOR_MARGIN
        half_w_nbr = inputs["ego_shape"][:, 2] / 2 + max(CORRIDOR_MARGIN, self.min_clearance)
        half_l = inputs["ego_shape"][:, 1] / 2

        lo, hi = self._corridor(inputs, xy, tan, nrm, half_w, half_w_nbr, half_l, wb)

        K = self.n_draws
        # One extra column ONLY when the toward-parked nudge is on, so at
        # toward_parked_prob = 0 the RNG stream is byte-for-byte what it has always been.
        toward_on = self.toward_parked_prob > 0.0
        r = torch.rand((B, 2 * K + 1 + int(toward_on)), generator=self.gen).to(dev)
        do_aug = r[:, 2 * K] < self._augment_prob
        # Low-speed / reverse gate (same threshold as the quintic augmenter): a
        # near-stationary ego makes every path-relative feasibility metric
        # degenerate (a pure sideways translation has zero lateral accel/jerk/yaw
        # rate), and a reversing ego would get motion-direction headings flipped
        # 180 deg by the polyline rewrite. Both train on plain GT instead.
        do_aug = do_aug & (inputs["ego_current_state"][:, 4] >= 2.0)
        dy = (r[:, :K] * 2 - 1) * self.dy_max
        dth = (r[:, K : 2 * K] * 2 - 1) * self.dth_max

        toward, toward_any, dy, t_obs = self._toward_parked_draws(r, K, P, half_w_nbr, do_aug, dy)

        L, in_corr, merges = self._candidate_profiles(
            combos, dy, dth, v0, speed, half_l, lo, hi, dev, dtype
        )

        feasible, jerk_fut_peak = self._feasibility(xy, nrm, L, in_corr, wb, P)

        # `feasible` is physics only: can a car drive this candidate. `admissible` adds the
        # two reasons a scene may be left alone regardless of physics -- it lost the
        # augment_prob coin, or it is below the low-speed gate -- so the two questions stay
        # separable when reading the selection below.
        admissible = feasible & do_aug[:, None, None]
        admissible, draw_ok, first = self._toward_parked_select(
            admissible, merges, dy, toward, toward_any, t_obs, P
        )
        has = draw_ok.any(-1)  # (B,)

        if not bool(has.any()):
            # nothing feasible this batch: every row trains on plain GT
            self._aug_rows = has
            return self.centric_transform(inputs, ego_future, neighbors_future)

        aug_xy = self._select_candidate(
            admissible, jerk_fut_peak, first, merges, L, xy, nrm, B, dev
        )
        g, heading = self._headings(aug_xy, tan)
        upd = self._veto_true_overlaps(inputs, has.clone(), aug_xy, heading)
        if self.recovery_rounds:
            aug_xy, g, heading, upd = self._recover_vetoed(
                inputs,
                admissible,
                jerk_fut_peak,
                first,
                merges,
                L,
                xy,
                nrm,
                tan,
                aug_xy,
                g,
                heading,
                upd,
                has,
                dy,
                toward,
                toward_any,
            )

        self._write_back(inputs, ego_future, aug_xy, g, heading, upd, wb, P)
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

    def _select_candidate(self, feasible, jerk_fut_peak, first, merges, L, xy, nrm, B, dev):
        """Sample a merge horizon per scene, take its lowest-jerk combo, build the polyline."""
        bi = torch.arange(B, device=dev)
        # P(M) ∝ n_feasible_combos(M) * exp(-(M-Mmin)/temp) — biased to fast
        # convergence, slower merges keep a chance; within the sampled M,
        # lowest peak future jerk wins. Shared with the post-veto retry.
        combo_idx, _ = self._sample_merge_then_jerk(
            feasible[bi, first], jerk_fut_peak[bi, first], merges
        )
        Lw = L[bi, first, combo_idx]  # (B, T)
        aug_xy = xy + Lw[..., None] * nrm
        return aug_xy

    def _sample_merge_then_jerk(self, feas_k, jerk_k, merges):
        """The merge-then-jerk pick, shared by first selection and recovery.

        Merge horizon sampled with weight exp(-(M - Mmin) / temp) over the
        horizons still feasible, then lowest peak future jerk within it.
        """
        BIG = torch.tensor(1e9, device=feas_k.device, dtype=jerk_k.dtype)
        m_feas = torch.where(feas_k, merges[None, :], BIG)
        w = torch.exp(-(merges[None, :] - m_feas.amin(-1, keepdim=True)) / self.ranked_temp_s)
        w = w * feas_k
        alive = w.sum(-1) > 0
        w = torch.where(alive[:, None], w, torch.ones_like(w))  # avoid all-zero rows
        pick_m = torch.multinomial(w.cpu().double(), 1, generator=self.gen).to(w.device)[:, 0]
        same_m = feas_k & (merges[None, :] == merges[pick_m][:, None])
        return torch.where(same_m, jerk_k, BIG).argmin(-1), alive

    def _recover_vetoed(
        self,
        inputs,
        admissible,
        jerk,
        first,
        merges,
        L,
        xy,
        nrm,
        tan,
        aug_xy,
        g,
        heading,
        upd,
        has,
        dy,
        toward,
        toward_any,
    ):
        """Re-draw for rows whose winner truly overlapped a recorded neighbour.

        The other 39 shapes of the losing draw carry the SAME lateral offset and
        would overlap the same neighbour, so a retry has to change the draw, not
        the shape: the losing draw is burned whole and the next surviving one is
        re-selected and re-checked. Measured on 105k scenes: round 1 recovers
        ~54% of vetoed rows, round 2 ~7% more, round 3 ~2% — the survivors are
        blocked geometrically, and re-rolling does not move geometry.

        The retry keeps each row's own selection rule: first-feasible everywhere, and
        LARGEST-feasible on the toward-parked rows. Falling back to first-feasible for a
        toward row would let it recover on a 3 cm offset and still be counted as a
        hardened example, which is the whole reason the rule exists (see
        :meth:`_toward_parked_select`).

        Rows that never recover keep upd False and train on plain GT, exactly as
        they do with recovery disabled.
        """
        adm = admissible.clone()
        B = xy.shape[0]
        adm[torch.arange(B, device=xy.device), first, :] = False
        for _ in range(self.recovery_rounds):
            live = has & ~upd
            if not bool(live.any()):
                break
            rows = torch.nonzero(live, as_tuple=True)[0]
            draw_ok = adm[rows].any(-1)  # (M, K) draws with a shape left to try
            if not bool(draw_ok.any()):
                break
            cur = draw_ok.float().argmax(-1)
            if toward_any:
                cur = torch.where(toward[rows], self._largest_offset_draw(draw_ok, dy[rows]), cur)
            feas_k = adm[rows, cur]
            combo_idx, alive = self._sample_merge_then_jerk(feas_k, jerk[rows, cur], merges)
            cand = xy[rows] + L[rows, cur, combo_idx][..., None] * nrm[rows]
            g_r, hd_r = self._headings(cand, tan[rows])
            keep = self._veto_true_overlaps_rows(inputs, alive.clone(), cand, hd_r, rows)
            sel = rows[keep]
            aug_xy[sel], g[sel], heading[sel] = cand[keep], g_r[keep], hd_r[keep]
            upd[sel] = True
            adm[rows, cur, :] = False
        return aug_xy, g, heading, upd

    def _veto_true_overlaps_rows(self, inputs, rows_mask, aug_xy, heading, rows):
        """:meth:`_veto_true_overlaps` restricted to a subset of scenes."""
        if self._nbr_st is None:
            return rows_mask
        hd_cs = torch.stack([heading.cos(), heading.sin()], dim=-1)
        return veto_overlapping(
            rows_mask,
            torch.cat([aug_xy, hd_cs], dim=-1),
            inputs["ego_shape"][rows],
            self._nbr_st[rows],
            self._nbr_valid[rows],
            inputs["neighbor_agents_past"][rows][:, :, -1][..., [6, 7]],
            self._nbr_near[rows],
            min_clearance=self.min_clearance,
        )

    def _write_back(self, inputs, ego_future, aug_xy, g, heading, upd, wb, P):
        """Write history / future / current state in place for the accepted rows."""
        gs = g.norm(dim=-1)
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
        recovery_rounds=int(args.frenet_recovery_rounds),
        toward_parked_prob=float(args.frenet_toward_parked_prob),
        min_clearance=float(args.frenet_min_clearance),
        ego_past_noise_std=float(args.ego_past_noise_std),
    )
