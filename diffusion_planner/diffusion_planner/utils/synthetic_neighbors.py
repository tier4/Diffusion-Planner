"""Synthetic adversarial neighbor generator for GRPO augmentation.

This builds *synthetic* neighbors that are **guaranteed to collide with the ego GT
trajectory** if the ego does nothing (i.e. follows its recorded GT future). The idea is to
give the collision-based GRPO reward a strong, always-present signal: every injected neighbor
forces the ego to deviate.

Each neighbor moves with **constant acceleration** (a person or a vehicle):

    p(t) = p0 + v0 * t + 0.5 * a * t**2

with a random initial position ``p0`` (sampled in an x/y box around the ego), a random
initial velocity ``v0`` (random heading, type-dependent speed), and an acceleration ``a``
solved so the neighbor passes through the ego's GT position at a randomly chosen collision
time ``t_c``:

    a = 2 * (ego_xy(t_c) - p0 - v0 * t_c) / t_c**2

Candidates whose required acceleration exceeds a realistic cap are rejected and re-sampled
(``max_tries``), so the kept neighbors are both colliding *and* physically plausible.

The output columns match ``neighbor_agents_past`` / ``neighbor_agents_future`` exactly, so an
injected neighbor flows through the normal preprocessing with no special handling. ``inject``
takes ``(inputs, inject_max, inject_prob)`` and mutates the raw batch in place, and is called
from ``grpo_epoch._grpo_step`` (and ``visualize_grpo_samples.py``).
"""

import math

import torch

from diffusion_planner.dimensions import INPUT_T, OUTPUT_T

# Neighbor past row layout (see loss.py / visualize_input.py).
_X, _Y, _COS, _SIN, _VX, _VY, _WIDTH, _LENGTH = 0, 1, 2, 3, 4, 5, 6, 7
_TYPE_BASE = 8  # one-hot [vehicle, pedestrian, bicycle] occupies columns 8..10

# Per-type kinematics. ``speed`` is (min, max) m/s, ``accel_max`` is m/s^2, ``idx`` matches the
# neighbor one-hot ordering [vehicle, pedestrian, bicycle] (cols 8..10). Size is either a fixed
# (width, length) in metres, or a ``size`` = ((width_min, length_min), (width_max, length_max))
# range that is sampled along a single factor so width and length scale together (a small car
# stays narrow, a long truck stays wide).
# ``min_turn_radius`` (metres) caps how sharply a synthetic path may bend: a constant-accel
# path with low initial speed and large acceleration degenerates into a near-cusp hairpin (an
# ~180-deg reversal from the agent's own past), which is unpredictable from its history. Every
# type gets a cap, looser for the more agile ones (pedestrian < bicycle < vehicle).
_TYPES = {
    "vehicle": dict(
        speed=(0.0, 14.0),
        accel_max=2.5,
        idx=0,
        size=((1.84, 4.34), (2.5649, 10.7462)),
        min_turn_radius=5.0,
    ),
    "pedestrian": dict(
        speed=(0.0, 2.0), accel_max=1.0, idx=1, width=0.7, length=0.7, min_turn_radius=1.0
    ),
    "bicycle": dict(
        speed=(0.0, 7.0), accel_max=1.5, idx=2, width=0.6, length=1.8, min_turn_radius=2.0
    ),
}


class SyntheticColliderInjector:
    """Inject synthetic constant-acceleration neighbors that collide with the ego GT."""

    def __init__(
        self,
        pedestrian_prob: float,
        bicycle_prob: float,
        keep_clear_radius: float,
        straight_line: bool,
    ):
        dt = 0.1
        self.dt = dt
        # spawn box (ego frame): x in [x_lo, x_hi] (forward-biased), y in [-y_half_width,
        # +y_half_width] (symmetric left/right).
        self.x_lo, self.x_hi = -10.0, 40.0
        self.y_half_width = 25.0
        # explicit categorical type mix; vehicle takes the remainder.
        assert pedestrian_prob + bicycle_prob <= 1.0 + 1e-6, "ped + bike prob must be <= 1"
        self.pedestrian_prob = pedestrian_prob
        self.bicycle_prob = bicycle_prob
        self.vehicle_prob = max(0.0, 1.0 - pedestrian_prob - bicycle_prob)
        self.type_probs = {
            "vehicle": self.vehicle_prob,
            "pedestrian": pedestrian_prob,
            "bicycle": bicycle_prob,
        }
        self.min_collision_time = 0.8
        # the synthetic path must stay this far from the ego's t=0 pose (origin), so a
        # stationary ego is never hit -> the forced collision is always avoidable.
        self.keep_clear_radius = keep_clear_radius
        self.max_tries = 50
        # straight_line: the collider drives at constant velocity straight at the collision
        # point (heading aimed at the target, zero acceleration -> zero curvature). This is the
        # easy, fully-history-predictable regime. False restores random-heading constant-accel
        # colliders (curved approaches, capped by each type's min_turn_radius).
        self.straight_line = straight_line

        self._tau_past = torch.arange(-INPUT_T, 1, dtype=torch.float32) * dt  # [31] in [-3,0]
        self._tau_future = torch.arange(1, OUTPUT_T + 1, dtype=torch.float32) * dt  # [80] in [.1,8]

    def _rand(self, lo, hi, device):
        return torch.rand((), device=device) * (hi - lo) + lo

    def _make_neighbor(self, ego_xy, device):
        """Build one (past[31,11], future[80,3]) colliding neighbor for a single scene.

        ego_xy: [80, 2] ego GT future positions (ego frame, metres).

        The collision is aimed at a GT waypoint that is **at least ``keep_clear_radius`` from
        the ego's t=0 position (the origin)**, and any candidate whose path passes within
        ``keep_clear_radius`` of the origin is rejected. This guarantees that simply *staying
        put* (the ego's current pose) always avoids the collision, so it is never unavoidable.

        Returns ``None`` if no avoidable colliding neighbor could be built (e.g. the ego
        barely moves, so every GT waypoint is too close to the origin).
        """
        tau_p = self._tau_past.to(device)
        tau_f = self._tau_future.to(device)

        r = torch.rand((), device=device).item()
        if r < self.pedestrian_prob:
            spec = _TYPES["pedestrian"]
        elif r < self.pedestrian_prob + self.bicycle_prob:
            spec = _TYPES["bicycle"]
        else:
            spec = _TYPES["vehicle"]

        # body size: fixed for ped/bike, sampled along one factor for vehicles so width and
        # length scale together (small car -> narrow, long truck -> wide).
        if "size" in spec:
            (w_lo, l_lo), (w_hi, l_hi) = spec["size"]
            f = float(torch.rand((), device=device))
            width, length = w_lo + f * (w_hi - w_lo), l_lo + f * (l_hi - l_lo)
        else:
            width, length = spec["width"], spec["length"]

        # candidate collision steps: non-padding GT waypoints that are far enough from the
        # ego's t=0 position that hitting them is avoidable by not driving there.
        non_pad = ego_xy.abs().sum(dim=-1) > 1e-3
        far = ego_xy.norm(dim=-1) >= self.keep_clear_radius
        late = tau_f >= self.min_collision_time
        cand = torch.nonzero(non_pad & far & late, as_tuple=False).squeeze(-1)
        if cand.numel() == 0:
            cand = torch.nonzero(non_pad & far, as_tuple=False).squeeze(-1)
        if cand.numel() == 0:
            return None  # ego stays within keep_clear_radius of origin -> no avoidable target

        best = None  # (excess_accel, p0, v0, a) among origin-clearing candidates
        for _ in range(self.max_tries):
            k_c = cand[torch.randint(0, cand.numel(), (), device=device)]
            t_c = float(tau_f[k_c])
            target = ego_xy[k_c]  # [2]

            p0 = torch.stack(
                [
                    self._rand(self.x_lo, self.x_hi, device),
                    self._rand(-self.y_half_width, self.y_half_width, device),
                ]
            )

            if self.straight_line:
                # Constant velocity straight at the target: v0 = (target - p0) / t_c, a = 0.
                # Heading is aimed at the collision point, so the whole path is a straight line.
                disp = target - p0
                v0 = disp / t_c
                a = torch.zeros_like(v0)
                # skip if the implied constant speed is unrealistic for this type
                if float(v0.norm()) > spec["speed"][1]:
                    continue
            else:
                speed = self._rand(spec["speed"][0], spec["speed"][1], device)
                ang = self._rand(-math.pi, math.pi, device)
                v0 = torch.stack([speed * torch.cos(ang), speed * torch.sin(ang)])
                a = 2.0 * (target - p0 - v0 * t_c) / (t_c * t_c)

            # reject if the path ever comes within keep_clear_radius of the ego's t=0 pose
            t = tau_f[:, None]
            path = p0[None, :] + v0[None, :] * t + 0.5 * a[None, :] * (t * t)  # [80,2]
            min_clear = min(float(p0.norm()), float(path.norm(dim=-1).min()))
            if min_clear < self.keep_clear_radius:
                continue

            # vehicles can't turn arbitrarily sharply: reject near-cusp constant-accel paths
            # whose minimum turn radius falls below the per-type cap. Curvature of a
            # constant-accel path is k(t) = |v0 x a| / |v(t)|^3, maximal where speed is lowest.
            min_turn_radius = spec.get("min_turn_radius", 0.0)
            if min_turn_radius > 0.0:
                tau_all = torch.cat([tau_p, tau_f])
                vel = v0[None, :] + a[None, :] * tau_all[:, None]  # [M, 2]
                cross = (vel[:, 0] * a[1] - vel[:, 1] * a[0]).abs()
                max_curv = float((cross / vel.norm(dim=-1).pow(3).clamp_min(1e-6)).max())
                if max_curv * min_turn_radius > 1.0:  # min radius < cap -> too sharp
                    continue

            excess = float(a.norm()) - spec["accel_max"]
            if excess <= 0.0:
                best = (0.0, p0, v0, a)
                break
            if best is None or excess < best[0]:
                best = (excess, p0, v0, a)

        if best is None:
            return None  # no origin-clearing candidate found within max_tries
        _, p0, v0, a = best
        return self._assemble(p0, v0, a, tau_p, tau_f, spec, width, length, device)

    def _assemble(self, p0, v0, a, tau_p, tau_f, spec, width, length, device):
        def motion(tau):  # tau [M] -> pos [M,2], vel [M,2]
            t = tau[:, None]
            pos = p0[None, :] + v0[None, :] * t + 0.5 * a[None, :] * (t * t)
            vel = v0[None, :] + a[None, :] * t
            return pos, vel

        past_pos, past_vel = motion(tau_p)  # [31,2]
        fut_pos, fut_vel = motion(tau_f)  # [80,2]

        # reference heading for ~stationary steps (atan2 of a ~0 vector is noisy): prefer the
        # initial-velocity direction, else the acceleration direction (handles v0 == 0).
        if float(v0.norm()) > 1e-3:
            ref_ang = torch.atan2(v0[1], v0[0])
        else:
            ref_ang = torch.atan2(a[1], a[0])

        def heading(vel):
            h = torch.atan2(vel[:, 1], vel[:, 0])
            slow = vel.norm(dim=-1) < 1e-3
            h[slow] = ref_ang
            return h

        h_past = heading(past_vel)
        h_fut = heading(fut_vel)

        past = torch.zeros(INPUT_T + 1, 11, device=device)
        past[:, _X], past[:, _Y] = past_pos[:, 0], past_pos[:, 1]
        past[:, _COS], past[:, _SIN] = torch.cos(h_past), torch.sin(h_past)
        past[:, _VX], past[:, _VY] = past_vel[:, 0], past_vel[:, 1]
        past[:, _WIDTH], past[:, _LENGTH] = width, length
        past[:, _TYPE_BASE + spec["idx"]] = 1.0

        future = torch.zeros(OUTPUT_T, 3, device=device)
        future[:, 0], future[:, 1], future[:, 2] = fut_pos[:, 0], fut_pos[:, 1], h_fut
        return past, future

    @torch.no_grad()
    def inject(self, inputs: dict, inject_max: int, inject_prob: float) -> dict:
        """Inject synthetic colliding neighbors into the batch (in place).

        Real neighbors are front-packed (low indices) with empty slots at the back, so always
        writing into an empty slot would place every collider at a high index and let the
        model key off position. Instead each collider goes to a **random index in
        ``[0, first_empty]`` inclusive** -- it may overwrite a real neighbor or take the first
        empty slot -- so colliders are interspersed with the real agents.

        Call on a *raw* batch before ``heading_to_cos_sin`` / normalization. The boolean mask
        of slots actually written is stored on ``self.last_injected_mask`` ([B, Pn]).
        """
        neighbor_past = inputs["neighbor_agents_past"]  # [B, Pn, 31, 11]
        neighbor_future = inputs["neighbor_agents_future"]  # [B, Pn, 80, 3]
        ego_future = inputs["ego_agent_future"]  # [B, 80, 3] (x, y, heading)
        device = neighbor_past.device
        B, Pn = neighbor_past.shape[:2]

        empty = (neighbor_past != 0.0).any(dim=(2, 3)).logical_not()  # [B, Pn]
        injected = torch.zeros(B, Pn, dtype=torch.bool, device=device)

        for b in range(B):
            if torch.rand((), device=device).item() > inject_prob:
                continue

            ego_xy = ego_future[b, :, :2]  # [80, 2]
            if (ego_xy.abs().sum(dim=-1) > 1e-3).sum() == 0:
                continue  # fully padded ego -> nothing meaningful to collide with

            # first empty slot (= number of real, front-packed neighbors). Candidate write
            # positions are [0, first_empty] inclusive.
            empty_idx = torch.nonzero(empty[b], as_tuple=False).squeeze(-1)
            first_empty = int(empty_idx.min().item()) if empty_idx.numel() > 0 else Pn - 1
            n_positions = first_empty + 1

            k = int(torch.randint(1, inject_max + 1, (), device=device).item())
            k = min(k, n_positions)
            chosen = torch.randperm(n_positions, device=device)[:k]

            for slot in chosen:
                out = self._make_neighbor(ego_xy, device)
                if out is None:
                    continue  # couldn't build an avoidable collider for this scene/slot
                past, future = out
                neighbor_past[b, slot] = past
                neighbor_future[b, slot] = future
                injected[b, slot] = True

        self.last_injected_mask = injected
        return inputs
