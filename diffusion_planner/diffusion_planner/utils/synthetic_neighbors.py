"""Synthetic adversarial neighbor generator for GRPO augmentation.

Instead of copying real neighbor patterns from a DB (see ``neighbor_db.py``), this builds
*synthetic* neighbors that are **guaranteed to collide with the ego GT trajectory** if the
ego does nothing (i.e. follows its recorded GT future). The idea is to give the
collision-based GRPO reward a strong, always-present signal: every injected neighbor forces
the ego to deviate.

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
injected neighbor flows through the normal preprocessing with no special handling -- the
``inject`` signature mirrors ``NeighborPatternDB.inject`` so it is a drop-in replacement in
``grpo_epoch._grpo_step``.
"""

import math

import torch

from diffusion_planner.dimensions import INPUT_T, OUTPUT_T

# Neighbor past row layout (see loss.py / visualize_input.py).
_X, _Y, _COS, _SIN, _VX, _VY, _WIDTH, _LENGTH = 0, 1, 2, 3, 4, 5, 6, 7
_TYPE_BASE = 8  # one-hot [vehicle, pedestrian, bicycle] occupies columns 8..10

# Per-type kinematics: (speed_min, speed_max) m/s, accel_max m/s^2, width m, length m, type idx.
# type idx matches the neighbor one-hot ordering [vehicle, pedestrian, bicycle] (cols 8..10).
_TYPES = {
    "vehicle": dict(speed=(2.0, 14.0), accel_max=2.5, width=2.0, length=4.6, idx=0),
    "pedestrian": dict(speed=(0.3, 2.0), accel_max=1.0, width=0.7, length=0.7, idx=1),
    "bicycle": dict(speed=(1.0, 7.0), accel_max=1.5, width=0.6, length=1.8, idx=2),
}


class SyntheticColliderInjector:
    """Inject synthetic constant-acceleration neighbors that collide with the ego GT."""

    def __init__(
        self,
        dt: float = 0.1,
        pos_range: tuple[float, float] = (-10.0, 40.0),
        pedestrian_prob: float = 0.3,
        bicycle_prob: float = 0.2,
        min_collision_time: float = 0.8,
        keep_clear_radius: float = 3.0,
        max_tries: int = 50,
    ):
        self.dt = dt
        self.pos_lo, self.pos_hi = pos_range
        # categorical type mix; the remainder (1 - ped - bike) is vehicle.
        self.pedestrian_prob = pedestrian_prob
        self.bicycle_prob = bicycle_prob
        self.min_collision_time = min_collision_time
        # the synthetic path must stay this far from the ego's t=0 pose (origin), so a
        # stationary ego is never hit -> the forced collision is always avoidable.
        self.keep_clear_radius = keep_clear_radius
        self.max_tries = max_tries
        self.num_patterns = float("inf")  # for parity with NeighborPatternDB logging

        self._tau_past = torch.arange(-INPUT_T, 1, dtype=torch.float32) * dt   # [31] in [-3,0]
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

            p0 = torch.stack([self._rand(self.pos_lo, self.pos_hi, device),
                              self._rand(self.pos_lo, self.pos_hi, device)])
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

            excess = float(a.norm()) - spec["accel_max"]
            if excess <= 0.0:
                best = (0.0, p0, v0, a)
                break
            if best is None or excess < best[0]:
                best = (excess, p0, v0, a)

        if best is None:
            return None  # no origin-clearing candidate found within max_tries
        _, p0, v0, a = best
        return self._assemble(p0, v0, a, tau_p, tau_f, spec, device)

    def _assemble(self, p0, v0, a, tau_p, tau_f, spec, device):
        def motion(tau):  # tau [M] -> pos [M,2], vel [M,2]
            t = tau[:, None]
            pos = p0[None, :] + v0[None, :] * t + 0.5 * a[None, :] * (t * t)
            vel = v0[None, :] + a[None, :] * t
            return pos, vel

        past_pos, past_vel = motion(tau_p)   # [31,2]
        fut_pos, fut_vel = motion(tau_f)      # [80,2]

        def heading(vel):
            h = torch.atan2(vel[:, 1], vel[:, 0])
            # freeze heading where the neighbor is ~stationary (atan2 of ~0 vector is noisy)
            slow = vel.norm(dim=-1) < 1e-3
            h[slow] = torch.atan2(v0[1], v0[0])
            return h

        h_past = heading(past_vel)
        h_fut = heading(fut_vel)

        past = torch.zeros(INPUT_T + 1, 11, device=device)
        past[:, _X], past[:, _Y] = past_pos[:, 0], past_pos[:, 1]
        past[:, _COS], past[:, _SIN] = torch.cos(h_past), torch.sin(h_past)
        past[:, _VX], past[:, _VY] = past_vel[:, 0], past_vel[:, 1]
        past[:, _WIDTH], past[:, _LENGTH] = spec["width"], spec["length"]
        past[:, _TYPE_BASE + spec["idx"]] = 1.0

        future = torch.zeros(OUTPUT_T, 3, device=device)
        future[:, 0], future[:, 1], future[:, 2] = fut_pos[:, 0], fut_pos[:, 1], h_fut
        return past, future

    @torch.no_grad()
    def inject(self, inputs: dict, inject_max: int, inject_prob: float) -> dict:
        """Fill empty neighbor slots with synthetic colliding neighbors (in place).

        Mirrors ``NeighborPatternDB.inject``: call on a *raw* batch before
        ``heading_to_cos_sin`` / normalization.
        """
        neighbor_past = inputs["neighbor_agents_past"]      # [B, Pn, 31, 11]
        neighbor_future = inputs["neighbor_agents_future"]  # [B, Pn, 80, 3]
        ego_future = inputs["ego_agent_future"]             # [B, 80, 3] (x, y, heading)
        device = neighbor_past.device
        B, Pn = neighbor_past.shape[:2]

        empty = (neighbor_past != 0.0).any(dim=(2, 3)).logical_not()  # [B, Pn]

        for b in range(B):
            if torch.rand((), device=device).item() > inject_prob:
                continue
            slots = torch.nonzero(empty[b], as_tuple=False).squeeze(-1)
            if slots.numel() == 0:
                continue

            ego_xy = ego_future[b, :, :2]  # [80, 2]
            if (ego_xy.abs().sum(dim=-1) > 1e-3).sum() == 0:
                continue  # fully padded ego -> nothing meaningful to collide with

            k = int(torch.randint(1, inject_max + 1, (), device=device).item())
            k = min(k, slots.numel())
            chosen = slots[torch.randperm(slots.numel(), device=device)[:k]]

            for slot in chosen:
                out = self._make_neighbor(ego_xy, device)
                if out is None:
                    continue  # couldn't build an avoidable collider for this scene/slot
                past, future = out
                neighbor_past[b, slot] = past
                neighbor_future[b, slot] = future

        return inputs
