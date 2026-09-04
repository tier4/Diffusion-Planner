# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the frenet augmenter DOES, on synthetic scenes with a known answer.

The factory tests next door pin the plumbing (a flag reaches the object). These
pin behaviour, and above all the invariant the opt-in flags are worth nothing
without: at their off values they change no output and consume no extra
randomness, so a run recorded at the defaults is the run that was trained.
"""

from __future__ import annotations

import pytest
import torch
from diffusion_planner.utils.augmentation_checks import DT, time_aligned_neighbor_tracks
from diffusion_planner.utils.data_augmentation_frenet import (
    PARKED_SPEED,
    FrenetStatePerturbationTensor,
)

P_STEPS = 21  # history samples, t = -2.0 s .. 0
F_STEPS = 80  # future samples, t = 0.1 s .. 8.0 s
EGO_V = 10.0  # m/s, straight and level
EGO_SHAPE = (2.75, 5.0, 2.0)  # wheel base, length, width
N_SLOTS = 4  # neighbour slots per scene


def _scene(batch: int = 1, border_y: float = 6.0, neighbours=(), pad_steps: int = 0):
    """A straight-road batch: ego at EGO_V along +x, kerbs at +-border_y.

    ``neighbours`` is a list of dicts, one per occupied slot, applied to EVERY
    scene in the batch:
      ``lon``/``lat``  metres ahead of / left of the ego's t=0 rear axle,
      ``wl``           (width, length),
      ``v``            recorded (vx, vy) at t=0,
      ``steps``        which time-grid indices hold a valid observation
                       (default: all of them),
      ``jitter``       amplitude of an alternating centroid wobble, which is what
                       makes a one-step position difference lie about the speed,
      ``drift``        metres the box travels over the recorded future.
    """
    t_past = torch.arange(-(P_STEPS - 1), 1, dtype=torch.float32) * DT
    t_fut = torch.arange(1, F_STEPS + 1, dtype=torch.float32) * DT
    t_all = torch.cat([t_past, t_fut])
    T = P_STEPS + F_STEPS

    past = torch.zeros(batch, P_STEPS, 4)
    past[..., 0] = EGO_V * t_past
    past[..., 2] = 1.0
    fut = torch.zeros(batch, F_STEPS, 3)
    fut[..., 0] = EGO_V * t_fut
    cur = torch.zeros(batch, 10)
    cur[:, 2] = 1.0
    cur[:, 4] = EGO_V

    # a scene starting near the beginning of a recording: the leading history is
    # all-zero, meaning "no history", not "the ego was at the origin"
    past[:, :pad_steps] = 0.0

    xs = torch.linspace(-40.0, 120.0, 20)
    line_strings = torch.zeros(batch, 2, 20, 4)
    for i, sgn in enumerate((1.0, -1.0)):
        line_strings[:, i, :, 0] = xs
        line_strings[:, i, :, 1] = sgn * border_y
        line_strings[:, i, :, 3] = 1.0  # road-border flag

    nb_past = torch.zeros(batch, N_SLOTS, P_STEPS, 11)
    nb_fut = torch.zeros(batch, N_SLOTS, F_STEPS, 4)
    for slot, nb in enumerate(neighbours):
        steps = nb.get("steps", range(T))
        w, ln = nb.get("wl", (1.8, 4.5))
        vx, vy = nb.get("v", (0.0, 0.0))
        jitter = nb.get("jitter", 0.0)
        drift = nb.get("drift", 0.0)
        for k in steps:
            # world x of the neighbour is fixed; the offsets are quoted at t=0
            x = nb["lon"] + jitter * (1.0 if k % 2 else -1.0)
            y = nb["lat"]
            if k >= P_STEPS:
                x = x + drift * (k - (P_STEPS - 1)) / F_STEPS
            row = torch.tensor([x, y, 1.0, 0.0, vx, vy, w, ln, 1.0, 0.0, 0.0])
            if k < P_STEPS:
                nb_past[:, slot, k] = row
            else:
                nb_fut[:, slot, k - P_STEPS] = row[:4]
    inputs = {
        "ego_agent_past": past,
        "ego_current_state": cur,
        "ego_shape": torch.tensor([EGO_SHAPE]).repeat(batch, 1),
        "line_strings": line_strings,
        "neighbor_agents_past": nb_past,
        "neighbor_agents_future": nb_fut,
        "goal_pose": torch.zeros(batch, 4),
        "lanes": torch.zeros(batch, 1, 20, 8),
        "route_lanes": torch.zeros(batch, 1, 20, 8),
        "polygons": torch.zeros(batch, 1, 20, 2),
        "static_objects": torch.zeros(batch, 5, 10),
    }
    return inputs, fut, t_all


def _run(aug, batch: int = 1, future=None, **scene_kw):
    """One augmentation pass on a fresh scene.

    Returns what the training loop reads: the augmenter's own return values, not the
    tensors handed in (a non-canonical future is narrowed to a fresh tensor on the way
    in, so the caller's copy is not the one that gets rewritten).
    """
    inputs, fut, _ = _scene(batch=batch, **scene_kw)
    inputs, fut, _ = aug(
        inputs, fut if future is None else future(fut, inputs), inputs["neighbor_agents_future"]
    )
    return inputs, fut


def _accepted(aug, batch, **scene_kw):
    _run(aug, batch=batch, **scene_kw)
    return aug._aug_rows.clone()


# ───────────────────────── 1. the defaults invariant ─────────────────────────


def _rand_spy(monkeypatch):
    """Record the shape of every torch.rand drawn from here on."""
    shapes = []
    real = torch.rand

    def spy(*args, **kwargs):
        shapes.append(tuple(args[0]) if args and not isinstance(args[0], int) else args)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", spy)
    return shapes


def test_new_flags_at_their_off_values_change_nothing(monkeypatch):
    """The whole point of the opt-in flags: OFF must be indistinguishable from absent.

    Same seed, same synthetic batch, plain defaults vs every new kwarg spelled out at
    its off value. Outputs must match exactly AND the generator must end in the same
    state, which is what fails the day the extra toward-parked ``torch.rand`` column
    stops being conditional.
    """
    nbr = [{"lon": 30.0, "lat": 1.6, "wl": (1.8, 4.5)}]

    shapes = _rand_spy(monkeypatch)
    a = FrenetStatePerturbationTensor(1.0, "cpu", seed=11)
    in_a, fut_a = _run(a, batch=8, neighbours=nbr)
    state_a = a.gen.get_state()

    shapes_a, _ = list(shapes), shapes.clear()
    b = FrenetStatePerturbationTensor(
        1.0, "cpu", seed=11, recovery_rounds=0, toward_parked_prob=0.0, min_clearance=0.0
    )
    in_b, fut_b = _run(b, batch=8, neighbours=nbr)
    shapes_b = list(shapes)

    assert shapes_a == shapes_b, "the off values changed how much randomness is drawn"
    assert torch.equal(state_a, b.gen.get_state())
    assert torch.equal(a._aug_rows, b._aug_rows)
    assert bool(a._aug_rows.any()), "the fixture augments nothing; the test proves nothing"
    for key in ("ego_agent_past", "ego_current_state"):
        assert torch.equal(in_a[key], in_b[key]), key
    assert torch.equal(fut_a, fut_b)


def test_toward_parked_off_draws_no_extra_column(monkeypatch):
    """Pinned separately from the output check: the RNG stream is the fragile part."""
    shapes = _rand_spy(monkeypatch)
    off = FrenetStatePerturbationTensor(1.0, "cpu", seed=3)
    _run(off, batch=2)
    width_off = [s for s in shapes if s == (2, 2 * off.n_draws + 1)]

    shapes.clear()
    on = FrenetStatePerturbationTensor(1.0, "cpu", seed=3, toward_parked_prob=0.5)
    _run(on, batch=2)
    width_on = [s for s in shapes if s == (2, 2 * on.n_draws + 2)]

    assert width_off and width_on


# ───────────────────────── 2. recovery after a veto ──────────────────────────

# A short box, valid for ONE timestep, 3.4 m ahead and 1.0 m left of the t=0 rear
# axle: outside the corridor's longitudinal window (half_l + ext_lon = 2.9 m) so it
# imposes no lateral cut, but inside the ego footprint, whose centre sits wb/2
# further forward. Exactly the ~1% of winners the exact-OBB veto exists to catch.
_VETO_NBR = [{"lon": 3.4, "lat": 1.0, "wl": (0.8, 0.8), "steps": [P_STEPS - 1]}]


def test_recovery_only_ever_adds_rows():
    """Recovery is a second chance, never a re-decision: accepted rows stay accepted."""
    flipped = 0
    for seed in range(6):
        off = _accepted(
            FrenetStatePerturbationTensor(1.0, "cpu", seed=seed), 8, neighbours=_VETO_NBR
        )
        on = _accepted(
            FrenetStatePerturbationTensor(1.0, "cpu", seed=seed, recovery_rounds=1),
            8,
            neighbours=_VETO_NBR,
        )
        assert bool((off & ~on).sum() == 0), f"seed {seed}: recovery dropped an accepted row"
        flipped += int((on & ~off).sum())
    assert flipped > 0, "the fixture vetoes nothing; the recovery path is untested"


# ───────────────────── 3. parked detection and the nudge ─────────────────────


def _parked(neighbours):
    inputs, _, _ = _scene(neighbours=neighbours)
    st, valid = time_aligned_neighbor_tracks(
        inputs["neighbor_agents_past"], inputs["neighbor_agents_future"]
    )
    return (
        FrenetStatePerturbationTensor._parked_mask(
            inputs["neighbor_agents_past"], st, valid, P_STEPS
        )[0, 0].item(),
        st,
        valid,
    )


def test_a_jittering_parked_box_is_recognised_by_its_recorded_velocity():
    """A 5 cm centroid wobble is 0.5 m/s to a one-step difference, and 0 to the data."""
    parked, st, valid = _parked([{"lon": 40.0, "lat": 2.0, "jitter": 0.05, "v": (0.0, 0.0)}])
    assert parked
    # the finite difference this replaced would have rejected the very same box
    step = (st[:, :, P_STEPS - 1, :2] - st[:, :, P_STEPS - 2, :2]).norm(dim=-1)
    assert float(step[0, 0]) / DT >= PARKED_SPEED


def test_a_vehicle_that_departs_is_not_parked():
    """Stopped at t=0 is not parked: the light turns green and the lead car leaves."""
    parked, _, _ = _parked([{"lon": 40.0, "lat": 2.0, "v": (0.2, 0.0), "drift": 30.0}])
    assert not parked


def test_a_stationary_vehicle_is_parked():
    parked, _, _ = _parked([{"lon": 40.0, "lat": 2.0, "v": (0.0, 0.0)}])
    assert parked


def _captured_dy(aug, batch, **scene_kw):
    """Run the augmenter, returning the drawn offsets it actually built profiles from."""
    seen = {}
    real = aug._candidate_profiles

    def spy(combos, dy, *args, **kwargs):
        seen["dy"] = dy.clone()
        return real(combos, dy, *args, **kwargs)

    aug._candidate_profiles = spy
    _run(aug, batch=batch, **scene_kw)
    return seen["dy"]


# a parked car 45 m ahead and 2.0 m left: reached at t ~ 4.5 s, later than the
# shortest merge, so the scene is eligible for the nudge
_PARKED_AHEAD = [{"lon": 45.0, "lat": 2.0, "jitter": 0.05, "v": (0.0, 0.0)}]


def test_toward_parked_points_every_offset_at_the_parked_car():
    """The car is on the +normal side, so a nudged scene draws only positive offsets."""
    on = FrenetStatePerturbationTensor(1.0, "cpu", seed=5, toward_parked_prob=1.0)
    dy_on = _captured_dy(on, 1, neighbours=_PARKED_AHEAD)
    assert bool((dy_on > 0).all()), dy_on

    # same augmenter, same RNG width, empty road: nothing to point at, nothing mirrored
    bare = FrenetStatePerturbationTensor(1.0, "cpu", seed=5, toward_parked_prob=1.0)
    dy_bare = _captured_dy(bare, 1)
    assert bool((dy_bare < 0).any()), "the unmirrored draw should straddle zero"
    assert torch.equal(dy_on, dy_bare.abs())


def test_toward_parked_ignores_a_moving_vehicle():
    """Same geometry, but the car drives off: no mirroring, the draw is untouched."""
    moving = [dict(_PARKED_AHEAD[0], v=(0.2, 0.0), drift=30.0)]
    on = FrenetStatePerturbationTensor(1.0, "cpu", seed=5, toward_parked_prob=1.0)
    bare = FrenetStatePerturbationTensor(1.0, "cpu", seed=5, toward_parked_prob=1.0)
    assert torch.equal(_captured_dy(on, 1, neighbours=moving), _captured_dy(bare, 1))


# ─────────────────────────── 4. the clearance floor ──────────────────────────

# 6.0 m ahead, dead centre: never overlapping, and 1.7 m clear of the ego's front
# face -- inside a 2.0 m floor but OUTSIDE the un-widened `near` prefilter, so this
# is also the regression test for the floor being silently unenforced.
_CLOSE_NBR = [{"lon": 6.0, "lat": 0.0, "wl": (0.8, 0.8), "steps": [P_STEPS - 1]}]


def test_min_clearance_rejects_a_candidate_inside_the_floor():
    loose = _accepted(FrenetStatePerturbationTensor(1.0, "cpu", seed=2), 8, neighbours=_CLOSE_NBR)
    tight = _accepted(
        FrenetStatePerturbationTensor(1.0, "cpu", seed=2, min_clearance=2.0),
        8,
        neighbours=_CLOSE_NBR,
    )
    assert bool(loose.any()), "nothing was accepted without the floor; the test proves nothing"
    assert not bool(tight.any()), "a candidate 1.7 m from a box passed a 2.0 m floor"


def test_min_clearance_leaves_the_border_corridor_alone():
    """The floor is a NEIGHBOUR rule. Kerbs are 1.6 m away and no car is present, so a
    scene that augments without the floor must augment identically with it -- widening
    the border cut instead would drop every kerb-hugging drive out of augmentation."""
    loose = _accepted(FrenetStatePerturbationTensor(1.0, "cpu", seed=4), 8, border_y=1.6)
    tight = _accepted(
        FrenetStatePerturbationTensor(1.0, "cpu", seed=4, min_clearance=2.0), 8, border_y=1.6
    )
    assert bool(loose.any()), "the tight corridor accepted nothing; the test proves nothing"
    assert torch.equal(loose, tight)


def test_min_clearance_does_not_move_the_border_bounds():
    """The same claim one level down, where the two half-widths are actually applied."""
    bounds = []
    for mc in (0.0, 2.0):
        aug = FrenetStatePerturbationTensor(1.0, "cpu", seed=4, min_clearance=mc)
        inputs, fut, _ = _scene(batch=2, border_y=1.6)
        xy = torch.cat([inputs["ego_agent_past"][..., :2], fut[..., :2]], dim=1)
        tan = torch.zeros_like(xy)
        tan[..., 0] = 1.0
        nrm = torch.stack([-tan[..., 1], tan[..., 0]], dim=-1)
        shape = inputs["ego_shape"]
        half_w = shape[:, 2] / 2 + 0.10
        bounds.append(
            aug._corridor(
                {"line_strings": inputs["line_strings"]},
                xy,
                tan,
                nrm,
                half_w,
                shape[:, 2] / 2 + max(0.10, mc),
                shape[:, 1] / 2,
                shape[:, 0],
            )
        )
    assert torch.equal(bounds[0][0], bounds[1][0])
    assert torch.equal(bounds[0][1], bounds[1][1])


# ─────────────────────────── 5. ego-history noise ────────────────────────────


def test_past_noise_scales_each_accepted_history_by_one_factor():
    """One scalar per scene, on the rewritten history and the state it implies."""
    plain = FrenetStatePerturbationTensor(1.0, "cpu", seed=7)
    in_ref, _ = _run(plain, batch=8, pad_steps=3)
    noisy = FrenetStatePerturbationTensor(1.0, "cpu", seed=7, ego_past_noise_std=0.2)
    in_new, _ = _run(noisy, batch=8, pad_steps=3)

    rows = plain._aug_rows
    assert torch.equal(rows, noisy._aug_rows), "the scale draw must not shift the decisions"
    assert bool(rows.any())

    past_ref, past_new = in_ref["ego_agent_past"], in_new["ego_agent_past"]
    cur_ref, cur_new = in_ref["ego_current_state"], in_new["ego_current_state"]

    # untouched rows train on plain ground truth, unscaled
    assert torch.equal(past_ref[~rows], past_new[~rows])
    assert torch.equal(cur_ref, cur_new), "the scale is history-only"

    for b in torch.nonzero(rows, as_tuple=True)[0].tolist():
        real = past_ref[b, :, 0].abs() > 1e-3  # the samples with a length to scale
        ratio = past_new[b, real, 0] / past_ref[b, real, 0]
        s = float(ratio[0])
        assert 1 - 2 * 0.2 <= s <= 1 + 2 * 0.2, s
        assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5), "not one factor"
        # ...and NOTHING but the history moves: the current state is an input in its own
        # right, not a summary of the history, and vx is a loss weight (see
        # _perturb_history), so scaling it would re-weight the loss rather than perturb
        assert torch.equal(cur_new[b], cur_ref[b])
        # t=0 is pinned by construction: the scale is taken about that sample, so it
        # is the SAMPLE, not the frame origin, that cannot move
        assert torch.equal(past_new[b, -1], past_ref[b, -1])
        # "no history" stays no history
        assert torch.equal(past_new[b, :3], torch.zeros(3, 4))


def test_past_noise_at_zero_changes_nothing(monkeypatch):
    shapes = _rand_spy(monkeypatch)
    off = FrenetStatePerturbationTensor(1.0, "cpu", seed=7)
    in_off, fut_off = _run(off, batch=8, pad_steps=3)
    shapes_off, _ = list(shapes), shapes.clear()

    explicit = FrenetStatePerturbationTensor(1.0, "cpu", seed=7, ego_past_noise_std=0.0)
    in_on, fut_on = _run(explicit, batch=8, pad_steps=3)

    assert shapes_off == list(shapes)
    assert torch.equal(off.gen.get_state(), explicit.gen.get_state())
    assert torch.equal(in_off["ego_agent_past"], in_on["ego_agent_past"])
    assert torch.equal(in_off["ego_current_state"], in_on["ego_current_state"])
    assert torch.equal(fut_off, fut_on)


# ────────────────────── 6. smooth ego-history jitter ─────────────────────────


def _straight_frame(batch: int):
    """A straight +x polyline with its path frame, as ``__call__`` builds them."""
    T = P_STEPS + F_STEPS
    xy = torch.zeros(batch, T, 2)
    xy[..., 0] = torch.arange(T, dtype=torch.float32) * EGO_V * DT
    tan = torch.zeros(batch, T, 2)
    tan[..., 0] = 1.0
    nrm = torch.stack([-tan[..., 1], tan[..., 0]], dim=-1)
    return xy, tan, nrm


def _jitter_offset(batch: int, seed: int = 5, **kw):
    aug = FrenetStatePerturbationTensor(1.0, "cpu", seed=seed, **kw)
    xy, tan, nrm = _straight_frame(batch)
    return aug._hist_jitter(xy, tan, nrm, P_STEPS), aug


def test_hist_jitter_is_exactly_zero_at_t0():
    """The current pose is what the model plans from; the history must not move it."""
    off, aug = _jitter_offset(64, hist_jitter_lat=0.4)
    # the basis itself, before any amplitude: every mode vanishes at u = 0
    phi = aug._hist_jitter_basis(P_STEPS, torch.device("cpu"), torch.float32)
    assert torch.equal(phi[:, -1], torch.zeros(phi.shape[0]))
    # and so does the offset actually applied, exactly -- not "to within a tolerance"
    assert torch.equal(off[:, P_STEPS - 1], torch.zeros(64, 2))
    assert torch.equal(off[:, P_STEPS:], torch.zeros(64, F_STEPS, 2)), "the future moved"
    assert float(off[:, 0].abs().max()) > 0.0, "nothing was jittered; the test proves nothing"


def test_hist_jitter_std_at_the_oldest_sample_is_the_requested_one():
    """The flag is defined as the std at the oldest sample: A = sigma / ||phi[:, 0]||."""
    sigma = 0.35
    n = 40000
    off, _ = _jitter_offset(n, hist_jitter_lat=sigma)
    emp = float(off[:, 0, 1].std())
    # Monte-Carlo error on a std from n samples is ~ sigma / sqrt(2n) = 1.2e-3 here;
    # 5% is many times that and still catches a wrong normalisation constant
    # (sqrt(2) = 1.41 or K = 3 would both be off by tens of percent).
    assert abs(emp - sigma) / sigma < 0.05, emp
    assert abs(float(off[:, 0, 0].mean())) < 1e-2, "lateral jitter moved the path tangent"


def test_hist_jitter_is_smooth_not_white():
    """Independent per-sample noise is a jagged track a model learns to ignore."""
    sigma = 0.3
    off, _ = _jitter_offset(4000, hist_jitter_lat=sigma)
    lat = off[:, :P_STEPS, 1]
    d2 = lat[:, 2:] - 2 * lat[:, 1:-1] + lat[:, :-2]
    # white noise of the same per-sample std has second-difference std sigma*sqrt(6)
    white = float(sigma * (6.0**0.5))
    assert float(d2.std()) < white / 20.0, (float(d2.std()), white)


def _spacing_ratio(**kw):
    """Mean along-path sample spacing of the jittered history, over the unjittered one."""
    xy, _, _ = _straight_frame(4000)
    off, _ = _jitter_offset(4000, **kw)
    step0 = (xy[:, 1:P_STEPS] - xy[:, : P_STEPS - 1]).norm(dim=-1)
    step1 = ((xy + off)[:, 1:P_STEPS] - (xy + off)[:, : P_STEPS - 1]).norm(dim=-1)
    return abs(float(step1.std() / step0.mean()))


def test_lateral_jitter_leaves_the_along_path_spacing_alone():
    """Bending the track sideways must not smuggle in a speed perturbation."""
    xy, _, _ = _straight_frame(4000)
    off, _ = _jitter_offset(4000, hist_jitter_lat=0.3)
    step0 = (xy[:, 1:P_STEPS] - xy[:, : P_STEPS - 1]).norm(dim=-1)
    step1 = ((xy + off)[:, 1:P_STEPS] - (xy + off)[:, : P_STEPS - 1]).norm(dim=-1)
    # a purely lateral bend only lengthens the step at second order in the bend angle
    assert abs(float(step1.mean() / step0.mean()) - 1.0) < 0.01


def test_longitudinal_jitter_is_what_moves_the_spacing():
    """The point of the second axis: the implied speed history wobbles."""
    lat_only = _spacing_ratio(hist_jitter_lat=0.3, hist_jitter_lon=0.0)
    lon_only = _spacing_ratio(hist_jitter_lat=0.0, hist_jitter_lon=0.3)
    assert lat_only < 0.01, lat_only
    assert lon_only > 20 * lat_only, (lon_only, lat_only)


def test_longitudinal_jitter_std_at_the_oldest_sample_is_the_requested_one():
    sigma = 0.35
    off, _ = _jitter_offset(40000, hist_jitter_lon=sigma)
    assert abs(float(off[:, 0, 0].std()) - sigma) / sigma < 0.05
    assert float(off[:, 0, 1].abs().max()) == 0.0, "longitudinal jitter moved the path normal"
    assert torch.equal(off[:, P_STEPS - 1], torch.zeros(40000, 2)), "t=0 moved"


def test_the_two_axes_are_drawn_independently():
    """A lateral-only run must not have its wobble mirrored along the path."""
    off, _ = _jitter_offset(20000, hist_jitter_lat=0.3, hist_jitter_lon=0.3)
    lon, lat = off[:, 0, 0], off[:, 0, 1]
    corr = float((lon * lat).mean() / (lon.std() * lat.std()))
    assert abs(corr) < 0.05, corr


def test_hist_jitter_headings_match_the_perturbed_polyline():
    """The jitter runs after the veto, so the stored cos/sin are re-derived from it."""
    aug = FrenetStatePerturbationTensor(1.0, "cpu", seed=13, hist_jitter_lat=0.3)
    inputs, _ = _run(aug, batch=8)
    rows = aug._aug_rows
    assert bool(rows.any()), "nothing was augmented; the test proves nothing"
    past = inputs["ego_agent_past"][rows]
    # interior history samples: ddt is a central difference there, and the batch-wide
    # re-centering is rigid, so the stored heading must be the polyline's own direction
    g = past[:, 2:, :2] - past[:, :-2, :2]
    stored = torch.atan2(past[:, 1:-1, 3], past[:, 1:-1, 2])
    assert torch.allclose(torch.atan2(g[..., 1], g[..., 0]), stored, atol=1e-4)
    # and the jitter really did bend it away from the straight recording
    assert float(stored.abs().max()) > 1e-3


def _normal_spy(monkeypatch):
    """Record the shape of every torch.normal drawn from here on."""
    shapes = []
    real = torch.normal

    def spy(*args, **kwargs):
        shapes.append(kwargs.get("size"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "normal", spy)
    return shapes


def test_hist_jitter_at_zero_changes_nothing(monkeypatch):
    shapes = _normal_spy(monkeypatch)
    off = FrenetStatePerturbationTensor(1.0, "cpu", seed=7)
    in_off, fut_off = _run(off, batch=8, pad_steps=3)
    shapes_off, _ = list(shapes), shapes.clear()

    explicit = FrenetStatePerturbationTensor(
        1.0, "cpu", seed=7, hist_jitter_lat=0.0, hist_jitter_lon=0.0
    )
    in_on, fut_on = _run(explicit, batch=8, pad_steps=3)

    assert shapes_off == list(shapes), "the off value changed how much randomness is drawn"
    assert torch.equal(off.gen.get_state(), explicit.gen.get_state())
    assert torch.equal(in_off["ego_agent_past"], in_on["ego_agent_past"])
    assert torch.equal(in_off["ego_current_state"], in_on["ego_current_state"])
    assert torch.equal(fut_off, fut_on)


def test_hist_jitter_refuses_a_negative_std():
    with pytest.raises(ValueError, match="hist_jitter_lat"):
        FrenetStatePerturbationTensor(1.0, "cpu", hist_jitter_lat=-0.1)
    with pytest.raises(ValueError, match="hist_jitter_lon"):
        FrenetStatePerturbationTensor(1.0, "cpu", hist_jitter_lon=-0.1)


# ───────────────────── layout robustness (3-col / 4-col) ─────────────────────


def test_a_3col_history_and_4col_future_are_accepted():
    """Scene-gen and the offline tools emit the other layout of each field."""
    ref = FrenetStatePerturbationTensor(1.0, "cpu", seed=9)
    in_ref, fut_ref = _run(ref, batch=4, neighbours=_PARKED_AHEAD)

    def to_alt_layout(fut, inputs):
        past = inputs["ego_agent_past"]
        inputs["ego_agent_past"] = torch.cat(
            [past[..., :2], torch.atan2(past[..., 3], past[..., 2])[..., None]], dim=-1
        )
        return torch.cat([fut[..., :2], fut[..., 2:3].cos(), fut[..., 2:3].sin()], dim=-1)

    alt = FrenetStatePerturbationTensor(1.0, "cpu", seed=9)
    in_alt, fut_alt = _run(alt, batch=4, future=to_alt_layout, neighbours=_PARKED_AHEAD)

    assert torch.equal(alt._aug_rows, ref._aug_rows)
    assert bool(ref._aug_rows.any()), "nothing was augmented; the test proves nothing"
    assert torch.allclose(in_alt["ego_agent_past"], in_ref["ego_agent_past"], atol=1e-5)
    assert torch.allclose(in_alt["ego_current_state"], in_ref["ego_current_state"], atol=1e-5)
    assert torch.allclose(fut_alt, fut_ref, atol=1e-5)


# ────────────── 7. both perturbations are input-only, post-veto ───────────────
#
# The property that makes an A/B between the two history perturbations valid: they are
# applied to the ACCEPTED history and nothing else, so they cannot move which scenes are
# augmented, cannot move the training target, and cannot move the pose the model plans
# from. Everything below is parametrised over both of them for exactly that reason.

_PERTURBATIONS = [
    pytest.param({"ego_past_noise_std": 0.2}, id="multiplicative"),
    pytest.param({"hist_jitter_lat": 0.3}, id="jitter_lat"),
    pytest.param({"hist_jitter_lat": 0.3, "hist_jitter_lon": 0.3}, id="jitter_lat_lon"),
]


def _clean_and_perturbed(seed=11, batch=8, pad_steps=3, **kw):
    """The same batch augmented twice: once clean, once with a history perturbation."""
    clean = FrenetStatePerturbationTensor(1.0, "cpu", seed=seed)
    in_ref, fut_ref = _run(clean, batch=batch, pad_steps=pad_steps)
    noisy = FrenetStatePerturbationTensor(1.0, "cpu", seed=seed, **kw)
    in_new, fut_new = _run(noisy, batch=batch, pad_steps=pad_steps)
    assert bool(clean._aug_rows.any()), "nothing was augmented; the test proves nothing"
    return (clean, in_ref, fut_ref), (noisy, in_new, fut_new)


@pytest.mark.parametrize("kw", _PERTURBATIONS)
def test_history_noise_cannot_move_which_scenes_are_augmented(kw):
    """THE point of applying both perturbations after the veto.

    Applied earlier they bend the history the plausibility-jerk screen and the exact-OBB
    veto judge, so turning the noise on quietly changes the training SET as well as its
    contents, and the two arms of an A/B are then not comparable.
    """
    (clean, _, _), (noisy, _, _) = _clean_and_perturbed(**kw)
    assert torch.equal(clean._aug_rows, noisy._aug_rows)


@pytest.mark.parametrize("kw", _PERTURBATIONS)
def test_history_noise_leaves_the_training_target_bit_identical(kw):
    """The future is the target: the model must learn it DESPITE the noisy history."""
    (_, _, fut_ref), (_, _, fut_new) = _clean_and_perturbed(**kw)
    assert torch.equal(fut_ref, fut_new)


@pytest.mark.parametrize("kw", _PERTURBATIONS)
def test_history_noise_leaves_the_t0_history_sample_bit_identical(kw):
    """Pose AND heading: the last history sample is the pose the model plans from."""
    (_, in_ref, _), (_, in_new, _) = _clean_and_perturbed(**kw)
    assert torch.equal(in_ref["ego_agent_past"][:, -1], in_new["ego_agent_past"][:, -1])
    # ...and something upstream of it really did move
    assert not torch.equal(in_ref["ego_agent_past"], in_new["ego_agent_past"])


@pytest.mark.parametrize("kw", _PERTURBATIONS)
def test_history_noise_leaves_the_current_state_bit_identical(kw):
    """Every arm perturbs what the encoder sees and NOTHING else.

    The current state is not a summary of the history the model could catch out --
    ego_agent_past is (x, y, cos, sin) and carries no velocity at all. Of the state,
    only the pose and vx are read anywhere, and vx is not an encoder input either: it is
    the divisor of the longitudinal loss weight in decoder.py. Scaling it would make the
    multiplicative arm a test of history noise AND of loss re-weighting at once, which
    the jitter arms have no counterpart for.
    """
    (_, in_ref, _), (_, in_new, _) = _clean_and_perturbed(**kw)
    assert torch.equal(in_ref["ego_current_state"], in_new["ego_current_state"])


@pytest.mark.parametrize("kw", _PERTURBATIONS)
def test_stored_history_headings_describe_the_perturbed_track(kw):
    """The bug the post-veto rewrite fixes: cos/sin used to be left describing the clean
    positions while the positions beside them had moved."""
    # no padded prefix here: a "no history" slot is held at exactly zero, so a
    # difference taken ACROSS it is a phantom step and describes nothing
    (_, _, _), (noisy, in_new, _) = _clean_and_perturbed(pad_steps=0, **kw)
    past = in_new["ego_agent_past"][noisy._aug_rows]
    # interior samples: ddt is a plain central difference there, and _headings keeps the
    # motion direction wherever the step is longer than 0.3 m (it is, at EGO_V)
    g = past[:, 2:, :2] - past[:, :-2, :2]
    stored = torch.atan2(past[:, 1:-1, 3], past[:, 1:-1, 2])
    assert torch.allclose(torch.atan2(g[..., 1], g[..., 0]), stored, atol=1e-4)
