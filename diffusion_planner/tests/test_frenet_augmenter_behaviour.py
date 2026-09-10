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


def test_the_toward_parked_flag_does_not_shift_the_corridor_stream(monkeypatch):
    """The corridor draw must be the same WIDTH whether the nudge is on or off.

    It used to be one column wider when the nudge was on, taken from the same
    generator, so enabling the flag shifted every later corridor and merge draw and
    changed which scenes were accepted -- even on a corpus where no scene is eligible
    to be nudged. An A/B on the flag then compared a reseed, not the feature. The coin
    now comes off a dedicated generator, so the shared stream is flag-independent.
    """
    shapes = _rand_spy(monkeypatch)
    off = FrenetStatePerturbationTensor(1.0, "cpu", seed=3)
    _run(off, batch=2)
    off_widths = {s for s in shapes if len(s) == 2 and s[0] == 2}

    shapes.clear()
    on = FrenetStatePerturbationTensor(1.0, "cpu", seed=3, toward_parked_prob=0.5)
    _run(on, batch=2)
    on_widths = {s for s in shapes if len(s) == 2 and s[0] == 2}

    corridor = (2, 2 * off.n_draws + 1)
    assert corridor in off_widths, f"corridor draw not seen with the nudge off: {off_widths}"
    assert corridor in on_widths, f"corridor draw not seen with the nudge on: {on_widths}"
    assert (2, 2 * off.n_draws + 2) not in on_widths, (
        "the nudge is back to widening the shared corridor draw"
    )


def test_the_toward_parked_flag_cannot_change_which_scenes_are_augmented():
    """The consequence of the above, asserted end to end.

    On a scene with no parked vehicle in reach the nudge can never fire, so switching
    it on must leave the accepted set and the augmented state untouched.
    """
    off = FrenetStatePerturbationTensor(1.0, "cpu", seed=3)
    in_off, fut_off = _run(off, batch=16)
    on = FrenetStatePerturbationTensor(1.0, "cpu", seed=3, toward_parked_prob=1.0)
    in_on, fut_on = _run(on, batch=16)

    assert bool(off._aug_rows.any()), "nothing was augmented; the test proves nothing"
    assert torch.equal(off._aug_rows, on._aug_rows), "the flag moved the accepted set"
    assert torch.equal(in_off["ego_current_state"], in_on["ego_current_state"])
    assert torch.equal(fut_off, fut_on)


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
                # the full input dict: _corridor now REQUIRES the neighbour tensors, and
                # passing a stripped dict is what used to silently give an unvetoed,
                # neighbour-free corridor. The scene here has no neighbours anyway, so
                # the border claim is unaffected.
                inputs,
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


def _scaled_pair(std=0.2, batch=8, pad_steps=3, seed=7):
    """The same batch augmented twice, with and without the history scale.

    Returns the shared accepted-row mask plus both (past, current_state) pairs, so each
    property below can be one short assertion instead of a loop with six of them.
    """
    plain = FrenetStatePerturbationTensor(1.0, "cpu", seed=seed)
    in_ref, _ = _run(plain, batch=batch, pad_steps=pad_steps)
    noisy = FrenetStatePerturbationTensor(1.0, "cpu", seed=seed, ego_past_noise_std=std)
    in_new, _ = _run(noisy, batch=batch, pad_steps=pad_steps)
    rows = plain._aug_rows
    assert torch.equal(rows, noisy._aug_rows), "the scale draw must not shift the decisions"
    assert bool(rows.any()), "nothing was augmented; the tests prove nothing"
    return rows, in_ref, in_new


def _row_ratios(rows, in_ref, in_new):
    """Per accepted row, the elementwise new/old ratio over the samples with a length."""
    past_ref, past_new = in_ref["ego_agent_past"], in_new["ego_agent_past"]
    for b in torch.nonzero(rows, as_tuple=True)[0].tolist():
        real = past_ref[b, :, 0].abs() > 1e-3  # the samples with a length to scale
        yield b, past_new[b, real, 0] / past_ref[b, real, 0]


def test_past_noise_is_one_factor_per_scene():
    """One scalar per scene, applied to the whole rewritten history."""
    rows, in_ref, in_new = _scaled_pair()
    for b, ratio in _row_ratios(rows, in_ref, in_new):
        assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5), (
            f"row {b}: not one factor"
        )


def test_past_noise_factor_stays_within_two_sigma():
    std = 0.2
    rows, in_ref, in_new = _scaled_pair(std=std)
    for b, ratio in _row_ratios(rows, in_ref, in_new):
        s = float(ratio[0])
        assert 1 - 2 * std <= s <= 1 + 2 * std, f"row {b}: {s} outside +-2 sigma"


def test_past_noise_leaves_the_current_state_alone():
    """``ego_current_state`` is an input in its own right, not a summary of the history,
    and its vx is a LOSS WEIGHT (see _perturb_history) -- scaling it would re-weight the
    scene's loss rather than perturb what the encoder reads."""
    rows, in_ref, in_new = _scaled_pair()
    assert torch.equal(in_ref["ego_current_state"], in_new["ego_current_state"])


def test_past_noise_leaves_unaugmented_rows_on_plain_ground_truth():
    rows, in_ref, in_new = _scaled_pair()
    assert torch.equal(in_ref["ego_agent_past"][~rows], in_new["ego_agent_past"][~rows])


def test_past_noise_pins_t0_and_the_zero_padding():
    """t=0 is pinned by construction: the scale is about that SAMPLE, not the frame
    origin. And "no history" has to stay no history."""
    rows, in_ref, in_new = _scaled_pair()
    past_ref, past_new = in_ref["ego_agent_past"], in_new["ego_agent_past"]
    for b in torch.nonzero(rows, as_tuple=True)[0].tolist():
        assert torch.equal(past_new[b, -1], past_ref[b, -1]), f"row {b}: t=0 moved"
        assert torch.equal(past_new[b, :3], torch.zeros(3, 4)), f"row {b}: padding written"


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
    off, aug = _jitter_offset(64, ego_past_noise_std=0.4, ego_past_noise_mode="jitter")
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
    off, _ = _jitter_offset(n, ego_past_noise_std=sigma, ego_past_noise_mode="jitter")
    emp = float(off[:, 0, 1].std())
    # Monte-Carlo error on a std from n samples is ~ sigma / sqrt(2n) = 1.2e-3 here;
    # 5% is many times that and still catches a wrong normalisation constant
    # (sqrt(2) = 1.41 or K = 3 would both be off by tens of percent).
    assert abs(emp - sigma) / sigma < 0.05, emp
    assert abs(float(off[:, 0, 0].mean())) < 1e-2, "lateral jitter moved the path tangent"


def test_hist_jitter_is_smooth_not_white():
    """Independent per-sample noise is a jagged track a model learns to ignore."""
    sigma = 0.3
    off, _ = _jitter_offset(4000, ego_past_noise_std=sigma, ego_past_noise_mode="jitter")
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
    off, _ = _jitter_offset(4000, ego_past_noise_std=0.3, ego_past_noise_mode="jitter")
    step0 = (xy[:, 1:P_STEPS] - xy[:, : P_STEPS - 1]).norm(dim=-1)
    step1 = ((xy + off)[:, 1:P_STEPS] - (xy + off)[:, : P_STEPS - 1]).norm(dim=-1)
    # a purely lateral bend only lengthens the step at second order in the bend angle
    assert abs(float(step1.mean() / step0.mean()) - 1.0) < 0.01


def test_there_is_no_longitudinal_axis():
    """The lon axis was measured and removed; the flag must not quietly come back.

    It made the arm the worst of the campaign in all four eval cells on both metrics,
    where lateral alone was the best on recovery.
    """
    aug = FrenetStatePerturbationTensor(
        1.0, "cpu", seed=5, ego_past_noise_std=0.3, ego_past_noise_mode="jitter"
    )
    assert not hasattr(aug, "hist_jitter_lon"), "the longitudinal knob came back"
    assert not hasattr(aug, "hist_jitter_lat"), "the separate jitter magnitude came back"
    with pytest.raises(TypeError):
        FrenetStatePerturbationTensor(1.0, "cpu", hist_jitter_lon=0.3)

    # The axis is still LISTED at a hard-zero std, on purpose: the coefficient draw is
    # shaped (B, len(axes), K), so dropping it would shift the generator stream and the
    # lateral jitter would give different numbers for the same seed.
    _, tan, nrm = _straight_frame(2)
    axes = aug._hist_jitter_axes(tan, nrm)
    assert len(axes) == 2, "the draw shape changed; the lateral jitter is no longer reproducible"
    assert axes[1][0] == 0.0, "the along-path axis has a non-zero std again"

    # and it really contributes nothing along the path
    off, _ = _jitter_offset(4000, ego_past_noise_std=0.3, ego_past_noise_mode="jitter")
    assert float(off[:, 0, 0].abs().max()) == 0.0, "the zero axis moved the path tangent"


def test_hist_jitter_headings_match_the_perturbed_polyline():
    """The jitter runs after the veto, so the stored cos/sin are re-derived from it."""
    aug = FrenetStatePerturbationTensor(
        1.0, "cpu", seed=13, ego_past_noise_std=0.3, ego_past_noise_mode="jitter"
    )
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

    explicit = FrenetStatePerturbationTensor(1.0, "cpu", seed=7, ego_past_noise_std=0.0)
    in_on, fut_on = _run(explicit, batch=8, pad_steps=3)

    assert shapes_off == list(shapes), "the off value changed how much randomness is drawn"
    assert torch.equal(off.gen.get_state(), explicit.gen.get_state())
    assert torch.equal(in_off["ego_agent_past"], in_on["ego_agent_past"])
    assert torch.equal(in_off["ego_current_state"], in_on["ego_current_state"])
    assert torch.equal(fut_off, fut_on)


def test_hist_jitter_refuses_a_negative_std():
    with pytest.raises(ValueError, match="ego_past_noise_std"):
        FrenetStatePerturbationTensor(1.0, "cpu", ego_past_noise_std=-0.1)


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
        # goal_pose has the same two layouts. This augmenter never reads it, but the
        # inherited centric_transform rotates its cols 2:4, so a 3-col tensor reaching
        # that far raises -- which is what happened before the canonicalisation was
        # added to the frenet __call__.
        goal = inputs["goal_pose"]
        inputs["goal_pose"] = torch.cat(
            [goal[..., :2], torch.atan2(goal[..., 3], goal[..., 2])[..., None]], dim=-1
        )
        return torch.cat([fut[..., :2], fut[..., 2:3].cos(), fut[..., 2:3].sin()], dim=-1)

    alt = FrenetStatePerturbationTensor(1.0, "cpu", seed=9)
    in_alt, fut_alt = _run(alt, batch=4, future=to_alt_layout, neighbours=_PARKED_AHEAD)

    assert torch.equal(alt._aug_rows, ref._aug_rows)
    assert bool(ref._aug_rows.any()), "nothing was augmented; the test proves nothing"
    assert torch.allclose(in_alt["ego_agent_past"], in_ref["ego_agent_past"], atol=1e-5)
    assert torch.allclose(in_alt["ego_current_state"], in_ref["ego_current_state"], atol=1e-5)
    assert torch.allclose(fut_alt, fut_ref, atol=1e-5)
    assert in_alt["goal_pose"].shape[-1] == 4, "a 3-col goal_pose was not widened"
    assert torch.allclose(in_alt["goal_pose"], in_ref["goal_pose"], atol=1e-5)


# ────────────── 7. both perturbations are input-only, post-veto ───────────────
#
# The property that makes an A/B between the two history perturbations valid: they are
# applied to the ACCEPTED history and nothing else, so they cannot move which scenes are
# augmented, cannot move the training target, and cannot move the pose the model plans
# from. Everything below is parametrised over both of them for exactly that reason.

_PERTURBATIONS = [
    pytest.param({"ego_past_noise_std": 0.2}, id="multiplicative"),
    pytest.param({"ego_past_noise_std": 0.3, "ego_past_noise_mode": "jitter"}, id="jitter"),
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


# ──────────────── 8. the toward-parked fallback and the floor mask ────────────────
# Regression cover for the three fixes in a00a98588 / a3a77f888 / 74609f900. The
# defect they answer was silent: the augmenter kept working and simply stopped
# augmenting the scene family the flag exists for.

_TIGHT_PASS = [{"lon": 25.0, "lat": 1.8, "v": (0.0, 0.0)}]


def test_toward_parked_does_not_delete_the_scenes_it_exists_to_harden():
    """The merge gate can strike out every horizon on a tight pass.

    Applying it unconditionally took this family from 91/128 accepted to 0/128 --
    the flag deleted its own target. The fallback keeps the row on its ungated
    candidates instead. It is a floor, not a repair: most rows are still lost to
    the exact footprint check (a 2.0 m ego does not fit a 1.8 m gap), so this
    asserts survival, not recovery.
    """
    off = _accepted(FrenetStatePerturbationTensor(1.0, "cpu", seed=7), 128, neighbours=_TIGHT_PASS)
    on = _accepted(
        FrenetStatePerturbationTensor(1.0, "cpu", seed=7, toward_parked_prob=1.0),
        128,
        neighbours=_TIGHT_PASS,
    )
    assert int(off.sum()) > 0, "nothing was accepted with the nudge off; the test proves nothing"
    assert int(on.sum()) > 0, (
        f"the toward-parked gate deleted the whole tight-pass family: {int(on.sum())}/128 "
        f"accepted with the nudge on vs {int(off.sum())}/128 without it"
    )


def test_a_row_that_fell_back_is_not_counted_as_hardened():
    """`hardened` is what the recovery path keys on, so it must exclude fallbacks.

    If a fallback row were reported as hardened, the first selection would take
    first-feasible while the retry took largest-offset -- an un-hardened row trained
    as if its merge were gated in front of the vehicle.
    """
    aug = FrenetStatePerturbationTensor(1.0, "cpu", seed=7, toward_parked_prob=1.0)
    seen = {}
    real = FrenetStatePerturbationTensor._toward_parked_select

    def spy(self, admissible, merges, dy, toward, toward_any, t_obs, P):
        out = real(self, admissible, merges, dy, toward, toward_any, t_obs, P)
        if toward_any:
            seen["toward"], seen["hardened"] = toward.clone(), out[3].clone()
        return out

    FrenetStatePerturbationTensor._toward_parked_select = spy
    try:
        _run(aug, batch=128, neighbours=_TIGHT_PASS)
    finally:
        FrenetStatePerturbationTensor._toward_parked_select = real

    assert seen, "the toward-parked branch never ran; the test proves nothing"
    assert bool((seen["toward"] & ~seen["hardened"]).any()), (
        "no row fell back on a scene built to empty the gate -- if the geometry changed, "
        "pick a tighter pass, otherwise the fallback is dead code"
    )
    assert bool((seen["hardened"] & ~seen["toward"]).sum() == 0), (
        "a row is hardened without being toward-parked"
    )


def test_the_clearance_floor_is_off_by_default_and_builds_no_mask():
    """The default path must not pay for, or be changed by, a feature that is off."""
    aug = FrenetStatePerturbationTensor(1.0, "cpu", seed=3)
    xy = torch.zeros(2, 5, 2)
    assert aug._floor_mask(xy + 1.0, xy) is None, "a mask was built with the floor off"


def test_the_floor_applies_only_where_the_candidate_left_the_recording():
    """A timestep identical to ground truth is judged at threshold 0, not at the floor.

    Without this, a scene is rejected for the clearance of the drive that was actually
    recorded -- the augmenter vetoing reality.
    """
    aug = FrenetStatePerturbationTensor(1.0, "cpu", seed=3, min_clearance=2.0)
    xy = torch.zeros(1, 4, 2)
    aug_xy = xy.clone()
    aug_xy[0, 2:] = 1.0  # departs only at the last two timesteps
    mask = aug._floor_mask(aug_xy, xy)
    assert mask is not None
    assert mask.tolist() == [[False, False, True, True]]


def test_recovery_cannot_see_the_toward_set_at_all():
    """Guard for the one invariant `_recover_vetoed` exists to hold.

    A fallback row took first-feasible in the first selection, so its retry must too;
    a genuinely gated row took largest-offset, so its retry must too. Keying the retry
    on `toward` instead of `hardened` breaks that, and no behavioural test caught it
    because the two sets differ on only a handful of rows.

    Rather than assert the selection indirectly, this removes the means: the recovery
    is not given `toward`, so keying on it cannot compile without also re-adding the
    parameter -- a visible edit rather than a one-word swap. The second assertion keeps
    the first from going vacuous by checking the two sets really do differ here.
    """
    import inspect

    params = list(inspect.signature(FrenetStatePerturbationTensor._recover_vetoed).parameters)
    assert "hardened" in params, "the recovery no longer takes the hardened set"
    assert "toward" not in params, (
        "`toward` is back in _recover_vetoed's signature -- it is the wrong set to key "
        "the retry on (a fallback row would retry with largest-offset after being "
        "selected with first-feasible); the recovery must only see `hardened`"
    )

    seen = {}
    real_sel = FrenetStatePerturbationTensor._toward_parked_select

    def sel_spy(self, admissible, merges, dy, toward, toward_any, t_obs, P):
        out = real_sel(self, admissible, merges, dy, toward, toward_any, t_obs, P)
        if toward_any:
            seen["toward"], seen["hardened"] = toward.clone(), out[3].clone()
        return out

    FrenetStatePerturbationTensor._toward_parked_select = sel_spy
    try:
        aug = FrenetStatePerturbationTensor(
            1.0, "cpu", seed=7, toward_parked_prob=1.0, recovery_rounds=2
        )
        _run(aug, batch=128, neighbours=_TIGHT_PASS)
    finally:
        FrenetStatePerturbationTensor._toward_parked_select = real_sel

    assert seen, "the toward-parked branch never ran; the test proves nothing"
    assert bool((seen["toward"] != seen["hardened"]).any()), (
        "toward and hardened coincide on this scene, so the guard above is vacuous -- "
        "pick a tighter pass or the distinction has stopped being reachable"
    )


# ──────────── generator streams must be disjoint across (rank, stream) ────────────


def _augmenter_at_rank(monkeypatch, rank, seed=3407):
    """Build the augmenter as DDP rank ``rank`` sees it, without a real process group."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    return FrenetStatePerturbationTensor(1.0, "cpu", seed=seed)


def test_no_generator_state_collides_across_ranks_or_streams(monkeypatch):
    """Every (rank, stream) pair must start from its own state.

    `seed + rank`, `+ rank + 1`, `+ rank + 2` separated the streams within a rank but
    made the per-rank blocks overlap, so rank 0's history generator started exactly
    where rank 1's main generator did (and rank 0's toward matched both rank 1's
    history and rank 2's main). Ranks in a multi-GPU job then drew correlated
    perturbations -- the opposite of what splitting the streams is for.
    """
    states = {}
    for rank in range(4):
        aug = _augmenter_at_rank(monkeypatch, rank)
        for name in ("gen", "hist_gen", "toward_gen"):
            key = bytes(getattr(aug, name).get_state().numpy().tobytes())
            assert key not in states, (
                f"rank {rank} {name} starts in the same state as {states[key]}"
            )
            states[key] = f"rank {rank} {name}"
    assert len(states) == 12


def test_the_main_stream_still_matches_the_unpatched_seeding(monkeypatch):
    """The main generator must stay `seed + rank`.

    That is what tier4-main uses, so changing it would alter the default path at every
    rank above 0 even with every new flag off. Only the ADDED streams are namespaced.
    """
    for rank in (0, 1, 5):
        aug = _augmenter_at_rank(monkeypatch, rank, seed=11)
        expected = torch.Generator(device="cpu").manual_seed(11 + rank)
        assert torch.equal(aug.gen.get_state(), expected.get_state())


def test_a_missing_neighbour_tensor_raises_instead_of_silently_disabling_the_veto():
    """Dropping a mandatory key used to be a silent no-op with severe consequences.

    Without the neighbour tensors the corridor gets no neighbour cuts, `_nbr_st` stays
    None so the exact footprint veto returns early, and `_nbr_lo` stays None so no scene
    is toward-parked eligible. A harness that popped `neighbor_agents_future` out of the
    dict -- which the positional argument invites -- therefore measured an empty world
    and reported 0 vetoes and 0 eligible scenes with no error.
    """
    aug = FrenetStatePerturbationTensor(1.0, "cpu", seed=3)
    inputs, fut, nbf = _scene(batch=2)
    stripped = {k: v for k, v in inputs.items() if k != "neighbor_agents_future"}
    with pytest.raises(KeyError, match="neighbor_agents_future"):
        aug(stripped, fut, nbf)

    for key in ("line_strings", "neighbor_agents_past"):
        aug2 = FrenetStatePerturbationTensor(1.0, "cpu", seed=3)
        inp, f2, n2 = _scene(batch=2)
        with pytest.raises(KeyError, match=key):
            aug2({k: v for k, v in inp.items() if k != key}, f2, n2)
