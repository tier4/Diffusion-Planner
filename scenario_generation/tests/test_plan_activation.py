from types import SimpleNamespace

import numpy as np

from scenario_generation.reproducer_rollout import (
    _attach_committed_prefix,
    _PlanSchedule,
)


def _plan(base: float, length: int = 8):
    xy = np.column_stack(
        [base + np.arange(length, dtype=np.float32), np.zeros(length, dtype=np.float32)]
    )
    heading = np.zeros(length, dtype=np.float32)
    return xy, heading


def test_pending_plans_activate_at_absolute_ticks_even_when_replan_is_faster():
    schedule = _PlanSchedule()
    xy0, h0 = _plan(0.0)
    schedule.enqueue(born_step=0, world_xy=xy0, world_heading=h0, delay_step=2, immediate=True)
    np.testing.assert_array_equal(
        schedule.reference(1, 2, strict_horizon=True)[0][:, 0], [1.0, 2.0]
    )

    xy1, h1 = _plan(100.0)
    schedule.enqueue(born_step=1, world_xy=xy1, world_heading=h1, delay_step=2)
    # Tick 2 still consumes plan 0; tick 3 switches to plan 1 row 2.
    np.testing.assert_array_equal(
        schedule.reference(2, 2, strict_horizon=True)[0][:, 0], [2.0, 102.0]
    )

    xy2, h2 = _plan(200.0)
    schedule.enqueue(born_step=2, world_xy=xy2, world_heading=h2, delay_step=2)
    np.testing.assert_array_equal(
        schedule.reference(3, 2, strict_horizon=True)[0][:, 0], [102.0, 202.0]
    )


def _model_args(*, velocity=False):
    normalizer = SimpleNamespace(
        mean=np.array([[[10.0, 20.0, 0.5, -0.5]], [[0.0, 0.0, 0.0, 0.0]]]),
        std=np.array([[[2.0, 4.0, 0.5, 0.25]], [[1.0, 1.0, 1.0, 1.0]]]),
    )
    return SimpleNamespace(
        state_normalizer=normalizer,
        predicted_neighbor_num=1,
        future_len=5,
        use_velocity_representation=velocity,
    )


def _state(schedule, *, k=0, velocity=False):
    return SimpleNamespace(
        delay_step=2,
        prefix_step=2,
        plan_schedule=schedule,
        k=k,
        live_pose=np.array([0.0, 0.0, 0.0]),
    )


def test_first_replan_has_zero_delay_fallback():
    scene = {}
    effective = _attach_committed_prefix(scene, _state(_PlanSchedule()), _model_args())

    assert effective == 0
    assert scene["delay"].tolist() == [0]
    assert not scene["sampled_trajectories"].any()


def test_prefix_is_live_ego_frame_then_state_normalized():
    schedule = _PlanSchedule()
    xy, heading = _plan(1.0)
    schedule.enqueue(born_step=0, world_xy=xy, world_heading=heading, delay_step=2, immediate=True)
    scene = {}

    effective = _attach_committed_prefix(scene, _state(schedule, k=1), _model_args())

    assert effective == 2
    assert scene["delay"].tolist() == [2]
    # At tick 1 the committed targets are world x=2 and x=3. With identity
    # ego pose those are already ego-frame positions, then ego state stats apply.
    expected = np.array(
        [
            [(2.0 - 10.0) / 2.0, (0.0 - 20.0) / 4.0, (1.0 - 0.5) / 0.5, (0.0 + 0.5) / 0.25],
            [(3.0 - 10.0) / 2.0, (0.0 - 20.0) / 4.0, (1.0 - 0.5) / 0.5, (0.0 + 0.5) / 0.25],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(scene["sampled_trajectories"][0, 0, 1:3], expected)
    assert not scene["sampled_trajectories"][0, 1].any()


def test_velocity_representation_uses_per_tick_displacements_before_normalization():
    schedule = _PlanSchedule()
    xy = np.array([[1.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    heading = np.zeros(3, dtype=np.float32)
    schedule.enqueue(born_step=0, world_xy=xy, world_heading=heading, delay_step=2, immediate=True)
    args = _model_args(velocity=True)
    # Identity normalization makes the displacement contract direct to inspect.
    args.state_normalizer.mean[0, 0] = 0.0
    args.state_normalizer.std[0, 0] = 1.0
    scene = {}

    _attach_committed_prefix(scene, _state(schedule), args)

    np.testing.assert_array_equal(
        scene["sampled_trajectories"][0, 0, 1:3, :2],
        [[1.0, 0.0], [2.0, 0.0]],
    )


def test_prefix_off_leaves_model_input_untouched_while_plan_is_still_delayed():
    schedule = _PlanSchedule()
    xy0, h0 = _plan(0.0)
    schedule.enqueue(born_step=0, world_xy=xy0, world_heading=h0, delay_step=2, immediate=True)
    state = _state(schedule, k=2)
    state.prefix_step = 0
    np_dict = {}
    assert _attach_committed_prefix(np_dict, state, _model_args()) == 0
    assert np_dict == {}
    # The simulator schedule is untouched by the model-input setting: the plan born at
    # tick 0 is still the one executing at tick 2.
    np.testing.assert_array_equal(schedule.reference(2, 1, strict_horizon=True)[0][:, 0], [2.0])
