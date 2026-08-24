import pytest

from scenario_generation.closed_loop_delay import (
    NAMED_PLANT_PARAMETERS,
    PlantParameters,
    resolve_plan_delay,
    resolve_plant_parameters,
    validate_delay_options,
)


def _valid(**overrides):
    values = {
        "timeline_progress_mode": "pose",
        "world_delay_mode": "none",
        "k_lag": 0,
        "delay_step": 0,
        "tracker_mode": "perfect",
        "neighbor_history_mode": "recorded",
        "replan_interval": 4,
        "controller_compensation": False,
    }
    values.update(overrides)
    validate_delay_options(**values)


def test_default_delay_contract_is_inert_and_valid():
    _valid()


@pytest.mark.parametrize("mode", ["lag_raw", "lag_extrapolated"])
def test_world_delay_requires_clock_recorded_and_positive_lag(mode):
    _valid(timeline_progress_mode="clock", world_delay_mode=mode, k_lag=3)
    with pytest.raises(ValueError, match="timeline_progress_mode='clock'"):
        _valid(world_delay_mode=mode, k_lag=3)
    with pytest.raises(ValueError, match="neighbor_history_mode='recorded'"):
        _valid(
            timeline_progress_mode="clock",
            world_delay_mode=mode,
            k_lag=3,
            neighbor_history_mode="sim",
        )
    with pytest.raises(ValueError, match="requires k_lag > 0"):
        _valid(timeline_progress_mode="clock", world_delay_mode=mode, k_lag=0)


def test_none_world_mode_rejects_hidden_lag():
    with pytest.raises(ValueError, match="k_lag must be 0"):
        _valid(k_lag=1)


def test_delay_step_is_limited_to_training_range():
    _valid(delay_step=5, tracker_mode="mpc")
    with pytest.raises(ValueError, match="plan_dead_time_step must be <= 5"):
        _valid(delay_step=6, tracker_mode="mpc")


def test_plan_dead_time_and_prefix_resolve_from_legacy_knob():
    assert resolve_plan_delay(None, None, None) == (0, 0)
    assert resolve_plan_delay(2, None, None) == (2, 2)
    assert resolve_plan_delay(2, None, 0) == (2, 0)
    assert resolve_plan_delay(None, 3, 1) == (3, 1)
    assert resolve_plan_delay(2, 4, None) == (4, 2)


def test_prefix_cannot_exceed_plan_dead_time():
    _valid(plan_dead_time_step=2, prefix_step=0, tracker_mode="delayed")
    _valid(plan_dead_time_step=2, prefix_step=2, tracker_mode="delayed")
    with pytest.raises(ValueError, match="prefix_step must be <= plan_dead_time_step"):
        _valid(plan_dead_time_step=1, prefix_step=2, tracker_mode="delayed")
    with pytest.raises(ValueError, match="prefix_step must be <= plan_dead_time_step"):
        _valid(prefix_step=1, tracker_mode="delayed")
    with pytest.raises(ValueError, match="prefix_step must be >= 0"):
        _valid(plan_dead_time_step=2, prefix_step=-1, tracker_mode="delayed")


def test_plan_delay_rejects_discontinuous_perfect_tracking():
    for tracker_mode in ("mpc", "mpc_batched", "delayed"):
        _valid(delay_step=2, tracker_mode=tracker_mode)
    with pytest.raises(ValueError, match="instantaneous pose jumps"):
        _valid(delay_step=2, tracker_mode="perfect")


def test_controller_compensation_is_delayed_tracker_only():
    _valid(tracker_mode="delayed", controller_compensation=True)
    with pytest.raises(ValueError, match="requires tracker_mode='delayed'"):
        _valid(controller_compensation=True)


def test_named_plant_parameters_match_the_fixed_contract():
    assert resolve_plant_parameters("official") == PlantParameters(0.17, 0.15, 0.10, 0.10)
    assert resolve_plant_parameters("measured") == PlantParameters(0.06, 0.03, 0.19, 0.25)
    assert set(NAMED_PLANT_PARAMETERS) == {"official", "measured"}


def test_custom_plant_parameters_require_all_values_and_nonnegative():
    custom = resolve_plant_parameters(
        "custom",
        steer_dead_time_s=0.2,
        steer_time_constant_s=0.3,
        accel_dead_time_s=0.4,
        accel_time_constant_s=0.5,
    )
    assert custom == PlantParameters(0.2, 0.3, 0.4, 0.5)
    with pytest.raises(ValueError, match="require all four"):
        resolve_plant_parameters("custom", steer_dead_time_s=0.1)
    with pytest.raises(ValueError, match="must be >= 0"):
        resolve_plant_parameters(
            "custom",
            steer_dead_time_s=-0.1,
            steer_time_constant_s=0.3,
            accel_dead_time_s=0.4,
            accel_time_constant_s=0.5,
        )


def test_named_set_rejects_silent_partial_override():
    with pytest.raises(ValueError, match="require plant_parameter_set='custom'"):
        resolve_plant_parameters("official", steer_dead_time_s=0.2)
