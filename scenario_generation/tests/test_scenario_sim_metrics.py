"""A scenario_sim row has to survive the aggregator it is built for."""

import numpy as np

from scenario_generation.closed_loop_eval import aggregate
from scenario_generation.scenario_sim_metrics import build_segment_row


def _row() -> dict:
    return build_segment_row(
        n_steps_run=3,
        terminated="goal",
        result_kind="Pass",
        clearances=[5.0, 4.0, 6.0],
        collisions=[False, False, False],
        rb_dists=np.array([2.0, 2.5, 3.0]),
        accels=np.array([0.0, -0.5, -0.2]),
        near_miss_thresh=0.5,
        strong_brake_mps2=-2.5,
        progress_m=12.0,
    )


def test_a_scenario_sim_row_aggregates():
    summary = aggregate([_row()], 1.0)

    assert summary["n_segments"] == 1
    assert summary["total_steps"] == 3


def test_off_path_collisions_report_as_never_measured():
    """Zero would claim the scenario had none; there is no recorded path to be off."""
    assert _row()["deviation_collision"]["measured"] is False
    assert aggregate([_row()], 1.0)["deviation_collision"]["count"] == 0
