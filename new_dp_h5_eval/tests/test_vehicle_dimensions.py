"""Native H5 is scored with the real wheelbase, not base_link_to_front.

Covers the per-project vehicle table the wheelbase is read from.
"""

import json

import numpy as np
import pytest

from new_dp_h5_eval.vehicle import (
    DEFAULT_VEHICLE_PARAMS,
    VehicleDimensions,
    load_vehicle_params,
    sim_ego_shape,
)

# Erga as in dataset/data_converter_param.json of the meta repository.
ERGA = {
    "ego_wheel_base": 4.9901,
    "ego_front_overhang": 2.5667,
    "ego_length": 10.8812,
    "ego_width": 2.8736,
}
ERGA_NATIVE = [7.5568, 10.8812, 2.8736]  # what the ML-Planner converter writes
ERGA_SIM = [4.9901, 10.8812, 2.8736]


def _write_params(path, entries):
    path.write_text(json.dumps(entries))
    return path


def test_load_vehicle_params_skips_entries_without_ego_dimensions(tmp_path):
    params = _write_params(
        tmp_path / "params.json",
        {"default": {"min_distance": 50.0}, "erga": {**ERGA, "min_distance": 50.0}},
    )
    vehicles = load_vehicle_params(params)
    assert list(vehicles) == ["erga"]
    assert vehicles["erga"] == VehicleDimensions(4.9901, 2.5667, 10.8812, 2.8736)
    assert vehicles["erga"].base_link_to_front == pytest.approx(7.5568)


def test_load_vehicle_params_rejects_a_table_without_any_vehicle(tmp_path):
    params = _write_params(tmp_path / "params.json", {"default": {"min_distance": 50.0}})
    with pytest.raises(ValueError, match="no project entry"):
        load_vehicle_params(params)


@pytest.mark.parametrize(
    "entries, message",
    [([ERGA], "json object"), ({"erga": {**ERGA, "ego_wheel_base": None}}, "non-numeric")],
)
def test_load_vehicle_params_reports_a_malformed_table(tmp_path, entries, message):
    with pytest.raises(ValueError, match=message):
        load_vehicle_params(_write_params(tmp_path / "params.json", entries))


def test_sim_ego_shape_replaces_base_link_to_front_with_the_wheelbase():
    vehicles = {"erga": VehicleDimensions(4.9901, 2.5667, 10.8812, 2.8736)}
    shape = sim_ego_shape(np.array([ERGA_NATIVE] * 2), "erga", vehicles, source="shard")
    assert shape.dtype == np.float32
    np.testing.assert_allclose(shape, ERGA_SIM, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "native, project, message",
    [
        ([4.9901, 10.8812, 2.8736], "erga", "disagree"),  # already a wheelbase
        ([7.5568, 10.0, 2.8736], "erga", "disagree"),  # converted with another length
        (ERGA_NATIVE, "x2_dev", "no ego dimensions"),
    ],
)
def test_sim_ego_shape_rejects_a_shard_that_does_not_match_the_table(native, project, message):
    vehicles = {"erga": VehicleDimensions(4.9901, 2.5667, 10.8812, 2.8736)}
    with pytest.raises(ValueError, match=message):
        sim_ego_shape(np.array([native]), project, vehicles, source="shard")


def test_default_vehicle_params_is_the_meta_repository_table():
    assert DEFAULT_VEHICLE_PARAMS.parts[-2:] == ("dataset", "data_converter_param.json")
