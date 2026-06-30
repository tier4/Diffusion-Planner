import torch

from planner_metrics.pdms_proxy import synthetic_epdms


_METRICS = {
    "no_at_fault_collision": 0.5,
    "drivable_area_compliance": 1.0,
    "driving_direction_compliance": 0.5,
    "traffic_light_compliance": 1.0,
    "ego_progress": 0.8,
    "time_to_collision_within_bound": 1.0,
    "lane_keeping": 0.0,
    "history_comfort": 1.0,
    "extended_comfort": 0.0,
}


def _snapshot(values):
    out = {}
    for key, value in values.items():
        out[key] = torch.tensor([value], dtype=torch.float32)
        out[f"{key}_available"] = torch.tensor([1.0], dtype=torch.float32)
    return out


def test_synthetic_epdms_matches_autoware_cpp_aggregation():
    agent = _snapshot(_METRICS)
    human = _snapshot({**_METRICS, "no_at_fault_collision": 1.0, "lane_keeping": 0.0, "extended_comfort": 1.0})

    result = synthetic_epdms(agent, human)

    raw_weighted = (5.0 * 0.8 + 5.0 * 1.0 + 2.0 * 0.0 + 2.0 * 1.0 + 2.0 * 0.0) / 16.0
    raw_multiplier = 0.5 * 1.0 * 0.5 * 1.0
    human_filtered_weighted = (
        5.0 * 0.8 + 5.0 * 1.0 + 2.0 * 1.0 + 2.0 * 1.0 + 2.0 * 0.0
    ) / 16.0

    assert torch.allclose(result.raw_available, torch.tensor([1.0]))
    assert torch.allclose(result.raw_multiplicative_metrics_prod, torch.tensor([raw_multiplier]))
    assert torch.allclose(result.raw_weighted_metrics, torch.tensor([raw_weighted]))
    assert torch.allclose(result.raw, torch.tensor([raw_multiplier * raw_weighted]))
    assert torch.allclose(result.human_filtered_available, torch.tensor([1.0]))
    assert torch.allclose(result.human_filtered_weighted_metrics, torch.tensor([human_filtered_weighted]))


def test_synthetic_epdms_is_unavailable_when_any_required_subscore_is_unavailable():
    agent = _snapshot(_METRICS)
    agent["traffic_light_compliance_available"] = torch.tensor([0.0])

    result = synthetic_epdms(agent)

    assert torch.allclose(result.raw_available, torch.tensor([0.0]))
