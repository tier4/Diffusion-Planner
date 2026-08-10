from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from human_match_prototype.score_scenes import score_one_scene

EXPECTED_ES_KEYS = [
    "es_obs_2s",
    "es_div_2s",
    "es_2s",
    "es_obs_4s",
    "es_div_4s",
    "es_4s",
    "es_obs_8s",
    "es_div_8s",
    "es_8s",
]
EXPECTED_FRENET_KEYS = [
    "es_lon_2s",
    "es_lon_4s",
    "es_lon_8s",
    "es_lat_2s",
    "es_lat_4s",
    "es_lat_8s",
]
EXPECTED_QA_KEYS = [
    "route_valid",
    "max_segment_gap",
    "total_interpolated_gap",
    "n_valid_segments",
    "route_arc_length",
    "route_coverage_insufficient",
    "human_max_proj_dist",
    "n_monotonic_violations",
    "frac_planner_proj_fail",
]


def _make_npz(tmp_path: Path, with_route: bool = True) -> str:
    """Create a minimal NPZ with human trajectory and route_lanes."""
    npz_path = tmp_path / "test_frame.npz"
    T = 80
    human = np.zeros((T, 3), dtype=np.float32)
    human[:, 0] = np.linspace(0, 40, T)  # straight +x

    route_lanes = np.zeros((25, 20, 33), dtype=np.float32)
    if with_route:
        for seg_i in range(3):
            x0 = seg_i * 19.0
            route_lanes[seg_i, :, 0] = np.linspace(x0, x0 + 19, 20)

    np.savez(npz_path, ego_agent_future=human, route_lanes=route_lanes)
    return str(npz_path)


def _mock_sampler(npz_path: str, num_samples: int = 64, seed: int = 0, temperature: float = 1.0):
    """Return a SampleResult-like object with samples near the human."""
    data = np.load(npz_path)
    human = data["ego_agent_future"][:, :3].astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.5, (num_samples, 80, 3)).astype(np.float32)
    samples = np.tile(human, (num_samples, 1, 1)) + noise
    result = MagicMock()
    result.ego_samples = samples
    result.human_future = human
    return result


class TestScoreOneScene:
    def test_all_keys_present(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=True)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        assert "npz_path" in row
        for k in EXPECTED_ES_KEYS:
            assert k in row, f"missing {k}"
        for k in EXPECTED_QA_KEYS:
            assert k in row, f"missing {k}"

    def test_valid_route_has_frenet(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=True)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        if row["route_valid"]:
            for k in EXPECTED_FRENET_KEYS:
                assert not np.isnan(row[k]), f"{k} should not be NaN with valid route"

    def test_no_route_frenet_nan(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=False)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        for k in EXPECTED_FRENET_KEYS:
            assert np.isnan(row[k]), f"{k} should be NaN with no route"

    def test_es_values_reasonable(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=True)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        # With small noise, obs should be positive and finite
        for h in ["2s", "4s", "8s"]:
            assert 0 < row[f"es_obs_{h}"] < 1000
            assert row[f"es_div_{h}"] >= 0
