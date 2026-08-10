import numpy as np
import pytest

from human_match_prototype.energy_score import per_scene_energy_score

HORIZONS = {"2s": 20, "4s": 40, "8s": 80}


def _straight_line(v: float = 5.0, T: int = 80, dt: float = 0.1) -> np.ndarray:
    """(T, 2) straight trajectory at constant speed v m/s along +x."""
    t = np.arange(T) * dt
    return np.stack([v * t, np.zeros(T)], axis=-1)


class TestPerSceneEnergyScore:
    def test_identical_samples_zero_divergence(self):
        """When all samples == human, obs ≈ 0, div ≈ 0, es ≈ 0."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert result[f"es_obs_{h}"] == pytest.approx(0.0, abs=1e-6)
            assert result[f"es_div_{h}"] == pytest.approx(0.0, abs=1e-6)
            assert result[f"es_{h}"] == pytest.approx(0.0, abs=1e-6)

    def test_shifted_samples_positive_obs(self):
        """Samples offset from human should have positive obs term."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        samples[:, :, 1] += 2.0  # shift 2m laterally
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert result[f"es_obs_{h}"] > 0.0
            # All samples identical -> div ≈ 0, es ≈ obs
            assert result[f"es_div_{h}"] == pytest.approx(0.0, abs=1e-6)
            assert result[f"es_{h}"] == pytest.approx(result[f"es_obs_{h}"], rel=1e-4)

    def test_diverse_samples_positive_diversity(self):
        """Spread-out samples should have positive div term."""
        rng = np.random.default_rng(42)
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1)) + rng.normal(0, 1.0, (64, 80, 2))
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert result[f"es_div_{h}"] > 0.0

    def test_output_keys_complete(self):
        """All 9 expected keys are present."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert f"es_obs_{h}" in result
            assert f"es_div_{h}" in result
            assert f"es_{h}" in result
        assert len(result) == 9

    def test_longer_horizon_geq_shorter(self):
        """For offset samples, obs at 4s >= obs at 2s (more points, larger norm)."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        samples[:, :, 1] += 1.0
        result = per_scene_energy_score(human, samples, HORIZONS)
        assert result["es_obs_4s"] >= result["es_obs_2s"]
        assert result["es_obs_8s"] >= result["es_obs_4s"]
