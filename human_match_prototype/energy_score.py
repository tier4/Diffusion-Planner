"""Per-scene Energy Score: 64 planner samples scored against one human trajectory."""

import numpy as np
from scipy.spatial.distance import cdist

HORIZONS = {"2s": 20, "4s": 40, "8s": 80}


def per_scene_energy_score(
    human_xy: np.ndarray,
    samples_xy: np.ndarray,
    horizons: dict[str, int] = HORIZONS,
) -> dict[str, float]:
    """Compute per-scene Energy Score at multiple horizons.

    Args:
        human_xy: (T, 2) human trajectory [x, y] in ego frame.
        samples_xy: (N, T, 2) planner sample trajectories.
        horizons: mapping of name -> number of timesteps.

    Returns:
        dict with keys es_obs_{h}, es_div_{h}, es_{h} for each horizon h.
        ES_h = obs_h - 0.5 * div_h where:
          obs_h = mean_m ||X_m[:h] - y[:h]||_2  (flattened trajectory norm)
          div_h = mean_{m!=n} ||X_m[:h] - X_n[:h]||_2  (distinct-pair mean)
    """
    N = len(samples_xy)
    out: dict[str, float] = {}

    for name, h in horizons.items():
        y_flat = human_xy[:h].reshape(1, -1)  # (1, h*2)
        x_flat = samples_xy[:, :h].reshape(N, -1)  # (N, h*2)

        obs = float(cdist(x_flat, y_flat).mean())

        pw = cdist(x_flat, x_flat)
        np.fill_diagonal(pw, 0.0)
        div = float(pw.sum() / (N * (N - 1)))

        out[f"es_obs_{name}"] = obs
        out[f"es_div_{name}"] = div
        out[f"es_{name}"] = obs - 0.5 * div

    return out
