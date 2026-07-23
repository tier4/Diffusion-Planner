# Dataset Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 6 independent experiments that characterize dataset difficulty, rarity, redundancy, outliers, and cluster structure for the Diffusion-Planner trajectory prediction dataset.

**Architecture:** A new `dataset_curation/` workspace member containing standalone scripts (one per experiment) that consume a JSON list of NPZ paths and produce analysis outputs (parquet, PNG, JSON). Experiments 1-3 work on hand-crafted features (CPU-only). Experiments 4-6 work on frozen encoder embeddings (GPU). The two tracks share no code except a common NPZ loading utility.

**Tech Stack:** scikit-learn (Isolation Forest, LOF, k-means), lightgbm, shap, umap-learn, pandas, pyarrow, matplotlib, seaborn, torch (for encoder inference)

## Global Constraints

- Python `>=3.10,<3.11` (matches workspace)
- All scripts follow the existing CLI pattern: `argparse`, `--scenes` for JSON scene list, `--output_dir` for results
- NPZ loading uses raw `np.load()` for feature extraction (no model needed), or `preference_optimization.utils.load_npz_data()` when model inference is required
- Tests use synthetic NPZ files in `tmp_path`, no real data required
- Working directory: `../Diffusion-Planner-dataset-characterization` (worktree on branch `dataset-characterization`)

---

## File Structure

```
dataset_curation/
├── pyproject.toml
├── __init__.py
├── features.py              # Experiment 1: feature extraction from NPZ
├── outlier_detection.py      # Experiment 2: Isolation Forest + LOF
├── difficulty_scoring.py     # Experiment 3: LightGBM difficulty classifier
├── embedding_extractor.py    # Experiment 4: frozen encoder embedding extraction
├── clustering.py             # Experiment 4: k-means + UMAP visualization
├── semdedup.py               # Experiment 5: SemDeDup on embeddings
├── prototypicality.py        # Experiment 6: rarity scoring via centroid distance
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared fixtures (synthetic NPZ factory)
    ├── test_features.py
    ├── test_outlier_detection.py
    ├── test_difficulty_scoring.py
    ├── test_embedding_extractor.py
    ├── test_clustering.py
    ├── test_semdedup.py
    └── test_prototypicality.py
```

---

### Task 1: Workspace Member Scaffolding

**Files:**
- Create: `dataset_curation/pyproject.toml`
- Create: `dataset_curation/__init__.py`
- Create: `dataset_curation/tests/__init__.py`
- Create: `dataset_curation/tests/conftest.py`
- Modify: `pyproject.toml` (root — add workspace member)

**Interfaces:**
- Consumes: nothing
- Produces: `make_synthetic_npz(path, *, ego_speed, heading_change, n_neighbors, has_traffic_light)` fixture factory; importable `dataset_curation` package

- [ ] **Step 1: Create `dataset_curation/pyproject.toml`**

```toml
[project]
name = "dataset-curation"
version = "0.1.0"
description = "Dataset characterization experiments for Diffusion-Planner"
requires-python = ">=3.10,<3.11"
dependencies = [
    "diffusion-planner",
    "numpy>=1.26,<2",
    "torch",
    "pandas>=2.0",
    "pyarrow",
    "scikit-learn>=1.3",
    "lightgbm>=4.0",
    "shap>=0.43",
    "umap-learn>=0.5",
    "matplotlib>=3.7",
    "seaborn>=0.13",
]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = [
    "dataset_curation",
    "dataset_curation.tests",
]
package-dir = { "dataset_curation" = "." }

[tool.uv.sources]
diffusion-planner = { workspace = true }
```

- [ ] **Step 2: Create `dataset_curation/__init__.py`**

```python
```

(Empty file.)

- [ ] **Step 3: Register in root `pyproject.toml`**

Add `"dataset_curation"` to `[tool.uv.workspace].members` list.

Add `dataset-curation = { workspace = true }` to `[tool.uv.sources]`.

Add `"dataset-curation"` to `[project].dependencies`.

- [ ] **Step 4: Create test fixtures in `dataset_curation/tests/conftest.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def make_npz(tmp_path: Path):
    """Factory fixture that creates a synthetic NPZ with controllable properties."""
    counter = 0

    def _make(
        *,
        ego_speed: float = 5.0,
        heading_change_deg: float = 0.0,
        n_neighbors: int = 3,
        has_traffic_light: bool = False,
        seed: int | None = None,
    ) -> Path:
        nonlocal counter
        rng = np.random.default_rng(seed if seed is not None else counter)
        counter += 1

        t_past = 31
        t_future = 80

        heading_rad = np.deg2rad(heading_change_deg)
        headings = np.linspace(0, heading_rad, t_future).astype(np.float32)
        dt = 0.1
        xs = np.cumsum(np.cos(headings) * ego_speed * dt).astype(np.float32)
        ys = np.cumsum(np.sin(headings) * ego_speed * dt).astype(np.float32)
        ego_future = np.stack([xs, ys, headings], axis=-1)

        ego_past = np.zeros((t_past, 4), dtype=np.float32)
        for i in range(t_past):
            t = -(t_past - 1 - i) * dt
            ego_past[i] = [t * ego_speed, 0.0, 1.0, 0.0]

        ego_current = np.array(
            [0.0, 0.0, 1.0, 0.0, ego_speed, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        nbr = np.zeros((320, t_past, 11), dtype=np.float32)
        for i in range(min(n_neighbors, 320)):
            offset_x = rng.uniform(5, 20)
            offset_y = rng.uniform(-5, 5)
            nbr[i, :, 0] = offset_x
            nbr[i, :, 1] = offset_y
            nbr[i, :, 2] = 1.0  # cos(0)
            nbr[i, :, 6] = 2.0  # width
            nbr[i, :, 7] = 4.5  # length
            nbr[i, :, 8] = 1.0  # is_vehicle

        lanes = np.zeros((140, 20, 33), dtype=np.float32)
        for seg in range(10):
            for pt in range(20):
                x = seg * 5.0 + pt * 0.25
                lanes[seg, pt, 0] = x
                lanes[seg, pt, 2] = 1.0  # dX
                lanes[seg, pt, 4] = -1.75  # LB_X offset
                lanes[seg, pt, 6] = 1.75  # RB_X offset
                if has_traffic_light:
                    lanes[seg, pt, 10] = 1.0  # red light

        lanes_speed_limit = np.full((140, 1), 13.9, dtype=np.float32)
        lanes_has_speed_limit = np.ones((140, 1), dtype=np.float32)

        route_lanes = np.zeros((25, 20, 33), dtype=np.float32)
        for seg in range(5):
            for pt in range(20):
                route_lanes[seg, pt, 0] = seg * 10.0 + pt * 0.5

        route_speed_limit = np.full((25, 1), 13.9, dtype=np.float32)
        route_has_speed_limit = np.ones((25, 1), dtype=np.float32)

        goal = np.array([xs[-1], ys[-1], np.cos(headings[-1]), np.sin(headings[-1])],
                        dtype=np.float32)

        ego_shape = np.array([2.75, 5.0, 2.0], dtype=np.float32)

        static_objects = np.zeros((5, 10), dtype=np.float32)
        polygons = np.zeros((10, 40, 3), dtype=np.float32)
        line_strings = np.zeros((60, 20, 4), dtype=np.float32)
        turn_indicators = np.zeros((30,), dtype=np.float32)

        path = tmp_path / f"scene_{counter:04d}.npz"
        np.savez(
            path,
            ego_agent_past=ego_past,
            ego_current_state=ego_current,
            ego_agent_future=ego_future,
            neighbor_agents_past=nbr,
            neighbor_agents_future=np.zeros((320, t_future, 3), dtype=np.float32),
            lanes=lanes,
            lanes_speed_limit=lanes_speed_limit,
            lanes_has_speed_limit=lanes_has_speed_limit,
            route_lanes=route_lanes,
            route_lanes_speed_limit=route_speed_limit,
            route_lanes_has_speed_limit=route_has_speed_limit,
            goal_pose=goal,
            ego_shape=ego_shape,
            static_objects=static_objects,
            polygons=polygons,
            line_strings=line_strings,
            turn_indicators=turn_indicators,
        )
        return path

    return _make


@pytest.fixture
def scene_list_json(tmp_path: Path, make_npz):
    """Create a JSON file listing N synthetic NPZ paths."""

    def _make(n: int = 20, **kwargs) -> Path:
        paths = [str(make_npz(**kwargs)) for _ in range(n)]
        json_path = tmp_path / "scenes.json"
        json_path.write_text(json.dumps(paths))
        return json_path

    return _make
```

- [ ] **Step 5: Create `dataset_curation/tests/__init__.py`**

```python
```

(Empty file.)

- [ ] **Step 6: Sync and verify**

Run:
```bash
cd /home/chenglin/workspace/Diffusion-Planner-dataset-characterization
uv sync
python -c "import dataset_curation; print('OK')"
```
Expected: no errors, prints `OK`.

- [ ] **Step 7: Verify fixtures work**

Run:
```bash
python -m pytest dataset_curation/tests/conftest.py --co -q
```
Expected: no collection errors (conftest fixtures are not collected as tests, but this verifies the file parses).

- [ ] **Step 8: Commit**

```bash
git add dataset_curation/ pyproject.toml
git commit -m "feat: scaffold dataset_curation workspace member with test fixtures"
```

---

### Task 2: Feature Extraction (Experiment 1)

**Files:**
- Create: `dataset_curation/features.py`
- Create: `dataset_curation/tests/test_features.py`

**Interfaces:**
- Consumes: NPZ files via `np.load()`
- Produces:
  - `extract_features(npz_path: str) -> dict[str, float]` — returns a flat dict of named features from one NPZ
  - `extract_features_batch(scene_paths: list[str], n_workers: int = 4) -> pd.DataFrame` — returns a DataFrame with one row per scene, columns = feature names, index = npz_path
  - CLI: `python -m dataset_curation.features --scenes <json> --output_dir <dir>` — writes `features.parquet` + histogram PNGs

- [ ] **Step 1: Write the failing tests**

Create `dataset_curation/tests/test_features.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


EXPECTED_FEATURE_NAMES = [
    "ego_speed",
    "ego_accel",
    "ego_yaw_rate",
    "ego_speed_past_mean",
    "ego_speed_past_std",
    "ego_speed_past_max",
    "ego_accel_past_mean",
    "ego_accel_past_std",
    "travel_distance",
    "endpoint_displacement",
    "heading_change_deg",
    "max_curvature",
    "path_straightness",
    "n_active_neighbors",
    "closest_neighbor_dist",
    "neighbor_vehicle_ratio",
    "neighbor_ped_ratio",
    "neighbor_bike_ratio",
    "mean_lane_curvature",
    "n_traffic_light_segments",
    "speed_limit_mean",
    "speed_limit_std",
    "goal_distance",
    "route_curvature",
    "route_length",
]


def test_extract_features_returns_dict(make_npz):
    from dataset_curation.features import extract_features

    npz_path = make_npz(ego_speed=10.0, heading_change_deg=30.0, n_neighbors=5)
    result = extract_features(str(npz_path))
    assert isinstance(result, dict)
    for name in EXPECTED_FEATURE_NAMES:
        assert name in result, f"Missing feature: {name}"
    for v in result.values():
        assert isinstance(v, float), f"Feature value should be float, got {type(v)}"


def test_extract_features_speed_matches_input(make_npz):
    from dataset_curation.features import extract_features

    npz_path = make_npz(ego_speed=12.0, heading_change_deg=0.0)
    result = extract_features(str(npz_path))
    assert abs(result["ego_speed"] - 12.0) < 1.0


def test_extract_features_heading_change(make_npz):
    from dataset_curation.features import extract_features

    straight = extract_features(str(make_npz(heading_change_deg=0.0)))
    turning = extract_features(str(make_npz(heading_change_deg=45.0)))
    assert turning["heading_change_deg"] > straight["heading_change_deg"]


def test_extract_features_neighbor_count(make_npz):
    from dataset_curation.features import extract_features

    few = extract_features(str(make_npz(n_neighbors=2)))
    many = extract_features(str(make_npz(n_neighbors=10)))
    assert few["n_active_neighbors"] < many["n_active_neighbors"]


def test_extract_features_traffic_light(make_npz):
    from dataset_curation.features import extract_features

    no_tl = extract_features(str(make_npz(has_traffic_light=False)))
    with_tl = extract_features(str(make_npz(has_traffic_light=True)))
    assert no_tl["n_traffic_light_segments"] == 0.0
    assert with_tl["n_traffic_light_segments"] > 0.0


def test_extract_features_batch(make_npz, tmp_path):
    from dataset_curation.features import extract_features_batch

    paths = [str(make_npz(ego_speed=5.0 + i)) for i in range(5)]
    json_path = tmp_path / "list.json"
    json_path.write_text(json.dumps(paths))

    df = extract_features_batch(paths, n_workers=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert set(EXPECTED_FEATURE_NAMES).issubset(set(df.columns))
    assert list(df.index) == paths


def test_extract_features_handles_corrupt_npz(tmp_path):
    from dataset_curation.features import extract_features

    bad_path = tmp_path / "bad.npz"
    bad_path.write_bytes(b"not a npz")
    with pytest.raises(Exception):
        extract_features(str(bad_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest dataset_curation/tests/test_features.py -v`
Expected: all FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `dataset_curation/features.py`**

```python
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _ego_speed_from_state(state: np.ndarray) -> float:
    return float(np.hypot(state[4], state[5]))


def _heading_from_past(past: np.ndarray) -> np.ndarray:
    return np.arctan2(past[:, 3], past[:, 2])


def _curvature(xy: np.ndarray) -> np.ndarray:
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return np.abs(dx * ddy - dy * ddx) / denom


def extract_features(npz_path: str) -> dict[str, float]:
    d = np.load(npz_path, allow_pickle=True)

    state = d["ego_current_state"]  # (10,)
    past = d["ego_agent_past"]  # (31, 4)
    future = d["ego_agent_future"]  # (80, 3)
    nbr = d["neighbor_agents_past"]  # (N, 31, 11)
    lanes = d["lanes"]  # (140, 20, 33)
    lanes_sl = d["lanes_speed_limit"]  # (140, 1)
    route = d["route_lanes"]  # (25, 20, 33)
    goal = d["goal_pose"]  # (4,) or (3,)

    ego_speed = _ego_speed_from_state(state)
    ego_accel = float(np.hypot(state[6], state[7]))
    ego_yaw_rate = float(abs(state[9]))

    past_dx = np.diff(past[:, 0])
    past_dy = np.diff(past[:, 1])
    past_speeds = np.hypot(past_dx, past_dy) / 0.1
    nonzero_mask = past_speeds > 0.01

    fut_xy = future[:, :2]
    diffs = np.diff(fut_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    travel_distance = float(seg_lengths.sum())
    endpoint_disp = float(np.hypot(fut_xy[-1, 0] - fut_xy[0, 0],
                                   fut_xy[-1, 1] - fut_xy[0, 1]))
    headings = future[:, 2]
    heading_change = float(np.rad2deg(abs(headings[-1] - headings[0])))
    curv = _curvature(fut_xy)
    max_curvature = float(curv.max()) if len(curv) > 0 else 0.0
    path_straightness = float(endpoint_disp / max(travel_distance, 1e-6))

    active_mask = np.any(np.abs(nbr[:, -1, :2]) > 0.01, axis=-1)
    n_active = int(active_mask.sum())
    if n_active > 0:
        dists = np.hypot(nbr[active_mask, -1, 0], nbr[active_mask, -1, 1])
        closest_dist = float(dists.min())
        n_veh = float(nbr[active_mask, -1, 8].sum())
        n_ped = float(nbr[active_mask, -1, 9].sum())
        n_bike = float(nbr[active_mask, -1, 10].sum())
        total_typed = max(n_veh + n_ped + n_bike, 1.0)
        veh_ratio = n_veh / total_typed
        ped_ratio = n_ped / total_typed
        bike_ratio = n_bike / total_typed
    else:
        closest_dist = 999.0
        veh_ratio = ped_ratio = bike_ratio = 0.0

    lane_active = np.any(np.abs(lanes[:, :, :2]) > 0.01, axis=(1, 2))
    active_lanes = lanes[lane_active]
    if len(active_lanes) > 0:
        lane_dirs = active_lanes[:, :, 2:4]
        lane_headings = np.arctan2(lane_dirs[:, :, 1], lane_dirs[:, :, 0])
        per_seg_curvature = np.std(lane_headings, axis=1)
        mean_lane_curv = float(per_seg_curvature.mean())
    else:
        mean_lane_curv = 0.0

    tl_active = lanes[:, :, 8:13]
    has_tl = np.any(tl_active[:, :, :3] > 0.5, axis=(1, 2))
    n_tl_segments = float(has_tl.sum())

    active_sl_mask = lanes_sl[:, 0] > 0.01
    if active_sl_mask.any():
        sl_vals = lanes_sl[active_sl_mask, 0]
        sl_mean = float(sl_vals.mean())
        sl_std = float(sl_vals.std())
    else:
        sl_mean = sl_std = 0.0

    goal_dist = float(np.hypot(goal[0], goal[1]))

    route_active = np.any(np.abs(route[:, :, :2]) > 0.01, axis=(1, 2))
    active_route = route[route_active]
    if len(active_route) > 0:
        route_pts = active_route[:, :, :2].reshape(-1, 2)
        nonzero_pts = route_pts[np.any(np.abs(route_pts) > 0.01, axis=1)]
        if len(nonzero_pts) > 2:
            rd = np.diff(nonzero_pts, axis=0)
            route_length_val = float(np.hypot(rd[:, 0], rd[:, 1]).sum())
            route_h = np.arctan2(rd[:, 1], rd[:, 0])
            route_curv = float(np.abs(np.diff(route_h)).sum())
        else:
            route_length_val = route_curv = 0.0
    else:
        route_length_val = route_curv = 0.0

    return {
        "ego_speed": ego_speed,
        "ego_accel": ego_accel,
        "ego_yaw_rate": ego_yaw_rate,
        "ego_speed_past_mean": float(past_speeds[nonzero_mask].mean()) if nonzero_mask.any() else 0.0,
        "ego_speed_past_std": float(past_speeds[nonzero_mask].std()) if nonzero_mask.any() else 0.0,
        "ego_speed_past_max": float(past_speeds.max()),
        "ego_accel_past_mean": float(np.abs(np.diff(past_speeds)).mean()),
        "ego_accel_past_std": float(np.abs(np.diff(past_speeds)).std()),
        "travel_distance": travel_distance,
        "endpoint_displacement": endpoint_disp,
        "heading_change_deg": heading_change,
        "max_curvature": max_curvature,
        "path_straightness": path_straightness,
        "n_active_neighbors": float(n_active),
        "closest_neighbor_dist": closest_dist,
        "neighbor_vehicle_ratio": veh_ratio,
        "neighbor_ped_ratio": ped_ratio,
        "neighbor_bike_ratio": bike_ratio,
        "mean_lane_curvature": mean_lane_curv,
        "n_traffic_light_segments": n_tl_segments,
        "speed_limit_mean": sl_mean,
        "speed_limit_std": sl_std,
        "goal_distance": goal_dist,
        "route_curvature": route_curv,
        "route_length": route_length_val,
    }


def extract_features_batch(
    scene_paths: list[str], n_workers: int = 4
) -> pd.DataFrame:
    if n_workers <= 1:
        rows = {p: extract_features(p) for p in scene_paths}
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(extract_features, scene_paths))
        rows = dict(zip(scene_paths, results))
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "npz_path"
    return df


def _plot_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_cols = 5
    n_features = len(df.columns)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[i], kde=True, bins=50)
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_histograms.png", dpi=150)
    plt.close(fig)

    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, ax=ax2, cmap="coolwarm", center=0, annot=False,
                xticklabels=True, yticklabels=True)
    ax2.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig2.savefig(output_dir / "feature_correlation.png", dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Extract hand-crafted features from NPZ scenes")
    parser.add_argument("--scenes", required=True, help="JSON file with list of NPZ paths")
    parser.add_argument("--output_dir", required=True, help="Directory to write results")
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print(f"Extracting features from {len(scene_paths)} scenes...")
    df = extract_features_batch(scene_paths, n_workers=args.n_workers)
    df.to_parquet(output_dir / "features.parquet")

    print("Generating visualizations...")
    _plot_histograms(df, output_dir)

    print(f"Summary statistics:")
    print(df.describe().to_string())
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest dataset_curation/tests/test_features.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dataset_curation/features.py dataset_curation/tests/test_features.py
git commit -m "feat: add feature extraction (experiment 1)"
```

---

### Task 3: Outlier Detection (Experiment 2)

**Files:**
- Create: `dataset_curation/outlier_detection.py`
- Create: `dataset_curation/tests/test_outlier_detection.py`

**Interfaces:**
- Consumes: `extract_features_batch()` from `dataset_curation.features` → `pd.DataFrame`
- Produces:
  - `detect_outliers(df: pd.DataFrame, method: str = "isolation_forest", contamination: float = 0.05) -> pd.Series` — returns boolean Series (True = outlier), indexed by npz_path
  - CLI: `python -m dataset_curation.outlier_detection --features <parquet> --output_dir <dir>` — writes `outlier_flags.parquet` + scatter PNGs

- [ ] **Step 1: Write the failing tests**

Create `dataset_curation/tests/test_outlier_detection.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_feature_df(n: int = 100, n_outliers: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n - n_outliers, 10))
    outliers = rng.normal(0, 1, (n_outliers, 10)) + 10.0  # far from cluster
    data = np.vstack([normal, outliers])
    paths = [f"/fake/scene_{i:04d}.npz" for i in range(n)]
    cols = [f"feat_{i}" for i in range(10)]
    return pd.DataFrame(data, index=paths, columns=cols)


def test_isolation_forest_returns_series():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df()
    result = detect_outliers(df, method="isolation_forest", contamination=0.05)
    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert len(result) == len(df)


def test_isolation_forest_detects_outliers():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df(n=200, n_outliers=10)
    result = detect_outliers(df, method="isolation_forest", contamination=0.1)
    n_flagged = result.sum()
    assert n_flagged > 0, "Should flag at least some outliers"
    last_10_flagged = result.iloc[-10:].sum()
    assert last_10_flagged > 5, f"Most of the injected outliers should be flagged, got {last_10_flagged}"


def test_lof_returns_series():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df()
    result = detect_outliers(df, method="lof", contamination=0.05)
    assert isinstance(result, pd.Series)
    assert result.dtype == bool


def test_lof_detects_outliers():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df(n=200, n_outliers=10)
    result = detect_outliers(df, method="lof", contamination=0.1)
    last_10_flagged = result.iloc[-10:].sum()
    assert last_10_flagged > 5


def test_invalid_method_raises():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df(n=20)
    with pytest.raises(ValueError, match="Unknown method"):
        detect_outliers(df, method="nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest dataset_curation/tests/test_outlier_detection.py -v`
Expected: all FAIL

- [ ] **Step 3: Implement `dataset_curation/outlier_detection.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def detect_outliers(
    df: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.05,
) -> pd.Series:
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    if method == "isolation_forest":
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = model.fit_predict(X)
    elif method == "lof":
        model = LocalOutlierFactor(contamination=contamination, n_neighbors=20, n_jobs=-1)
        preds = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'isolation_forest' or 'lof'.")

    return pd.Series(preds == -1, index=df.index, name=f"outlier_{method}")


def _plot_outlier_scatter(df: pd.DataFrame, flags: pd.Series, output_dir: Path, method: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_features = df.std().nlargest(6).index.tolist()
    n_pairs = min(len(top_features) * (len(top_features) - 1) // 2, 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    pair_idx = 0
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            if pair_idx >= len(axes):
                break
            ax = axes[pair_idx]
            normal = ~flags
            ax.scatter(df.loc[normal, top_features[i]], df.loc[normal, top_features[j]],
                       s=5, alpha=0.3, label="normal")
            ax.scatter(df.loc[flags, top_features[i]], df.loc[flags, top_features[j]],
                       s=20, c="red", alpha=0.8, label="outlier")
            ax.set_xlabel(top_features[i])
            ax.set_ylabel(top_features[j])
            ax.legend(fontsize=7)
            pair_idx += 1
    for k in range(pair_idx, len(axes)):
        axes[k].set_visible(False)
    plt.suptitle(f"Outlier Detection: {method}")
    plt.tight_layout()
    fig.savefig(output_dir / f"outlier_scatter_{method}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Detect outliers in feature matrix")
    parser.add_argument("--features", required=True, help="Parquet file from experiment 1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features)

    results = pd.DataFrame(index=df.index)
    for method in ["isolation_forest", "lof"]:
        flags = detect_outliers(df, method=method, contamination=args.contamination)
        results[f"outlier_{method}"] = flags
        n_flagged = flags.sum()
        print(f"{method}: {n_flagged}/{len(df)} flagged ({100*n_flagged/len(df):.1f}%)")
        _plot_outlier_scatter(df, flags, output_dir, method)

    results["outlier_both"] = results["outlier_isolation_forest"] & results["outlier_lof"]
    results.to_parquet(output_dir / "outlier_flags.parquet")

    both_count = results["outlier_both"].sum()
    print(f"\nFlagged by both methods: {both_count}")
    flagged_paths = results.index[results["outlier_both"]].tolist()
    if flagged_paths:
        import json
        with open(output_dir / "outliers_both.json", "w") as f:
            json.dump(flagged_paths[:50], f, indent=2)
        print(f"Top {min(50, len(flagged_paths))} outlier paths saved to outliers_both.json")
        print("Inspect with: python -m scenario_generation.visualize <path>")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest dataset_curation/tests/test_outlier_detection.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dataset_curation/outlier_detection.py dataset_curation/tests/test_outlier_detection.py
git commit -m "feat: add outlier detection with Isolation Forest and LOF (experiment 2)"
```

---

### Task 4: LightGBM Difficulty Scoring (Experiment 3)

**Files:**
- Create: `dataset_curation/difficulty_scoring.py`
- Create: `dataset_curation/tests/test_difficulty_scoring.py`

**Interfaces:**
- Consumes: `pd.DataFrame` from `extract_features_batch()` + maneuver labels (either a pre-computed JSON or derived from trajectory features)
- Produces:
  - `train_difficulty_classifier(df: pd.DataFrame, labels: pd.Series) -> tuple[lgb.Booster, np.ndarray]` — returns trained model + per-sample entropy scores
  - `compute_difficulty_scores(df: pd.DataFrame, model: lgb.Booster) -> np.ndarray` — returns entropy array
  - CLI: `python -m dataset_curation.difficulty_scoring --features <parquet> --output_dir <dir> [--labels <json>]`

- [ ] **Step 1: Write the failing tests**

Create `dataset_curation/tests/test_difficulty_scoring.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_labeled_df(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n_per_class = n // 4

    features_list = []
    labels_list = []
    for cls_idx, cls_name in enumerate(["straight", "left_turn", "right_turn", "lane_follow"]):
        center = rng.standard_normal(10) * (cls_idx + 1)
        data = rng.normal(center, 0.5, (n_per_class, 10))
        features_list.append(data)
        labels_list.extend([cls_name] * n_per_class)

    data = np.vstack(features_list)
    paths = [f"/fake/scene_{i:04d}.npz" for i in range(len(labels_list))]
    cols = [f"feat_{i}" for i in range(10)]
    df = pd.DataFrame(data, index=paths, columns=cols)
    labels = pd.Series(labels_list, index=paths, name="maneuver")
    return df, labels


def test_train_returns_model_and_scores():
    from dataset_curation.difficulty_scoring import train_difficulty_classifier

    df, labels = _make_labeled_df()
    model, scores = train_difficulty_classifier(df, labels)
    assert model is not None
    assert len(scores) == len(df)
    assert all(s >= 0 for s in scores)


def test_entropy_higher_for_ambiguous_samples():
    from dataset_curation.difficulty_scoring import train_difficulty_classifier

    rng = np.random.default_rng(0)
    n = 300
    cols = [f"feat_{i}" for i in range(10)]
    paths = [f"/fake/{i}.npz" for i in range(n)]

    clear_a = rng.normal([5] * 10, 0.1, (100, 10))
    clear_b = rng.normal([-5] * 10, 0.1, (100, 10))
    ambiguous = rng.normal([0] * 10, 0.1, (100, 10))  # between clusters
    data = np.vstack([clear_a, clear_b, ambiguous])

    labels = pd.Series(
        ["A"] * 100 + ["B"] * 100 + ["A"] * 50 + ["B"] * 50,
        index=paths,
    )
    df = pd.DataFrame(data, index=paths, columns=cols)

    _, scores = train_difficulty_classifier(df, labels)
    clear_scores = np.concatenate([scores[:100], scores[100:200]])
    ambig_scores = scores[200:]
    assert ambig_scores.mean() > clear_scores.mean(), (
        f"Ambiguous mean {ambig_scores.mean():.3f} should exceed clear mean {clear_scores.mean():.3f}"
    )


def test_compute_difficulty_scores_shape():
    from dataset_curation.difficulty_scoring import (
        compute_difficulty_scores,
        train_difficulty_classifier,
    )

    df, labels = _make_labeled_df(n=100)
    model, _ = train_difficulty_classifier(df, labels)
    scores = compute_difficulty_scores(df, model)
    assert len(scores) == len(df)


def test_feature_importance_not_empty():
    from dataset_curation.difficulty_scoring import train_difficulty_classifier

    df, labels = _make_labeled_df()
    model, _ = train_difficulty_classifier(df, labels)
    importance = model.feature_importance(importance_type="gain")
    assert len(importance) == len(df.columns)
    assert importance.sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest dataset_curation/tests/test_difficulty_scoring.py -v`
Expected: all FAIL

- [ ] **Step 3: Implement `dataset_curation/difficulty_scoring.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import entropy


def _heuristic_labels(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series("lane_follow", index=df.index, name="maneuver")
    if "heading_change_deg" in df.columns:
        labels[df["heading_change_deg"] > 20] = "left_turn"
        labels[df["heading_change_deg"] < -20] = "right_turn"
        labels[df["heading_change_deg"].abs() > 20] = np.where(
            df.loc[df["heading_change_deg"].abs() > 20, "heading_change_deg"] > 0,
            "left_turn",
            "right_turn",
        )
    if "closest_neighbor_dist" in df.columns:
        labels[(df["closest_neighbor_dist"] < 5) & (labels == "lane_follow")] = "avoidance"
    return labels


def train_difficulty_classifier(
    df: pd.DataFrame,
    labels: pd.Series,
) -> tuple[lgb.Booster, np.ndarray]:
    label_map = {name: idx for idx, name in enumerate(sorted(labels.unique()))}
    y = labels.map(label_map).values
    n_classes = len(label_map)

    ds = lgb.Dataset(df.values, label=y, feature_name=list(df.columns))
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "verbose": -1,
        "seed": 42,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
    }
    model = lgb.train(params, ds, num_boost_round=200)
    probs = model.predict(df.values)
    scores = np.array([entropy(p) for p in probs])
    return model, scores


def compute_difficulty_scores(
    df: pd.DataFrame,
    model: lgb.Booster,
) -> np.ndarray:
    probs = model.predict(df.values)
    return np.array([entropy(p) for p in probs])


def _plot_difficulty(df: pd.DataFrame, scores: np.ndarray, labels: pd.Series,
                     model: lgb.Booster, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(scores, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Prediction Entropy (Difficulty)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Difficulty Score Distribution")

    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.Series(importance, index=df.columns).sort_values(ascending=True)
    feat_imp.tail(15).plot.barh(ax=axes[1])
    axes[1].set_title("Top 15 Feature Importance (Gain)")

    unique_labels = sorted(labels.unique())
    for lbl in unique_labels:
        mask = labels == lbl
        axes[2].hist(scores[mask], bins=30, alpha=0.5, label=lbl)
    axes[2].legend()
    axes[2].set_xlabel("Difficulty Score")
    axes[2].set_title("Difficulty by Maneuver Type")

    plt.tight_layout()
    fig.savefig(output_dir / "difficulty_analysis.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LightGBM difficulty scoring")
    parser.add_argument("--features", required=True, help="Parquet from experiment 1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels", default=None, help="JSON mapping npz_path -> maneuver label")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features)

    if args.labels:
        with open(args.labels) as f:
            label_dict = json.load(f)
        labels = pd.Series(label_dict).reindex(df.index).fillna("unknown")
    else:
        print("No labels provided, deriving heuristic labels from features...")
        labels = _heuristic_labels(df)

    print(f"Label distribution:\n{labels.value_counts().to_string()}\n")
    model, scores = train_difficulty_classifier(df, labels)

    result = pd.DataFrame({"difficulty_score": scores, "maneuver_label": labels}, index=df.index)
    result.to_parquet(output_dir / "difficulty_scores.parquet")
    model.save_model(str(output_dir / "difficulty_model.lgb"))

    _plot_difficulty(df, scores, labels, model, output_dir)

    print(f"\nDifficulty score stats:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")

    top_hard = result.nlargest(50, "difficulty_score")
    top_easy = result.nsmallest(50, "difficulty_score")
    top_hard.index.tolist()
    json.dump(top_hard.index.tolist(), open(output_dir / "top50_hardest.json", "w"), indent=2)
    json.dump(top_easy.index.tolist(), open(output_dir / "top50_easiest.json", "w"), indent=2)
    print(f"\nTop-50 hardest/easiest scene paths saved.")
    print(f"Inspect with: python -m scenario_generation.visualize <path>")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest dataset_curation/tests/test_difficulty_scoring.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dataset_curation/difficulty_scoring.py dataset_curation/tests/test_difficulty_scoring.py
git commit -m "feat: add LightGBM difficulty scoring (experiment 3)"
```

---

### Task 5: Embedding Extraction (Experiment 4, Part 1)

**Files:**
- Create: `dataset_curation/embedding_extractor.py`
- Create: `dataset_curation/tests/test_embedding_extractor.py`

**Interfaces:**
- Consumes: model checkpoint via `torch.load()`, NPZ data via `preference_optimization.utils.load_npz_data()`
- Produces:
  - `extract_embeddings(model_path: str, scene_paths: list[str], device: str = "cuda") -> np.ndarray` — returns `[N, D_enc]` array where `D_enc` is the encoder's hidden_dim (256)
  - CLI: `python -m dataset_curation.embedding_extractor --model_path <pth> --scenes <json> --output_dir <dir>` — writes `embeddings.npy` + `embedding_paths.json`

- [ ] **Step 1: Write the failing tests**

Create `dataset_curation/tests/test_embedding_extractor.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def test_pool_embedding_mean():
    from dataset_curation.embedding_extractor import _pool_embedding

    enc = torch.randn(1, 50, 256)
    pooled = _pool_embedding(enc)
    assert pooled.shape == (256,)
    expected = enc[0].mean(dim=0)
    torch.testing.assert_close(pooled, expected)


def test_pool_embedding_batch():
    from dataset_curation.embedding_extractor import _pool_embedding

    enc = torch.randn(4, 50, 256)
    pooled = _pool_embedding(enc)
    assert pooled.shape == (4, 256)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest dataset_curation/tests/test_embedding_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `dataset_curation/embedding_extractor.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _pool_embedding(enc: torch.Tensor) -> torch.Tensor:
    if enc.dim() == 3 and enc.shape[0] == 1:
        return enc[0].mean(dim=0)
    elif enc.dim() == 3:
        return enc.mean(dim=1)
    return enc


def extract_embeddings(
    model_path: str,
    scene_paths: list[str],
    device: str = "cuda",
    batch_size: int = 1,
) -> np.ndarray:
    from diffusion_planner.train_config import TrainConfig
    from diffusion_planner.utils.train_utils import openjson
    from exploration_policy.utils import get_frozen_encoder, run_frozen_encoder
    from preference_optimization.utils import load_npz_data

    args_path = Path(model_path).parent / "args.json"
    if args_path.exists():
        config = TrainConfig(**openjson(str(args_path)))
    else:
        config = TrainConfig()

    from diffusion_planner.model.diffusion_planner import Diffusion_Planner

    model = Diffusion_Planner(config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    encoder = get_frozen_encoder(model)
    embeddings = []
    dev = torch.device(device)

    for i, sp in enumerate(scene_paths):
        try:
            data = load_npz_data(sp, dev)
            enc = run_frozen_encoder(model, data)
            pooled = _pool_embedding(enc).cpu().numpy()
            embeddings.append(pooled)
        except Exception as e:
            print(f"[WARN] Failed on {sp}: {e}")
            embeddings.append(None)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(scene_paths)}")

    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    valid_embeddings = np.stack([embeddings[i] for i in valid_indices])
    if len(valid_indices) < len(scene_paths):
        print(f"[WARN] {len(scene_paths) - len(valid_indices)} scenes failed embedding extraction")

    return valid_embeddings, [scene_paths[i] for i in valid_indices]


def main():
    parser = argparse.ArgumentParser(description="Extract frozen encoder embeddings")
    parser.add_argument("--model_path", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print(f"Extracting embeddings from {len(scene_paths)} scenes...")
    embeddings, valid_paths = extract_embeddings(
        args.model_path, scene_paths, device=args.device,
    )
    np.save(output_dir / "embeddings.npy", embeddings)
    with open(output_dir / "embedding_paths.json", "w") as f:
        json.dump(valid_paths, f)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest dataset_curation/tests/test_embedding_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dataset_curation/embedding_extractor.py dataset_curation/tests/test_embedding_extractor.py
git commit -m "feat: add frozen encoder embedding extraction (experiment 4, part 1)"
```

---

### Task 6: Clustering + UMAP Visualization (Experiment 4, Part 2)

**Files:**
- Create: `dataset_curation/clustering.py`
- Create: `dataset_curation/tests/test_clustering.py`

**Interfaces:**
- Consumes: `embeddings.npy` from Task 5, optionally `features.parquet` from Task 2
- Produces:
  - `cluster_embeddings(embeddings: np.ndarray, k: int = 50) -> tuple[np.ndarray, KMeans]` — returns labels array + fitted KMeans model
  - `compute_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray` — returns `[N, 2]` UMAP coordinates
  - CLI: `python -m dataset_curation.clustering --embeddings <npy> --paths <json> --output_dir <dir> [--features <parquet>]`

- [ ] **Step 1: Write the failing tests**

Create `dataset_curation/tests/test_clustering.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


def _make_clustered_embeddings(n_per: int = 30, k: int = 3, dim: int = 256, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, dim)) * 10
    parts = [rng.normal(c, 0.5, (n_per, dim)) for c in centers]
    return np.vstack(parts).astype(np.float32)


def test_cluster_embeddings_labels_shape():
    from dataset_curation.clustering import cluster_embeddings

    emb = _make_clustered_embeddings()
    labels, model = cluster_embeddings(emb, k=3)
    assert labels.shape == (len(emb),)
    assert set(labels) == {0, 1, 2}


def test_cluster_embeddings_finds_structure():
    from dataset_curation.clustering import cluster_embeddings

    emb = _make_clustered_embeddings(n_per=50, k=3)
    labels, _ = cluster_embeddings(emb, k=3)
    for cluster_id in range(3):
        count = (labels == cluster_id).sum()
        assert count >= 30, f"Cluster {cluster_id} has only {count} members"


def test_compute_umap_shape():
    from dataset_curation.clustering import compute_umap

    emb = _make_clustered_embeddings(n_per=20)
    coords = compute_umap(emb, n_components=2)
    assert coords.shape == (len(emb), 2)


def test_compute_umap_preserves_structure():
    from dataset_curation.clustering import compute_umap

    emb = _make_clustered_embeddings(n_per=30, k=2, dim=50)
    coords = compute_umap(emb, n_components=2)
    group_a = coords[:30]
    group_b = coords[30:]
    intra_a = np.mean(np.linalg.norm(group_a - group_a.mean(axis=0), axis=1))
    intra_b = np.mean(np.linalg.norm(group_b - group_b.mean(axis=0), axis=1))
    inter = np.linalg.norm(group_a.mean(axis=0) - group_b.mean(axis=0))
    assert inter > (intra_a + intra_b) / 2, "UMAP should preserve cluster separation"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest dataset_curation/tests/test_clustering.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `dataset_curation/clustering.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_embeddings(
    embeddings: np.ndarray,
    k: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, KMeans]:
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def _plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    paths: list[str],
    features_df=None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plots = 2
    color_by = {"cluster_id": labels}
    if features_df is not None:
        for col in ["heading_change_deg", "ego_speed", "n_active_neighbors", "travel_distance"]:
            if col in features_df.columns:
                vals = features_df[col].reindex([p for p in paths if p in features_df.index])
                if len(vals) == len(coords):
                    color_by[col] = vals.values
                    n_plots += 1

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, vals) in enumerate(color_by.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=3, alpha=0.5, cmap="tab20" if name == "cluster_id" else "viridis")
        ax.set_title(f"UMAP colored by {name}")
        plt.colorbar(sc, ax=ax)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "umap_visualization.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    unique, counts = np.unique(labels, return_counts=True)
    ax2.bar(unique, counts)
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Count")
    ax2.set_title("Cluster Size Distribution")
    plt.tight_layout()
    fig2.savefig(output_dir / "cluster_sizes.png", dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings + UMAP visualization")
    parser.add_argument("--embeddings", required=True, help="embeddings.npy from experiment 4")
    parser.add_argument("--paths", required=True, help="embedding_paths.json from experiment 4")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--features", default=None, help="features.parquet for coloring UMAP")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    with open(args.paths) as f:
        paths = json.load(f)

    print(f"Clustering {len(embeddings)} embeddings into {args.k} clusters...")
    labels, model = cluster_embeddings(embeddings, k=args.k)

    print("Computing UMAP projection...")
    coords = compute_umap(embeddings)

    import pandas as pd
    features_df = None
    if args.features:
        features_df = pd.read_parquet(args.features)

    _plot_umap(coords, labels, output_dir, paths, features_df)

    result = pd.DataFrame({
        "npz_path": paths,
        "cluster_id": labels,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
    })
    result.to_parquet(output_dir / "clustering_results.parquet", index=False)
    np.save(output_dir / "umap_coords.npy", coords)

    print(f"\nCluster distribution:")
    for cid in sorted(set(labels)):
        count = (labels == cid).sum()
        print(f"  Cluster {cid}: {count} scenes ({100*count/len(labels):.1f}%)")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest dataset_curation/tests/test_clustering.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dataset_curation/clustering.py dataset_curation/tests/test_clustering.py
git commit -m "feat: add embedding clustering + UMAP visualization (experiment 4, part 2)"
```

---

### Task 7: SemDeDup (Experiment 5)

**Files:**
- Create: `dataset_curation/semdedup.py`
- Create: `dataset_curation/tests/test_semdedup.py`

**Interfaces:**
- Consumes: `embeddings.npy` + `clustering_results.parquet` from Tasks 5-6
- Produces:
  - `semdedup(embeddings: np.ndarray, labels: np.ndarray, threshold: float = 0.95) -> np.ndarray` — returns boolean mask (True = keep, False = duplicate)
  - CLI: `python -m dataset_curation.semdedup --embeddings <npy> --clustering <parquet> --output_dir <dir> --thresholds 0.85,0.90,0.95`

- [ ] **Step 1: Write the failing tests**

Create `dataset_curation/tests/test_semdedup.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


def test_semdedup_keeps_unique():
    from dataset_curation.semdedup import semdedup

    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 64)).astype(np.float32)
    labels = np.zeros(50, dtype=int)
    keep = semdedup(emb, labels, threshold=0.99)
    assert keep.sum() >= 45, "Most unique samples should be kept"


def test_semdedup_removes_near_duplicates():
    from dataset_curation.semdedup import semdedup

    rng = np.random.default_rng(42)
    base = rng.standard_normal((10, 64)).astype(np.float32)
    dupes = base + rng.normal(0, 0.001, base.shape).astype(np.float32)
    unique = rng.standard_normal((10, 64)).astype(np.float32) * 5
    emb = np.vstack([base, dupes, unique])
    labels = np.array([0] * 20 + [1] * 10)

    keep = semdedup(emb, labels, threshold=0.95)
    assert keep.sum() < len(emb), "Should remove some near-duplicates"
    assert keep[20:].all(), "Unique cluster members should all be kept"


def test_semdedup_threshold_monotonic():
    from dataset_curation.semdedup import semdedup

    rng = np.random.default_rng(42)
    base = rng.standard_normal((20, 64)).astype(np.float32)
    dupes = base + rng.normal(0, 0.01, base.shape).astype(np.float32)
    emb = np.vstack([base, dupes])
    labels = np.zeros(40, dtype=int)

    kept_95 = semdedup(emb, labels, threshold=0.95).sum()
    kept_90 = semdedup(emb, labels, threshold=0.90).sum()
    kept_85 = semdedup(emb, labels, threshold=0.85).sum()
    assert kept_85 <= kept_90 <= kept_95, (
        f"Lower threshold should keep fewer: {kept_85}, {kept_90}, {kept_95}"
    )


def test_semdedup_returns_correct_shape():
    from dataset_curation.semdedup import semdedup

    emb = np.random.default_rng(0).standard_normal((100, 32)).astype(np.float32)
    labels = np.arange(100) % 5
    keep = semdedup(emb, labels, threshold=0.95)
    assert keep.shape == (100,)
    assert keep.dtype == bool
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest dataset_curation/tests/test_semdedup.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `dataset_curation/semdedup.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def semdedup(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    n = len(embeddings)
    keep = np.ones(n, dtype=bool)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normed = embeddings / norms

    for cluster_id in np.unique(labels):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) <= 1:
            continue

        cluster_emb = normed[indices]
        centroid = cluster_emb.mean(axis=0)
        centroid_dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        sorted_order = np.argsort(centroid_dists)

        sim = cosine_similarity(cluster_emb)
        removed = set()
        for i in reversed(sorted_order):
            if i in removed:
                continue
            for j in sorted_order:
                if j == i or j in removed:
                    continue
                if sim[i, j] >= threshold:
                    removed.add(i)
                    break

        for local_idx in removed:
            keep[indices[local_idx]] = False

    return keep


def main():
    parser = argparse.ArgumentParser(description="SemDeDup: semantic deduplication")
    parser.add_argument("--embeddings", required=True, help="embeddings.npy")
    parser.add_argument("--clustering", required=True, help="clustering_results.parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--thresholds", default="0.85,0.90,0.95",
                        help="Comma-separated similarity thresholds to evaluate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    cluster_df = pd.read_parquet(args.clustering)
    labels = cluster_df["cluster_id"].values
    paths = cluster_df["npz_path"].tolist()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"Running SemDeDup on {len(embeddings)} samples...")
    for thresh in thresholds:
        keep = semdedup(embeddings, labels, threshold=thresh)
        n_kept = keep.sum()
        n_removed = len(keep) - n_kept
        print(f"\nThreshold {thresh}:")
        print(f"  Kept: {n_kept} ({100*n_kept/len(keep):.1f}%)")
        print(f"  Removed: {n_removed} ({100*n_removed/len(keep):.1f}%)")

        kept_paths = [p for p, k in zip(paths, keep) if k]
        removed_paths = [p for p, k in zip(paths, keep) if not k]
        json.dump(kept_paths, open(output_dir / f"kept_t{thresh}.json", "w"))
        json.dump(removed_paths[:100], open(output_dir / f"removed_t{thresh}_sample.json", "w"), indent=2)

    print(f"\nResults saved to {output_dir}")
    print("Inspect removed pairs with: python -m scenario_generation.visualize <path1> <path2>")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest dataset_curation/tests/test_semdedup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dataset_curation/semdedup.py dataset_curation/tests/test_semdedup.py
git commit -m "feat: add SemDeDup semantic deduplication (experiment 5)"
```

---

### Task 8: Prototypicality / Rarity Scoring (Experiment 6)

**Files:**
- Create: `dataset_curation/prototypicality.py`
- Create: `dataset_curation/tests/test_prototypicality.py`

**Interfaces:**
- Consumes: `embeddings.npy` + `clustering_results.parquet` from Tasks 5-6
- Produces:
  - `compute_rarity_scores(embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray` — returns per-sample rarity score (L2 distance to centroid)
  - CLI: `python -m dataset_curation.prototypicality --embeddings <npy> --clustering <parquet> --output_dir <dir>`

- [ ] **Step 1: Write the failing tests**

Create `dataset_curation/tests/test_prototypicality.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


def test_rarity_scores_shape():
    from dataset_curation.prototypicality import compute_rarity_scores

    rng = np.random.default_rng(42)
    emb = rng.standard_normal((100, 64)).astype(np.float32)
    labels = np.arange(100) % 5
    centroids = np.stack([emb[labels == k].mean(axis=0) for k in range(5)])

    scores = compute_rarity_scores(emb, labels, centroids)
    assert scores.shape == (100,)
    assert (scores >= 0).all()


def test_centroid_sample_has_lowest_score():
    from dataset_curation.prototypicality import compute_rarity_scores

    rng = np.random.default_rng(42)
    center = np.zeros(64, dtype=np.float32)
    near = rng.normal(0, 0.01, (10, 64)).astype(np.float32)
    far = rng.normal(0, 5.0, (10, 64)).astype(np.float32)
    emb = np.vstack([center[None], near, far])
    labels = np.zeros(21, dtype=int)
    centroids = emb.mean(axis=0, keepdims=True)

    scores = compute_rarity_scores(emb, labels, centroids)
    near_mean = scores[1:11].mean()
    far_mean = scores[11:].mean()
    assert far_mean > near_mean, "Far samples should have higher rarity"


def test_rarity_scores_all_finite():
    from dataset_curation.prototypicality import compute_rarity_scores

    emb = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    labels = np.zeros(50, dtype=int)
    centroids = emb.mean(axis=0, keepdims=True)

    scores = compute_rarity_scores(emb, labels, centroids)
    assert np.all(np.isfinite(scores))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest dataset_curation/tests/test_prototypicality.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `dataset_curation/prototypicality.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_rarity_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(embeddings))
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        scores[i] = np.linalg.norm(emb - centroids[label])
    return scores


def main():
    parser = argparse.ArgumentParser(description="Prototypicality / rarity scoring")
    parser.add_argument("--embeddings", required=True, help="embeddings.npy")
    parser.add_argument("--clustering", required=True, help="clustering_results.parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    cluster_df = pd.read_parquet(args.clustering)
    labels = cluster_df["cluster_id"].values
    paths = cluster_df["npz_path"].tolist()

    n_clusters = len(np.unique(labels))
    centroids = np.stack([embeddings[labels == k].mean(axis=0) for k in range(n_clusters)])

    scores = compute_rarity_scores(embeddings, labels, centroids)

    result = pd.DataFrame({
        "npz_path": paths,
        "rarity_score": scores,
        "cluster_id": labels,
    })
    result.to_parquet(output_dir / "rarity_scores.parquet", index=False)

    sorted_idx = np.argsort(scores)
    top_rare = [paths[i] for i in sorted_idx[-args.top_k:]][::-1]
    top_typical = [paths[i] for i in sorted_idx[:args.top_k]]
    json.dump(top_rare, open(output_dir / f"top{args.top_k}_rarest.json", "w"), indent=2)
    json.dump(top_typical, open(output_dir / f"top{args.top_k}_typical.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(scores, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Rarity Score (L2 to centroid)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Rarity Score Distribution")

    if "umap_x" in cluster_df.columns:
        sc = axes[1].scatter(
            cluster_df["umap_x"], cluster_df["umap_y"],
            c=scores, s=3, alpha=0.5, cmap="plasma",
        )
        axes[1].set_title("UMAP colored by rarity score")
        plt.colorbar(sc, ax=axes[1])
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "rarity_analysis.png", dpi=150)
    plt.close(fig)

    print(f"Rarity score stats:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")
    print(f"\nTop-{args.top_k} rarest / most typical scene paths saved.")
    print(f"Inspect with: python -m scenario_generation.visualize <path>")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest dataset_curation/tests/test_prototypicality.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dataset_curation/prototypicality.py dataset_curation/tests/test_prototypicality.py
git commit -m "feat: add prototypicality rarity scoring (experiment 6)"
```

---

### Task 9: Run All Tests + Final Verification

**Files:**
- No new files

**Interfaces:**
- Consumes: all previous tasks
- Produces: passing test suite

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest dataset_curation/tests/ -v
```
Expected: all tests PASS

- [ ] **Step 2: Run linting**

```bash
ruff check dataset_curation/ --fix
ruff format dataset_curation/
```
Expected: no errors after fix

- [ ] **Step 3: Verify all scripts have working `--help`**

```bash
python -m dataset_curation.features --help
python -m dataset_curation.outlier_detection --help
python -m dataset_curation.difficulty_scoring --help
python -m dataset_curation.embedding_extractor --help
python -m dataset_curation.clustering --help
python -m dataset_curation.semdedup --help
python -m dataset_curation.prototypicality --help
```
Expected: each prints usage without errors

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -u
git commit -m "style: apply ruff formatting to dataset_curation"
```

- [ ] **Step 5: Push branch and create draft PR**

```bash
git push -u origin dataset-characterization
gh pr create --title "Dataset characterization experiments" --body "$(cat <<'EOF'
## Summary
- Six independent experiments for dataset characterization
- Experiment 1: Hand-crafted feature profiling (ego dynamics, trajectory shape, neighbors, map, route)
- Experiment 2: Outlier detection (Isolation Forest + LOF)
- Experiment 3: LightGBM difficulty scoring (prediction entropy as complexity proxy)
- Experiment 4: Frozen encoder embeddings + k-means clustering + UMAP visualization
- Experiment 5: SemDeDup semantic deduplication
- Experiment 6: Prototypicality/rarity scoring via centroid distance

## Test plan
- [ ] Run `python -m pytest dataset_curation/tests/ -v` — all pass
- [ ] Run experiment 1 on a small scene list and inspect histograms
- [ ] Run experiment 2 and manually inspect top-50 outliers
- [ ] Run experiment 3 and check if difficulty scores correlate with scene complexity
- [ ] Run experiment 4 with a model checkpoint and verify UMAP clusters make sense
- [ ] Run experiment 5 at different thresholds and inspect duplicate pairs
- [ ] Run experiment 6 and verify rare scenes are genuinely unusual
EOF
)" --draft
```
