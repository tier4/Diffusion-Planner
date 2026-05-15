# Sampling Package

## Overview

The Sampling Package provides tools to cluster trajectory datasets used for Diffusion Planner training.
By grouping ego future trajectories into clusters, it helps avoid sampling bias and ensures a balanced data distribution across training scenarios.

## File Structure

```
sampling/
├── cluster.py            # CLI entry point — argument parsing and file I/O
├── sampling.py           # Balanced sampling from cluster result JSON
├── visualize_cluster.py  # Visualize clustering results as trajectory plots
├── utils/
│   ├── elbow.py          # WCSS computation, elbow detection, KMeans fitting
│   └── pipeline.py       # Feature extraction, ClusteringStrategy interface, and pipeline
└── README.md
```

### File Roles

| File | Role |
|---|---|
| `cluster.py` | CLI entry point. Reads an NPZ file list, runs the clustering pipeline, and writes the result JSON. |
| `sampling.py` | Reads the cluster result JSON and samples an equal number of files from each cluster (equal to the smallest cluster size). Outputs a JSON list suitable for `train_run.sh`. |
| `visualize_cluster.py` | Reads the result JSON from `cluster.py` and produces a grid of subplots, one per cluster, showing overlaid ego future trajectories. |
| `utils/elbow.py` | Utilities for computing WCSS (within-cluster sum of squares), finding the elbow point, and fitting KMeans. |
| `utils/pipeline.py` | Feature extraction from NPZ files, the `ClusteringStrategy` abstract interface, the `ElbowKMeansStrategy` concrete implementation, and the `cluster_trajectories` pipeline function. |

---

## Usage

### Step 1: Clustering (`cluster.py`)

```bash
python cluster.py \
    --data_list /path/to/data_list.json \
    --output    /path/to/result.json \
    [--k_max 20] \
    [--pca_components 50] \
    [--seed 42]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_list` | ✓ | — | Path to a JSON file listing NPZ file paths |
| `--output` | ✓ | — | Output path for the clustering result JSON |
| `--k_max` | | `20` | Upper bound on the number of clusters to evaluate |
| `--pca_components` | | `50` | Number of PCA components for dimensionality reduction |
| `--seed` | | `42` | Random seed |

**Input JSON format (`--data_list`)**

```json
[
    "/path/to/sample_0000.npz",
    "/path/to/sample_0001.npz",
    ...
]
```

### Step 2: Balanced Sampling (`sampling.py`)

```bash
# Specify seed directly
python sampling.py \
    --cluster_json /path/to/cluster_result.json \
    --output       /path/to/sampled.json \
    --seed 42

# Read seed from a previous sampling output (for reproducibility)
python sampling.py \
    --cluster_json /path/to/cluster_result.json \
    --output       /path/to/sampled.json \
    --seed_json    /path/to/previous_sampled.json
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--cluster_json` | ✓ | — | Clustering result JSON produced by `cluster.py` |
| `--output` | ✓ | — | Output path for the sampled file list JSON |
| `--seed` | ✓ (one of) | — | Random seed value (integer) |
| `--seed_json` | ✓ (one of) | — | JSON file containing a `"seed"` key (e.g. a previous `sampling.py` output) |

`--seed` and `--seed_json` are mutually exclusive. Exactly one must be provided.

**Output JSON format**

```json
{
    "seed": 42,
    "files": [
        "/path/to/sample_0.npz",
        "/path/to/sample_1.npz"
    ]
}
```

Pass the output directly to `train_run.sh`:

```bash
bash train_run.sh my_exp /path/to/sampled.json /path/to/valid.json /path/to/sft.json
```

### Step 3: Visualization (`visualize_cluster.py`)

```bash
python visualize_cluster.py \
    --cluster_json /path/to/result.json \
    --output       /path/to/figure.png \
    [--max_samples 200] \
    [--seed 42]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--cluster_json` | ✓ | — | Clustering result JSON produced by `cluster.py` |
| `--output` | | — | Output path (PNG / PDF / SVG). If omitted, the figure is shown interactively. |
| `--max_samples` | | `200` | Maximum number of trajectories to draw per cluster |
| `--seed` | | `42` | Random seed for trajectory sampling |

---

## Processing Pipeline

```
NPZ files
     │
     ▼
Extract ego_agent_future (80, 3) → flatten → (240,)
     │
     ▼
Z-score normalization
     │
     ▼
PCA  (240-dim → pca_components-dim)
     │
     ▼
ClusteringStrategy.fit_predict(features)
     │                 │
     │    ElbowKMeansStrategy (default)
     │      Determine optimal k via elbow method (k = 1 .. k_max)
     │      Fit KMeans with k = optimal_k
     │                 │
     ▼                 ▼
result.json  (NPZ paths grouped by cluster ID)
```

The clustering step is implemented as a **Strategy pattern**.
The preprocessing steps (feature extraction, Z-score normalization, PCA) are fixed,
while the clustering algorithm is delegated to a `ClusteringStrategy` instance.
This makes it straightforward to swap in a different algorithm without touching the pipeline.

---

## Extending with a Custom Clustering Strategy

To use a different clustering algorithm, subclass `ClusteringStrategy` and implement `fit_predict`.
The method must set `self.n_clusters_` as a side-effect so the pipeline can report the number of clusters used.

```python
import numpy as np
from utils.pipeline import ClusteringStrategy, cluster_trajectories

class MyStrategy(ClusteringStrategy):
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        # ... your algorithm ...
        self.n_clusters_ = k  # must be set
        return labels          # integer array of shape (n_samples,)

strategy = MyStrategy()
result = cluster_trajectories(npz_paths, strategy, pca_components=50)
print(f"Clusters: {strategy.n_clusters_}")
```

### Built-in strategies

| Class | Description |
|---|---|
| `ElbowKMeansStrategy(k_max, random_state)` | Selects the number of clusters automatically via the elbow method, then fits KMeans. |

---

## Output Files

### Clustering result JSON (`cluster.py`)

A dictionary mapping cluster IDs to lists of NPZ file paths.
Keys are sorted as `cluster_id0`, `cluster_id1`, …

```json
{
    "cluster_id0": [
        "/path/to/sample_0000.npz",
        "/path/to/sample_0003.npz"
    ],
    "cluster_id1": [
        "/path/to/sample_0001.npz"
    ],
    "cluster_id2": [
        "/path/to/sample_0002.npz",
        "/path/to/sample_0004.npz"
    ]
}
```

- Every input file appears in exactly one cluster.
- The number of clusters is determined automatically within `[1, k_max]`.

### Visualization figure (`visualize_cluster.py`)

A PNG (or PDF / SVG) with one subplot per cluster arranged in a grid.
Each subplot overlays the `(x, y)` trajectories of the samples assigned to that cluster.

---

## Troubleshooting

### `OpenBLAS: Program is Terminated. Because you tried to allocate too many memory regions.`

This occurs when OpenBLAS exhausts the OS memory-map region limit during the elbow-method loop.
Set the following environment variables before running:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python cluster.py ...
```
