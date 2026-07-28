# Sampling Package

## Overview

The Sampling Package provides tools to cluster trajectory datasets used for Diffusion Planner training.
By grouping ego future trajectories into clusters, it helps avoid sampling bias and ensures a balanced data distribution across training scenarios.

## File Structure

```
sampling/
├── cluster.py                    # CLI entry point — argument parsing and file I/O
├── sampling.py                   # Balanced sampling from cluster result JSON
├── visualize_cluster.py          # Visualize clustering results as trajectory plots
├── visualize_cluster_report.py   # HTML diagnostic report with BEV videos per cluster
├── utils/
│   ├── elbow.py                  # WCSS computation, elbow detection, KMeans fitting
│   └── pipeline.py               # Feature extraction, ClusteringStrategy interface, and pipeline
└── README.md
```

### File Roles

| File | Role |
|---|---|
| `cluster.py` | CLI entry point. Reads an NPZ file list, runs the clustering pipeline, and writes the result JSON. |
| `sampling.py` | Reads the cluster result JSON and samples an equal number of files from each cluster (equal to the smallest cluster size). Outputs a JSON list suitable for `train_run.py`. |
| `visualize_cluster.py` | Reads the result JSON from `cluster.py` and produces a grid of subplots, one per cluster, showing overlaid ego future trajectories. |
| `visualize_cluster_report.py` | Generates an HTML diagnostic report with cluster stats, sampling weights, and BEV video examples per cluster via clip-review-tool. Supports `--standalone` mode for self-contained shareable HTML with embedded GIFs. |
| `utils/elbow.py` | Utilities for computing WCSS (within-cluster sum of squares), finding the elbow point, and fitting KMeans. |
| `utils/pipeline.py` | Feature extraction from NPZ files (`extract_features`, `extract_features_enriched`), the `ClusteringStrategy` abstract interface, the `ElbowKMeansStrategy` concrete implementation, and the `cluster_trajectories` / `cluster_trajectories_enriched` pipeline functions. |

---

## Usage

### Step 1: Clustering (`cluster.py`)

```bash
python cluster.py \
    --data_list /path/to/data_list.json \
    --output    /path/to/result.json \
    [--k_max 20] \
    [--pca_components 50] \
    [--seed 42] \
    [--mode trajectory] \
    [--top_k_neighbors 20] \
    [--neighbor_pca_components 50] \
    [--temporal_hz 2]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_list` | ✓ | — | Path to a JSON file listing NPZ file paths |
| `--output` | ✓ | — | Output path for the clustering result JSON |
| `--k_max` | | `20` | Upper bound on the number of clusters to evaluate |
| `--pca_components` | | `50` | Number of PCA components for the final dimensionality reduction |
| `--seed` | | `42` | Random seed (reaches KMeans; the PCA solver is separately fixed at 0) |
| `--mode` | | `trajectory` | `trajectory` = ego future only; `enriched` = ego + neighbors + ego state |
| `--top_k_neighbors` | | `20` | **Enriched only.** Number of nearest neighbors to keep per frame |
| `--neighbor_pca_components` | | `50` | **Enriched only.** PCA components for the neighbor block (stage 1) |
| `--temporal_hz` | | `2` | **Enriched only.** Temporal downsample rate; must evenly divide 10 |

> **Enriched mode is opt-in and less tested than trajectory mode.** It reads
> `neighbor_agents_past` / `neighbor_agents_future` from every NPZ, so it is far more
> I/O and memory hungry than trajectory mode, and it assumes every NPZ in the data list
> shares one array layout. A corpus mixing npz v2 (3-col `neighbor_agents_future`) with
> npz v3 (4-col `[x, y, cos, sin]`) produces different-width feature vectors and fails
> when they are stacked. Trajectory mode reads only `ego_agent_future`, whose
> `[x, y, heading]` layout is stable across both versions.

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

Pass the output directly to `train_run.py`:

```bash
python3 train_run.py \
  --exp_name my_exp \
  --train_set_list /path/to/sampled.json \
  --valid_set_list /path/to/valid.json \
  --resume_model_path /path/to/sft.pth
```

### Step 3: Cluster Diagnostic Report (`visualize_cluster_report.py`)

Generates an HTML diagnostic report with cluster statistics, sampling behavior documentation, and BEV video examples per cluster rendered by [clip-review-tool](https://github.com/tier4/clip-review-tool).

```bash
# Standard mode: report.html + videos/ directory with MP4s
python visualize_cluster_report.py \
    --cluster_json /path/to/cluster_result.json \
    --output_dir   /path/to/report_output/ \
    --max_videos 3 --workers 4

# Standalone mode: single self-contained HTML with embedded GIFs
# Shareable via Slack, Google Drive, email — no extra files needed
python visualize_cluster_report.py \
    --cluster_json /path/to/cluster_result.json \
    --output_dir   /path/to/report_output/ \
    --max_videos 3 --workers 4 --standalone
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--cluster_json` | ✓ | — | Cluster assignment JSON from `cluster.py` |
| `--output_dir` | ✓ | — | Output directory for report.html (and videos/ in standard mode) |
| `--max_videos` | | `3` | Max BEV video examples to render per cluster |
| `--workers` | | `1` | Parallel video rendering workers |
| `--seed` | | `42` | Random seed for video subsampling |
| `--standalone` | | `False` | Embed GIFs (240px, 3fps) as base64 in a single HTML file |
| `--cluster_weight_alpha` | | `1.0` | Weight exponent to model in the report, in `[0, 1]`. Must match training's `--cluster_weight_alpha` for the Weight column to reflect actual oversampling. |

**Prerequisites:**
- `render-video-txt` on PATH (`pip install -e /path/to/clip-review-tool`)
- `ffmpeg` on PATH (for video rendering; also used for GIF conversion in standalone mode)

**Caveat — the report and training must describe the same population:**

The report derives cluster frequencies from the cluster JSON's own totals, while
training derives them from the live, post-subsample training file list. The two
agree only when the cluster JSON and the training file list cover the same set of
samples. In particular, `--train_subsample_step > 1`, or a cluster JSON that is a
superset of `--train_set_list`, will make the report's **Weight** column diverge
from the multipliers training actually applies — even when `--cluster_weight_alpha`
matches. Training's startup log (`Cluster distribution ... alpha=...`, printed with
one `Nx` multiplier per cluster) is the authoritative record of what was applied.

**Report contents:**
- Pipeline overview (trajectory → PCA → KMeans)
- Cluster distribution table + bar chart (sorted by sample count)
- Sampling behavior summary (oversampling, unmatched samples, total draws)
- Per-cluster BEV video gallery (3 examples each by default)
- Render error diagnostics

**Sharing the report:**

The standalone HTML file (~30-40MB) can be shared via Slack, Google Drive, email, or any file-sharing tool. To publish to Confluence with inline GIFs:

```bash
# 1. Generate the report with videos
python visualize_cluster_report.py \
    --cluster_json /path/to/cluster_result.json \
    --output_dir   /path/to/report_output/ \
    --max_videos 3 --workers 4

# 2. Convert MP4s to GIFs (240px, 3fps)
find /path/to/report_output/videos -name "*.mp4" -exec sh -c '
    ffmpeg -i "$1" -vf "fps=3,scale=240:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -y "${1%.mp4}.gif"
' _ {} \;

# 3. Create a Confluence page and upload chart + GIFs as attachments
#    Then reference them in the page body with:
#    <ac:image ac:width="240"><ri:attachment ri:filename="cluster_id0_sample0.gif" /></ac:image>
#
#    See the Confluence REST API docs for creating pages and uploading attachments:
#    POST /wiki/rest/api/content                          (create page)
#    POST /wiki/rest/api/content/{pageId}/child/attachment (upload files)
#    PUT  /wiki/rest/api/content/{pageId}                  (update page body)
#
#    Authentication: Basic auth with your Atlassian email + API token.
```

### Weighted Sampling During Training

Instead of pre-sampling a balanced file list with `sampling.py`, training can
consume the full dataset and oversample rare clusters on the fly. Pass the
cluster result JSON to `train_run.py`:

```bash
python3 train_run.py \
  --exp_name my_exp \
  --train_set_list /path/to/train.json \
  --valid_set_list /path/to/valid.json \
  --cluster_json /path/to/cluster_result.json \
  --cluster_weight_alpha 0.5
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--cluster_json` | | — | Cluster assignment JSON from `cluster.py`. Enables weighted sampling. |
| `--cluster_weight_alpha` | | `1.0` | Exponent on the inverse-frequency weights, in `[0, 1]`. `1.0` gives every cluster an equal share of draws; `0.0` is uniform sampling. Values outside `[0, 1]` are rejected. |

Each sample's weight is `(1 / cluster_frequency) ** alpha`, normalized to mean
1.0 — so a sample's weight *is* its oversampling multiplier. Lowering `alpha`
softens the reweighting: the multiplier ratio between any two clusters goes
from `R` at `alpha=1.0` to `R ** alpha`.

`alpha` is capped at 1.0 on purpose. Draws per cluster scale as
`matched ** alpha * n_c ** (1 - alpha)`, so above 1.0 the exponent on `n_c` turns
negative and the weighting *inverts* — the largest cluster is starved. On an
18,000 + 10 split over 18,010 draws per epoch, `alpha=2.0` sends 7 draws to the
18,000-sample cluster and leaves 17 distinct samples in the epoch; `alpha=5.0`
leaves 10. Loss still falls (the model memorizes) and DDP stays healthy, so the
failure is invisible without this guard.

Note that `alpha=0.0` makes every sample equally likely but still draws *with
replacement*, so it is not the same as omitting `--cluster_json` (which uses a
plain `DistributedSampler` and draws without replacement).

Training prints the resulting multipliers at startup, which is how you tune
`alpha`:

```
Using cluster-weighted sampling from /path/to/cluster_result.json
Cluster distribution (matched 48231/48231 data paths, alpha=0.50):
  cluster_id0: 18402 samples  0.78x
  cluster_id7: 1204 samples  1.53x
```

The same numbers are written to `<save_dir>/cluster_sampling.json` and pushed into
`wandb.config` under `cluster_sampling`. Prefer that file over the log: the
multipliers are computed from the *live*, post-`--train_subsample_step` data list,
so they cannot be reconstructed from `args.json`, which records only `cluster_json`
and `cluster_weight_alpha`.

The cluster JSON must reference the same NPZ files as `--train_set_list`.
Paths are canonicalized before matching, so differing prefixes (absolute vs
relative, different mount points) are handled automatically. Paths present in
the data list but absent from the JSON receive the mean matched weight and a
warning is emitted.

Cluster assignments must be **disjoint**. If one canonical path appears under two
cluster ids, training refuses to start: last-write-wins would let JSON iteration
order decide the applied cluster while `matched_count` still reported a full match.
Note that canonicalization anchors on the first `train`/`valid` path component, so
two NPZs under different dataset roots that share a `location/split/date/time/frame`
layout collide and trip the same check.

### Step 4: Trajectory Visualization (`visualize_cluster.py`)

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

### `--mode trajectory` (default)

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

### `--mode enriched`

```
NPZ files
     │
     ├──────────────────────────────┐
     ▼                              ▼
ego block                       neighbor block
  ego_agent_past    (downsampled to temporal_hz)
  ego_agent_future  (downsampled, last step always kept)
  ego_current_state              neighbor_agents_past   (top_k nearest)
  top_k neighbor distances       neighbor_agents_future (top_k nearest)
     │                              │
     ▼                              ▼
Z-score                         Z-score
     │                              │
     │                              ▼
     │                    PCA (→ neighbor_pca_components)   ← stage 1
     │                              │
     └──────────► hstack ◄──────────┘
                    │
                    ▼
              Z-score  →  PCA (→ pca_components)            ← stage 2
                    │
                    ▼
      ClusteringStrategy.fit_predict(features)
```

Both stages re-standardize, so every retained neighbor PCA component carries equal
weight in the final space, and the ego:neighbor influence ratio is set implicitly by
the two blocks' dimension counts (at defaults the ego block is the larger of the two).
Tune `--neighbor_pca_components` with that in mind.

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
