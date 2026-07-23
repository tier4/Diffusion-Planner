# Dataset Characterization Design

**Goal**: Build a set of independent experiments to understand dataset properties — difficulty, rarity, redundancy, outliers, and cluster structure — for the Diffusion-Planner trajectory prediction dataset. Results inform downstream decisions about curriculum learning, cluster-weighted sampling, data filtering, and gap analysis.

**Scope**: Exploratory analysis. Each experiment is standalone and evaluated on its own merits. Methods that don't produce useful insights get dropped.

**Dataset**: ~80K scenes now (vectorized NPZ, no images), scaling to ~10M. Each scene contains ego state, neighbor agents, lanes, route, traffic lights, goal pose, and GT future trajectory.

**Existing assets**:
- Rule-based maneuver classifier (right turn, left turn, lane follow, avoidance)
- Frozen encoder producing scene embeddings (`exploration_policy/utils.py:run_frozen_encoder`)
- Extensive reward/metrics infrastructure (`rlvr/reward.py`, `planner_metrics/`)
- Spatial search + scene search tooling (`scene_search/`, `search_scenes.py`)
- Many filtering/curation scripts in `rlvr/autoresearch/tools/`

---

## Experiment 1: Hand-Crafted Feature Profiling

**Question**: What does our dataset actually look like?

**Method**: Extract ~40-60 interpretable features per scene from NPZ fields, then visualize distributions.

### Features to extract

| Category | Features | Source field |
|---|---|---|
| Ego dynamics | speed, accel magnitude, yaw_rate at t=0; max/mean/std over past 30 steps | `ego_current_state`, `ego_agent_past` |
| Trajectory shape | travel distance, endpoint displacement, total heading change (deg), max curvature, path length ratio (travel/displacement = straightness) | `ego_agent_future` |
| Neighbor complexity | count of active neighbors, closest neighbor distance at t=0, mean/min TTC estimate, neighbor type ratios (vehicle/ped/bike), max neighbor speed | `neighbor_agents_past` |
| Map context | mean lane curvature across nearby lanes, number of lane segments with active traffic lights, speed limit mean/variance | `lanes`, `lanes_speed_limit` |
| Route | goal distance, route curvature (sum of heading changes along route), route length | `route_lanes`, `goal_pose` |
| Maneuver label | from existing rule-based classifier | pre-computed or re-derived |

### Visualization

- Per-feature histograms (log-scale for heavy-tailed features like neighbor_count)
- Correlation matrix across all features
- Feature distributions broken down by maneuver type (violin plots or overlaid histograms)
- Scatter matrix of most interesting feature pairs (e.g. neighbor_count vs heading_change, speed vs curvature)

### What we're looking for

- Class imbalance (e.g. 90% lane-follow)
- Near-zero-variance features (useless, drop them)
- Unexpected bimodal distributions (potential data quality issues or meaningful sub-populations)
- Obvious clusters or gaps in feature space

### Success criterion

The feature profiles reveal non-obvious structure in the dataset. If everything looks uniform and boring, the features need redesigning. If we see clear clusters, imbalances, or gaps, this is immediately useful for understanding what we have.

---

## Experiment 2: Outlier Detection on Features

**Question**: Are there bad/corrupted/nonsensical scenes hiding in the data?

**Method**: Run two complementary outlier detectors on the feature matrix from Experiment 1:

1. **Isolation Forest** — finds globally unusual points (anomalies in the overall feature distribution)
2. **Local Outlier Factor (LOF)** — finds points that are unusual relative to their local neighborhood (contextual anomalies)

### Evaluation

Manually inspect the top-50 flagged scenes from each method using `scenario_generation.visualize`. Categorize each:

- **Genuinely bad data** — perception failures, impossible kinematics (e.g. teleporting ego), empty neighbor arrays where there should be agents, corrupted lane data
- **Rare but valid** — unusual maneuvers, edge cases worth keeping (and maybe upsampling)
- **False positives** — normal scenes with unusual but valid feature combinations

### Decision matrix

| Outcome | Action |
|---|---|
| Most flagged samples are genuinely bad | Use this method for data cleaning |
| Most flagged samples are rare-but-valid | Repurpose as a rarity detector |
| Mostly false positives | Drop this method |

---

## Experiment 3: LightGBM Difficulty Scoring

**Question**: Can we score intrinsic scene complexity using a simple classifier?

**Method**: Train LightGBM to predict maneuver type (from rule-based classifier labels) using the hand-crafted features.

### Analysis

1. **Prediction entropy** — high entropy = model unsure which maneuver type = scene is ambiguous/sits at decision boundaries = intrinsically complex
2. **Feature importance** — which features most distinguish maneuver types (tells us what "makes a scene different")
3. **Misclassified samples** — scenes where LightGBM gets the label wrong = boundary cases between maneuver types
4. **SHAP values** — per-sample explanations (e.g. "this scene is complex because neighbor_count=12 AND curvature=high")

### Evaluation

Take the top-50 highest-entropy scenes and top-50 lowest-entropy scenes. Visualize both groups with `scenario_generation.visualize`.

- Do high-entropy scenes actually look harder to a human?
- Do low-entropy scenes look trivially easy (straight highway, no neighbors)?
- Does the entropy ranking correlate with features we'd intuitively associate with difficulty?

### Success criterion

Entropy ranking matches human intuition about scene difficulty. If yes, this gives us a cheap, interpretable difficulty score for curriculum learning or weighted sampling. If not, the maneuver labels may be too coarse and we need a different prediction target.

---

## Experiment 4: Embedding Clustering + UMAP

**Question**: Does the learned encoder capture scene structure beyond what hand-crafted features and rule-based labels show?

**Method**:

1. Run frozen encoder on all ~80K scenes → embedding matrix `[80K, D_enc]`
2. k-means clustering with k in {20, 50, 100} (compare via silhouette score and elbow plot)
3. UMAP projection to 2D

### Visualization

Color the UMAP projection by:
- Maneuver type label (from rule-based classifier)
- Cluster ID (from k-means)
- Individual features from Experiment 1 (speed, curvature, neighbor_count, etc.)
- Outlier flags from Experiment 2
- Difficulty scores from Experiment 3

### What we're looking for

| Observation | Interpretation |
|---|---|
| UMAP by maneuver type shows clean separation | Encoder captures maneuver structure, but doesn't add much over rule-based classifier |
| Clusters split a single maneuver type into sub-clusters | Encoder captures finer-grained structure (e.g. "left turn with oncoming traffic" vs "left turn at empty intersection") — valuable new information |
| Clusters correlate with specific feature combinations | Encoder has learned composite scene representations — embeddings are useful |
| Clusters look random / don't correlate with anything | Encoder embeddings may not be useful for characterization |

### Success criterion

The embedding space reveals meaningful structure not captured by hand-crafted features alone. This determines whether the embedding layer is worth pursuing for Experiments 5 and 6.

---

## Experiment 5: SemDeDup on Embeddings

**Question**: How much semantic redundancy exists in the dataset, and can we safely remove it?

**Depends on**: Experiment 4 (need embeddings + clusters). Only run if Experiment 4 shows embeddings are meaningful.

**Method**: Within each k-means cluster, compute pairwise cosine similarity between embeddings. Flag pairs above a threshold as semantic duplicates. From each duplicate group, keep the most representative sample (closest to cluster centroid).

### Threshold sweep

| Threshold | Expected behavior |
|---|---|
| 0.95 | Very conservative — only near-identical scenes |
| 0.90 | Moderate — similar scenes |
| 0.85 | Aggressive — loosely similar scenes |

### Evaluation

At each threshold:
- How many samples get flagged? (dedup ratio)
- Inspect 20 flagged pairs — are they truly redundant?
- Are flagged pairs from the same recording bag (temporal redundancy) or different bags (cross-drive redundancy)?
- Plot: dedup ratio vs threshold → the redundancy curve

### Success criterion

At threshold 0.95, we're removing >10% of data with clearly redundant pairs. If barely anything gets flagged, the dataset is already diverse enough and dedup isn't needed.

---

## Experiment 6: Prototypicality / Rarity Scoring

**Question**: Can embedding distance to cluster centroids identify genuinely rare scenarios?

**Depends on**: Experiment 4.

**Method**: For each scene, compute L2 distance to its assigned cluster centroid. High distance = atypical within its cluster.

### Evaluation

- Take the top-50 most atypical (highest centroid distance) and bottom-50 most typical. Visualize both.
- Cross-reference with maneuver types: do rare scenes concentrate in certain maneuver types?
- Cross-reference with Experiment 2 outlier flags:
  - Outlier in features AND far from embedding clusters → likely bad data
  - Outlier in features BUT close to embedding cluster → unusual features, semantically normal
  - Far from embedding clusters BUT features look normal → rare-but-valid, candidate for upsampling
- Cross-reference with Experiment 3 difficulty scores: do rare scenes tend to be difficult?

### Success criterion

The rarity ranking surfaces genuinely unusual driving scenarios (not just noise). Top-rare scenes should be interpretably different from typical ones.

---

## Implementation Notes

- All experiments live in `dataset_curation/` as standalone scripts
- Each experiment outputs results to a configurable output directory
- Experiment 1 (feature extraction) is the foundation — run it first, others consume its output
- Experiments 2 and 3 depend on Experiment 1's feature matrix
- Experiments 4, 5, and 6 depend on the frozen encoder and form a chain (4 → 5, 4 → 6)
- Experiments {1,2,3} and {4} can run in parallel
- Use existing `scenario_generation.visualize` for manual inspection of flagged scenes
- At 80K scenes, all methods are cheap enough to run on a single GPU + CPU

### Dependencies (Python)

- `scikit-learn` — Isolation Forest, LOF, k-means, UMAP (via `umap-learn`)
- `lightgbm` — difficulty classifier
- `shap` — per-sample explanations
- `pandas` / `pyarrow` — feature matrix storage (parquet)
- `matplotlib` / `seaborn` — visualization
- Existing project deps — `torch`, `numpy` for encoder inference and NPZ loading

### Dependency graph

```
Experiment 1 (features)
├── Experiment 2 (outlier detection)
├── Experiment 3 (LightGBM difficulty)
│
Experiment 4 (embeddings + clustering)
├── Experiment 5 (SemDeDup)
├── Experiment 6 (prototypicality)
```
