# Latent Scoring Methods: Multi-Method Similarity Comparison

## Goal

Compare multiple scoring methods for measuring scene similarity in the diffusion planner's latent space. The planner's encoder produces embeddings that capture its understanding of a scene; we want to find which distance/similarity method best captures meaningful differences between scenes from the planner's perspective.

Currently only kNN with cosine distance is implemented. We add three alternatives — Mahalanobis distance, Spherical K-means, and GMM log-likelihood — each with its own appropriate preprocessing, then compare all four on the same data.

## Non-Goals

- This is NOT about safety or EPDMS integration. The goal is understanding what the planner sees as similar/different.
- We are not changing the pooling strategy (mean pool stays). Only the scoring method varies.
- We are not choosing a "winner" automatically — the comparison tooling provides data for manual inspection.
- Calibration (percentile thresholds, warning/high levels) is out of scope for the new methods. The existing kNN calibration is untouched. If a method proves useful, calibration can be added later.

## Architecture

### Pipeline Overview

```
npz files
    ↓ EncoderInference.encode_batch()
[B, token_num, 256] → mean(dim=1) → [B, 256]  (raw, un-normalized)
    ↓
    ├─ Method 0 (kNN):         L2-normalize → cosine-distance kNN
    ├─ Method 1 (Mahalanobis): L2-normalize → Mahalanobis distance
    ├─ Method 2 (K-means):     L2-normalize → Spherical K-means centroid distance
    └─ Method 3 (GMM):         z-score standardize per dim → GMM negative log-likelihood
```

### Normalization flow

Currently, L2 normalization happens in two places:
1. `EncoderInference.encode_batch()` normalizes after mean pooling (line 158-159)
2. `LatentOODScorer.build()` normalizes again when building the bank (line 59-61)

This double normalization is harmless today (idempotent), but must change:
- `encode_batch()` will return the **raw** mean-pooled vector (remove the `F.normalize` call)
- Each scoring method applies its own preprocessing
- The `LatentOODScorer.build()` normalization stays for the kNN path
- New methods handle their own normalization/standardization

### Bank Storage

The bank directory gains new files alongside the existing ones:

```
latent_ood_bank/
    embeddings.npy          # raw mean-pooled vectors (NOT L2-normalized)
    embeddings_l2.npy       # L2-normalized (for kNN, Mahalanobis, K-means)
    metadata.json           # existing, gains "bank_format_version": 2
    paths.jsonl             # existing
    calibration.json        # existing (kNN calibration)
    mahalanobis.npz         # μ, Σ⁻¹, Σ^{-1/2} (fitted on L2-normalized vectors)
    kmeans.npz              # centroids (L2-normalized), K, assignments [N]
    gmm.npz                 # weights, means, covariances, assignments [N]
    zscore_params.npz       # per-dim mean, std (for GMM standardization)
```

**Migration**: Old banks (format version 1 or absent) store L2-normalized vectors in `embeddings.npy`. The loader detects old-format banks by checking for `bank_format_version` in `metadata.json`. Old banks are rejected with a clear error message instructing the user to rebuild. No automatic migration — rebuilding is straightforward and avoids silent correctness issues.

## Method Details

### Method 0: kNN (existing, no change)

- **Preprocessing**: L2 normalization
- **Scoring**: k=10 nearest neighbors, cosine distance (= L2 on unit sphere)
- **Output**: `knn_mean`, `knn_min`, `knn_kth`
- **Neighbor retrieval**: top-k indices + distances from the bank
- **Reference**: Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors" (ICML 2022)

### Method 1: Mahalanobis Distance

- **Preprocessing**: L2 normalization (per Mahalanobis++, ICML 2025)
- **Fit (bank build time)**:
  1. Compute mean μ and covariance Σ from L2-normalized bank embeddings
  2. Regularize: `Σ_reg = Σ + εI` (ε configurable, default 1e-5). Log the condition number of Σ_reg so users can verify numerical health.
  3. Precompute and store both `Σ⁻¹` (for scoring) and `Σ^{-1/2}` (for neighbor retrieval)
  4. Precompute whitened bank: `W = Σ^{-1/2} @ (embeddings_l2 - μ).T` and store for neighbor retrieval
- **Score**: squared Mahalanobis distance `(x-μ)ᵀ Σ⁻¹ (x-μ)` — single scalar per frame. This is the squared form (no sqrt), consistent with OOD detection convention.
- **Neighbor retrieval**: L2 distance in whitened space between `Σ^{-1/2}(x - μ)` and the precomputed whitened bank. This is O(N×d) per query, same as kNN — not O(1).
- **Reference**: Lee et al. (NeurIPS 2018); Müller et al. "Mahalanobis++" (ICML 2025)

### Method 2: Spherical K-means

- **Preprocessing**: L2 normalization
- **Fit (bank build time)**: Run spherical K-means (K configurable, default=32) on L2-normalized bank embeddings. Cosine similarity as the distance metric. Centroids are L2-normalized after each iteration. Store centroids and per-embedding cluster assignments.
- **Score**: cosine distance to nearest centroid
- **Additional outputs**: assigned cluster ID, distance to 2nd-nearest centroid (margin)
- **Neighbor retrieval**: search the top-2 nearest clusters, return top-k bank members across both by cosine distance. This avoids missing cross-boundary neighbors.
- **Implementation**: Use FAISS `Kmeans(d, k, spherical=True)` which correctly implements spherical K-means (re-normalizes centroids after each M-step). Do NOT use scikit-learn KMeans — its centroid update computes arithmetic mean without re-normalizing, which is not equivalent to spherical K-means.
- **Reference**: Dhillon & Modha, "Concept Decompositions for Large Sparse Text Data using Clustering" (2001)

### Method 3: GMM Log-Likelihood

- **Preprocessing**: z-score standardization (zero mean, unit variance per dimension). Fit scaler on bank, apply to both bank and query.
- **Fit (bank build time)**: Fit a GMM with K components (default=16), diagonal covariance, on z-score-standardized bank embeddings. Store per-embedding component assignments.
- **Score**: negative log-likelihood `−log Σᵢ πᵢ N(x|μᵢ, Σᵢ)`
- **Additional outputs**: most-likely component ID, posterior probability of most-likely component
- **Neighbor retrieval**: search the top-2 most-likely components, return top-k bank members across both by Euclidean distance in standardized space. This avoids missing cross-boundary neighbors.
- **Implementation**: scikit-learn `GaussianMixture` with `covariance_type='diag'`. Diagonal covariance is chosen because full covariance with K=16 components in 256 dims would mean ~528K covariance parameters per component — identifiable at 150K samples but slow for EM and prone to component collapse. Diagonal is stable and sufficient for a comparison experiment.
- **Reference**: Standard; Reynolds, "Gaussian Mixture Models" (2009)

## Comparison Tooling

### Script: `scripts/compare_scoring_methods.py`

**Inputs**:
- `--bank_dir`: pre-built bank directory (with all method artifacts fitted)
- `--eval_list`: JSON list of eval npz paths
- `--query_npz`: 2 specific npz paths to use as query scenes for retrieval visualization (if not provided, pick 2 deterministically from eval_list using `--seed`)
- `--seed`: random seed for reproducible query selection (default 42)
- `--output_dir`: directory for all outputs
- `--model_path`, `--args_path`: for encoder inference

**Outputs**:

#### 1. Per-frame scores (`scores.jsonl`)
Each line:
```json
{
  "npz_path": "...",
  "knn_score": 0.123,
  "mahalanobis_score": 45.6,
  "kmeans_score": 0.089,
  "gmm_score": -123.4
}
```

#### 2. Rank correlation matrix (`rank_correlation.csv`)
4×4 Spearman rank correlation between methods' score rankings.

#### 3. Outlier agreement (`outlier_agreement.csv`)
Top-50 most OOD frames per method. Columns: `npz_path`, `flagged_by` (list of methods that include it in top-50).

#### 4. Retrieval visualization (`comparison_report.html`)
Self-contained local HTML file. For each of the 4 methods:
- **2 query scenes** (same queries across all methods for fair comparison)
- **3 most similar training frames** per query (retrieved by that method)
- BEV images rendered from npz files using `clip-review-tool`'s `visualization.py`

Layout per method:
```
Method: kNN
  Query A: [BEV img] → Similar: [BEV 1] [BEV 2] [BEV 3]
  Query B: [BEV img] → Similar: [BEV 1] [BEV 2] [BEV 3]

Method: Mahalanobis
  Query A: [BEV img] → Similar: [BEV 1] [BEV 2] [BEV 3]
  ...
```

Images are rendered as PNG, base64-encoded inline in the HTML (no external files).

Score distributions (histograms) and rank correlation heatmap are also embedded inline in the HTML as base64 PNGs.

### Script: `scripts/build_latent_ood_bank.py` (modified)

Extended to also fit Mahalanobis, K-means, and GMM artifacts at bank-build time. New flags:
- `--kmeans_k` (default 32)
- `--gmm_k` (default 16)
- `--gmm_covariance_type` (default `diag`)
- `--mahalanobis_eps` (default 1e-5)

The bank build step becomes: extract embeddings → save raw + L2-normalized → fit all 4 method artifacts → save.

## Data Scale

- Local smoke test: 5K bank embeddings
- Real evaluation on cloud: 50K–150K bank embeddings
- Eval set: OR scene npz files (variable count per experiment)

All methods must handle 150K × 256 without memory issues. Scoring complexity per query:
- **kNN**: O(N×d) — batched matrix multiply `z @ bank.T`
- **Mahalanobis scoring**: O(d²) — one matrix-vector multiply, no bank scan
- **Mahalanobis neighbor retrieval**: O(N×d) — L2 distance in whitened space against precomputed whitened bank
- **K-means scoring**: O(K×d) — distance to K centroids
- **K-means neighbor retrieval**: O(N_cluster×d) — scan members of top-2 clusters
- **GMM scoring**: O(K×d) — evaluate K component likelihoods
- **GMM neighbor retrieval**: O(N_component×d) — scan members of top-2 components

## Dependencies

- `scikit-learn`: GaussianMixture (already available or add to requirements)
- `faiss-cpu` (or `faiss-gpu`): for spherical K-means
- `clip-review-tool/src/visualization.py`: BEV rendering. Imported via `sys.path` manipulation pointing to the sibling repo at `../clip-review-tool/src/`. The comparison script accepts `--viz_module_path` to override this default.
- `matplotlib`: for score distribution plots and BEV rendering

## File Changes

| File | Change |
|------|--------|
| `diffusion_planner/utils/encoder_inference.py` | Return raw mean-pooled vector; move L2 norm out |
| `diffusion_planner/utils/latent_ood.py` | Add Mahalanobis, K-means, GMM scoring classes |
| `scripts/build_latent_ood_bank.py` | Fit all 4 methods at bank-build time; save raw + L2-normalized embeddings |
| `scripts/score_latent_ood.py` | Accept `--method` flag to select scoring method |
| `scripts/compare_scoring_methods.py` | New: run all 4, generate comparison HTML report |
