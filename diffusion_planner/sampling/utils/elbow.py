# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.cluster import KMeans


def compute_wcss(features: np.ndarray, k_max: int, random_state: int = 42) -> list:
    """Compute within-cluster sum of squares (inertia) for k = 1 .. k_max."""
    wcss = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(features)
        wcss.append(km.inertia_)
    return wcss


def find_elbow(wcss: list) -> int:
    """Return the optimal k (1-indexed) using the maximum second-derivative criterion."""
    if len(wcss) < 3:
        return len(wcss)
    second_diff = np.diff(np.array(wcss), n=2)
    # argmax gives 0-indexed position among the second-diff array;
    # add 2 to convert back to the original k index.
    return int(np.argmax(second_diff) + 2)


def elbow_kmeans(
    features: np.ndarray,
    k_max: int = 20,
    random_state: int = 42,
) -> tuple:
    """Determine k via the elbow method, then fit KMeans.

    Args:
        features: 2-D array of shape (n_samples, n_features).
        k_max: upper bound on the number of clusters to evaluate.
        random_state: random seed for reproducibility.

    Returns:
        labels: integer cluster assignment for each sample, shape (n_samples,).
        optimal_k: chosen number of clusters.
        wcss: list of inertia values for k = 1 .. k_max (useful for plotting).
    """
    k_max = min(k_max, len(features))
    wcss = compute_wcss(features, k_max, random_state)
    optimal_k = find_elbow(wcss)

    km = KMeans(n_clusters=optimal_k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(features)
    return labels, optimal_k, wcss
