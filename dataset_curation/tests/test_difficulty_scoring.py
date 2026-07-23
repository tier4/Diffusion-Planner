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
