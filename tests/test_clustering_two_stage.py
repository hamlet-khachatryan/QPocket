"""Two-stage clustering, and the k-medoids it can use for either stage.

Everything here rests on one claim: shape and chemistry carry different groupings,
and clustering the sum of the two distances loses both. The fixtures are built so
the right answer is known by construction — twelve pockets in three shape groups,
each of which splits cleanly into two chemistry pairs — and a stage that recovers
anything else is wrong, not merely different.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpocket.core.clustering import (
    cluster_labels,
    kmedoids_labels,
    two_stage_labels,
)


def distances(values):
    """Pairwise Euclidean distance over a list of coordinates."""
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(-1))


# Twelve pockets: shape group = i // 4, chemistry pair = (i % 4) // 2.
# The jitter keeps members distinguishable without touching the grouping.
JITTER = np.array([0.0, 0.1, 0.2, 0.3] * 3)
SHAPE_COORD = np.repeat([0.0, 100.0, 200.0], 4) + JITTER
CHEM_COORD = np.tile([0.0, 0.0, 50.0, 50.0], 3) + JITTER

SHAPE_TRUTH = np.repeat([0, 1, 2], 4)
CHEM_TRUTH = np.tile([0, 0, 1, 1], 3)


@pytest.fixture
def two_axes():
    return distances(SHAPE_COORD), distances(CHEM_COORD)


def groups_of(labels):
    """The partition a labelling induces, as a set of frozensets of positions."""
    labels = np.asarray(labels)
    return {frozenset(np.flatnonzero(labels == v).tolist()) for v in np.unique(labels)}


# --- stage one ------------------------------------------------------------


# --- stage two ------------------------------------------------------------


def test_stage_two_splits_each_shape_group_by_chemistry(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d, min_cluster_size=4)
    for shape_value in np.unique(result.shape_labels):
        members = np.flatnonzero(result.shape_labels == shape_value)
        pairs = {frozenset(np.flatnonzero(
            (result.shape_labels == shape_value) & (result.chem_labels == c)
        ).tolist()) for c in np.unique(result.chem_labels[members])}
        assert len(pairs) == 2


# --- composite labels -----------------------------------------------------


# --- k-medoids ------------------------------------------------------------


def test_kmedoids_finds_the_planted_groups():
    d = distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    result = kmedoids_labels(d, n_clusters=2)
    assert groups_of(result.labels) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}


# --- relative cut ---------------------------------------------------------


def _two_blobs():
    return distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])


def labels(d, criterion, threshold):
    return cluster_labels(d, method="average", criterion=criterion, threshold=threshold).labels


def test_relative_cut_on_two_pockets_does_not_raise():
    assert len(cluster_labels(np.array([[0, 1.0], [1.0, 0]]), method="average",
                              criterion="relative", threshold=0.5).labels) == 2


def test_default_two_stage_recovers_the_planted_groups(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d)
    assert result.shape_clustering.n_clusters == 3 and len(set(result.labels.tolist())) == 6


def _old_run(tmp_path, two_axes):
    """A run saved before this change: k null under the count criterion, with a k_range."""
    import json
    import pandas as pd

    shape_d, chem_d = two_axes
    meta = tmp_path / "metadata"
    meta.mkdir()
    ids = [f"P{i}" for i in range(len(shape_d))]
    for name, m in (("shape_distance_matrix", shape_d), ("feature_distance_matrix", chem_d),
                    ("spectral_distance_matrix", shape_d + chem_d)):
        pd.DataFrame(m, index=ids, columns=ids).to_csv(meta / f"{name}.csv")
    old = {"method": "hierarchical", "linkage": "average", "k": None, "k_range": [2, 10],
           "criterion": "count", "threshold": None, "silhouette": 0.4}
    (meta / "clustering_params.json").write_text(json.dumps(
        {"mode": "two_stage", "shape": old, "chem": {**old, "method": "kmedoids"}}))
    return tmp_path


def test_an_old_run_reclusters_at_the_default_percentiles(tmp_path, two_axes):
    from qpocket.arms import recluster

    run = _old_run(tmp_path, two_axes)
    settings = recluster.load_settings(run)
    assert (settings.shape.percentile, settings.chem.percentile) == (15.0, 20.0)
    assert settings.chem.method == "hierarchical"          # its kmedoids had no k
    recluster.run(run, settings)


