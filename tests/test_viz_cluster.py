"""Tests for qpocket.viz.cluster.

Re-clustering is only useful if it is honest: the cut has to do what it says, the
labels have to stay attached to the same groups as the cut moves, and the
comparison against the stored labels has to survive the fact that cluster numbers
mean nothing across two groupings.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpocket.viz.cluster import (
    affinity_labels,
)


def distances(points):
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(-1))


@pytest.fixture
def three_groups():
    """Sixteen points in three tight, well-separated groups of 6, 6 and 4."""
    rng = np.random.default_rng(0)
    points = np.vstack(
        [
            rng.normal(0.0, 0.3, size=(6, 3)),
            rng.normal(20.0, 0.3, size=(6, 3)),
            rng.normal(-20.0, 0.3, size=(4, 3)),
        ]
    )
    truth = np.array([0] * 6 + [1] * 6 + [2] * 4)
    return distances(points), truth


def same_partition(a, b):
    """True when two labelings group the structures identically."""
    a, b = np.asarray(a), np.asarray(b)
    iu = np.triu_indices(len(a), 1)
    return np.array_equal(a[iu[0]] == a[iu[1]], b[iu[0]] == b[iu[1]])


# ---------------------------------------------------------------------------
# linkage_matrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# cluster_labels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# threshold_for_clusters
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# compare_labelings
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# block_order
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# exemplars
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# dendrogram_segments
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# affinity propagation: the pipeline's own method
# ---------------------------------------------------------------------------


def test_affinity_finds_the_planted_groups(three_groups):
    d, truth = three_groups
    result = affinity_labels(d)
    assert same_partition(result.labels, truth)


