"""Percentile cuts: a cut is a quantile of the site's own pairwise distances."""
import numpy as np
import pytest
from scipy.spatial.distance import squareform

from qpocket.core.clustering import cluster_labels, pair_distances, percentile_height, to_percentile


def _random_matrix(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return np.linalg.norm(pts[:, None] - pts[None], axis=2)


def test_pair_distances_are_the_upper_triangle():
    d = squareform([1.0, 2.0, 3.0])
    assert sorted(pair_distances(d)) == [1.0, 2.0, 3.0]


def test_percentile_height_is_the_quantile_of_pairs():
    d = squareform([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])          # 4 structures
    assert percentile_height(d, 50) == pytest.approx(3.5)
    assert percentile_height(np.zeros((1, 1)), 50) == 0.0


def test_to_percentile_is_monotone_and_bounded():
    d = squareform(np.arange(1.0, 11.0))                   # 5 structures
    p = to_percentile(d, np.array([0.0, 1.0, 5.5, 10.0, 99.0]))
    assert list(p) == sorted(p) and p[0] == 0 and p[-1] == 100


def test_percentile_cut_matches_the_threshold_cut():
    d = _random_matrix(30, seed=0)
    a = cluster_labels(d, method="average", criterion="percentile", threshold=15)
    b = cluster_labels(d, method="average", criterion="threshold", threshold=percentile_height(d, 15))
    assert np.array_equal(a.labels, b.labels)


def test_percentile_cut_on_constant_matrix():
    d = np.ones((6, 6)) - np.eye(6)
    assert cluster_labels(d, method="average", criterion="percentile", threshold=15).n_clusters == 1


def test_percentile_cut_two_structures():
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert cluster_labels(d, method="average", criterion="percentile", threshold=50).n_clusters in (1, 2)


def test_merges_at_the_cut_height_are_kept_together():
    """Average linkage can put a merge at 0.1 + 1 ulp; a p30 cut at 0.1 must still join it."""
    n, a, b = 15, list(range(10)), list(range(10, 15))
    d = np.full((n, n), 10.0)
    for group in (a, b):
        for i in group:
            for j in group:
                d[i, j] = 0.1
    np.fill_diagonal(d, 0.0)
    assert cluster_labels(d, method="average", criterion="percentile", threshold=30).n_clusters == 2
