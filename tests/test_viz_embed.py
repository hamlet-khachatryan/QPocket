"""Tests for qpocket.viz.embed and the vendored viewer asset.

Layouts of a distance matrix are easy to get subtly wrong: a flipped eigenvector
makes the same data look different between runs, and a parameter scikit-learn
rejects only shows up at call time. These tests pin the invariants that make two
runs of the notebook comparable.
"""

from __future__ import annotations


import numpy as np
import pytest

from qpocket.viz.embed import (
    UMAP,
    embed_distances,
)


def distances_from_points(points):
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(-1))


@pytest.fixture
def two_groups():
    """Two well-separated groups of points, and their group labels."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.3, size=(8, 3))
    b = rng.normal(10.0, 0.3, size=(8, 3))
    labels = np.array([0] * 8 + [1] * 8)
    return distances_from_points(np.vstack([a, b])), labels


def group_separation(coords, labels):
    """Mean within-group distance over mean between-group distance."""
    d = distances_from_points(coords)
    iu = np.triu_indices(len(coords), 1)
    same = labels[iu[0]] == labels[iu[1]]
    return d[iu][same].mean() / d[iu][~same].mean()


# ---------------------------------------------------------------------------
# symmetrise
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# classical MDS
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# embed_distances
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# vendored 3Dmol asset
# ---------------------------------------------------------------------------


def test_a_tiny_run_falls_back_instead_of_crashing():
    """Three structures broke UMAP's spectral start deep inside scipy."""
    import numpy as np

    from qpocket.viz.embed import PCA_MDS, TSNE, UMAP, embed_distances

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    for method in (UMAP, TSNE):
        result = embed_distances(d, method=method)
        assert result.coords.shape == (3, 2)
        assert result.method == PCA_MDS
        assert "too few" in result.note


def test_umap_layouts_never_run_at_the_same_time(two_groups, monkeypatch):
    """The explorer serves every session from one process; numba aborts it on concurrent UMAP fits."""
    import threading
    import time

    import umap

    d, _ = two_groups
    running, most = [0], [0]
    guard = threading.Lock()

    class OneAtATime:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, matrix):
            with guard:
                running[0] += 1
                most[0] = max(most[0], running[0])
            time.sleep(0.2)
            with guard:
                running[0] -= 1
            return np.zeros((len(matrix), 2))

    monkeypatch.setattr(umap, "UMAP", OneAtATime)
    threads = [threading.Thread(target=embed_distances, args=(d, UMAP)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most[0] == 1
