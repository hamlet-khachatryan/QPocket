"""Three clustering methods, each adjustable, all returning the same shape."""
import numpy as np
import pytest

from qpocket.viz import clusterings as cl
from qpocket.viz.two_stage import RunMatrices


def _matrices(n=12, seed=0):
    """Two clear blobs, so every method should find structure."""
    rng = np.random.default_rng(seed)
    points = np.vstack([rng.normal(0, 0.3, (n // 2, 2)), rng.normal(6, 0.3, (n // 2, 2))])
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(d, 0.0)
    ids = [f"P{i:02d}" for i in range(n)]
    return RunMatrices(ids=ids, shape=d.copy(), chem=d.copy(), combined=d.copy())


def test_three_methods_are_offered():
    assert set(cl.METHODS) == {"affinity", "hierarchical", "two_stage"}


@pytest.mark.parametrize("build", [
    lambda m: cl.affinity(m),
    lambda m: cl.hierarchical(m, threshold=3.0),
])
def test_every_method_returns_the_same_columns(build):
    m = _matrices()
    grouping = build(m)
    assert list(grouping.frame.columns) == list(cl.COLUMNS)
    assert grouping.frame["pdb_id"].tolist() == m.ids
    assert grouping.n_clusters >= 1
    assert grouping.caption


def test_affinity_separates_two_blobs():
    grouping = cl.affinity(_matrices())
    assert grouping.n_clusters >= 2
    assert grouping.silhouette is not None and grouping.silhouette > 0.5


def test_affinity_damping_is_adjustable():
    """Different damping is accepted and reported in the caption."""
    loose = cl.affinity(_matrices(), damping=0.5)
    tight = cl.affinity(_matrices(), damping=0.95)
    assert "0.50" in loose.caption and "0.95" in tight.caption


def test_a_lower_hierarchical_cut_gives_more_clusters():
    m = _matrices()
    assert cl.hierarchical(m, 0.2).n_clusters > cl.hierarchical(m, 4.0).n_clusters


def test_the_hierarchical_cut_height_reaches_the_caption():
    assert "2.5" in cl.hierarchical(_matrices(), 2.5).caption


def test_agreement_is_perfect_against_itself():
    grouping = cl.affinity(_matrices())
    stored = dict(zip(grouping.frame["pdb_id"], grouping.frame["cluster_id"]))
    assert cl.agreement(grouping.frame, stored).ari == pytest.approx(1.0)


def test_agreement_is_none_without_a_stored_clustering():
    assert cl.agreement(cl.affinity(_matrices()).frame, {}) is None


def test_silhouette_is_none_when_everything_lands_in_one_cluster():
    m = _matrices()
    assert cl.hierarchical(m, threshold=1e6).silhouette is None
