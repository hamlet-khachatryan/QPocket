"""Two-stage clustering: hierarchical threshold cuts per stage, affinity settings honoured."""
import numpy as np
import pytest

from qpocket.core.clustering import THRESHOLD, cluster_labels, two_stage_labels


def _two_blobs_with_substructure():
    rng = np.random.default_rng(0)
    centres = [(0, 0), (0, 1.5), (8, 0), (8, 1.5)]
    pts = np.vstack([rng.normal(c, 0.1, (5, 2)) for c in centres])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    return d, d.copy()


def test_shape_threshold_matches_a_single_hierarchical_cut():
    shape, chem = _two_blobs_with_substructure()
    ts = two_stage_labels(shape, chem, shape_criterion="threshold", shape_threshold=4.0,
                          chem_method="hierarchical", chem_criterion="threshold", chem_threshold=100.0)
    ref = cluster_labels(shape, method="average", criterion=THRESHOLD, threshold=4.0)
    assert len(set(ts.shape_labels)) == ref.n_clusters == 2
    assert len(set(ts.labels)) == 2                      # a huge chemistry cut never splits


def test_lower_chem_threshold_splits_each_shape_group():
    shape, chem = _two_blobs_with_substructure()
    loose = two_stage_labels(shape, chem, shape_criterion="threshold", shape_threshold=4.0,
                             chem_method="hierarchical", chem_criterion="threshold", chem_threshold=5.0)
    tight = two_stage_labels(shape, chem, shape_criterion="threshold", shape_threshold=4.0,
                             chem_method="hierarchical", chem_criterion="threshold", chem_threshold=0.8)
    assert len(set(loose.labels)) == 2 and len(set(tight.labels)) == 4


def test_count_criterion_is_the_default_and_unchanged():
    shape, chem = _two_blobs_with_substructure()
    a = two_stage_labels(shape, chem)
    b = two_stage_labels(shape, chem, shape_criterion="count", chem_criterion="count")
    assert np.array_equal(a.labels, b.labels)


def test_affinity_stage_receives_its_damping(monkeypatch):
    from qpocket.core import clustering as cl
    seen = {}
    real = cl.affinity_labels

    def spy(d, damping=cl.PIPELINE_DAMPING, preference_percentile=cl.PIPELINE_PREFERENCE_PCT,
            max_iter=cl.PIPELINE_MAX_ITER):
        seen.update(damping=damping, max_iter=max_iter)
        return real(d, damping=damping, preference_percentile=preference_percentile, max_iter=max_iter)

    monkeypatch.setattr(cl, "affinity_labels", spy)
    shape, chem = _two_blobs_with_substructure()
    two_stage_labels(shape, chem, shape_method="affinity", shape_damping=0.7, shape_max_iter=123)
    assert seen == {"damping": 0.7, "max_iter": 123}


def test_threshold_needs_a_hierarchical_stage():
    shape, chem = _two_blobs_with_substructure()
    with pytest.raises(ValueError, match="threshold"):
        two_stage_labels(shape, chem, shape_method="kmedoids", shape_criterion="threshold", shape_threshold=1.0)
