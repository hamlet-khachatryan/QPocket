"""Colouring a dendrogram by the clusters its cut produces."""
import numpy as np
import pytest

from qpocket.core.clustering import dendrogram_segments, linkage_matrix, symmetrise
from qpocket.viz import two_stage as ts


def _tree(n=8, seed=0):
    """Two tight blobs, so a mid cut gives exactly two clusters."""
    rng = np.random.default_rng(seed)
    points = np.vstack([rng.normal(0, 0.2, (n // 2, 2)), rng.normal(8, 0.2, (n // 2, 2))])
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(d, 0.0)
    return symmetrise(d)


def test_branches_above_the_cut_have_no_cluster():
    d = _tree()
    link = linkage_matrix(d, "average")
    data = dendrogram_segments(link, threshold=4.0)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    colours = ts.segment_clusters(data, 4.0, labels)
    assert len(colours) == len(data.segments)
    # The root joins the two blobs, so it sits above any sensible cut.
    assert colours[-1] is None


def test_branches_below_the_cut_take_their_leaves_cluster():
    d = _tree()
    link = linkage_matrix(d, "average")
    data = dendrogram_segments(link, threshold=4.0)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    colours = ts.segment_clusters(data, 4.0, labels)
    assigned = [c for c in colours if c is not None]
    assert assigned, "nothing was coloured"
    assert set(assigned) <= {0, 1}


def test_a_lower_cut_leaves_fewer_coloured_branches():
    d = _tree()
    link = linkage_matrix(d, "average")
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    high = ts.segment_clusters(dendrogram_segments(link, 4.0), 4.0, labels)
    low = ts.segment_clusters(dendrogram_segments(link, 0.2), 0.2, labels)
    assert sum(c is not None for c in low) < sum(c is not None for c in high)


def test_every_segment_gets_an_answer():
    d = _tree()
    link = linkage_matrix(d, "average")
    data = dendrogram_segments(link, 4.0)
    colours = ts.segment_clusters(data, 4.0, np.zeros(8, dtype=int))
    assert len(colours) == len(data.segments)


def _matrices(n=8, seed=0):
    from qpocket.viz.two_stage import RunMatrices

    d = _tree(n, seed)
    return RunMatrices(ids=[f"P{i}" for i in range(n)], shape=d.copy(),
                       chem=d.copy(), combined=d.copy())


def _settings():
    from qpocket.config import ClusteringParams

    return ClusteringParams()


def test_all_three_methods_produce_settings_the_pipeline_accepts():
    base = _settings()
    m = _matrices()
    for method, kw in (("affinity", {}),
                       ("hierarchical", {"combined_cut": 4.0}),
                       ("two_stage", {"shape_cut": 4.0, "chem_cut": 0.6})):
        result = ts.preview(m, ts.settings_for(base, method, **kw))
        assert result.n_clusters >= 1
        assert list(result.frame.columns) == [
            "pdb_id", "cluster_id", "shape_cluster_id", "chem_cluster_id"]


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown method"):
        ts.settings_for(_settings(), "telepathy")


def test_the_settings_are_what_recluster_would_save():
    """Preview and Apply must run the same clustering, so they share the type."""
    from qpocket.config import ClusteringParams

    settings = ts.settings_for(_settings(), "two_stage", shape_cut=4.0, chem_cut=0.6)
    assert isinstance(settings, ClusteringParams)
    assert settings.mode == "two_stage"
    assert settings.shape.criterion == "threshold" and settings.shape.threshold == 4.0
    assert settings.chem.criterion == "threshold" and settings.chem.threshold == 0.6


def test_affinity_settings_use_the_single_stage_path():
    settings = ts.settings_for(_settings(), "affinity", damping=0.77)
    assert settings.mode == "single"
    assert settings.single.method == "affinity"
    assert settings.single.damping == 0.77


def test_the_mirrored_figure_has_two_panels_and_two_cuts():
    m = _matrices()
    base = _settings()
    result = ts.preview(m, ts.settings_for(base, "two_stage", shape_cut=4.0, chem_cut=0.6))
    figure = ts.mirrored_figure(m, result, 4.0, 0.6)
    assert len(figure.data) > 0
    assert len(figure.layout.shapes) == 2, "expected a cut line in each panel"
    assert [a.text for a in figure.layout.annotations] == [
        "shape", "chemistry, within each group"]


def test_the_chemistry_panel_grows_inward():
    """Mirrored means its height axis runs the other way."""
    m = _matrices()
    result = ts.preview(m, ts.settings_for(_settings(), "two_stage",
                                           shape_cut=4.0, chem_cut=0.6))
    figure = ts.mirrored_figure(m, result, 4.0, 0.6)
    assert figure.layout.xaxis2.autorange == "reversed"


def test_a_group_of_one_contributes_no_chemistry_tree():
    m = _matrices()
    singles = np.arange(len(m.ids))          # every pocket its own shape group
    assert all(data is None for _g, _members, data in ts.chem_subtrees(m, singles))


def test_relative_cuts_write_relative_settings():
    settings = ts.settings_for(_settings(), "two_stage", shape_cut=0.5, chem_cut=0.3,
                               criterion="relative")
    assert (settings.shape.criterion, settings.shape.threshold) == ("relative", 0.5)
    assert (settings.chem.criterion, settings.chem.threshold) == ("relative", 0.3)
    single = ts.settings_for(_settings(), "hierarchical", combined_cut=0.4, criterion="relative")
    assert (single.single.criterion, single.single.threshold) == ("relative", 0.4)
    whole = ts.settings_for(_settings(), "two_stage", shape_cut=0.5, chem_cut=1.0, criterion="relative")
    assert ts.preview(_matrices(), whole).n_clusters == 2   # a full-height chemistry cut never splits


def test_a_relative_shape_only_cut():
    settings = ts.shape_only_settings(_settings(), 0.5, criterion="relative")
    assert (settings.mode, settings.shape.criterion, settings.shape.threshold) == (
        "shape_only", "relative", 0.5)


def test_the_preview_carries_no_silhouette():
    assert "silhouette" not in ts.Preview._fields
