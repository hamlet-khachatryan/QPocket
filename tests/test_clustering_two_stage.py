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
    KMEDOIDS,
    _stage_labels,
    cluster_labels,
    exemplars,
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


def test_stage_one_recovers_the_shape_groups(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d)
    assert groups_of(result.shape_labels) == groups_of(SHAPE_TRUTH)


def test_stage_one_ignores_the_chemistry_axis(two_axes):
    """Swapping the chemistry matrix cannot move a single structure in stage one."""
    shape_d, chem_d = two_axes
    baseline = two_stage_labels(shape_d, chem_d)
    shuffled = two_stage_labels(shape_d, distances(np.zeros(12)))
    assert groups_of(baseline.shape_labels) == groups_of(shuffled.shape_labels)


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


def test_chem_labels_are_numbered_within_their_shape_cluster(two_axes):
    """Every subdivided shape cluster restarts its chemistry numbering at zero."""
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d, min_cluster_size=4)
    for shape_value in np.unique(result.shape_labels):
        members = np.flatnonzero(result.shape_labels == shape_value)
        assert result.chem_labels[members].min() == 0


def test_a_shape_cluster_below_min_size_is_left_whole(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d, min_cluster_size=99)
    assert set(result.chem_labels.tolist()) == {0}


def test_a_cluster_left_whole_records_no_stage_two_result(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d, min_cluster_size=99)
    assert all(v is None for v in result.chem_clusterings.values())


# --- composite labels -----------------------------------------------------


def test_composite_labels_recover_all_six_true_groups(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d, min_cluster_size=4)
    truth = {frozenset(np.flatnonzero((SHAPE_TRUTH == g) & (CHEM_TRUTH == c)).tolist())
             for g in range(3) for c in range(2)}
    assert groups_of(result.labels) == truth


def test_composite_labels_are_contiguous_from_zero(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d, min_cluster_size=4)
    assert sorted(set(result.labels.tolist())) == list(range(6))


def test_composite_labels_are_ordered_largest_cluster_first():
    """A lopsided split must still number the biggest group zero."""
    shape = distances([0.0, 0.1, 0.2, 0.3, 0.4, 100.0])
    chem = distances(np.zeros(6))
    result = two_stage_labels(shape, chem, shape_k=2, min_cluster_size=99)
    counts = [int((result.labels == v).sum()) for v in sorted(set(result.labels.tolist()))]
    assert counts == sorted(counts, reverse=True)


def test_exemplars_are_drawn_from_the_combined_matrix(two_axes):
    """The medoid reported must be the medoid under the matrix handed in."""
    shape_d, chem_d = two_axes
    combined = 0.5 * shape_d + 0.5 * chem_d
    result = two_stage_labels(shape_d, chem_d, min_cluster_size=4,
                              combined_distances=combined)
    expected = exemplars(combined, result.labels)
    assert set(result.exemplar_positions) == set(expected.values())


def test_mismatched_matrices_are_refused(two_axes):
    shape_d, _ = two_axes
    with pytest.raises(ValueError, match="structures"):
        two_stage_labels(shape_d, distances(np.zeros(5)))


# --- k-medoids ------------------------------------------------------------


def test_kmedoids_finds_the_planted_groups():
    d = distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    result = kmedoids_labels(d, n_clusters=2)
    assert groups_of(result.labels) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}


def test_kmedoids_picks_the_central_member_as_the_medoid():
    """The medoid of 0, 1, 2 is 1; of 100, 101, 102 is 4. Nothing else is central."""
    d = distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    result = kmedoids_labels(d, n_clusters=2)
    assert set(result.exemplar_positions) == {1, 4}


def test_kmedoids_medoids_are_real_members_of_their_cluster():
    d = distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    result = kmedoids_labels(d, n_clusters=2)
    for position in result.exemplar_positions:
        assert 0 <= position < len(d)


def test_kmedoids_is_reproducible():
    d = distances([0.0, 1.0, 2.0, 50.0, 51.0, 52.0, 100.0, 101.0, 102.0])
    first = kmedoids_labels(d, n_clusters=3)
    second = kmedoids_labels(d, n_clusters=3)
    assert np.array_equal(first.labels, second.labels)


def test_kmedoids_survives_a_matrix_of_identical_points():
    """Zero distances make the seeding weights vanish; it must not divide by zero."""
    result = kmedoids_labels(distances(np.zeros(5)), n_clusters=2)
    assert len(result.labels) == 5


def test_kmedoids_clips_k_above_the_sample_count():
    result = kmedoids_labels(distances([0.0, 1.0, 2.0]), n_clusters=99)
    assert result.n_clusters <= 3


def test_kmedoids_is_reachable_through_cluster_labels():
    d = distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    result = cluster_labels(d, method=KMEDOIDS, n_clusters=2)
    assert result.method == KMEDOIDS


# --- relative cut ---------------------------------------------------------


def _two_blobs():
    return distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])


def labels(d, criterion, threshold):
    return cluster_labels(d, method="average", criterion=criterion, threshold=threshold).labels


def test_relative_cut_is_scale_free():
    d = _two_blobs()
    assert (labels(d, "relative", 0.5) == labels(d * 1000, "relative", 0.5)).all()
    assert len(set(labels(d, "relative", 0.5))) == 2


def test_relative_cut_on_two_pockets_does_not_raise():
    assert len(cluster_labels(np.array([[0, 1.0], [1.0, 0]]), method="average",
                              criterion="relative", threshold=0.5).labels) == 2


@pytest.mark.parametrize("n", [1, 2])
def test_relative_cut_on_a_tiny_stage_is_one_cluster(n):
    result = cluster_labels(distances(np.arange(float(n))), method="average",
                            criterion="relative", threshold=0.5)
    assert result.n_clusters == 1 and len(result.labels) == n


def test_relative_cut_on_identical_pockets_is_one_cluster():
    assert set(labels(distances(np.zeros(5)), "relative", 0.5).tolist()) == {0}


def test_count_without_k_is_refused():
    with pytest.raises(ValueError, match="set k or use criterion"):
        _stage_labels(_two_blobs(), "hierarchical", "average", k=None, criterion="count")


def test_kmedoids_without_k_is_refused():
    with pytest.raises(ValueError, match="set k or use criterion"):
        _stage_labels(_two_blobs(), "kmedoids", "average", k=None, criterion="count")


def test_relative_needs_a_hierarchical_stage():
    with pytest.raises(ValueError, match="hierarchical"):
        _stage_labels(_two_blobs(), "kmedoids", "average", k=None, criterion="relative",
                      threshold=0.5)


def test_no_silhouette_anywhere():
    import qpocket.core.clustering as cl
    assert not hasattr(cl, "silhouette") and not hasattr(cl, "best_k_by_silhouette")
    assert "silhouette" not in cl.Clustering._fields


def test_defaults_are_percentile_hierarchical():
    from qpocket.config import QPocketConfig

    c = QPocketConfig().distance.clustering
    assert (c.shape.method, c.shape.percentile) == ("hierarchical", 15.0)
    assert (c.chem.method, c.chem.percentile) == ("hierarchical", 20.0)
    assert c.single.method == "affinity"


def test_default_two_stage_recovers_the_planted_groups(two_axes):
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d)
    assert result.shape_clustering.n_clusters == 3 and len(set(result.labels.tolist())) == 6


def test_clustering_files_carry_no_silhouette(tmp_path, two_axes):
    from qpocket.config import ClusteringParams
    from qpocket.pipelines.clustering import cluster_pockets, write_clustering

    shape_d, chem_d = two_axes
    combined = shape_d + chem_d
    ids = [f"P{i}" for i in range(len(shape_d))]
    assignment = cluster_pockets(ClusteringParams(), shape_d, chem_d, combined)
    write_clustering(tmp_path, ids, assignment, combined, {})
    for path in tmp_path.iterdir():
        assert "silhouette" not in path.read_text().lower(), path.name
    import json
    params = json.loads((tmp_path / "clustering_params.json").read_text())
    for stage, default in (("shape", 15.0), ("chem", 20.0)):
        assert params[stage]["percentile"] == default and params[stage]["cut_height"] is not None
        assert "k_range" not in params[stage] and "criterion" not in params[stage]


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


def test_cli_recluster_takes_percentiles(tmp_path, two_axes):
    import json
    from click.testing import CliRunner
    from qpocket.cli import cli

    run = _old_run(tmp_path, two_axes)
    res = CliRunner().invoke(cli, ["recluster", str(run), "--shape-percentile", "30",
                                   "--chem-percentile", "20"])
    assert res.exit_code == 0, res.output
    params = json.loads((run / "metadata/clustering_params.json").read_text())
    assert (params["shape"]["percentile"], params["chem"]["percentile"]) == (30.0, 20.0)
    assert params["chem"]["method"] == "hierarchical"


def test_cli_recluster_on_an_old_run_works_without_options(tmp_path, two_axes):
    from click.testing import CliRunner
    from qpocket.cli import cli

    res = CliRunner().invoke(cli, ["recluster", str(_old_run(tmp_path, two_axes))])
    assert res.exit_code == 0, res.output


@pytest.mark.parametrize("criterion", ["relative", "threshold"])
def test_a_height_cut_without_a_threshold_says_so(criterion):
    with pytest.raises(ValueError, match=f"a '{criterion}' cut needs a threshold"):
        cluster_labels(_two_blobs(), method="average", criterion=criterion, threshold=None)
    with pytest.raises(ValueError, match=f"a '{criterion}' cut needs a threshold"):
        _stage_labels(_two_blobs(), "hierarchical", "average", k=None, criterion=criterion)


def test_cli_percentile_out_of_range_is_refused(tmp_path, two_axes):
    from click.testing import CliRunner
    from qpocket.cli import cli

    res = CliRunner().invoke(cli, ["recluster", str(_old_run(tmp_path, two_axes)), "--shape-percentile", "150"])
    assert res.exit_code != 0 and "--shape-percentile" in res.output and "Traceback" not in res.output


def test_cli_k_pins_a_kmedoids_stage(tmp_path, two_axes):
    import json
    from click.testing import CliRunner
    from qpocket.cli import cli

    run = _old_run(tmp_path, two_axes)
    res = CliRunner().invoke(cli, ["recluster", str(run), "--shape-k", "2"])
    assert res.exit_code == 0, res.output
    params = json.loads((run / "metadata/clustering_params.json").read_text())
    assert (params["shape"]["method"], params["shape"]["k"]) == ("kmedoids", 2)


def test_a_run_saved_before_percentiles_takes_the_defaults(tmp_path, two_axes):
    import json
    from qpocket.arms import recluster

    run = _old_run(tmp_path, two_axes)
    path = run / "metadata/clustering_params.json"
    params = json.loads(path.read_text())
    for stage in ("shape", "chem"):
        params[stage].pop("criterion")
        params[stage].pop("threshold")
    path.write_text(json.dumps(params))
    settings = recluster.load_settings(run)
    assert (settings.shape.percentile, settings.chem.percentile) == (15.0, 20.0)


@pytest.mark.parametrize("mode", ["two_stage", "single"])
def test_cluster_pockets_without_chem_raises(mode):
    from dataclasses import replace
    from qpocket.config import ClusteringParams
    from qpocket.pipelines.clustering import cluster_pockets

    d = distances([0, 1, 2, 10, 11, 12])
    with pytest.raises(ValueError, match="needs featurised pockets"):
        cluster_pockets(replace(ClusteringParams(), mode=mode), d, None, None)


def test_chemistry_percentile_is_site_wide():
    """Group B is chemically tight against the site: a site-wide p20 leaves it whole."""
    from dataclasses import replace

    from qpocket.config import ClusteringParams
    from qpocket.pipelines.clustering import cluster_pockets

    rng = np.random.default_rng(1)
    a, b = list(range(10)), list(range(10, 15))
    n = 15
    d_shape = np.full((n, n), 10.0)
    d_chem = np.full((n, n), 5.0)
    for group in (a, b):
        for i in group:
            for j in group:
                d_shape[i, j] = 0.1
    for i in a:
        for j in a:
            if i < j:
                d_chem[i, j] = d_chem[j, i] = rng.uniform(1.0, 10.0)
    for k, (i, j) in enumerate((i, j) for i in b for j in b if i < j):
        d_chem[i, j] = d_chem[j, i] = 0.10 + 0.01 * k
    np.fill_diagonal(d_shape, 0.0)
    np.fill_diagonal(d_chem, 0.0)
    base = ClusteringParams()
    settings = replace(base, mode="two_stage", shape=replace(base.shape, percentile=30.0),
                       chem=replace(base.chem, percentile=20.0))
    out = cluster_pockets(settings, d_shape, d_chem, (d_shape + d_chem) / 2)
    assert len(set(out.shape_labels[a])) == 1 and len(set(out.shape_labels[b])) == 1
    assert len(set(out.chem_labels[b])) == 1          # tight on the site's scale: not split
    assert len(set(out.chem_labels[a])) > 1
