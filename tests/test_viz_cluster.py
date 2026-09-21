"""Tests for qpocket_pipeline.viz.cluster.

Re-clustering is only useful if it is honest: the cut has to do what it says, the
labels have to stay attached to the same groups as the cut moves, and the
comparison against the stored labels has to survive the fact that cluster numbers
mean nothing across two groupings.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpocket_pipeline.viz.cluster import (
    AFFINITY,
    COUNT,
    CRITERIA,
    LINKAGES,
    METHODS,
    PIPELINE_DAMPING,
    PIPELINE_PREFERENCE_PCT,
    THRESHOLD,
    affinity_labels,
    block_order,
    cluster_labels,
    compare_labelings,
    dendrogram_segments,
    exemplars,
    linkage_matrix,
    silhouette,
    threshold_for_clusters,
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


@pytest.mark.parametrize("method", LINKAGES)
def test_every_offered_linkage_works_on_precomputed_distances(method, three_groups):
    d, _ = three_groups
    link = linkage_matrix(d, method)
    assert link.shape == (len(d) - 1, 4)


@pytest.mark.parametrize("method", ["ward", "centroid", "median", "nonsense"])
def test_euclidean_only_linkages_are_refused(method, three_groups):
    """These assume coordinates and silently return nonsense for distances."""
    d, _ = three_groups
    with pytest.raises(ValueError, match="not defined for a precomputed"):
        linkage_matrix(d, method)


def test_linkage_tolerates_a_slightly_asymmetric_matrix():
    d = np.array([[0.0, 1.0, 2.0], [1.000000001, 0.0, 2.0], [2.0, 2.0, 0.0]])
    assert linkage_matrix(d, "average").shape == (2, 4)


# ---------------------------------------------------------------------------
# cluster_labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_cluster_count_criterion_returns_that_many_clusters(k, three_groups):
    d, _ = three_groups
    result = cluster_labels(d, "average", COUNT, n_clusters=k)
    assert result.n_clusters == k
    assert len(result.labels) == len(d)


def test_cluster_count_recovers_planted_groups(three_groups):
    d, truth = three_groups
    result = cluster_labels(d, "average", COUNT, n_clusters=3)
    assert same_partition(result.labels, truth)


def test_labels_are_zero_based_and_contiguous(three_groups):
    d, _ = three_groups
    labels = cluster_labels(d, "average", COUNT, n_clusters=3).labels
    assert sorted(np.unique(labels)) == [0, 1, 2]


def test_largest_cluster_is_labelled_zero(three_groups):
    """Cluster numbers from fcluster follow tree order, which shuffles as the cut
    moves; ordering by size keeps a colour attached to the same kind of group."""
    d, _ = three_groups
    labels = cluster_labels(d, "average", COUNT, n_clusters=3).labels
    counts = [int(np.sum(labels == v)) for v in sorted(np.unique(labels))]
    assert counts == sorted(counts, reverse=True)


def test_cluster_count_is_clamped_to_the_structure_count(three_groups):
    d, _ = three_groups
    assert cluster_labels(d, "average", COUNT, n_clusters=999).n_clusters == len(d)


def test_threshold_criterion_cuts_at_the_given_height(three_groups):
    d, _ = three_groups
    link = linkage_matrix(d, "average")
    cut = threshold_for_clusters(link, 3)
    result = cluster_labels(d, "average", THRESHOLD, threshold=cut)
    assert result.n_clusters == 3
    assert result.threshold == pytest.approx(cut)


def test_a_high_threshold_merges_everything(three_groups):
    d, _ = three_groups
    result = cluster_labels(d, "average", THRESHOLD, threshold=1e6)
    assert result.n_clusters == 1


def test_a_zero_threshold_splits_everything(three_groups):
    d, _ = three_groups
    result = cluster_labels(d, "average", THRESHOLD, threshold=0.0)
    assert result.n_clusters == len(d)


def test_both_criteria_agree_when_asked_for_the_same_cut(three_groups):
    d, _ = three_groups
    by_count = cluster_labels(d, "complete", COUNT, n_clusters=3)
    by_height = cluster_labels(d, "complete", THRESHOLD, threshold=by_count.threshold)
    assert same_partition(by_count.labels, by_height.labels)


def test_unknown_criterion_is_reported(three_groups):
    d, _ = three_groups
    with pytest.raises(ValueError, match="unknown criterion"):
        cluster_labels(d, "average", "elbow", n_clusters=2)


def test_criteria_names_are_the_two_offered():
    assert CRITERIA == (COUNT, THRESHOLD)


def test_reclustering_is_reproducible(three_groups):
    d, _ = three_groups
    first = cluster_labels(d, "average", COUNT, n_clusters=3)
    second = cluster_labels(d, "average", COUNT, n_clusters=3)
    assert np.array_equal(first.labels, second.labels)
    assert first.threshold == pytest.approx(second.threshold)


def test_linkage_matrix_is_returned_for_reuse(three_groups):
    d, _ = three_groups
    result = cluster_labels(d, "average", COUNT, n_clusters=3)
    assert result.linkage_matrix.shape == (len(d) - 1, 4)
    assert result.method == "average"


# ---------------------------------------------------------------------------
# threshold_for_clusters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_threshold_for_clusters_round_trips(k, three_groups):
    from scipy.cluster.hierarchy import fcluster

    d, _ = three_groups
    link = linkage_matrix(d, "average")
    cut = threshold_for_clusters(link, k)
    assert len(np.unique(fcluster(link, t=cut, criterion="distance"))) == k


def test_threshold_for_one_cluster_sits_above_the_tallest_merge(three_groups):
    d, _ = three_groups
    link = linkage_matrix(d, "average")
    assert threshold_for_clusters(link, 1) > link[:, 2].max()


# ---------------------------------------------------------------------------
# silhouette
# ---------------------------------------------------------------------------


def test_silhouette_is_high_for_the_planted_grouping(three_groups):
    d, truth = three_groups
    mean, per_structure = silhouette(d, truth)
    assert mean > 0.9
    assert per_structure.shape == (len(d),)


def test_silhouette_prefers_the_right_cluster_count(three_groups):
    d, _ = three_groups
    scores = {
        k: cluster_labels(d, "average", COUNT, n_clusters=k).silhouette for k in (2, 3, 4, 5)
    }
    assert max(scores, key=lambda k: scores[k]) == 3


def test_silhouette_is_undefined_for_a_single_cluster(three_groups):
    d, _ = three_groups
    assert silhouette(d, np.zeros(len(d), dtype=int)) == (None, None)


def test_silhouette_is_undefined_when_every_structure_is_its_own_cluster(three_groups):
    d, _ = three_groups
    assert silhouette(d, np.arange(len(d))) == (None, None)


def test_clustering_reports_no_silhouette_when_it_is_undefined(three_groups):
    d, _ = three_groups
    result = cluster_labels(d, "average", COUNT, n_clusters=1)
    assert result.silhouette is None
    assert result.silhouette_per_structure is None


# ---------------------------------------------------------------------------
# compare_labelings
# ---------------------------------------------------------------------------


def test_identical_groupings_agree_perfectly(three_groups):
    _, truth = three_groups
    result = compare_labelings(truth, truth)
    assert result.ari == pytest.approx(1.0)
    assert result.n_moved == 0
    assert (result.pairs_split, result.pairs_joined) == (0, 0)


def test_renumbered_groupings_still_agree_perfectly(three_groups):
    """Cluster numbers carry no meaning across two groupings."""
    _, truth = three_groups
    renumbered = np.array([{0: 2, 1: 0, 2: 1}[v] for v in truth])
    result = compare_labelings(truth, renumbered)
    assert result.ari == pytest.approx(1.0)
    assert result.n_moved == 0


def test_one_structure_moving_is_counted_as_one(three_groups):
    _, truth = three_groups
    moved = truth.copy()
    moved[0] = 1
    result = compare_labelings(truth, moved)
    assert result.n_moved == 1
    assert result.moved == (0,)
    assert result.ari < 1.0


def test_splitting_a_cluster_is_reported_as_split_pairs(three_groups):
    _, truth = three_groups
    split = truth.copy()
    split[:3] = 3  # tear the first group in half
    result = compare_labelings(truth, split)
    assert result.pairs_split > 0
    assert result.pairs_joined == 0
    assert result.n_clusters_a == 3 and result.n_clusters_b == 4


def test_merging_clusters_is_reported_as_joined_pairs(three_groups):
    _, truth = three_groups
    merged = np.where(truth == 2, 1, truth)
    result = compare_labelings(truth, merged)
    assert result.pairs_joined > 0
    assert result.pairs_split == 0


def test_comparison_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="structures"):
        compare_labelings([0, 1], [0, 1, 1])


# ---------------------------------------------------------------------------
# block_order
# ---------------------------------------------------------------------------


def test_block_order_is_a_permutation(three_groups):
    d, truth = three_groups
    order = block_order(d, truth)
    assert sorted(order) == list(range(len(d)))


def test_block_order_keeps_clusters_contiguous(three_groups):
    d, truth = three_groups
    order = block_order(d, truth)
    runs = [truth[i] for i in order]
    changes = sum(1 for a, b in zip(runs, runs[1:]) if a != b)
    assert changes == len(np.unique(truth)) - 1


def test_block_order_puts_similar_structures_next_to_each_other(three_groups):
    d, truth = three_groups
    ordered = block_order(d, truth)
    naive = list(range(len(d)))
    neighbour = lambda idx: np.mean([d[idx[i], idx[i + 1]] for i in range(len(idx) - 1)])  # noqa: E731
    assert neighbour(ordered) <= neighbour(naive) * 1.001


def test_block_order_without_labels_still_orders_by_similarity(three_groups):
    d, truth = three_groups
    order = block_order(d)
    assert sorted(order) == list(range(len(d)))
    runs = [truth[i] for i in order]
    # Similar structures adjacent means the planted groups come out contiguous.
    changes = sum(1 for a, b in zip(runs, runs[1:]) if a != b)
    assert changes == len(np.unique(truth)) - 1


def test_block_order_can_keep_input_order_inside_a_cluster(three_groups):
    d, truth = three_groups
    order = block_order(d, truth, within_cluster=False)
    assert order == [i for value in sorted(np.unique(truth)) for i in np.flatnonzero(truth == value)]


def test_block_order_handles_tiny_clusters(three_groups):
    d, _ = three_groups
    singletons = np.arange(len(d))
    assert sorted(block_order(d, singletons)) == list(range(len(d)))


# ---------------------------------------------------------------------------
# exemplars
# ---------------------------------------------------------------------------


def test_exemplar_is_the_most_central_member(three_groups):
    d, truth = three_groups
    picks = exemplars(d, truth)
    assert sorted(picks) == [0, 1, 2]
    for cluster, position in picks.items():
        members = np.flatnonzero(truth == cluster)
        assert position in members
        totals = {int(m): d[np.ix_([m], members)].sum() for m in members}
        assert totals[position] == pytest.approx(min(totals.values()))


def test_exemplar_of_a_singleton_is_itself():
    d = distances([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    assert exemplars(d, [0, 1]) == {0: 0, 1: 1}


# ---------------------------------------------------------------------------
# dendrogram_segments
# ---------------------------------------------------------------------------


def test_dendrogram_has_one_segment_per_merge(three_groups):
    d, _ = three_groups
    link = linkage_matrix(d, "average")
    data = dendrogram_segments(link)
    assert len(data.segments) == len(d) - 1
    assert len(data.leaf_order) == len(d)
    assert sorted(data.leaf_order) == list(range(len(d)))


def test_dendrogram_segments_have_four_points_each(three_groups):
    d, _ = three_groups
    for xs, ys in dendrogram_segments(linkage_matrix(d, "average")).segments:
        assert len(xs) == len(ys) == 4


def test_dendrogram_leaf_positions_match_scipy_convention(three_groups):
    d, _ = three_groups
    data = dendrogram_segments(linkage_matrix(d, "average"))
    assert data.leaf_positions[0] == pytest.approx(5.0)
    assert np.allclose(np.diff(data.leaf_positions), 10.0)


def test_dendrogram_height_matches_the_tallest_merge(three_groups):
    d, _ = three_groups
    link = linkage_matrix(d, "average")
    data = dendrogram_segments(link)
    assert data.max_height == pytest.approx(link[:, 2].max())
    assert max(max(ys) for _, ys in data.segments) == pytest.approx(data.max_height)


def test_dendrogram_leaf_order_matches_block_order_without_labels(three_groups):
    """Both come from the same average-linkage tree, so they cannot disagree."""
    d, _ = three_groups
    data = dendrogram_segments(linkage_matrix(d, "average"))
    assert list(data.leaf_order) == block_order(d)


# ---------------------------------------------------------------------------
# affinity propagation: the pipeline's own method
# ---------------------------------------------------------------------------


def test_affinity_is_the_default_method(three_groups):
    d, _ = three_groups
    assert cluster_labels(d).method == AFFINITY


def test_methods_lead_with_the_pipelines_own(three_groups):
    assert METHODS[0] == AFFINITY
    assert set(METHODS[1:]) == set(LINKAGES)


def test_affinity_finds_the_planted_groups(three_groups):
    d, truth = three_groups
    result = affinity_labels(d)
    assert same_partition(result.labels, truth)


def test_affinity_chooses_its_own_cluster_count(three_groups):
    """It takes no k, which is why its knob is a preference percentile."""
    d, _ = three_groups
    result = cluster_labels(d, AFFINITY, n_clusters=99)
    assert result.n_clusters == 3


def test_affinity_nominates_exemplars(three_groups):
    d, truth = three_groups
    result = affinity_labels(d)
    assert result.exemplar_positions is not None
    assert len(result.exemplar_positions) == result.n_clusters
    # One exemplar per cluster, and each inside the cluster it represents.
    assert len({result.labels[p] for p in result.exemplar_positions}) == result.n_clusters


def test_affinity_builds_no_tree(three_groups):
    d, _ = three_groups
    result = affinity_labels(d)
    assert result.linkage_matrix is None
    assert np.isnan(result.threshold)


def test_affinity_is_refused_by_linkage_matrix(three_groups):
    d, _ = three_groups
    with pytest.raises(ValueError, match="not defined for a precomputed"):
        linkage_matrix(d, AFFINITY)


def test_median_preference_matches_scikit_learns_default(three_groups):
    """The pipeline passes no preference, so percentile 50 has to be identical."""
    from sklearn.cluster import AffinityPropagation

    d, _ = three_groups
    theirs = AffinityPropagation(
        affinity="precomputed", damping=PIPELINE_DAMPING, max_iter=300, random_state=42
    ).fit_predict(-d)
    ours = affinity_labels(d, preference_percentile=PIPELINE_PREFERENCE_PCT)
    assert same_partition(theirs, ours.labels)


def test_preference_trends_towards_more_clusters(three_groups):
    """The extremes bracket, but the count can jump between them.

    Message passing settles on a different fixed point for a small change in
    preference, so real data does not climb smoothly; only the direction between
    the extremes is a property worth asserting.
    """
    d, _ = three_groups
    counts = [affinity_labels(d, preference_percentile=p).n_clusters for p in (5.0, 50.0, 95.0)]
    assert counts[0] <= counts[-1]
    assert len(set(counts)) > 1, "the knob has to do something"


def test_affinity_labels_are_size_ordered(three_groups):
    d, _ = three_groups
    labels = affinity_labels(d).labels
    counts = [int(np.sum(labels == v)) for v in sorted(np.unique(labels))]
    assert counts == sorted(counts, reverse=True)


def test_affinity_is_reproducible(three_groups):
    d, _ = three_groups
    first, second = affinity_labels(d), affinity_labels(d)
    assert np.array_equal(first.labels, second.labels)
    assert first.exemplar_positions == second.exemplar_positions


def test_affinity_reports_a_silhouette(three_groups):
    d, _ = three_groups
    result = affinity_labels(d)
    assert result.silhouette is not None and result.silhouette > 0.9


def test_unknown_method_is_reported(three_groups):
    d, _ = three_groups
    with pytest.raises(ValueError, match="not defined for a precomputed"):
        cluster_labels(d, "k-means")
