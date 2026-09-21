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
    AFFINITY,
    KMEDOIDS,
    best_k_by_silhouette,
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


# --- silhouette-chosen k --------------------------------------------------


def test_best_k_finds_the_planted_cluster_count():
    """Three tight, far-apart blobs: silhouette peaks at three and nowhere else."""
    d = distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 200.0, 201.0, 202.0])
    assert best_k_by_silhouette(d, method="average", k_range=(2, 6)).n_clusters == 3


def test_best_k_respects_the_upper_bound():
    d = distances([0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 200.0, 201.0, 202.0])
    assert best_k_by_silhouette(d, method="average", k_range=(2, 2)).n_clusters == 2


def test_best_k_still_returns_a_grouping_for_a_tiny_run():
    """Three structures leave no k where silhouette is defined; it must not raise."""
    result = best_k_by_silhouette(distances([0.0, 1.0, 50.0]), method="average")
    assert result.n_clusters >= 1


def test_best_k_refuses_affinity_propagation():
    d = distances([0.0, 1.0, 100.0, 101.0])
    with pytest.raises(ValueError, match="affinity"):
        best_k_by_silhouette(d, method=AFFINITY)


def test_two_stage_resolves_k_by_silhouette_when_it_is_not_pinned(two_axes):
    """With k unpinned, stage one must still land on the three real shape groups."""
    shape_d, chem_d = two_axes
    result = two_stage_labels(shape_d, chem_d, shape_k=None, shape_k_range=(2, 6))
    assert result.shape_clustering.n_clusters == 3
