"""Tests for qpocket_pipeline.viz.pockets.

Everything here rests on one claim: pocket clouds share a lattice, so comparing
them is set arithmetic rather than interpolation. These tests check that claim's
machinery on synthetic lattices with answers known by hand, and pin the failure
mode that a wrong spacing would cause — silent nonsense rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpocket_pipeline.viz.pockets import (
    LATTICE_TOLERANCE,
    NOISE_FLOOR,
    field_difference,
    infer_spacing,
    jaccard,
    lattice_deviation,
    occupancy,
    overlap_matrix,
    pocket_metrics,
    shares_lattice,
    volume_table,
    voxel_index,
)

SPACING = 0.7


def lattice(*sites, spacing=SPACING):
    """Coordinates for integer lattice sites, e.g. ``lattice((0, 0, 0), (1, 0, 0))``."""
    return np.array(sites, dtype=float) * spacing


def cube(n, spacing=SPACING, offset=(0, 0, 0)):
    """An ``n × n × n`` block of sites, offset by whole lattice steps."""
    r = range(n)
    sites = [(i + offset[0], j + offset[1], k + offset[2]) for i in r for j in r for k in r]
    return lattice(*sites, spacing=spacing)


# ---------------------------------------------------------------------------
# infer_spacing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing", [0.35, 0.7, 1.0, 1.5])
def test_spacing_is_recovered_from_a_clean_cloud(spacing):
    assert infer_spacing(cube(4, spacing=spacing)) == pytest.approx(spacing)


def test_spacing_survives_float_noise_between_clouds():
    """The failure this function exists for.

    Pooling clouds written at different times puts 28.699997 beside 28.700001, and
    the naive smallest-gap rule then returns 4e-6 instead of 0.7.
    """
    rng = np.random.default_rng(0)
    clouds = [cube(4) + rng.normal(0.0, 2e-6, size=(64, 3)) for _ in range(5)]
    pooled = np.vstack(clouds)
    assert infer_spacing(pooled) == pytest.approx(SPACING, abs=1e-4)


def test_spacing_ignores_gaps_below_the_noise_floor():
    """A duplicated site jittered by less than the floor must not become the step."""
    points = np.vstack([cube(4), cube(4)[:5] + NOISE_FLOOR / 2])
    spacing = infer_spacing(points)
    assert spacing > NOISE_FLOOR
    assert spacing == pytest.approx(SPACING, rel=0.01)


def test_spacing_falls_back_to_the_smallest_gap():
    """A cloud where multiples of the step outnumber the step itself."""
    sites = [(0, 0, 0), (1, 0, 0), (3, 0, 0), (5, 0, 0), (7, 0, 0), (9, 0, 0)]
    assert infer_spacing(lattice(*sites)) == pytest.approx(SPACING)


def test_spacing_needs_more_than_one_point():
    with pytest.raises(ValueError, match="at least two points"):
        infer_spacing(np.zeros((1, 3)))


def test_spacing_refuses_a_cloud_with_no_usable_gaps():
    with pytest.raises(ValueError, match="noise floor"):
        infer_spacing(np.zeros((5, 3)))


def test_spacing_refuses_coordinates_that_are_not_on_a_lattice():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="on a lattice"):
        infer_spacing(rng.uniform(0, 10, size=(200, 3)))


# ---------------------------------------------------------------------------
# voxel_index, lattice_deviation, shares_lattice
# ---------------------------------------------------------------------------


def test_voxel_index_is_exact_for_lattice_points():
    sites = [(0, 0, 0), (1, 2, 3), (-4, 5, -6)]
    assert np.array_equal(voxel_index(lattice(*sites), SPACING), np.array(sites))


def test_voxel_index_absorbs_float_noise():
    points = lattice((3, -2, 7)) + 1e-5
    assert np.array_equal(voxel_index(points, SPACING), np.array([[3, -2, 7]]))


def test_voxel_index_rejects_a_nonpositive_spacing():
    with pytest.raises(ValueError, match="must be positive"):
        voxel_index(cube(2), 0.0)


def test_lattice_deviation_is_zero_on_the_lattice_and_half_off_it():
    assert lattice_deviation(cube(3), SPACING) == pytest.approx(0.0, abs=1e-12)
    assert lattice_deviation(cube(3) + SPACING / 2, SPACING) == pytest.approx(0.5)


def test_lattice_deviation_of_nothing_is_zero():
    assert lattice_deviation(np.empty((0, 3)), SPACING) == 0.0


def test_shares_lattice_spots_a_shifted_cloud():
    on = cube(3)
    off = cube(3) + SPACING * 0.4  # a fraction of a step: a different lattice
    assert shares_lattice([on, on + 5 * SPACING], SPACING)
    assert not shares_lattice([on, off], SPACING)


def test_shares_lattice_tolerates_noise_within_tolerance():
    rng = np.random.default_rng(2)
    noisy = cube(3) + rng.normal(0, SPACING * LATTICE_TOLERANCE / 4, size=(27, 3))
    assert shares_lattice([cube(3), noisy], SPACING)


# ---------------------------------------------------------------------------
# pocket_metrics and volume_table
# ---------------------------------------------------------------------------


def test_volume_is_site_count_times_voxel_volume():
    metrics = pocket_metrics(cube(3), SPACING)
    assert metrics.n_points == 27
    assert metrics.volume == pytest.approx(27 * SPACING**3)


def test_repeated_sites_are_counted_once():
    doubled = np.vstack([cube(2), cube(2)])
    assert pocket_metrics(doubled, SPACING).n_points == 8


def test_volume_of_an_empty_cloud_is_zero():
    metrics = pocket_metrics(np.empty((0, 3)), SPACING)
    assert (metrics.n_points, metrics.volume) == (0, 0.0)


def test_volume_table_keeps_its_keys():
    table = volume_table({"a": cube(2), "b": cube(3)}, SPACING)
    assert list(table) == ["a", "b"]
    assert table["a"].n_points == 8 and table["b"].n_points == 27


# ---------------------------------------------------------------------------
# occupancy
# ---------------------------------------------------------------------------


def test_identical_pockets_are_open_everywhere():
    result = occupancy([cube(2), cube(2), cube(2)], SPACING)
    assert result.n_pockets == 3
    assert len(result.coords) == 8
    assert np.all(result.counts == 3)
    assert np.all(result.fraction == 1.0)


def test_occupancy_counts_partial_sites():
    """Two pockets sharing half their sites: the shared half reads 1.0."""
    a = lattice((0, 0, 0), (1, 0, 0))
    b = lattice((1, 0, 0), (2, 0, 0))
    result = occupancy([a, b], SPACING)
    assert len(result.coords) == 3
    assert sorted(result.counts) == [1, 1, 2]
    assert sorted(result.fraction) == pytest.approx([0.5, 0.5, 1.0])


def test_occupancy_coordinates_come_back_in_angstrom():
    result = occupancy([lattice((2, 3, 4))], SPACING)
    assert result.coords[0] == pytest.approx([2 * SPACING, 3 * SPACING, 4 * SPACING])


def test_occupancy_ignores_empty_clouds_when_counting_pockets():
    result = occupancy([cube(2), np.empty((0, 3))], SPACING)
    assert result.n_pockets == 1
    assert np.all(result.fraction == 1.0)


def test_occupancy_of_nothing_is_empty():
    result = occupancy([], SPACING)
    assert result.n_pockets == 0
    assert result.coords.shape == (0, 3)


def test_occupancy_deduplicates_within_a_pocket():
    """A repeated site must not let one pocket count twice."""
    doubled = np.vstack([cube(2), cube(2)])
    result = occupancy([doubled, cube(2)], SPACING)
    assert np.all(result.counts == 2)
    assert np.all(result.fraction == 1.0)


# ---------------------------------------------------------------------------
# jaccard and overlap_matrix
# ---------------------------------------------------------------------------


def test_identical_pockets_overlap_completely():
    assert jaccard(cube(3), cube(3), SPACING) == 1.0


def test_disjoint_pockets_do_not_overlap():
    assert jaccard(cube(2), cube(2) + 10 * SPACING, SPACING) == 0.0


def test_overlap_is_intersection_over_union():
    a = lattice((0, 0, 0), (1, 0, 0), (2, 0, 0))
    b = lattice((2, 0, 0), (3, 0, 0))
    # one shared site, four in the union
    assert jaccard(a, b, SPACING) == pytest.approx(0.25)


def test_two_empty_pockets_are_called_identical():
    assert jaccard(np.empty((0, 3)), np.empty((0, 3)), SPACING) == 1.0


def test_overlap_matrix_is_symmetric_with_a_unit_diagonal():
    clouds = [cube(2), cube(2) + SPACING, cube(3)]
    matrix = overlap_matrix(clouds, SPACING)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_overlap_matrix_agrees_with_the_pairwise_function():
    clouds = [cube(2), cube(2) + SPACING, cube(3) + 2 * SPACING]
    matrix = overlap_matrix(clouds, SPACING)
    for i in range(3):
        for j in range(3):
            assert matrix[i, j] == pytest.approx(jaccard(clouds[i], clouds[j], SPACING))


# ---------------------------------------------------------------------------
# field_difference
# ---------------------------------------------------------------------------


def test_difference_is_zero_against_itself():
    points = cube(2)
    values = np.arange(len(points), dtype=float)
    result = field_difference(points, values, points, values, SPACING)
    assert result.n_shared == len(points)
    assert result.n_only_reference == result.n_only_other == 0
    assert np.allclose(result.delta, 0.0)


def test_difference_is_other_minus_reference():
    points = lattice((0, 0, 0), (1, 0, 0))
    result = field_difference(points, [1.0, 2.0], points, [4.0, 6.0], SPACING)
    assert sorted(result.delta) == pytest.approx([3.0, 4.0])


def test_difference_only_covers_shared_sites():
    ref = lattice((0, 0, 0), (1, 0, 0))
    other = lattice((1, 0, 0), (2, 0, 0))
    result = field_difference(ref, [1.0, 1.0], other, [5.0, 9.0], SPACING)
    assert result.n_shared == 1
    assert result.n_only_reference == 1 and result.n_only_other == 1
    assert result.delta == pytest.approx([4.0])
    assert result.coords[0] == pytest.approx([SPACING, 0.0, 0.0])


def test_difference_of_disjoint_pockets_is_empty_not_an_error():
    result = field_difference(cube(2), np.zeros(8), cube(2) + 9 * SPACING, np.ones(8), SPACING)
    assert result.n_shared == 0
    assert result.coords.shape == (0, 3)
    assert result.delta.shape == (0,)


def test_difference_averages_repeated_sites():
    points = np.vstack([lattice((0, 0, 0)), lattice((0, 0, 0))])
    result = field_difference(points, [0.0, 0.0], points, [1.0, 3.0], SPACING)
    assert result.delta == pytest.approx([2.0])


def test_difference_checks_the_value_count():
    with pytest.raises(ValueError, match="values for"):
        field_difference(cube(2), np.zeros(3), cube(2), np.zeros(8), SPACING)


def test_a_cloud_off_lattice_by_under_half_a_step_is_snapped_silently():
    """Why shares_lattice has to be checked before trusting a difference map.

    voxel_index snaps to the nearest site, so a cloud offset by 0.4 of a step maps
    onto the same sites and the difference looks perfectly well defined. Nothing in
    the set arithmetic can notice; only the deviation check can.
    """
    ref = cube(3)
    shifted = cube(3) + SPACING * 0.4

    result = field_difference(ref, np.zeros(27), shifted, np.ones(27), SPACING)
    assert result.n_shared == 27  # looks fine, and is not
    assert not shares_lattice([ref, shifted], SPACING)  # the only warning available
    assert lattice_deviation(shifted, SPACING) == pytest.approx(0.4)


def test_a_cloud_off_lattice_by_over_half_a_step_shares_nothing():
    ref = lattice((0, 0, 0))
    shifted = ref + SPACING * 0.6
    result = field_difference(ref, [0.0], shifted, [1.0], SPACING)
    assert result.n_shared == 0
