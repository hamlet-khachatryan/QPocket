"""Set arithmetic over pocket point clouds."""

from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

#: How far off a lattice site a coordinate may sit, as a fraction of the spacing,
#: before the clouds cannot be treated as sharing one lattice.
LATTICE_TOLERANCE = 0.05


class Occupancy(NamedTuple):
    """How many pockets of a set occupy each lattice site.

    Attributes:
        coords: ``(m, 3)`` site coordinates, in Ångström.
        counts: Pockets containing each site.
        fraction: ``counts`` over the number of pockets, in ``(0, 1]``.
        n_pockets: Pockets the count is over.
    """

    coords: np.ndarray
    counts: np.ndarray
    fraction: np.ndarray
    n_pockets: int


class FieldDifference(NamedTuple):
    """One pocket's field minus another's, where both are defined.

    Attributes:
        coords: ``(m, 3)`` coordinates of the shared sites.
        delta: ``other - reference`` at those sites.
        n_shared: Sites both pockets occupy.
        n_only_reference: Sites only the reference occupies.
        n_only_other: Sites only the other pocket occupies.
    """

    coords: np.ndarray
    delta: np.ndarray
    n_shared: int
    n_only_reference: int
    n_only_other: int


class PocketMetrics(NamedTuple):
    """Size of one pocket.

    Attributes:
        n_points: Occupied lattice sites.
        volume: ``n_points`` times the voxel volume, in Å³.
    """

    n_points: int
    volume: float


def lattice_deviation(points: np.ndarray, spacing: float) -> float:
    """Largest distance from a lattice site, as a fraction of the spacing.

    Args:
        points: ``(n, 3)`` coordinates.
        spacing: Lattice spacing.

    Returns:
        ``0.0`` for points exactly on the lattice, ``0.5`` for the worst possible case of sitting halfway between sites.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(pts):
        return 0.0
    scaled = pts / spacing
    return float(np.abs(scaled - np.rint(scaled)).max())


#: Gaps below this are float noise, not lattice steps. Pooling clouds written by
#: different runs of the sampler puts 28.699997 next to 28.700001, and the naive
#: smallest gap would then be 4e-6 rather than the 0.7 Å step.
NOISE_FLOOR = 1e-3


def infer_spacing(points: np.ndarray, tolerance: float = LATTICE_TOLERANCE) -> float:
    """Infer the lattice spacing from one or more pooled point clouds.

    Args:
        points: ``(n, 3)`` coordinates, possibly several clouds stacked.
        tolerance: Allowed deviation from the lattice, as a fraction of the spacing.

    Returns:
        The spacing in Ångström.

    Raises:
        ValueError: If the cloud is too small to tell, or no candidate spacing puts the coordinates on a lattice.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        raise ValueError("need at least two points to infer a lattice spacing")

    gaps: List[float] = []
    for axis in range(3):
        unique = np.unique(pts[:, axis])
        if len(unique) < 2:
            continue
        diffs = np.diff(unique)
        gaps.extend(float(g) for g in diffs[diffs > NOISE_FLOOR])
    if not gaps:
        raise ValueError("no coordinate gaps above the noise floor; cannot infer a lattice")

    rounded = np.round(np.asarray(gaps), 4)
    values, counts = np.unique(rounded, return_counts=True)
    # Most common gap first, smallest gap as the fallback: a cloud with few
    # neighbouring sites can have its step outnumbered by multiples of the step.
    for candidate in (float(values[np.argmax(counts)]), float(values.min())):
        if candidate > 0 and lattice_deviation(pts, candidate) <= tolerance:
            return candidate
    raise ValueError(
        f"no candidate spacing puts these coordinates on a lattice within"
        f" {tolerance:g} of a site; tried {values[np.argmax(counts)]:g} and {values.min():g}"
    )


def voxel_index(points: np.ndarray, spacing: float) -> np.ndarray:
    """Map coordinates to integer lattice indices.

    Args:
        points: ``(n, 3)`` coordinates.
        spacing: Lattice spacing.

    Returns:
        ``(n, 3)`` integer indices.

    Raises:
        ValueError: If ``spacing`` is not positive.
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.rint(pts / spacing).astype(np.int64)


def shares_lattice(
    clouds: Iterable[np.ndarray], spacing: float, tolerance: float = LATTICE_TOLERANCE
) -> bool:
    """Whether every cloud sits on one common lattice.

    Args:
        clouds: Point clouds.
        spacing: Lattice spacing.
        tolerance: Allowed deviation, as a fraction of the spacing.

    Returns:
        ``True`` when set arithmetic across the clouds is exact.
    """
    return all(lattice_deviation(cloud, spacing) <= tolerance for cloud in clouds)


def _keys(index: np.ndarray) -> np.ndarray:
    """View integer index rows as scalars, so set operations work on them."""
    contiguous = np.ascontiguousarray(index.astype(np.int64))
    return contiguous.view([("i", np.int64), ("j", np.int64), ("k", np.int64)]).ravel()


def pocket_metrics(points: np.ndarray, spacing: float) -> PocketMetrics:
    """Occupied site count and volume of one pocket.

    Args:
        points: ``(n, 3)`` coordinates.
        spacing: Lattice spacing.

    Returns:
        The size of the pocket.
    """
    index = voxel_index(points, spacing)
    unique = np.unique(index, axis=0) if len(index) else index
    return PocketMetrics(len(unique), float(len(unique) * spacing**3))


def occupancy(clouds: Sequence[np.ndarray], spacing: float) -> Occupancy:
    """Count how many pockets occupy each site across a set of pockets.

    Args:
        clouds: One point cloud per pocket.
        spacing: Lattice spacing.

    Returns:
        Sites, counts and fractions over the union of the clouds.
    """
    per_cloud = [np.unique(voxel_index(c, spacing), axis=0) for c in clouds if len(np.asarray(c))]
    n_pockets = len(per_cloud)
    if not n_pockets:
        empty = np.empty((0, 3))
        return Occupancy(empty, np.empty(0, dtype=int), np.empty(0), 0)

    stacked = np.vstack(per_cloud)
    sites, counts = np.unique(stacked, axis=0, return_counts=True)
    return Occupancy(
        coords=sites.astype(float) * spacing,
        counts=counts,
        fraction=counts / float(n_pockets),
        n_pockets=n_pockets,
    )


def jaccard(a: np.ndarray, b: np.ndarray, spacing: float) -> float:
    """Overlap of two pockets as intersection over union of occupied sites.

    Args:
        a: First point cloud.
        b: Second point cloud.
        spacing: Lattice spacing.

    Returns:
        ``1.0`` for identical pockets, ``0.0`` for disjoint ones, and ``1.0`` when both are empty, since nothing distinguishes them.
    """
    keys_a = np.unique(_keys(voxel_index(a, spacing)))
    keys_b = np.unique(_keys(voxel_index(b, spacing)))
    if not len(keys_a) and not len(keys_b):
        return 1.0
    union = len(np.union1d(keys_a, keys_b))
    if not union:
        return 1.0
    return float(len(np.intersect1d(keys_a, keys_b)) / union)


def overlap_matrix(clouds: Sequence[np.ndarray], spacing: float) -> np.ndarray:
    """Pairwise Jaccard overlap between pockets.

    Args:
        clouds: One point cloud per pocket.
        spacing: Lattice spacing.

    Returns:
        Symmetric ``(n, n)`` matrix with ones on the diagonal.
    """
    keys = [np.unique(_keys(voxel_index(c, spacing))) for c in clouds]
    n = len(keys)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            union = len(np.union1d(keys[i], keys[j]))
            value = 1.0 if not union else len(np.intersect1d(keys[i], keys[j])) / union
            out[i, j] = out[j, i] = value
    return out


def _site_values(points: np.ndarray, values: Sequence[float], spacing: float):
    """Site keys and their values, averaging any repeats."""
    index = voxel_index(points, spacing)
    keys = _keys(index)
    vals = np.asarray(values, dtype=float)
    if len(vals) != len(keys):
        raise ValueError(f"got {len(vals)} values for {len(keys)} points")
    unique, inverse = np.unique(keys, return_inverse=True)
    totals = np.bincount(inverse, weights=vals, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    return unique, totals / counts, index, inverse


def field_difference(
    reference_points: np.ndarray,
    reference_values: Sequence[float],
    other_points: np.ndarray,
    other_values: Sequence[float],
    spacing: float,
) -> FieldDifference:
    """Difference between two pockets' fields, on the sites they share.

    Args:
        reference_points: ``(n, 3)`` coordinates of the reference pocket.
        reference_values: Field values at those points.
        other_points: ``(m, 3)`` coordinates of the pocket being compared.
        other_values: Field values at those points.
        spacing: Lattice spacing.

    Returns:
        Shared-site coordinates and ``other - reference`` there.
    """
    ref_keys, ref_vals, _, _ = _site_values(reference_points, reference_values, spacing)
    other_keys, other_vals, _, _ = _site_values(other_points, other_values, spacing)

    shared, ref_pos, other_pos = np.intersect1d(ref_keys, other_keys, return_indices=True)
    coords = (
        np.stack([shared["i"], shared["j"], shared["k"]], axis=-1).astype(float) * spacing
        if len(shared)
        else np.empty((0, 3))
    )
    return FieldDifference(
        coords=coords,
        delta=other_vals[other_pos] - ref_vals[ref_pos],
        n_shared=len(shared),
        n_only_reference=len(ref_keys) - len(shared),
        n_only_other=len(other_keys) - len(shared),
    )


def volume_table(
    clouds: Dict[str, np.ndarray], spacing: float
) -> Dict[str, PocketMetrics]:
    """Size of every pocket in a set.

    Args:
        clouds: Point cloud per pocket identifier.
        spacing: Lattice spacing.

    Returns:
        Metrics per identifier, in the input order.
    """
    return {key: pocket_metrics(cloud, spacing) for key, cloud in clouds.items()}
