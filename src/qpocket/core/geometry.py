"""Point-cloud geometry: boxes, overlap tests and cropping (pure)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def check_overlap(
    points_obj: np.ndarray,
    box_points: np.ndarray,
    tolerance: float = 2.0,
    use_majority: bool = True,
) -> bool:
    """Checks whether *points_obj* lies within the bounding box of *box_points*.

    Args:
        points_obj: Point(s) to test — 1-D for a single point, 2-D for many.
        box_points: Points that define the reference bounding box.
        tolerance: Padding (Å) added to every face of the bounding box.
        use_majority: When ``True`` (default), passes if > 50 % of atoms are inside; when ``False``, uses a 70 % threshold (legacy behaviour).

    Returns:
        ``True`` if the overlap criterion is satisfied.
    """
    if len(box_points) == 0:
        return False

    min_b = np.min(box_points, axis=0) - tolerance
    max_b = np.max(box_points, axis=0) + tolerance

    if points_obj.ndim == 1:
        return bool(np.all(points_obj >= min_b) and np.all(points_obj <= max_b))

    in_bounds = (points_obj >= min_b) & (points_obj <= max_b)
    mask = np.all(in_bounds, axis=1)
    num_inside = int(np.sum(mask))
    total = len(points_obj)

    threshold = 0.5 if use_majority else 0.7
    return num_inside > (total * threshold)


def create_bounding_box_pocket(
    center: List[float],
    dimensions: List[float],
    tolerance: float = 2.0,
) -> np.ndarray:
    """Creates an 8-corner point cloud representing a bounding box pocket.

    Args:
        center: ``[x, y, z]`` coordinates of the box centre.
        dimensions: ``[dx, dy, dz]`` half-dimensions of the box.
        tolerance: Additional padding (Å) on every face.

    Returns:
        Array of shape ``(8, 3)`` — the corners of the padded bounding box.
    """
    c = np.array(center)
    half = np.array(dimensions) / 2.0 + tolerance

    corners = [
        c + np.array([sx * half[0], sy * half[1], sz * half[2]])
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    return np.array(corners)


def crop_pocket_by_interactions(
    pocket_cloud: np.ndarray,
    interacting_coords: np.ndarray,
    padding: float = 5.0,
) -> np.ndarray:
    """Crops a pocket point cloud to points within *padding* Å of interacting residues.

    Args:
        pocket_cloud: Full pocket point cloud ``(N, 3)``.
        interacting_coords: Atom coordinates of the interacting residues.
        padding: Distance cutoff in Å.

    Returns:
        Subset of *pocket_cloud* within the cutoff.
    """
    distances = cdist(pocket_cloud, interacting_coords)
    mask = distances.min(axis=1) <= padding
    return pocket_cloud[mask]


def compute_pocket_bounding_box(
    pocket_coords: np.ndarray,
    padding: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives an axis-aligned bounding box from a pocket point cloud.

    Args:
        pocket_coords: Point cloud ``(N, 3)`` of the pocket cavity.
        padding: Expansion (Å) added to every face of the box.

    Returns:
        ``(min_corner, max_corner)`` each of shape ``(3,)``.
    """
    return (
        np.min(pocket_coords, axis=0) - padding,
        np.max(pocket_coords, axis=0) + padding,
    )

