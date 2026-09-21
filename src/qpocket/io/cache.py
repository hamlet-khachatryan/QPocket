"""On-disk caches: geodesic matrices and fitted scalers."""
from pathlib import Path
from typing import Tuple

import numpy as np

from qpocket.types import Scaler


def save_geodesic(path: Path, d_geo: np.ndarray, features: np.ndarray) -> None:
    """Write a pocket's geodesic matrix and raw features.

    Args:
        path: Target .npz file.
        d_geo: (N, N) geodesic matrix.
        features: (N, D) raw features.
    """
    n_points = len(d_geo)
    weights = np.ones(n_points) / n_points  # kept for readers of older caches
    np.savez_compressed(path, D_geo=d_geo, features=features, weights=weights)


def load_geodesic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a geodesic cache.

    Args:
        path: .npz written by save_geodesic.

    Returns:
        (geodesic matrix, raw features).
    """
    with np.load(path) as data:
        return data["D_geo"], data["features"]


def save_scaler(path: Path, scaler: Scaler) -> None:
    """Write a fitted scaler.

    Args:
        path: Target .npz file.
        scaler: Scaler to store.
    """
    np.savez(path, mean=scaler.mean, scale=scaler.scale)


def load_scaler(path: Path) -> Scaler:
    """Read a scaler written by save_scaler.

    Args:
        path: .npz file.

    Returns:
        The scaler.
    """
    with np.load(path) as data:
        return Scaler(mean=data["mean"], scale=data["scale"])
