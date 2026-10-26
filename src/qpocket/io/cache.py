"""On-disk caches: geodesic matrices and fitted scalers."""
from pathlib import Path
import numpy as np

from qpocket.types import Scaler


def save_geodesic(path: Path, d_geo: np.ndarray,
                  source_hash: str = "", radius: float = float("nan")) -> None:
    """Write a pocket's geodesic matrix, with what it came from.

    Args:
        path: Target .npz file.
        d_geo: (N, N) geodesic matrix.
        source_hash: sha256 of the binding_site.csv it was computed from.
        radius: Neighbour radius used.
    """
    n_points = len(d_geo)
    weights = np.ones(n_points) / n_points  # kept for readers of older caches
    np.savez_compressed(path, D_geo=d_geo, weights=weights,
                        source_hash=np.array(source_hash), radius=np.array(radius))


def is_current(path: Path, source_hash: str, radius: float) -> bool:
    """Whether a geodesic cache was built from this input and radius.

    Args:
        path: Cache .npz.
        source_hash: sha256 of the current binding_site.csv.
        radius: Neighbour radius wanted.

    Returns:
        False for a missing, unreadable or provenance-free cache.
    """
    try:
        with np.load(path) as data:
            return (str(data["source_hash"]) == source_hash
                    and float(data["radius"]) == float(radius))
    except Exception:
        return False


def load_geodesic(path: Path) -> np.ndarray:
    """Read a geodesic cache.

    Args:
        path: .npz written by save_geodesic.

    Returns:
        The (N, N) geodesic matrix; features of older caches are ignored.
    """
    with np.load(path) as data:
        return data["D_geo"]


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
