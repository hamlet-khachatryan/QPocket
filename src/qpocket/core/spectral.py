"""Geodesic graphs, spectral fingerprints and pocket distances (pure)."""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from qpocket.types import Fingerprint, Scaler

#: Fewer points than this cannot give a meaningful spectrum.
MIN_POINTS = 5


def geodesic_matrix(points: np.ndarray, neighbor_radius: float) -> Optional[np.ndarray]:
    """Normalised all-pairs geodesic distances over a radius graph.

    Args:
        points: (N, 3) pocket coordinates.
        neighbor_radius: Largest edge length (A).

    Returns:
        (N, N) matrix scaled to max 1, or None for fewer than 5 points or no edges.
    """
    n_points = len(points)
    if n_points < MIN_POINTS:
        return None

    pairs = cKDTree(points).query_pairs(r=neighbor_radius, output_type="ndarray")
    if len(pairs) == 0:
        return None

    rows_idx, cols_idx = pairs[:, 0], pairs[:, 1]
    edge_dists = np.linalg.norm(points[rows_idx] - points[cols_idx], axis=1)
    graph = csr_matrix(
        (np.concatenate([edge_dists, edge_dists]),
         (np.concatenate([rows_idx, cols_idx]), np.concatenate([cols_idx, rows_idx]))),
        shape=(n_points, n_points),
    )
    d_geo = shortest_path(csgraph=graph, directed=False, method="D")

    # Disconnected pairs get 1.5x the largest finite distance.
    finite_vals = d_geo[np.isfinite(d_geo)]
    max_finite = float(np.max(finite_vals)) if len(finite_vals) > 0 else 100.0
    d_geo[~np.isfinite(d_geo)] = max_finite * 1.5
    if d_geo.max() > 0:
        d_geo /= d_geo.max()
    return d_geo


def pad_or_truncate(vector: np.ndarray, target_k: int) -> np.ndarray:
    """Zero-pad or cut a vector to exactly target_k elements.

    Args:
        vector: 1-D array.
        target_k: Wanted length.

    Returns:
        Array of length target_k.
    """
    curr_k = len(vector)
    if curr_k < target_k:
        return np.pad(vector, (0, target_k - curr_k), "constant")
    return vector[:target_k]


def spectral_fingerprint(
    d_geo: np.ndarray, features: np.ndarray, global_k: int, sigma: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Laplacian eigenvalues and GFT power spectrum of already-scaled features.

    Args:
        d_geo: (N, N) geodesic matrix.
        features: (N, D) standardised features.
        global_k: Fingerprint length.
        sigma: Heat-kernel width.

    Returns:
        (eigenvalues, flattened spectrum), or (None, None) if the graph is too small.
    """
    n = d_geo.shape[0]

    # Gaussian heat kernel W_ij = exp(-d^2 / sigma^2), normalised Laplacian.
    w = np.exp(-(d_geo ** 2) / (sigma ** 2))
    np.fill_diagonal(w, 0)
    degrees = np.sum(w, axis=1)
    d_inv_sqrt = np.where(degrees > 1e-12, np.power(degrees, -0.5, where=degrees > 1e-12), 0.0)
    d_mat = np.diag(d_inv_sqrt)
    laplacian = np.eye(n) - (d_mat @ w @ d_mat)

    # Skip the trivial mode 0.
    k_req = min(global_k + 1, n - 1)
    if k_req < 1:
        return None, None
    eigvals, eigvecs = linalg.eigh(laplacian, subset_by_index=[0, k_req])
    lam, u = eigvals[1:], eigvecs[:, 1:]

    # Power spectrum is sign-invariant.
    spec_feat = np.abs(u.T @ features) ** 2
    curr_k = spec_feat.shape[0]
    if curr_k < global_k:
        spec_feat = np.pad(spec_feat, ((0, global_k - curr_k), (0, 0)), "constant")
    else:
        spec_feat = spec_feat[:global_k, :]
    return pad_or_truncate(lam, global_k), spec_feat.flatten()


def fit_scaler(feature_blocks: Sequence[np.ndarray]) -> Scaler:
    """Fit feature standardisation over all points of all pockets.

    Args:
        feature_blocks: Per-pocket (N_i, D) feature arrays.

    Returns:
        Mean and scale per feature.
    """
    fitted = StandardScaler().fit(np.vstack(list(feature_blocks)))
    return Scaler(mean=fitted.mean_.copy(), scale=fitted.scale_.copy())


def apply_scaler(features: np.ndarray, scaler: Scaler, clip: float = 5.0) -> np.ndarray:
    """Standardise features and clip to +/- clip.

    Args:
        features: (N, D) features.
        scaler: Output of fit_scaler.
        clip: Absolute clip value.

    Returns:
        Standardised copy.
    """
    return np.clip((features - scaler.mean) / scaler.scale, -clip, clip)


def fingerprint(
    d_geo: np.ndarray, features: np.ndarray, scaler: Scaler, global_k: int, sigma: float
) -> Optional[Fingerprint]:
    """Spectral fingerprint of one pocket.

    Args:
        d_geo: (N, N) geodesic matrix.
        features: (N, D) raw features.
        scaler: Feature standardisation to apply.
        global_k: Fingerprint length.
        sigma: Heat-kernel width.

    Returns:
        The fingerprint, or None if the pocket is too small.
    """
    lam, spec = spectral_fingerprint(d_geo, apply_scaler(features, scaler), global_k, sigma)
    return None if lam is None else Fingerprint(lam, spec)


def distance_matrices(
    fingerprints: Sequence[Fingerprint], shape_weight: float, chem_scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shape, chemistry and combined pairwise distances.

    Args:
        fingerprints: Fingerprints in row order.
        shape_weight: Weight of the shape term.
        chem_scale: Scale of the chemistry term.

    Returns:
        (shape, chem, combined) square matrices; combined is float32.
    """
    lam_matrix = np.array([fp.eigenvalues for fp in fingerprints], dtype=np.float64)
    feat_matrix = np.array([fp.spectrum for fp in fingerprints], dtype=np.float64)
    d_shape = squareform(pdist(lam_matrix, metric="euclidean"))
    d_chem = squareform(pdist(feat_matrix, metric="euclidean"))
    combined = (shape_weight * d_shape + (1.0 - shape_weight) * chem_scale * d_chem).astype(np.float32)
    return d_shape, d_chem, combined


def pocket_distance(
    a: Fingerprint, b: Fingerprint, shape_weight: float = 0.85, chem_scale: float = 12.5
) -> Dict[str, float]:
    """Shape, chemistry and combined distance between two fingerprints.

    Args:
        a: First fingerprint.
        b: Second fingerprint.
        shape_weight: Weight of the shape term.
        chem_scale: Scale of the chemistry term.

    Returns:
        Dict with keys shape, chem, combined.
    """
    shape, chem, combined = distance_matrices([a, b], shape_weight, chem_scale)
    return {"shape": float(shape[0, 1]), "chem": float(chem[0, 1]),
            "combined": float(combined[0, 1])}
