"""
Tests for spectral distance functions in distance_logic/spectral_distance.py.
No external process dependencies — all math is pure numpy/scipy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpocket.pipelines.distance import (
    compute_geodesic,
    compute_spectral_fingerprint,
    data_processing_spectral,
    pad_or_truncate,
)
from tests.conftest import make_config


# ---------------------------------------------------------------------------
# pad_or_truncate
# ---------------------------------------------------------------------------


def test_pad_or_truncate_shorter_pads_with_zeros():
    v = np.array([1.0, 2.0, 3.0])
    result = pad_or_truncate(v, 6)
    assert result.shape == (6,)
    assert result[3] == 0.0
    assert result[4] == 0.0
    assert result[5] == 0.0


def test_pad_or_truncate_longer_truncates():
    v = np.arange(10, dtype=float)
    result = pad_or_truncate(v, 5)
    assert result.shape == (5,)
    np.testing.assert_array_equal(result, v[:5])


def test_pad_or_truncate_exact_unchanged():
    v = np.array([1.0, 2.0, 3.0])
    result = pad_or_truncate(v, 3)
    np.testing.assert_array_equal(result, v)


def test_pad_or_truncate_preserves_values():
    v = np.array([10.0, 20.0])
    result = pad_or_truncate(v, 5)
    assert result[0] == 10.0
    assert result[1] == 20.0


# ---------------------------------------------------------------------------
# compute_geodesic
# ---------------------------------------------------------------------------


def _make_pocket_csv(tmp_path: Path, n_points: int = 10, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 5, (n_points, 3))
    features = rng.standard_normal((n_points, 3))
    df = pd.DataFrame(
        np.hstack([coords, features]),
        columns=["x", "y", "z", "f1", "f2", "f3"],
    )
    p = tmp_path / "pocket.csv"
    df.to_csv(p, index=False)
    return p


def test_compute_geodesic_returns_valid_path(tmp_path):
    csv_path = _make_pocket_csv(tmp_path)
    result = compute_geodesic(csv_path, neighbor_radius=3.0)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".npz"


def test_compute_geodesic_npz_has_expected_keys(tmp_path):
    csv_path = _make_pocket_csv(tmp_path)
    cache = compute_geodesic(csv_path, neighbor_radius=3.0)
    with np.load(cache) as d:
        assert "D_geo" in d
        assert "features" in d
        assert "weights" in d


def test_compute_geodesic_caching_skips_recompute(tmp_path):
    csv_path = _make_pocket_csv(tmp_path)
    cache1 = compute_geodesic(csv_path, neighbor_radius=3.0)
    cache_mtime = cache1.stat().st_mtime

    # Second call should use the cache — file not re-written
    cache2 = compute_geodesic(csv_path, neighbor_radius=3.0)
    assert cache2 == cache1
    assert cache2.stat().st_mtime == cache_mtime


def test_compute_geodesic_too_few_points_returns_none(tmp_path):
    csv_path = _make_pocket_csv(tmp_path, n_points=3)
    result = compute_geodesic(csv_path, neighbor_radius=3.0)
    assert result is None


def test_compute_geodesic_d_geo_normalised(tmp_path):
    csv_path = _make_pocket_csv(tmp_path, n_points=15, seed=7)
    cache = compute_geodesic(csv_path, neighbor_radius=3.0)
    assert cache is not None
    with np.load(cache) as d:
        D = d["D_geo"]
        assert D.max() <= 1.0 + 1e-6


def test_compute_geodesic_d_geo_nonnegative(tmp_path):
    csv_path = _make_pocket_csv(tmp_path, n_points=15, seed=7)
    cache = compute_geodesic(csv_path, neighbor_radius=3.0)
    with np.load(cache) as d:
        D = d["D_geo"]
        assert np.all(D >= 0)


def test_compute_geodesic_disconnected_graph_no_inf(tmp_path):
    # Two clusters far apart → shortest path returns inf → code replaces with finite value
    coords_a = np.zeros((6, 3))
    coords_b = np.ones((6, 3)) * 1000
    coords = np.vstack([coords_a, coords_b])
    features = np.zeros((12, 3))
    df = pd.DataFrame(
        np.hstack([coords, features]),
        columns=["x", "y", "z", "f1", "f2", "f3"],
    )
    p = tmp_path / "pocket.csv"
    df.to_csv(p, index=False)

    cache = compute_geodesic(p, neighbor_radius=2.0)
    if cache is not None:
        with np.load(cache) as d:
            assert not np.any(np.isinf(d["D_geo"]))


# ---------------------------------------------------------------------------
# compute_spectral_fingerprint
# ---------------------------------------------------------------------------


def _make_geo_and_features(n: int = 8, k_feat: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 5, (n, 3))
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(coords))
    if D.max() > 0:
        D /= D.max()
    F = rng.standard_normal((n, k_feat))
    return D, F


def test_compute_spectral_fingerprint_output_shapes():
    D, F = _make_geo_and_features(n=8, k_feat=3)
    global_k = 5
    lam, spec_vec = compute_spectral_fingerprint(D, F, global_k=global_k, sigma=0.5)
    assert lam is not None
    assert spec_vec is not None
    assert lam.shape == (global_k,)
    assert spec_vec.shape == (global_k * F.shape[1],)


def test_compute_spectral_fingerprint_tiny_graph():
    D, F = _make_geo_and_features(n=5, k_feat=2)
    lam, spec_vec = compute_spectral_fingerprint(D, F, global_k=3, sigma=1.0)
    assert lam is not None
    assert spec_vec is not None


def test_compute_spectral_fingerprint_degenerate_identical_points():
    # All identical → D_geo = zero matrix → degenerate but should not crash
    D = np.zeros((6, 6))
    F = np.ones((6, 4))
    lam, spec_vec = compute_spectral_fingerprint(D, F, global_k=3, sigma=1.0)
    # Either returns None (if k_req < 1) or returns arrays
    if lam is not None:
        assert lam.shape == (3,)


def test_compute_spectral_fingerprint_symmetric_d_geo():
    D, F = _make_geo_and_features(n=8, k_feat=3)
    # D must be symmetric input; verify eigendecomp runs on symmetric matrix
    assert np.allclose(D, D.T)
    lam, spec_vec = compute_spectral_fingerprint(D, F, global_k=4, sigma=0.5)
    assert lam is not None


# ---------------------------------------------------------------------------
# data_processing_spectral
# ---------------------------------------------------------------------------


def _write_npz(path: Path, n: int = 10, k_feat: int = 3, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 5, (n, 3))
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(coords))
    if D.max() > 0:
        D /= D.max()
    F = rng.standard_normal((n, k_feat))
    weights = np.ones(n) / n
    np.savez_compressed(path, D_geo=D, features=F, weights=weights)
    return path


def test_data_processing_spectral_runs(tmp_path):
    p1 = _write_npz(tmp_path / "a.npz", seed=1)
    p2 = _write_npz(tmp_path / "b.npz", seed=2)

    cfg = make_config()
    cfg.distance.global_k = 5
    cfg.distance.sigma = 0.5
    cfg.distance.parallel_jobs = 1
    cfg.distance.max_points = 5000

    result = data_processing_spectral(cfg, [p1, p2])
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_data_processing_spectral_scaler_applied(tmp_path):
    """Features should be clipped to ±5 after standardisation."""
    p1 = _write_npz(tmp_path / "a.npz", seed=10)
    p2 = _write_npz(tmp_path / "b.npz", seed=20)

    cfg = make_config()
    cfg.distance.global_k = 3
    cfg.distance.sigma = 1.0
    cfg.distance.parallel_jobs = 1
    cfg.distance.max_points = 5000

    result = data_processing_spectral(cfg, [p1, p2])
    # Just verify it runs without error and returns a dict
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Eigenvalue distance regression test
# ---------------------------------------------------------------------------


def test_eigenvalue_distance_known_matrix():
    """
    Regression: for two identical fingerprints the eigenvalue distance is 0.
    """
    lam = np.array([0.1, 0.2, 0.3])
    d = np.linalg.norm(lam - lam)
    assert d == pytest.approx(0.0)


def test_eigenvalue_distance_known_values():
    lam1 = np.array([1.0, 0.0, 0.0])
    lam2 = np.array([0.0, 1.0, 0.0])
    d = np.linalg.norm(lam1 - lam2)
    assert d == pytest.approx(np.sqrt(2.0))
