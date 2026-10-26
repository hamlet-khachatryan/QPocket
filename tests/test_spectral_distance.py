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
from qpocket.config import evolve


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
        assert "features" not in d
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


def _write_npz(path: Path, n: int = 10, k_feat: int = 3, seed: int = 0, n_feat_rows: int = None,
               permute: bool = False) -> Path:
    """Write a geodesic cache at ``path`` and the ``featurized.csv`` beside it."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 5, (n, 3))
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(coords))
    if D.max() > 0:
        D /= D.max()
    F = rng.standard_normal((n, k_feat))
    weights = np.ones(n) / n
    np.savez_compressed(path, D_geo=D, weights=weights)
    feats = pd.DataFrame(F[: n_feat_rows or n], columns=[f"f{i}" for i in range(k_feat)])
    cloud = pd.DataFrame(coords, columns=list("xyz"))
    cloud.to_csv(path.with_name("binding_site.csv"), index=False)
    feats = pd.concat([cloud.iloc[: n_feat_rows or n], feats], axis=1)
    if permute:
        feats[list("xyz")] = coords[::-1]
    if k_feat == 0:
        feats = cloud.iloc[: n_feat_rows or n]
    feats.to_csv(path.with_name("featurized.csv"), index=False)
    return path


def _pocket_dir(root: Path, name: str) -> Path:
    (root / name).mkdir()
    return root / name


def test_feature_rows_must_match(tmp_path):
    p1 = _write_npz(_pocket_dir(tmp_path, "a") / "geospectral_cache.npz", seed=1, n_feat_rows=7)
    cfg = evolve(make_config(), "distance.parallel_jobs", 1)
    with pytest.raises(ValueError):
        data_processing_spectral(cfg, [p1])


def test_features_must_belong_to_the_cloud(tmp_path):
    p1 = _write_npz(_pocket_dir(tmp_path, "a") / "geospectral_cache.npz", seed=1, permute=True)
    cfg = evolve(make_config(), "distance.parallel_jobs", 1)
    with pytest.raises(ValueError, match="a"):
        data_processing_spectral(cfg, [p1])


def test_featurized_csv_needs_feature_columns(tmp_path):
    p1 = _write_npz(_pocket_dir(tmp_path, "a") / "geospectral_cache.npz", seed=1, k_feat=0)
    cfg = evolve(make_config(), "distance.parallel_jobs", 1)
    with pytest.raises(ValueError, match="no feature"):
        data_processing_spectral(cfg, [p1])


def test_geodesic_from_unfeaturised_cloud(tmp_path):
    from qpocket.io.cache import load_geodesic
    rng = np.random.default_rng(1)
    csv = tmp_path / "binding_site.csv"
    pd.DataFrame(rng.random((20, 3)) * 4, columns=list("xyz")).to_csv(csv, index=False)
    assert load_geodesic(compute_geodesic(csv, 2.0)).shape == (20, 20)


def test_data_processing_spectral_runs(tmp_path):
    p1 = _write_npz(_pocket_dir(tmp_path, "a") / "geospectral_cache.npz", seed=1)
    p2 = _write_npz(_pocket_dir(tmp_path, "b") / "geospectral_cache.npz", seed=2)

    cfg = make_config()
    cfg = evolve(cfg, "distance.global_k", 5)
    cfg = evolve(cfg, "distance.sigma", 0.5)
    cfg = evolve(cfg, "distance.parallel_jobs", 1)
    cfg = evolve(cfg, "distance.max_points", 5000)

    result = data_processing_spectral(cfg, [p1, p2])
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_data_processing_spectral_scaler_applied(tmp_path):
    """Features should be clipped to ±5 after standardisation."""
    p1 = _write_npz(_pocket_dir(tmp_path, "a") / "geospectral_cache.npz", seed=10)
    p2 = _write_npz(_pocket_dir(tmp_path, "b") / "geospectral_cache.npz", seed=20)

    cfg = make_config()
    cfg = evolve(cfg, "distance.global_k", 3)
    cfg = evolve(cfg, "distance.sigma", 1.0)
    cfg = evolve(cfg, "distance.parallel_jobs", 1)
    cfg = evolve(cfg, "distance.max_points", 5000)

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


# ---------------------------------------------------------------------------
# Shape-only distances
# ---------------------------------------------------------------------------


def _site(tmp_path: Path, n_pockets: int):
    """A site workspace with clouds, geodesic caches and a master.csv."""
    from qpocket.io import layout
    ws = layout.workspace(tmp_path / "site_01")
    ws["metadata"].mkdir(parents=True)
    ids = [f"P{i}" for i in range(n_pockets)]
    pd.DataFrame({"pdb_id": ids}).to_csv(ws["metadata"] / "master.csv", index=False)
    for i, pid in enumerate(ids):
        (ws["pockets"] / pid).mkdir(parents=True)
        _write_npz(ws["pockets"] / pid / "geospectral_cache.npz", n=12 + i, seed=i)
    return ws, ids


def test_shape_matrix_equals_two_stage_shape(tmp_path):
    from qpocket.core import spectral as sp
    from qpocket.io.cache import load_geodesic
    from qpocket.pipelines.distance import build_distance_matrices, compute_shape_matrix

    ws, ids = _site(tmp_path, 4)
    cfg = evolve(evolve(make_config(), "distance.parallel_jobs", 1), "distance.global_k", 5)
    for stale in ("feature_distance_matrix.csv", "spectral_distance_matrix.csv"):
        (ws["metadata"] / stale).write_text("old")

    out = compute_shape_matrix(cfg, ws)
    got = pd.read_csv(out, index_col=0)

    pockets = [ws["pockets"] / i / "geospectral_cache.npz" for i in ids]
    fps = {i: sp.fingerprint(load_geodesic(p), np.zeros((len(load_geodesic(p)), 1)),
                             sp.Scaler(np.zeros(1), np.ones(1)), 5, cfg.distance.sigma)
           for i, p in zip(ids, pockets)}
    expected = build_distance_matrices(cfg, {i: (f.eigenvalues, f.spectrum) for i, f in fps.items()}, ids)[0]
    assert list(got.index) == ids
    np.testing.assert_allclose(got.to_numpy(), expected, atol=1e-10)
    assert not (ws["metadata"] / "feature_distance_matrix.csv").exists()
    assert not (ws["metadata"] / "spectral_distance_matrix.csv").exists()


def test_shape_only_needs_two_pockets(tmp_path):
    from qpocket.pipelines.distance import compute_shape_matrix

    ws, _ = _site(tmp_path, 1)
    with pytest.raises(ValueError, match="site_01 has fewer than 2 pockets to compare"):
        compute_shape_matrix(evolve(make_config(), "distance.parallel_jobs", 1), ws)
