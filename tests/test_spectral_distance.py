"""
Tests for spectral distance functions in distance_logic/spectral_distance.py.
No external process dependencies — all math is pure numpy/scipy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.conftest import make_config
from qpocket.config import evolve


# ---------------------------------------------------------------------------
# pad_or_truncate
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Eigenvalue distance regression test
# ---------------------------------------------------------------------------


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


