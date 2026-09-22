"""The split spectral functions reproduce the old monolithic ones exactly."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qpocket.core import spectral as sp
from qpocket.io import cache
from qpocket.types import Fingerprint, Scaler


def _blocks():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(40, 6)), rng.normal(size=(25, 6)) * 3 + 1]


def test_fit_scaler_equals_sklearn_on_stacked_points():
    blocks = _blocks()
    ref = StandardScaler().fit(np.vstack(blocks))
    sc = sp.fit_scaler(blocks)
    assert np.array_equal(sc.mean, ref.mean_) and np.array_equal(sc.scale, ref.scale_)


def test_apply_scaler_is_bitwise_sklearn_transform_then_clip():
    blocks = _blocks()
    ref = StandardScaler().fit(np.vstack(blocks))
    got = sp.apply_scaler(blocks[1], sp.fit_scaler(blocks))
    assert np.array_equal(got, np.clip(ref.transform(blocks[1]), -5, 5))


def test_apply_scaler_clips():
    sc = sp.fit_scaler([np.array([[0.0], [1.0]])])
    assert sp.apply_scaler(np.array([[100.0]]), sc).max() == 5.0


def test_geodesic_matrix_is_normalised_and_symmetric():
    pts = np.random.default_rng(1).random((60, 3)) * 6
    d = sp.geodesic_matrix(pts, 2.0)
    assert d.shape == (60, 60) and d.max() == 1.0 and np.allclose(d, d.T)


def test_geodesic_matrix_rejects_tiny_or_disconnected_clouds():
    assert sp.geodesic_matrix(np.zeros((4, 3)), 2.0) is None
    far = np.arange(30, dtype=float).reshape(10, 3) * 100
    assert sp.geodesic_matrix(far, 2.0) is None


def test_fingerprint_does_not_modify_inputs():
    rng = np.random.default_rng(3)
    d = rng.random((20, 20)); d = (d + d.T) / 2; np.fill_diagonal(d, 0)
    f = rng.random((20, 4))
    d0, f0 = d.copy(), f.copy()
    fp = sp.fingerprint(d, f, sp.fit_scaler([f]), 10, 10.0)
    assert np.array_equal(d, d0) and np.array_equal(f, f0)
    assert fp.eigenvalues.shape == (10,) and fp.spectrum.shape == (40,)


def test_pocket_distance_matches_distance_matrices():
    rng = np.random.default_rng(2)
    fps = [Fingerprint(rng.random(10), rng.random(60)) for _ in range(2)]
    shape, chem, combined = sp.distance_matrices(fps, 0.85, 12.5)
    got = sp.pocket_distance(*fps)
    assert got["shape"] == shape[0, 1] and got["chem"] == chem[0, 1]
    assert got["combined"] == float(combined[0, 1]) and combined.dtype == np.float32


def test_geodesic_and_scaler_caches_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    d, f = rng.random((5, 5)), rng.random((5, 3))
    cache.save_geodesic(tmp_path / "g.npz", d, f)
    d2, f2 = cache.load_geodesic(tmp_path / "g.npz")
    assert np.array_equal(d, d2) and np.array_equal(f, f2)
    sc = Scaler(mean=rng.random(3), scale=rng.random(3))
    cache.save_scaler(tmp_path / "s.npz", sc)
    back = cache.load_scaler(tmp_path / "s.npz")
    assert np.array_equal(back.mean, sc.mean) and np.array_equal(back.scale, sc.scale)


def test_compute_geodesic_still_writes_the_same_cache(tmp_path):
    from qpocket.pipelines.distance import compute_geodesic
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.random((60, 3)) * 6, columns=list("xyz")); df["apbs"] = rng.random(60)
    df.to_csv(tmp_path / "featurized.csv", index=False)
    with np.load(compute_geodesic(tmp_path / "featurized.csv", 2.0)) as z:
        pts = pd.read_csv(tmp_path / "featurized.csv")[list("xyz")].to_numpy(float)
        assert np.array_equal(z["D_geo"], sp.geodesic_matrix(pts, 2.0))
        assert set(z.files) == {"D_geo", "features", "weights", "source_hash", "radius"}
