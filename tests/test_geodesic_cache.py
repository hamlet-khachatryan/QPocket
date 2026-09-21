"""The geodesic cache is reused only for the same featurized.csv and radius."""
import numpy as np
import pandas as pd

from qpocket.pipelines.distance import compute_geodesic


def _write(path, n=40):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, 3)) * 5, columns=list("xyz"))
    df["apbs"] = 0.0
    df.to_csv(path, index=False)


def _geodesic(csv, radius):
    with np.load(compute_geodesic(csv, radius)) as z:
        return z["D_geo"].copy(), {k: z[k] for k in z.files}


def test_unchanged_input_reuses_the_cache(tmp_path):
    csv = tmp_path / "featurized.csv"
    _write(csv)
    first, _ = _geodesic(csv, 2.0)
    cache = tmp_path / "geospectral_cache.npz"
    stamp = cache.stat().st_mtime_ns
    second, _ = _geodesic(csv, 2.0)
    assert np.array_equal(first, second) and cache.stat().st_mtime_ns == stamp


def test_rewritten_pocket_is_recomputed(tmp_path):
    csv = tmp_path / "featurized.csv"
    _write(csv)
    _geodesic(csv, 2.0)
    pd.read_csv(csv).iloc[:30].to_csv(csv, index=False)   # the Interactions pass crops the pocket
    assert _geodesic(csv, 2.0)[0].shape == (30, 30)


def test_new_radius_is_recomputed(tmp_path):
    csv = tmp_path / "featurized.csv"
    _write(csv)
    _geodesic(csv, 2.0)
    assert float(_geodesic(csv, 3.0)[1]["radius"]) == 3.0


def test_old_caches_without_provenance_are_stale(tmp_path):
    csv = tmp_path / "featurized.csv"
    _write(csv)
    np.savez_compressed(tmp_path / "geospectral_cache.npz", D_geo=np.zeros((2, 2)), features=np.zeros((2, 1)))
    assert _geodesic(csv, 2.0)[0].shape == (40, 40)
