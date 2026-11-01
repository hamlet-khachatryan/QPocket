"""The split spectral functions reproduce the old monolithic ones exactly."""
import numpy as np

from qpocket.core import spectral as sp
from qpocket.types import Fingerprint


def _blocks():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(40, 6)), rng.normal(size=(25, 6)) * 3 + 1]


def test_pocket_distance_matches_distance_matrices():
    rng = np.random.default_rng(2)
    fps = [Fingerprint(rng.random(10), rng.random(60)) for _ in range(2)]
    shape, chem, combined = sp.distance_matrices(fps, 0.85, 12.5)
    got = sp.pocket_distance(*fps)
    assert got["shape"] == shape[0, 1] and got["chem"] == chem[0, 1]
    assert got["combined"] == float(combined[0, 1]) and combined.dtype == np.float32


