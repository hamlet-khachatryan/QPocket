"""Pure featurisation: column order, crossed H-bond fields, DX parsing."""
import numpy as np

from qpocket.core.features import FEATURE_COLUMNS, parse_dx, pocket_features
from qpocket.types import StructureData

EMPTY = np.zeros((0, 3))


def _data(donors=EMPTY, acceptors=EMPTY, aromatics=EMPTY):
    rng = np.random.default_rng(0)
    return StructureData(rng.random((30, 3)) * 10, rng.normal(size=30), rng.random(30),
                         donors, acceptors, aromatics)


def test_columns_and_order():
    out = pocket_features(np.random.default_rng(1).random((12, 3)) * 10, _data(), np.zeros(12))
    assert tuple(out) == FEATURE_COLUMNS


def test_donor_field_comes_from_protein_acceptors():
    pts = np.zeros((4, 3))
    out = pocket_features(pts, _data(donors=np.zeros((2, 3))), np.zeros(4))
    assert np.all(out["h_acceptor"] > 0) and np.all(out["h_donor"] == 0)


def test_parse_dx_reads_grid_and_axes():
    text = ("object 1 class gridpositions counts 2 2 2\norigin 1.0 2.0 3.0\n"
            "delta 0.5 0.0 0.0\ndelta 0.0 0.5 0.0\ndelta 0.0 0.0 0.5\n"
            "object 3 class array type double rank 0 items 8 data follows\n"
            "1 2 3\n4 5 6\n7 8\nattribute \"dep\" string \"positions\"\n")
    grid, (x, y, z) = parse_dx(text)
    assert grid.shape == (2, 2, 2) and grid[1, 1, 1] == 8.0
    assert x.tolist() == [1.0, 1.5] and z.tolist() == [3.0, 3.5]
