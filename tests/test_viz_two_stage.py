"""The notebook's two-stage clustering helpers."""
from pathlib import Path

import numpy as np
import pytest

from qpocket.config import ClusteringParams
from qpocket.viz import two_stage as ts

GOLD = Path(__file__).resolve().parent / "golden/expected/7KG6"


def _toy():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(c, 0.1, (5, 2)) for c in [(0, 0), (0, 1.5), (8, 0), (8, 1.5)]])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    return ts.RunMatrices([f"P{i}" for i in range(20)], d, d.copy(), d.copy())


def test_slider_range_covers_every_merge():
    lo, hi = ts.height_range(_toy().shape)
    assert 0 < lo < 1 < 4 < hi


def test_regroup_follows_both_cuts():
    m = _toy()
    table = ts.regroup(m, ts.percentile_settings(ClusteringParams(), 60.0, 25.0))
    assert table["shape_cluster_id"].nunique() == 2 and table["cluster_id"].nunique() == 4
    assert table.groupby("cluster_id")["is_representative"].sum().eq(1).all()


def test_dendrogram_figure_draws_the_cut():
    fig = ts.dendrogram_figure(_toy().shape, [f"P{i}" for i in range(20)], 4.0, "shape")
    assert len(fig.data) == 19 and fig.layout.shapes[0].y0 == 4.0


@pytest.mark.skipif(not (GOLD / "metadata/spectral_distance_matrix.csv").exists(), reason="needs golden run")
def test_loads_a_real_run():
    m = ts.load_matrices(GOLD.parent / "7KG6")
    assert len(m.ids) == 7 and m.shape.shape == (7, 7)
