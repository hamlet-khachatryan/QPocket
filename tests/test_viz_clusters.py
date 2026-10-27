"""Collapsed trees and the cluster table for large sites."""
import numpy as np
import pandas as pd
import pytest

from qpocket.config import ClusteringParams
from qpocket.viz import two_stage as ts
from qpocket.viz.clusters import cluster_table, shared_volume
from qpocket.viz.style import GREY, cluster_colours


def _toy(n_small=0):
    """Two shape groups of ten (each two chemistry subgroups of five), plus far-away singletons."""
    rng = np.random.default_rng(0)
    centres = [(0, 0), (0, 1.5), (8, 0), (8, 1.5)]
    pts = np.vstack([rng.normal(c, 0.1, (5, 2)) for c in centres] + [rng.normal((40 + 9 * i, 40), 0.1, (1, 2))
                                                                     for i in range(n_small)])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    return ts.RunMatrices([f"P{i}" for i in range(len(pts))], d, d.copy(), d.copy())


def _preview(m, shape=60.0, chem=25.0):
    return ts.preview(m, ts.percentile_settings(ClusteringParams(), shape, chem))


def _colours(result, min_size=4):
    return cluster_colours(result.frame["shape_cluster_id"], result.frame["chem_cluster_id"], min_size)


def _leaf_markers(fig):
    return [t for t in fig.data if getattr(t, "name", None) == "clusters"]


def test_collapsed_tree_has_one_leaf_per_cluster():
    m = _toy(n_small=2)
    result = _preview(m)
    fig = ts.collapsed_figure(m, result, 60.0, 25.0, "average", _colours(result))
    (markers,) = _leaf_markers(fig)
    assert len(markers.x) == result.frame["shape_cluster_id"].nunique() == 4
    assert sorted(markers.customdata) == sorted(set(result.shape_labels.tolist()))
    assert sum(c == GREY for c in markers.marker.color) == 2          # the two singletons


def test_collapsed_tree_axis_is_percentile():
    m = _toy()
    result = _preview(m)
    fig = ts.collapsed_figure(m, result, 60.0, 25.0, "average", _colours(result))
    ys = [y for t in fig.data if t.y is not None for y in t.y if y is not None]
    assert -100 <= min(ys) and max(ys) <= 100
    assert "percentile" in fig.layout.yaxis.title.text.lower()


def test_collapsed_tree_two_leaves():
    m = ts.RunMatrices(["A", "B"], np.array([[0.0, 1.0], [1.0, 0.0]]), None, np.array([[0.0, 1.0], [1.0, 0.0]]))
    result = ts.preview(m, ts.shape_only_settings(ClusteringParams(), 50.0))
    fig = ts.collapsed_figure(m, result, 50.0, None, "average", cluster_colours(result.shape_labels, None, 1))
    assert len(_leaf_markers(fig)) == 1


def test_chemistry_line_is_one_height_across_subtrees():
    m = _toy()
    result = _preview(m)
    fig = ts.collapsed_figure(m, result, 60.0, 25.0, "average", _colours(result))
    lines = [s for s in fig.layout.shapes if s.line.dash == "dash"]
    assert sorted(round(s.y0, 6) for s in lines) == [-25.0, 60.0]


def test_subtree_figure_has_no_leaf_labels():
    m = _toy()
    fig = ts.subtree_figure(m, m.ids[:10], "shape", "average")
    assert not fig.layout.xaxis.ticktext


def test_shared_volume_identical_and_disjoint():
    a = np.array([[0.0, 0, 0], [0.7, 0, 0]])
    assert shared_volume(a, a + 1e-4, 0.7) == 1.0
    assert shared_volume(a, a + 10, 0.7) == 0.0


def _info(m, result):
    frame = result.frame.copy()
    frame["protein_state"] = ["BOUND_STATE" if i % 2 else "UNBOUND_STATE" for i in range(len(frame))]
    return frame


def test_cluster_table_columns_and_pooling():
    m = _toy(n_small=2)
    result = _preview(m)
    clouds = {pid: np.array([[i * 0.7, 0, 0], [i * 0.7 + 0.7, 0, 0]]) for i, pid in enumerate(m.ids)}
    table = cluster_table(_info(m, result), m, clouds, min_size=4)
    assert list(table.columns) == ["cluster", "level", "parent", "n", "holo", "shape_within", "chem_within",
                                   "shared_volume", "representative"]
    small = table[table["cluster"] == "small"]
    assert len(small) == 1 and small["n"].iloc[0] == 2
    assert set(table.loc[table["level"] == "chem", "parent"]) <= set(table.loc[table["level"] == "shape", "cluster"])


def test_within_is_a_site_percentile():
    m = _toy()
    result = _preview(m)
    table = cluster_table(_info(m, result), m, {}, min_size=4).set_index("cluster")
    shape_rows = table[table["level"] == "shape"]
    assert ((shape_rows["shape_within"] >= 0) & (shape_rows["shape_within"] <= 100)).all()
    chem_rows = table[table["level"] == "chem"]
    assert chem_rows["chem_within"].max() < shape_rows["chem_within"].min()   # subclusters are tighter
