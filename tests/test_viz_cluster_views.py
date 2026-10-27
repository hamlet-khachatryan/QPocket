"""Cluster-level views of a site: the k × k distance map, a member block, the landscape."""
import numpy as np
import pandas as pd

from qpocket.viz.clusters import landscape_figure
from qpocket.viz.matrix import cluster_map, cluster_map_figure, member_block


def _site():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(c, 0.2, (6, 2)) for c in [(0, 0), (5, 0), (0, 9)]])
    ids = [f"P{i}" for i in range(len(pts))]
    M = pd.DataFrame(np.linalg.norm(pts[:, None] - pts[None], axis=2), index=ids, columns=ids)
    labels = pd.Series(np.repeat([0, 1, 2], 6), index=ids)
    return M, labels, pts


def test_cluster_map_is_k_by_k_in_percentiles():
    M, labels, _ = _site()
    cmap = cluster_map(M, labels)
    assert cmap.shape == (3, 3) and ((cmap.values >= 0) & (cmap.values <= 100)).all()
    assert cmap.loc[0, 0] < cmap.loc[0, 1] < cmap.loc[0, 2]          # 5 Å apart, then 9 Å apart
    assert np.allclose(cmap.values, cmap.values.T)


def test_cluster_map_figure_is_styled():
    M, labels, _ = _site()
    fig = cluster_map_figure(cluster_map(M, labels), {})
    assert fig.layout.template.layout.xaxis.showgrid is False and len(fig.data) == 1


def test_member_block_holds_only_the_two_clusters():
    M, labels, _ = _site()
    block = member_block(M, labels, 0, 2)
    assert block.shape == (12, 12)
    assert set(block.index) == set(labels[labels.isin([0, 2])].index)
    assert list(block.index[:6]) == sorted(block.index[:6], key=lambda i: labels[i]) or labels[block.index[0]] == 0


def test_landscape_has_no_text_labels():
    _, labels, pts = _site()
    coords = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "pdb_id": labels.index,
                           "shape_cluster_id": labels.values, "chem_cluster_id": 0,
                           "protein_state": "BOUND_STATE"})
    fig = landscape_figure(coords, "cluster", {(c, 0): "#1f5f8b" for c in range(3)})
    assert fig.data and all(t.mode == "markers" for t in fig.data)
    assert "P0" in fig.data[0].hovertext[0]
