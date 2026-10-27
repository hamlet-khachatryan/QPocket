"""The pairwise distance heatmap: symmetrised, ordered so similar pockets sit together."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def symmetrised(dist_df: pd.DataFrame) -> pd.DataFrame:
    """The distance matrix made symmetric, with a zero diagonal.

    Args:
        dist_df: Square distance table, ids as index and columns.

    Returns:
        The symmetrised table, in the input's id order.
    """
    ids = list(dist_df.index)
    raw = dist_df.loc[ids, ids].to_numpy(dtype=float)
    sym = 0.5 * (raw + raw.T)
    np.fill_diagonal(sym, 0.0)
    return pd.DataFrame(sym, index=ids, columns=ids)


def leaf_order(M: pd.DataFrame, ids: Sequence[str]) -> List[str]:
    """Average-linkage leaf order of a subset — puts similar pockets adjacent.

    Args:
        M: Symmetrised distance table.
        ids: The subset.

    Returns:
        The ids, reordered; fewer than three keep their order.
    """
    ids = list(ids)
    if len(ids) < 3:
        return ids
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    sub = M.loc[ids, ids].to_numpy(dtype=float)
    Z = linkage(squareform(sub, checks=False), method="average")
    return [ids[i] for i in leaves_list(Z)]


def ordering(M: pd.DataFrame, active_info: pd.DataFrame,
             how: str) -> Tuple[List[str], List[Tuple[object, List[str]]]]:
    """Row/column order for the heatmap, and the cluster blocks to outline.

    Args:
        M: Symmetrised distance table.
        active_info: Run table with pdb_id and cluster_id.
        how: "as loaded", "hierarchical", or cluster blocks (anything else).

    Returns:
        (order, [(cluster, members)]); blocks only in cluster-block order.
    """
    ids = list(M.index)
    cluster_of: Dict[str, object] = dict(zip(active_info["pdb_id"].astype(str),
                                             active_info["cluster_id"]))
    blocks: List[Tuple[object, List[str]]] = []
    if how == "as loaded":
        return ids, blocks
    if how == "hierarchical":
        return leaf_order(M, ids), blocks
    order: List[str] = []  # cluster-major, hierarchical inside each cluster
    for c in sorted({cluster_of.get(str(i), -1) for i in ids}):
        members = leaf_order(M, [i for i in ids if cluster_of.get(str(i), -1) == c])
        blocks.append((c, members))
        order.extend(members)
    return order, blocks


def heatmap(M: pd.DataFrame, order: Sequence[str], blocks, how: str):
    """The heatmap figure, with cluster blocks outlined.

    Args:
        M: Symmetrised distance table.
        order: Row/column order.
        blocks: (cluster, members) to outline.
        how: Ordering name, for the title.

    Returns:
        A plotly figure.
    """
    import plotly.express as px

    order = list(order)
    fig = px.imshow(M.loc[order, order], color_continuous_scale="Blues", aspect="auto",
                    labels=dict(color="spectral dist"), height=620)
    fig.update_layout(
        title=f"Pairwise spectral distance between pockets · ordered by {how}",
        xaxis=dict(tickmode="array", tickvals=list(range(len(order))), ticktext=order, tickangle=-90),
        yaxis=dict(tickmode="array", tickvals=list(range(len(order))), ticktext=order),
        margin=dict(t=70),
    )
    cum = 0
    for c, members in blocks:
        lo, hi = cum - 0.5, cum + len(members) - 0.5
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=lo, y1=hi,
                      line=dict(color="#1a1a19", width=2), fillcolor="rgba(0,0,0,0)", layer="above")
        fig.add_annotation(x=(lo + hi) / 2, y=1.0, xref="x", yref="paper", text=f"cluster {c}",
                           showarrow=False, yanchor="bottom", font=dict(size=12, color="#52514e"))
        cum += len(members)
    return fig


def cluster_map(M: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Median distance between every pair of clusters, as a percentile of the site's pairs.

    Args:
        M: Symmetrised distance table of the site.
        labels: Cluster per pocket, indexed by pocket id.

    Returns:
        k × k table indexed by cluster; the diagonal is the median within-cluster distance.
    """
    from qpocket.core.clustering import sorted_pairs, to_percentile

    full = M.to_numpy(dtype=float)
    labels = labels.reindex(M.index)
    clusters = sorted(labels.dropna().unique())
    masks = [(labels == c).to_numpy() for c in clusters]
    medians = np.zeros((len(clusters), len(clusters)))
    for i, rows in enumerate(masks):
        for j in range(i, len(clusters)):
            block = full[np.ix_(rows, masks[j])]
            values = block[np.triu_indices(len(block), k=1)] if i == j else block.ravel()
            medians[i, j] = medians[j, i] = float(np.median(values)) if len(values) else 0.0
    # one sort of the site's pairs, then every cell placed against it
    pct = to_percentile(None, medians.ravel(), pairs=sorted_pairs(full)).reshape(medians.shape)
    return pd.DataFrame(pct, index=clusters, columns=clusters)


def cluster_map_figure(cmap: pd.DataFrame, colours: Dict, height: int = 380, names: Optional[Dict] = None):
    """The k × k cluster map as a styled heatmap, cluster swatches along both axes.

    Args:
        cmap: From :func:`cluster_map`.
        colours: Colour per cluster label in ``cmap.index`` (missing ones are drawn grey).
        height: Figure height (px).
        names: Display name per cluster label, e.g. S1·c2; default "S<label + 1>".

    Returns:
        A plotly Figure; hover shows both clusters and the percentile.
    """
    import plotly.graph_objects as go

    from qpocket.viz.style import GREY, apply

    names = [(names or {}).get(c) or (f"S{int(c) + 1}" if isinstance(c, (int, np.integer)) else str(c))
             for c in cmap.index]
    fig = go.Figure(go.Heatmap(
        z=cmap.to_numpy(dtype=float), x=names, y=names, zmin=0, zmax=100,
        colorscale=[[0, "#123a5a"], [0.5, "#9cc0da"], [1, "#f4f8fb"]],
        colorbar=dict(title=dict(text="pct", font=dict(size=9)), thickness=8, len=0.8),
        hovertemplate="%{y} × %{x}: p%{z:.0f}<extra></extra>"))
    fig = apply(fig)
    fig.update_layout(height=height,
                      yaxis=dict(autorange="reversed", showline=False, ticks="", tickfont=dict(size=9)),
                      xaxis=dict(showline=False, ticks="", tickfont=dict(size=9)))
    # cluster swatches along the axes, the same colours as the tree and the landscape
    for i, cluster in enumerate(cmap.index):
        colour = colours.get(cluster, GREY)
        fig.add_shape(type="rect", xref="paper", yref="y", x0=-0.035, x1=-0.012, y0=i - 0.4, y1=i + 0.4,
                      fillcolor=colour, line=dict(width=0))
    return fig


def member_block(M: pd.DataFrame, labels: pd.Series, a, b) -> pd.DataFrame:
    """The distances among the members of two clusters, ordered by cluster then by tree order.

    Args:
        M: Symmetrised distance table.
        labels: Cluster per pocket, indexed by pocket id.
        a: One cluster.
        b: The other (may equal ``a``).

    Returns:
        Square table over the two clusters' members.
    """
    labels = labels.reindex(M.index)
    order: List[str] = []
    for cluster in ([a] if a == b else [a, b]):
        order.extend(leaf_order(M, [i for i in M.index if labels[i] == cluster]))
    return M.loc[order, order]
