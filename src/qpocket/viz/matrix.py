"""The pairwise distance heatmap: symmetrised, ordered so similar pockets sit together."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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
