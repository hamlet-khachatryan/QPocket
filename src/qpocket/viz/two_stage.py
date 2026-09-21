"""Interactive two-stage clustering: slider ranges, live regrouping and dendrograms."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from qpocket.core.clustering import dendrogram_segments, linkage_matrix, symmetrise, threshold_for_clusters
from qpocket.pipelines.distance import cluster_pockets


class RunMatrices(NamedTuple):
    """A run's three distance matrices in one row order.

    Attributes:
        ids: Pocket IDs.
        shape: Shape distances.
        chem: Physicochemical distances.
        combined: Blended distances.
    """

    ids: List[str]
    shape: np.ndarray
    chem: np.ndarray
    combined: np.ndarray


def load_matrices(run_dir: Path) -> Optional[RunMatrices]:
    """Read shape, chemistry and combined matrices from metadata/.

    Args:
        run_dir: Run root.

    Returns:
        The matrices, or None if the run has none.
    """
    meta = Path(run_dir) / "metadata"
    names = ("shape_distance_matrix", "feature_distance_matrix", "spectral_distance_matrix")
    if not all((meta / f"{n}.csv").exists() for n in names):
        return None
    combined = pd.read_csv(meta / "spectral_distance_matrix.csv", index_col=0)
    order = combined.index
    shape, chem = (pd.read_csv(meta / f"{n}.csv", index_col=0).loc[order, order] for n in names[:2])
    return RunMatrices([str(i) for i in order], shape.to_numpy(float), chem.to_numpy(float),
                       combined.to_numpy(float))


def height_range(distances: np.ndarray, linkage: str = "average") -> Tuple[float, float]:
    """Lowest and highest merge height of a hierarchical tree, for a cut slider.

    Args:
        distances: Square distance matrix.
        linkage: Linkage method.

    Returns:
        (min height, max height).
    """
    heights = linkage_matrix(symmetrise(distances), linkage)[:, 2]
    return float(heights.min()), float(heights.max())


def height_for_count(distances: np.ndarray, n_clusters: int, linkage: str = "average") -> float:
    """Cut height that yields a given number of clusters.

    Args:
        distances: Square distance matrix.
        n_clusters: Wanted count.
        linkage: Linkage method.

    Returns:
        The height.
    """
    return float(threshold_for_clusters(linkage_matrix(symmetrise(distances), linkage), n_clusters))


def chem_height_range(m: RunMatrices, shape_labels: np.ndarray, linkage: str = "average") -> Tuple[float, float]:
    """Merge-height range pooled over the chemistry trees of every shape cluster.

    Args:
        m: The run's matrices.
        shape_labels: Stage-one label per pocket.
        linkage: Linkage method.

    Returns:
        (min height, max height); (0, 1) when no cluster can be split.
    """
    lows, highs = [], []
    for value in np.unique(shape_labels):
        members = np.flatnonzero(shape_labels == value)
        if len(members) >= 3:
            lo, hi = height_range(m.chem[np.ix_(members, members)], linkage)
            lows.append(lo)
            highs.append(hi)
    return (min(lows), max(highs)) if lows else (0.0, 1.0)


def threshold_settings(settings, shape_threshold: float, chem_threshold: Optional[float],
                       min_cluster_size: Optional[int] = None):
    """Clustering settings with hierarchical threshold cuts on both stages.

    Args:
        settings: Starting ClusteringParams (usually the run's own).
        shape_threshold: Shape tree cut height.
        chem_threshold: Chemistry tree cut height, or None to keep the run's chemistry stage.
        min_cluster_size: Override for the smallest shape cluster that is split.

    Returns:
        New ClusteringParams in two-stage mode.
    """
    shape = replace(settings.shape, method="hierarchical", criterion="threshold", threshold=float(shape_threshold))
    chem = settings.chem if chem_threshold is None else replace(
        settings.chem, method="hierarchical", criterion="threshold", threshold=float(chem_threshold))
    return replace(settings, mode="two_stage", shape=shape, chem=chem,
                   min_cluster_size=min_cluster_size or settings.min_cluster_size)


def regroup(m: RunMatrices, settings) -> pd.DataFrame:
    """Cluster a run's matrices and tabulate it like clustering_metadata.csv.

    Args:
        m: The run's matrices.
        settings: ClusteringParams.

    Returns:
        pdb_id, cluster_id, shape_cluster_id, chem_cluster_id, is_representative, dist_to_exemplar.
    """
    a = cluster_pockets(settings, m.shape, m.chem, m.combined)
    exemplar_of: Dict[int, int] = {int(a.labels[p]): p for p in a.exemplar_positions}
    rows = []
    for i, pid in enumerate(m.ids):
        ex = exemplar_of.get(int(a.labels[i]), i)
        rows.append({"pdb_id": pid, "cluster_id": int(a.labels[i]),
                     "shape_cluster_id": int(a.shape_labels[i]), "chem_cluster_id": int(a.chem_labels[i]),
                     "is_representative": i == ex, "dist_to_exemplar": float(m.combined[i, ex])})
    return pd.DataFrame(rows)


def dendrogram_figure(distances: np.ndarray, ids: List[str], threshold: Optional[float],
                      title: str, linkage: str = "average", height: int = 320):
    """Plotly dendrogram with a dashed line at the cut height.

    Args:
        distances: Square distance matrix.
        ids: Leaf labels, same order.
        threshold: Cut height to draw, or None.
        title: Figure title.
        linkage: Linkage method.
        height: Figure height (px).

    Returns:
        A plotly Figure.
    """
    import plotly.graph_objects as go

    tree = dendrogram_segments(linkage_matrix(symmetrise(distances), linkage), threshold)
    fig = go.Figure()
    for xs, ys in tree.segments:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#555", width=1),
                                 hoverinfo="skip", showlegend=False))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color="#d62728", dash="dash"))
    fig.update_layout(title=title, height=height, margin=dict(l=40, r=10, t=40, b=80),
                      xaxis=dict(tickmode="array", tickvals=list(tree.leaf_positions),
                                 ticktext=[ids[i] for i in tree.leaf_order], tickangle=-60),
                      yaxis=dict(title="merge height"), plot_bgcolor="white")
    return fig


def shape_groups(m: RunMatrices, threshold: float, linkage: str = "average") -> np.ndarray:
    """Stage-one labels for a shape cut height.

    Args:
        m: The run's matrices.
        threshold: Cut height.
        linkage: Linkage method.

    Returns:
        Label per pocket.
    """
    from qpocket.core.clustering import THRESHOLD, cluster_labels

    return cluster_labels(m.shape, method=linkage, criterion=THRESHOLD, threshold=threshold).labels
