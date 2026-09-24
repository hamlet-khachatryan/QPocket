"""Three ways to cluster a run's pockets, each with its own adjustable knobs."""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from qpocket.core.clustering import (
    THRESHOLD,
    affinity_labels,
    cluster_labels,
    compare_labelings,
    exemplars,
    silhouette,
    symmetrise,
)
from qpocket.viz.two_stage import RunMatrices, regroup

#: Method key to the label shown in the picker.
METHODS: Dict[str, str] = {
    "affinity": "Affinity propagation on the combined score",
    "hierarchical": "One hierarchical cut on the combined score",
    "two_stage": "Two stages: shape, then chemistry",
}

#: Columns every method returns, matching metadata/clustering_metadata.csv.
COLUMNS = ("pdb_id", "cluster_id", "shape_cluster_id", "chem_cluster_id",
           "is_representative", "dist_to_exemplar")


class Grouping(NamedTuple):
    """One clustering of a run's pockets.

    Attributes:
        frame: One row per pocket, with the columns in COLUMNS.
        method: The method key that produced it.
        caption: One line describing what was done, for the notebook.
        n_clusters: Number of clusters.
        silhouette: Mean silhouette on the combined distances, or None.
    """

    frame: pd.DataFrame
    method: str
    caption: str
    n_clusters: int
    silhouette: Optional[float]


def _frame(ids: List[str], labels, distances: np.ndarray,
           shape=None, chem=None) -> pd.DataFrame:
    """Assemble the common column layout, nominating one exemplar per cluster."""
    labels = np.asarray(labels, dtype=int)
    frame = pd.DataFrame({
        "pdb_id": list(ids),
        "cluster_id": labels,
        "shape_cluster_id": labels if shape is None else np.asarray(shape, dtype=int),
        "chem_cluster_id": np.zeros_like(labels) if chem is None else np.asarray(chem, dtype=int),
    })
    return frame.assign(**representatives(distances, labels))


def representatives(distances: np.ndarray, labels) -> Dict[str, np.ndarray]:
    """The medoid of each cluster, and every pocket's distance to its own.

    Args:
        distances: Square distance matrix, in the same order as the labels.
        labels: Cluster index per pocket.

    Returns:
        Mapping with the is_representative and dist_to_exemplar columns.
    """
    labels = np.asarray(labels, dtype=int)
    d = symmetrise(distances)
    chosen = exemplars(d, labels)
    is_rep = np.zeros(len(labels), dtype=bool)
    to_exemplar = np.zeros(len(labels), dtype=float)
    for label, position in chosen.items():
        members = np.flatnonzero(labels == label)
        is_rep[position] = True
        to_exemplar[members] = d[members, position]
    return {"is_representative": is_rep, "dist_to_exemplar": to_exemplar}


def _score(distances: np.ndarray, labels) -> Optional[float]:
    """Mean silhouette, or None when it is undefined."""
    labels = np.asarray(labels, dtype=int)
    if len(set(labels.tolist())) < 2 or len(labels) < 3:
        return None
    try:
        mean, _per_structure = silhouette(symmetrise(distances), labels)
    except Exception:
        return None
    return None if mean is None else float(mean)


def affinity(m: RunMatrices, damping: float = 0.9,
             preference_percentile: float = 50.0) -> Grouping:
    """Cluster the combined score by affinity propagation.

    Args:
        m: The run's distance matrices.
        damping: Damping factor, in [0.5, 1.0).
        preference_percentile: Percentile of the similarities used as the preference.

    Returns:
        The grouping.
    """
    result = affinity_labels(m.combined, damping=damping,
                             preference_percentile=preference_percentile)
    return Grouping(
        frame=_frame(m.ids, result.labels, m.combined),
        method="affinity",
        caption=(f"affinity propagation · damping {damping:.2f} · "
                 f"preference {preference_percentile:.0f}th percentile"),
        n_clusters=int(result.n_clusters),
        silhouette=_score(m.combined, result.labels),
    )


def hierarchical(m: RunMatrices, threshold: float, linkage: str = "average") -> Grouping:
    """Cut the combined score's tree at one height.

    Args:
        m: The run's distance matrices.
        threshold: Cut height.
        linkage: Linkage method.

    Returns:
        The grouping.
    """
    result = cluster_labels(m.combined, method=linkage, criterion=THRESHOLD,
                            threshold=float(threshold))
    return Grouping(
        frame=_frame(m.ids, result.labels, m.combined),
        method="hierarchical",
        caption=f"{linkage} linkage · cut at {threshold:.3g}",
        n_clusters=int(result.n_clusters),
        silhouette=_score(m.combined, result.labels),
    )


def two_stage(m: RunMatrices, settings) -> Grouping:
    """Group by shape, then split each group by chemistry.

    Args:
        m: The run's distance matrices.
        settings: Two-stage settings, as built by viz.two_stage.threshold_settings.

    Returns:
        The grouping.
    """
    frame = regroup(m, settings).copy()
    for column, values in representatives(m.combined, frame["cluster_id"].to_numpy()).items():
        frame[column] = values
    shape_cut = getattr(settings.shape, "threshold", None)
    chem_cut = getattr(settings.chem, "threshold", None)
    parts = [f"shape cut {shape_cut:.3g}" if shape_cut else "shape as stored",
             f"chemistry cut {chem_cut:.3g}" if chem_cut else "chemistry as stored"]
    return Grouping(
        frame=frame[list(COLUMNS)].copy(),
        method="two_stage",
        caption=" · ".join(parts),
        n_clusters=int(frame["cluster_id"].nunique()),
        silhouette=_score(m.combined, frame["cluster_id"].to_numpy()),
    )


def agreement(frame: pd.DataFrame, stored: Dict[str, int]):
    """How far a grouping sits from the one stored with the run.

    Args:
        frame: A grouping's frame.
        stored: Stored cluster per pocket ID.

    Returns:
        The comparison, or None when the run stores no clustering.
    """
    if not stored:
        return None
    theirs = [stored.get(str(pid), -1) for pid in frame["pdb_id"]]
    return compare_labelings(theirs, frame["cluster_id"].tolist())
