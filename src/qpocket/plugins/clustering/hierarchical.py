"""Hierarchical clustering: cut a linkage tree by cluster count or by height."""
import numpy as np

from qpocket.core import clustering as cl
from qpocket.core.ports import CLUSTERING


@CLUSTERING.register("hierarchical")
def hierarchical(distances: np.ndarray, stage) -> cl.Clustering:
    """Cluster by a linkage tree.

    Args:
        distances: Square distance matrix.
        stage: linkage, k, k_range, criterion and threshold.

    Returns:
        The clustering.
    """
    if stage.criterion in ("threshold", cl.THRESHOLD):
        return cl.cluster_labels(distances, method=stage.linkage, criterion=cl.THRESHOLD,
                                 threshold=stage.threshold)
    if stage.k is None:
        return cl.best_k_by_silhouette(distances, method=stage.linkage, k_range=stage.k_range)
    return cl.cluster_labels(distances, method=stage.linkage, n_clusters=int(stage.k))
