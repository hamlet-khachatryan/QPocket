"""Hierarchical clustering: cut a linkage tree by cluster count, height or relative height."""
import numpy as np

from qpocket.core import clustering as cl
from qpocket.core.ports import CLUSTERING


@CLUSTERING.register("hierarchical")
def hierarchical(distances: np.ndarray, stage) -> cl.Clustering:
    """Cluster by a linkage tree.

    Args:
        distances: Square distance matrix.
        stage: linkage, k, criterion and threshold.

    Returns:
        The clustering.

    Raises:
        ValueError: If a count cut has no k.
    """
    if stage.criterion in ("threshold", "relative", cl.THRESHOLD, cl.RELATIVE):
        return cl.cluster_labels(distances, method=stage.linkage, criterion=stage.criterion,
                                 threshold=stage.threshold)
    if stage.k is None:
        raise ValueError(cl.NEEDS_K)
    return cl.cluster_labels(distances, method=stage.linkage, n_clusters=int(stage.k))
