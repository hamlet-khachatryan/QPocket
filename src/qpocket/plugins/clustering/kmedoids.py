"""k-medoids: partition around representative pockets."""
import numpy as np

from qpocket.core import clustering as cl
from qpocket.core.ports import CLUSTERING


@CLUSTERING.register("kmedoids")
def kmedoids(distances: np.ndarray, stage) -> cl.Clustering:
    """Cluster by k-medoids.

    Args:
        distances: Square distance matrix.
        stage: k and k_range.

    Returns:
        The clustering.

    Raises:
        ValueError: If a threshold cut is asked for.
    """
    if stage.criterion in ("threshold", cl.THRESHOLD):
        raise ValueError("a threshold cut needs a hierarchical stage, not 'kmedoids'")
    if stage.k is None:
        return cl.best_k_by_silhouette(distances, method=cl.KMEDOIDS, k_range=stage.k_range)
    return cl.cluster_labels(distances, method=cl.KMEDOIDS, n_clusters=int(stage.k))
