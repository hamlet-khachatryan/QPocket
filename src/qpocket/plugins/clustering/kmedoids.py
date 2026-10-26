"""k-medoids: partition around representative pockets."""
import numpy as np

from qpocket.core import clustering as cl
from qpocket.core.ports import CLUSTERING


@CLUSTERING.register("kmedoids")
def kmedoids(distances: np.ndarray, stage) -> cl.Clustering:
    """Cluster by k-medoids.

    Args:
        distances: Square distance matrix.
        stage: k and criterion.

    Returns:
        The clustering.

    Raises:
        ValueError: If a threshold or relative cut is asked for, or k is not set.
    """
    if stage.criterion in ("threshold", "relative", cl.THRESHOLD, cl.RELATIVE):
        raise ValueError(f"a {stage.criterion} cut needs a hierarchical stage, not 'kmedoids'")
    if stage.k is None:
        raise ValueError(cl.NEEDS_K)
    return cl.cluster_labels(distances, method=cl.KMEDOIDS, n_clusters=int(stage.k))
