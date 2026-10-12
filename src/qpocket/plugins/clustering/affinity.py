"""Affinity Propagation: the method chooses its own cluster count."""
import numpy as np

from qpocket.core import clustering as cl
from qpocket.core.ports import CLUSTERING


@CLUSTERING.register("affinity")
def affinity(distances: np.ndarray, stage) -> cl.Clustering:
    """Cluster by Affinity Propagation.

    Args:
        distances: Square distance matrix.
        stage: damping and max_iter.

    Returns:
        The clustering.

    Raises:
        ValueError: If a threshold cut is asked for.
    """
    if stage.criterion in ("threshold", cl.THRESHOLD):
        raise ValueError("a threshold cut needs a hierarchical stage, not 'affinity'")
    return cl.affinity_labels(distances, damping=stage.damping, max_iter=stage.max_iter)
