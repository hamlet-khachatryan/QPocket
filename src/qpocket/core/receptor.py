"""Which missing pieces of a receptor to build (pure)."""
from typing import Dict, Hashable, Set

import numpy as np


def gaps_to_build(flanks: Dict[Hashable, np.ndarray], pocket: np.ndarray, radius: float) -> Set[Hashable]:
    """Gaps worth modelling: those with a flanking atom near the pocket.

    Args:
        flanks: Gap key to (M, 3) coordinates of the residues either side of it.
        pocket: (N, 3) pocket points.
        radius: Distance limit, Å.

    Returns:
        Keys of the gaps to build.
    """
    if radius <= 0 or len(pocket) == 0:
        return set()
    keep = set()
    for key, xyz in flanks.items():
        if len(xyz) and np.min(np.linalg.norm(xyz[:, None, :] - pocket[None, :, :], axis=-1)) <= radius:
            keep.add(key)
    return keep
