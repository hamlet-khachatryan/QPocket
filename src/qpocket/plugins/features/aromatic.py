"""Geometric aromatic field from ring centres."""
from typing import Dict

import numpy as np

from qpocket.constants import SIGMA_AROM
from qpocket.core.features import calculate_geometric_potentials
from qpocket.core.ports import FEATURES
from qpocket.types import FeatureContext


@FEATURES.register("aromatic", columns=("aromatic",), requires=("structure",))
def aromatic(points: np.ndarray, ctx: FeatureContext, params) -> Dict[str, np.ndarray]:
    """Aromatic field at each point.

    Args:
        points: (N, 3) pocket points.
        ctx: Featurisation context.
        params: None needed.

    Returns:
        Column aromatic.
    """
    return {"aromatic": calculate_geometric_potentials(points, ctx.structure_data.aromatics,
                                                       sigma=SIGMA_AROM)}
