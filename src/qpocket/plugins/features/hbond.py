"""Geometric hydrogen-bond donor and acceptor fields."""
from typing import Dict

import numpy as np

from qpocket.constants import SIGMA_HBOND
from qpocket.core.features import calculate_geometric_potentials
from qpocket.core.ports import FEATURES
from qpocket.types import FeatureContext


@FEATURES.register("hbond", columns=("h_donor", "h_acceptor"), requires=("structure",))
def hbond(points: np.ndarray, ctx: FeatureContext, params) -> Dict[str, np.ndarray]:
    """Donor and acceptor fields at each point.

    Args:
        points: (N, 3) pocket points.
        ctx: Featurisation context.
        params: None needed.

    Returns:
        Columns h_donor and h_acceptor.
    """
    data = ctx.structure_data
    # Deliberately crossed: a donor field is felt where protein acceptors are.
    return {
        "h_donor": calculate_geometric_potentials(points, data.acceptors, sigma=SIGMA_HBOND),
        "h_acceptor": calculate_geometric_potentials(points, data.donors, sigma=SIGMA_HBOND),
    }
