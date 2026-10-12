"""Charge and lipophilic potentials from the PQR atoms."""
from typing import Dict

import numpy as np

from qpocket.core.features import calculate_cp_lp
from qpocket.core.ports import FEATURES
from qpocket.types import FeatureContext


@FEATURES.register("cp_lp", columns=("cp", "lp"), requires=("pqr",))
def cp_lp(points: np.ndarray, ctx: FeatureContext, params) -> Dict[str, np.ndarray]:
    """Charge and lipophilic potential at each point.

    Args:
        points: (N, 3) pocket points.
        ctx: Featurisation context.
        params: None needed.

    Returns:
        Columns cp and lp.
    """
    data = ctx.structure_data
    cp, lp = calculate_cp_lp(points, data.coords, data.charges, data.hydrophobicity)
    return {"cp": cp, "lp": lp}
