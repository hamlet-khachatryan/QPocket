"""Electrostatic potential, computed by APBS before the channels run."""
from typing import Dict

import numpy as np

from qpocket.core.ports import FEATURES
from qpocket.types import FeatureContext


@FEATURES.register("apbs", columns=("apbs",), requires=("pqr",))
def apbs(points: np.ndarray, ctx: FeatureContext, params) -> Dict[str, np.ndarray]:
    """Potential at each point.

    Args:
        points: (N, 3) pocket points.
        ctx: Featurisation context; options["apbs"] holds the grid values.
        params: None needed.

    Returns:
        Column apbs.
    """
    # The pipeline runs the tool and puts the result in the context.
    values = ctx.options.get("apbs")
    return {"apbs": np.zeros(len(points)) if values is None else values}
