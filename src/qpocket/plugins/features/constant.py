"""A worked example channel: copy this file to add a feature source."""
from dataclasses import dataclass
from typing import Dict

import numpy as np

from qpocket.core.ports import FEATURES
from qpocket.types import FeatureContext


@dataclass(frozen=True)
class ConstantParams:
    """Parameters of the template channel.

    Attributes:
        value: The value every point gets.
    """

    value: float = 1.0


@FEATURES.register("template", params=ConstantParams, columns=("template",))
def constant(points: np.ndarray, ctx: FeatureContext, params: ConstantParams) -> Dict[str, np.ndarray]:
    """One value per pocket point.

    Args:
        points: (N, 3) pocket points.
        ctx: Cleaned PDB, PQR, parsed atoms and precomputed values.
        params: This channel's parameters.

    Returns:
        The channel's columns, each (N,).
    """
    # A real channel does its work here: a QM single point on ctx.pdb_path, a model
    # over ctx.structure_data. Keep it pure: anything that runs a tool belongs in
    # qpocket.io, with its result passed through ctx.options.
    return {"template": np.full(len(points), params.value, dtype=float)}
