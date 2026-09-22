"""A worked example provider: copy this file to add a feature source."""
from typing import Dict, Tuple

import numpy as np

from qpocket.core.providers.base import FeatureContext, register


class ConstantProvider:
    """Constant per-point value; the shape a real provider takes.

    Attributes:
        name: Name used in features.enabled.
        columns: Columns this provider produces.
        requires: What it needs: "pqr", "structure", or nothing.
    """

    name: str = "template"
    columns: Tuple[str, ...] = ("template",)
    requires: Tuple[str, ...] = ()

    def compute(self, points: np.ndarray, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        """One value per pocket point.

        Args:
            points: (N, 3) pocket points.
            ctx: Cleaned PDB, PQR, parsed atoms and this provider's options.

        Returns:
            The provider's columns, each (N,).
        """
        # A real provider does its work here: a QM single point on ctx.pdb_path, a
        # trained model over ctx.structure_data, a grid named in ctx.options.
        # Keep it pure: no logging, and write no files. Anything that runs a tool
        # belongs in qpocket.io, with its result passed through ctx.options.
        value = float(ctx.options.get("value", 1.0))
        return {"template": np.full(len(points), value, dtype=float)}


register(ConstantProvider())
