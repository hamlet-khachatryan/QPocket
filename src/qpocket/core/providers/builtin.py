"""The four providers behind today's six feature columns."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from qpocket.constants import SIGMA_AROM, SIGMA_HBOND
from qpocket.core.features import calculate_cp_lp, calculate_geometric_potentials
from qpocket.core.providers.base import FeatureContext, register


class ApbsProvider:
    """Electrostatic potential, computed by APBS before the providers run.

    Attributes:
        name: Name used in features.enabled.
        columns: Columns this provider produces.
        requires: Inputs it needs.
    """

    name: str = "apbs"
    columns: Tuple[str, ...] = ("apbs",)
    requires: Tuple[str, ...] = ("pqr",)

    def compute(self, points: np.ndarray, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        """Potential at each point.

        Args:
            points: (N, 3) pocket points.
            ctx: Featurisation context; options["apbs"] holds the grid values.

        Returns:
            Column apbs.
        """
        # core/ runs no tools: the pipeline puts the APBS result in the context.
        values = ctx.options.get("apbs")
        return {"apbs": np.zeros(len(points)) if values is None else values}


class CpLpProvider:
    """Charge and lipophilic potentials from the PQR atoms.

    Attributes:
        name: Name used in features.enabled.
        columns: Columns this provider produces.
        requires: Inputs it needs.
    """

    name: str = "cp_lp"
    columns: Tuple[str, ...] = ("cp", "lp")
    requires: Tuple[str, ...] = ("pqr",)

    def compute(self, points: np.ndarray, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        """Charge and lipophilic potential at each point.

        Args:
            points: (N, 3) pocket points.
            ctx: Featurisation context.

        Returns:
            Columns cp and lp.
        """
        data = ctx.structure_data
        cp, lp = calculate_cp_lp(points, data.coords, data.charges, data.hydrophobicity)
        return {"cp": cp, "lp": lp}


class HBondProvider:
    """Geometric hydrogen-bond donor and acceptor fields.

    Attributes:
        name: Name used in features.enabled.
        columns: Columns this provider produces.
        requires: Inputs it needs.
    """

    name: str = "hbond"
    columns: Tuple[str, ...] = ("h_donor", "h_acceptor")
    requires: Tuple[str, ...] = ("structure",)

    def compute(self, points: np.ndarray, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        """Donor and acceptor fields at each point.

        Args:
            points: (N, 3) pocket points.
            ctx: Featurisation context.

        Returns:
            Columns h_donor and h_acceptor.
        """
        data = ctx.structure_data
        # Deliberately crossed: a donor field is felt where protein acceptors are.
        return {
            "h_donor": calculate_geometric_potentials(points, data.acceptors, sigma=SIGMA_HBOND),
            "h_acceptor": calculate_geometric_potentials(points, data.donors, sigma=SIGMA_HBOND),
        }


class AromaticProvider:
    """Geometric aromatic field from ring centres.

    Attributes:
        name: Name used in features.enabled.
        columns: Columns this provider produces.
        requires: Inputs it needs.
    """

    name: str = "aromatic"
    columns: Tuple[str, ...] = ("aromatic",)
    requires: Tuple[str, ...] = ("structure",)

    def compute(self, points: np.ndarray, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        """Aromatic field at each point.

        Args:
            points: (N, 3) pocket points.
            ctx: Featurisation context.

        Returns:
            Column aromatic.
        """
        return {"aromatic": calculate_geometric_potentials(points, ctx.structure_data.aromatics,
                                                           sigma=SIGMA_AROM)}


for _provider in (ApbsProvider(), CpLpProvider(), HBondProvider(), AromaticProvider()):
    register(_provider)
