"""Receptor protonation and charges from pdb2pqr with PROPKA."""
from dataclasses import dataclass
from pathlib import Path

from qpocket.core.ports import RECEPTORS
from qpocket.io.tools import protonate_receptor


@dataclass(frozen=True)
class Pdb2pqrReceptorParams:
    """Protonation settings.

    Attributes:
        ph: Protonation pH.
        forcefield: pdb2pqr force field for radii and charges.
    """

    ph: float = 7.4
    forcefield: str = "AMBER"


@RECEPTORS.register("pdb2pqr", params=Pdb2pqrReceptorParams, tools=("pdb2pqr",))
def pdb2pqr(pdb_path: Path, out_pqr: Path, params: Pdb2pqrReceptorParams) -> Path:
    """A protonated, charged receptor PQR.

    Args:
        pdb_path: Repaired receptor.
        out_pqr: Where the PQR goes.
        params: pH and force field.

    Returns:
        The PQR path.
    """
    return protonate_receptor(pdb_path, out_pqr, ph=params.ph, forcefield=params.forcefield)
