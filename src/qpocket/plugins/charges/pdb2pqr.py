"""Classical charges and radii from pdb2pqr."""
from pathlib import Path

from qpocket.core.ports import CHARGES
from qpocket.io import tables, tools
from qpocket.types import StructureData


@CHARGES.register("pdb2pqr", tools=("pdb2pqr",))
def pdb2pqr(pdb_path: Path, pqr_path: Path, params) -> StructureData:
    """Atoms, charges and pharmacophore sites of a protein.

    Args:
        pdb_path: Cleaned protein.
        pqr_path: Where the PQR goes; reused when newer than the PDB.
        params: None needed.

    Returns:
        The structure's atom data.
    """
    # One pdb2pqr run per structure; the Interactions pass reuses it.
    if not pqr_path.exists() or pqr_path.stat().st_mtime < Path(pdb_path).stat().st_mtime:
        tools.protonate(pdb_path, pqr_path)
    return tools.structure_data(pdb_path, tables.read_pqr(pqr_path))
