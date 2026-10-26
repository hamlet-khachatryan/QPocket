"""Binding-site discovery: which ligands a structure holds (pure)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from qpocket.core.types import Ligand


def find_ligands(structure, pdb_id: str, allowed_cofactors: Sequence[str],
                 min_heavy_atoms: int) -> List[Tuple[Ligand, object]]:
    """Every non-water, carbon-containing HET residue of the first model that the run does not clean away.

    Args:
        structure: Parsed Biopython structure (first model only, so NMR ensembles do not repeat ligands).
        pdb_id: Identifier stamped on each ligand.
        allowed_cofactors: Residue names never reported.
        min_heavy_atoms: Residues with fewer non-hydrogen atoms are skipped.

    Returns:
        ``(Ligand, residue)`` pairs in chain, then residue, order.
    """
    found: List[Tuple[Ligand, object]] = []
    for chain in next(iter(structure)):
        for res in chain:
            if res.id[0] in (" ", "W") or res.get_resname().strip().upper() in allowed_cofactors:
                continue
            if not any((a.element or "").strip().upper() == "C" for a in res):
                continue  # ions and inorganic species such as SO4 are never ligands
            n_heavy = sum(1 for a in res if (a.element or "").strip().upper() not in ("H", "D"))
            if n_heavy < min_heavy_atoms:
                continue
            _, resseq, icode = res.id
            found.append((Ligand(pdb_id, chain.id, res.get_resname().strip(), resseq, icode, n_heavy), res))
    return found
