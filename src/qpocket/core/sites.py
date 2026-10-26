"""Binding-site discovery: which ligands a structure holds (pure)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from qpocket.core.pocket import is_residue_ligand
from qpocket.core.types import Ligand


def find_ligands(structure, pdb_id: str, allowed_cofactors: Sequence[str], min_heavy_atoms: int,
                 include: Sequence[str] = (), exclude: Sequence[str] = ()) -> List[Tuple[Ligand, object]]:
    """Every drug-like ligand residue in a structure's first model.

    Args:
        structure: Parsed Biopython structure (first model only, so NMR ensembles do not repeat ligands).
        pdb_id: Identifier stamped on each ligand.
        allowed_cofactors: Residue names never reported.
        min_heavy_atoms: Residues with fewer non-hydrogen atoms are skipped.
        include: Names reported even if normally filtered out (cofactors still win).
        exclude: Names never reported.

    Returns:
        ``(Ligand, residue)`` pairs in chain, then residue, order.
    """
    found: List[Tuple[Ligand, object]] = []
    for chain in next(iter(structure)):
        for res in chain:
            if not is_residue_ligand(res, allowed_cofactors, include, exclude):
                continue
            n_heavy = sum(1 for a in res if (a.element or "").strip().upper() not in ("H", "D"))
            if n_heavy < min_heavy_atoms:
                continue
            _, resseq, icode = res.id
            found.append((Ligand(pdb_id, chain.id, res.get_resname().strip(), resseq, icode, n_heavy), res))
    return found
