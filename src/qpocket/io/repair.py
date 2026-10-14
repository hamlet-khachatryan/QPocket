"""Receptor repair with PDBFixer: side chains everywhere, loops near the pocket."""
from pathlib import Path
from typing import Dict

import numpy as np
from eliot import log_message

from qpocket.core.receptor import gaps_to_build


def repair(pdb_in: Path, pdb_out: Path, pocket: np.ndarray, loop_radius: float) -> Dict[str, int]:
    """Complete missing side-chain atoms and the missing loops that border the pocket.

    Args:
        pdb_in: Cleaned receptor.
        pdb_out: Where the repaired receptor goes.
        pocket: (N, 3) pocket points, Å.
        loop_radius: Build a missing loop only this close to the pocket, Å.

    Returns:
        Counts: loops_built, loops_skipped, atoms_added.
    """
    from openmm import unit
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    xyz = np.array(fixer.positions.value_in_unit(unit.angstrom))
    chains = list(fixer.topology.chains())
    flanks = {}
    for key in fixer.missingResidues:
        chain_index, residue_index = key
        residues = list(chains[chain_index].residues())
        if residue_index == 0 or residue_index >= len(residues):
            continue  # terminal gaps are never built
        atoms = [a.index for r in (residues[residue_index - 1], residues[residue_index]) for a in r.atoms()]
        flanks[key] = xyz[atoms]
    build = gaps_to_build(flanks, np.asarray(pocket, dtype=float), loop_radius)
    skipped = len(fixer.missingResidues) - len(build)
    fixer.missingResidues = {k: v for k, v in fixer.missingResidues.items() if k in build}
    n_before = fixer.topology.getNumAtoms()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()  # adds atoms only; existing coordinates are kept
    Path(pdb_out).parent.mkdir(parents=True, exist_ok=True)
    with open(pdb_out, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    stats = {"loops_built": len(build), "loops_skipped": skipped,
             "atoms_added": fixer.topology.getNumAtoms() - n_before}
    log_message(message_type="receptor_repaired", pdb=str(pdb_in), **stats)
    return stats
