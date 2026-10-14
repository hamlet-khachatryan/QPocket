"""Receptor repair with PDBFixer: side chains everywhere, loops near the pocket."""
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
from eliot import log_message

from qpocket.core.receptor import gaps_to_build


def chain_sequences(structure_path: Path) -> Dict[str, List[str]]:
    """Full polymer sequence of every chain, from an entry's SEQRES or entity records.

    Args:
        structure_path: The ingested PDB or mmCIF file.

    Returns:
        Author chain name to residue names, unmodelled residues included.
    """
    import gemmi

    st = gemmi.read_structure(str(structure_path))
    st.setup_entities()
    out: Dict[str, List[str]] = {}
    for chain in st[0]:
        polymer = chain.get_polymer()
        entity = st.get_entity_of(polymer) if len(polymer) else None
        if entity is not None and entity.full_sequence:
            out[chain.name] = [gemmi.Entity.first_mon(m) for m in entity.full_sequence]
    return out


def repair(pdb_in: Path, pdb_out: Path, pocket: np.ndarray, loop_radius: float,
           sequences: Optional[Mapping[str, List[str]]] = None) -> Dict[str, int]:
    """Complete missing side-chain atoms and the missing loops that border the pocket.

    Args:
        pdb_in: Cleaned receptor.
        pdb_out: Where the repaired receptor goes.
        pocket: (N, 3) pocket points, Å.
        loop_radius: Build a missing loop only this close to the pocket, Å.
        sequences: Chain to full sequence; without it no missing loop can be seen, because cleaned files carry no SEQRES.

    Returns:
        Counts: loops_built, loops_skipped, atoms_added.
    """
    from openmm import unit
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_in))
    if sequences:
        from pdbfixer.pdbfixer import Sequence

        fixer.sequences = [Sequence(c.id, list(sequences[c.id]))
                           for c in fixer.topology.chains() if c.id in sequences]
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
