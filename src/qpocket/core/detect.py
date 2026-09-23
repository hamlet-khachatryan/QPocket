"""Detect protein-ligand interactions with ProLIF and name them for the pipeline."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: Requested from ProLIF by default. PiStacking is deliberately absent: it is the
#: union of FaceToFace and EdgeToFace, so requesting all three triple-counts stacks.
INTERACTION_NAMES: Tuple[str, ...] = (
    "Hydrophobic", "HBDonor", "HBAcceptor", "FaceToFace", "EdgeToFace",
    "Cationic", "Anionic", "CationPi", "PiCation",
    "XBDonor", "XBAcceptor", "MetalDonor", "MetalAcceptor",
)

#: ProLIF names an interaction from the ligand's side; the stored strings name it
#: from the protein's, so the directional types are inverted here.
LABELS: Dict[str, str] = {
    "Hydrophobic": "Hydrophobic",
    "HBDonor": "H-bond Acceptor",
    "HBAcceptor": "H-bond Donor",
    "FaceToFace": "Pi-stacking (P)",
    "EdgeToFace": "Pi-stacking (T)",
    "Cationic": "Saltbridge (-)",
    "Anionic": "Saltbridge (+)",
    "CationPi": "Pi-Cation (Pi)",
    "PiCation": "Pi-Cation (Cat)",
    "XBDonor": "Halogen Bond",
    "XBAcceptor": "Halogen Bond",
    "MetalDonor": "Metal Complex",
    "MetalAcceptor": "Metal Complex",
}


def format_entry(res_name: str, res_seq: int, chain_id: str, prolif_name: str) -> str:
    """One stored interaction string.

    Args:
        res_name: Protein residue name.
        res_seq: Residue number.
        chain_id: Chain identifier.
        prolif_name: The ProLIF interaction name.

    Returns:
        A string of the form RESNAME_RESNUM_CHAIN|InteractionType.
    """
    return f"{res_name.upper()}_{res_seq}_{chain_id}|{LABELS[prolif_name]}"


def interactions(ligand, protein,
                 enabled: Optional[Sequence[str]] = None,
                 parameters: Optional[Mapping[str, Mapping]] = None) -> List[str]:
    """Interactions between one ligand and one protein.

    Args:
        ligand: A prolif Molecule holding the ligand.
        protein: A prolif Molecule holding the protein.
        enabled: ProLIF interaction names, defaulting to INTERACTION_NAMES.
        parameters: Per-interaction parameter overrides.

    Returns:
        Sorted, de-duplicated interaction strings.
    """
    # Imported here because prolif pulls in MDAnalysis, which is slow to import
    # and must not be paid by every qpocket import.
    import prolif as plf

    names = list(enabled or INTERACTION_NAMES)
    fingerprint = plf.Fingerprint(names, parameters=dict(parameters or {}), count=True)
    fingerprint.run_from_iterable([ligand], protein, progress=False)

    found: List[str] = []
    for _frame, residues in fingerprint.ifp.items():
        for (_ligand_id, residue_id), detected in residues.items():
            for name, occurrences in detected.items():
                if name not in LABELS:
                    continue
                for _occurrence in occurrences:
                    found.append(format_entry(residue_id.name, residue_id.number,
                                              residue_id.chain or "A", name))
    return sorted(set(found))
