"""Structure cleaning: keeps protein residues and allowed cofactors.

Biopython rather than PyMOL; see :mod:`qpocket.core.structure`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from eliot import log_message, start_action

from qpocket.core import structure as st
from qpocket.io import structures as sio


def clean_structure(
    input_pdb_path: str,
    output_pdb_path: str,
    allowed_cofactors: Optional[str],
) -> bool:
    """Filters a PDB file to retain only protein residues and allowed cofactors.

    Removes water, ions, ligands and modified residues not named in
    *allowed_cofactors*. Assumes *input_pdb_path* already holds only the
    relevant chain(s).

    Args:
        input_pdb_path: Path to the aligned PDB file.
        output_pdb_path: Destination path for the cleaned PDB file.
        allowed_cofactors: ``"|"``-separated residue names to keep
            (e.g. ``"HEM|MG"``). ``None`` keeps only standard amino acids.

    Returns:
        ``True`` on success, ``False`` if cleaning failed or left nothing.
    """
    allowed_set = allowed_cofactors.split("|") if allowed_cofactors else []
    struct_name = Path(input_pdb_path).stem

    with start_action(action_type="clean_structure", pdb_id=struct_name):

        if not Path(input_pdb_path).exists():
            log_message(
                message_type="error",
                stage="clean_structure",
                pdb_id=struct_name,
                error="input file not found",
                path=input_pdb_path,
            )
            return False

        if Path(output_pdb_path).exists():
            log_message(
                message_type="skip",
                stage="clean_structure",
                pdb_id=struct_name,
                reason="output already exists",
            )
            return True

        try:
            structure, n_dropped = st.cleaned(sio.load(Path(input_pdb_path)), allowed_set)

            log_message(
                message_type="cleaning_stats",
                pdb_id=struct_name,
                residues_removed=n_dropped,
                allowed_cofactors=allowed_set,
            )

            if st.count_atoms(structure) == 0:
                log_message(
                    message_type="error",
                    stage="clean_structure",
                    pdb_id=struct_name,
                    error="structure empty after cleaning",
                )
                return False

            # Links come from the aligned file's CONECT records, so a kept
            # covalent cofactor stays bonded to the protein.
            sio.save(structure, Path(output_pdb_path), sio.read_links(Path(input_pdb_path)))
            return True

        except Exception as e:
            log_message(
                message_type="error",
                stage="clean_structure",
                pdb_id=struct_name,
                error=str(e),
            )
            return False
