"""Structure cleaning: removes non-standard residues and cofactors via PyMOL."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from eliot import log_message, start_action
from pymol import cmd

from qpocket_pipeline.global_vars import STANDARD_AA

cmd.feedback("disable", "all", "everything")


def clean_structure(
    input_pdb_path: str,
    output_pdb_path: str,
    allowed_cofactors: Optional[str],
) -> bool:
    """Filters a PDB file to retain only protein residues and allowed cofactors.

    Removes water molecules, ions, ligands, and modified residues that are
    not in *allowed_cofactors*.  Assumes *input_pdb_path* already contains
    only the relevant chain.

    Args:
        input_pdb_path: Path to the aligned PDB file.
        output_pdb_path: Destination path for the cleaned PDB file.
        allowed_cofactors: ``"|"``-separated residue names to keep
            (e.g. ``"HEM|MG"``).  Pass ``None`` to keep only standard AA.

    Returns:
        ``True`` on success, ``False`` if the cleaning failed or the output
        structure is empty.
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
            cmd.reinitialize()
            cmd.load(input_pdb_path, "target")

            valid_res_list = list(STANDARD_AA) + allowed_set
            valid_res_str = "+".join(valid_res_list)
            to_remove_sel = f"target and not resn {valid_res_str}"

            myspace: dict = {"dropped_ids": set()}
            cmd.iterate(
                to_remove_sel,
                "dropped_ids.add((chain, resi))",
                space=myspace,
            )
            n_dropped = len(myspace["dropped_ids"])

            cmd.remove(to_remove_sel)

            log_message(
                message_type="cleaning_stats",
                pdb_id=struct_name,
                residues_removed=n_dropped,
                allowed_cofactors=allowed_set,
            )

            if cmd.count_atoms("target") == 0:
                log_message(
                    message_type="error",
                    stage="clean_structure",
                    pdb_id=struct_name,
                    error="structure empty after cleaning",
                )
                return False

            cmd.save(output_pdb_path, "target")
            cmd.delete("target")
            return True

        except Exception as e:
            log_message(
                message_type="error",
                stage="clean_structure",
                pdb_id=struct_name,
                error=str(e),
            )
            return False