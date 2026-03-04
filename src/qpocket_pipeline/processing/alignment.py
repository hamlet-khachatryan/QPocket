"""Structure alignment against a reference using PyMOL and UniProt-based renumbering."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from Bio import Align
from eliot import log_message, start_action
from pymol import cmd

from qpocket_pipeline.utils import calculate_sequence_identity
from qpocket_pipeline.global_vars import THREE_TO_ONE_MAPPING, FASTA_UNIPROT_TEMPLATE

cmd.feedback("disable", "all", "everything")


def fetch_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """Fetches the canonical amino acid sequence for a UniProt accession.

    Args:
        uniprot_id: UniProt accession (e.g. ``"P0DTC2"``).

    Returns:
        Single-letter amino acid sequence string, or ``None`` on failure.
    """
    url = FASTA_UNIPROT_TEMPLATE.format(uniprot_id)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        if len(lines) > 1:
            return "".join(lines[1:])
    except Exception as e:
        log_message(
            message_type="error",
            stage="fetch_uniprot_sequence",
            uniprot_id=uniprot_id,
            error=str(e),
        )
    return None


def generate_renumbering_map(
    pdb_seq: str,
    uniprot_seq: str,
    pdb_start_resi: int = 1,
) -> Dict[int, int]:
    """Builds a PDB residue number → UniProt residue number mapping.

    Performs a local pairwise alignment of *pdb_seq* against *uniprot_seq*
    and maps each aligned position.

    Args:
        pdb_seq: Single-letter sequence extracted from the PDB structure.
        uniprot_seq: Canonical UniProt single-letter sequence.
        pdb_start_resi: Residue index of the first PDB residue (0-based
            offset into *pdb_seq*).

    Returns:
        ``{old_pdb_residue_number: new_uniprot_residue_number}`` dictionary.
        Empty if no alignment could be produced.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(uniprot_seq, pdb_seq)
    try:
        alignment = next(alignments)
    except StopIteration:
        log_message(
            message_type="warning",
            stage="generate_renumbering_map",
            warning="no alignment produced",
        )
        return {}

    mapping: Dict[int, int] = {}
    uni_segments, pdb_segments = alignment.aligned

    for (u_start, u_end), (p_start, p_end) in zip(uni_segments, pdb_segments):
        length = u_end - u_start
        if (p_end - p_start) != length:
            continue
        for i in range(length):
            u_idx = u_start + i
            p_idx = p_start + i
            mapping[pdb_start_resi + p_idx] = u_idx + 1  # 1-based UniProt

    return mapping


def process_structure(
    target_path: str | Path,
    output_dir: str | Path,
    ref_obj_name: str,
    ref_seq: str,
    uniprot_id: Optional[str] = None,
    renumber: bool = True,
) -> Tuple[bool, float, str]:
    """Aligns a target structure to the reference and optionally renumbers residues.

    Steps:

    1. Load target CIF into PyMOL, strip waters.
    2. Select the chain with ≥ 90 % sequence identity to the reference.
    3. Align selected chain to *ref_obj_name* via ``cmd.align``.
    4. If *renumber* and *uniprot_id* are given, remap residue numbers to
       UniProt positions.
    5. Save the aligned (and possibly renumbered) structure as a PDB file.

    Args:
        target_path: Path to the input CIF or PDB file.
        output_dir: Directory to write the output PDB.
        ref_obj_name: Name of the reference PyMOL object.
        ref_seq: Single-letter sequence of the reference chain.
        uniprot_id: UniProt accession used for renumbering (optional).
        renumber: Whether to apply UniProt-based renumbering.

    Returns:
        ``(success, rmsd, output_file_path)`` — on failure rmsd is ``-1.0``
        and path is an empty string.
    """
    target_path = Path(target_path)
    output_dir = Path(output_dir)
    name = target_path.stem
    output_path = output_dir / f"{name}_aligned.pdb"

    with start_action(action_type="process_structure", pdb_id=name):
        try:
            cmd.load(str(target_path), "dynamic_full")
            cmd.remove("dynamic_full and (solvent or resn HOH)")

            chains = cmd.get_chains("dynamic_full")
            best_chain: Optional[str] = None
            best_identity = 0.0

            for chain in chains:
                chain_seq = cmd.get_fastastr(selection=f"dynamic_full and chain {chain}")
                chain_seq = "".join(chain_seq.split()[1:])
                identity = calculate_sequence_identity(ref_seq, chain_seq)
                if identity >= 90.0 and identity > best_identity:
                    best_identity = identity
                    best_chain = chain

            if best_chain is None:
                log_message(
                    message_type="skip",
                    stage="process_structure",
                    pdb_id=name,
                    reason="no chain with >=90% sequence identity",
                )
                cmd.delete("dynamic_full")
                return False, -1.0, ""

            cmd.create(name, f"dynamic_full and chain {best_chain}")
            cmd.delete("dynamic_full")

            aln_result = cmd.align(name, ref_obj_name)
            rmsd = float(aln_result[0])

            log_message(
                message_type="alignment_stats",
                pdb_id=name,
                chain=best_chain,
                identity=best_identity,
                rmsd=rmsd,
            )

            if renumber and uniprot_id:
                model: Dict = {"resis": [], "seq": []}
                cmd.iterate(
                    f"{name} and name CA",
                    "resis.append(resi); seq.append(resn)",
                    space=model,
                )

                pdb_seq_str = "".join(
                    THREE_TO_ONE_MAPPING.get(r, "X") for r in model["seq"]
                )

                try:
                    pdb_resis = [
                        int("".join(filter(str.isdigit, r))) for r in model["resis"]
                    ]
                except ValueError:
                    pdb_resis = []

                if pdb_resis:
                    uni_seq = fetch_uniprot_sequence(uniprot_id)
                    if uni_seq:
                        relative_map = generate_renumbering_map(
                            pdb_seq_str, uni_seq, pdb_start_resi=0
                        )
                        for seq_idx, new_uni_res in relative_map.items():
                            if seq_idx < len(pdb_resis):
                                old_res_id = pdb_resis[seq_idx]
                                cmd.alter(
                                    f"{name} and resi {old_res_id}",
                                    f"resi='{new_uni_res}'",
                                )
                        cmd.sort(name)

                        log_message(
                            message_type="renumbering_applied",
                            pdb_id=name,
                            n_mapped=len(relative_map),
                        )

            cmd.save(str(output_path), name)
            cmd.delete(name)
            return True, rmsd, str(output_path)

        except Exception as e:
            log_message(
                message_type="error",
                stage="process_structure",
                pdb_id=name,
                error=str(e),
            )
            cmd.delete(name)
            cmd.delete("dynamic_full")
            return False, -1.0, ""