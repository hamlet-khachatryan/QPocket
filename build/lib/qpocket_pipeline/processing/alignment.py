"""Structure alignment against a reference using PyMOL and UniProt-based renumbering."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from Bio import Align
from eliot import log_message, start_action
from pymol import cmd

from qpocket_pipeline.utils import calculate_sequence_identity, find_auxiliary_chain
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
    pdb_start_resi: int = 0,
) -> Dict[int, int]:
    """Builds a sequential-index → UniProt residue number mapping.

    Performs a local pairwise alignment of *pdb_seq* against *uniprot_seq*
    and maps each aligned position.

    Args:
        pdb_seq: Single-letter sequence extracted from the PDB structure.
        uniprot_seq: Canonical UniProt single-letter sequence.
        pdb_start_resi: 0-based offset added to alignment position indices
            when constructing the output keys.  Callers typically pass 0 so
            that keys are simple list indices into the ``pdb_resis`` array.

    Returns:
        ``{list_index: uniprot_residue_number_1based}`` dictionary where
        *list_index* is an index into the ordered list of CA residue numbers
        extracted from the structure.  Empty if no alignment was produced.
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
                if identity >= 20.0 and identity > best_identity:
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
                _apply_uniprot_renumbering(
                    obj_name=name,
                    chain_id=best_chain,
                    uniprot_id=uniprot_id,
                    pdb_id=name,
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


def process_structure_interface(
    target_path: str | Path,
    output_dir: str | Path,
    ref_obj_name: str,
    ref_seq: str,
    ref_aux_seqs: Dict[str, str],
    ref_aux_centroids: Dict[str, np.ndarray],
    uniprot_id: Optional[str] = None,
    auxiliary_uniprot_ids: Optional[Dict[str, str]] = None,
    renumber: bool = True,
    min_auxiliary_identity: float = 0.70,
    max_auxiliary_centroid_deviation: float = 15.0,
    require_all_chains: bool = True,
) -> Tuple[bool, float, str, Dict[str, str]]:
    """Aligns a multi-chain interface structure to the reference.

    Extends :func:`process_structure` for binding pockets formed at the
    interface between two or more chains.  The alignment strategy is:

    1. Load the full target structure.
    2. Identify the primary chain by sequence identity (≥ 90 %) against
       *ref_seq*.
    3. Align the **entire object** using the primary chain as the anchor,
       so all auxiliary chains are carried into the reference frame.
    4. For each reference auxiliary chain, find the matching target chain via
       :func:`~qpocket_pipeline.utils.find_auxiliary_chain` (sequence filter
       followed by Cα centroid proximity for homodimer disambiguation).
    5. Skip the structure if *require_all_chains* is ``True`` and any
       auxiliary chain cannot be matched.
    6. Save a multi-chain PDB containing the primary chain and all matched
       auxiliary chains.
    7. Optionally renumber each chain to canonical UniProt residue positions.

    Args:
        target_path: Path to the input CIF or PDB file.
        output_dir: Directory to write the output PDB.
        ref_obj_name: Name of the reference PyMOL object (primary chain only).
        ref_seq: Single-letter sequence of the reference primary chain.
        ref_aux_seqs: Mapping ``{ref_chain_id: sequence}`` for each reference
            auxiliary chain.
        ref_aux_centroids: Mapping ``{ref_chain_id: centroid_xyz}`` — Cα mean
            coordinates of each reference auxiliary chain in the reference
            coordinate frame.
        uniprot_id: UniProt accession for the primary chain (renumbering).
        auxiliary_uniprot_ids: Per-chain UniProt overrides keyed by reference
            chain ID (e.g. ``{"B": "P00533"}``).  Chains absent from this
            mapping are not renumbered.
        renumber: Whether to apply UniProt-based renumbering.
        min_auxiliary_identity: Minimum sequence identity fraction to accept an
            auxiliary chain match.
        max_auxiliary_centroid_deviation: Cα centroid deviation threshold (Å)
            above which a match triggers a warning.
        require_all_chains: When ``True``, return failure if any auxiliary
            chain cannot be matched.

    Returns:
        ``(success, rmsd, output_file_path, aux_mapping)`` where *aux_mapping*
        maps each reference auxiliary chain ID to its matched target chain ID.
        On failure, *rmsd* is ``-1.0``, *output_file_path* is empty, and
        *aux_mapping* is an empty dict.
    """
    target_path = Path(target_path)
    output_dir = Path(output_dir)
    name = target_path.stem
    output_path = output_dir / f"{name}_aligned.pdb"

    with start_action(action_type="process_structure_interface", pdb_id=name):
        try:
            cmd.load(str(target_path), name)
            cmd.remove(f"{name} and (solvent or resn HOH)")

            chains = cmd.get_chains(name)

            # ── Step 1: identify primary chain ────────────────────────────────
            best_chain: Optional[str] = None
            best_identity = 0.0

            for chain in chains:
                raw_fasta = cmd.get_fastastr(selection=f"{name} and chain {chain}")
                chain_seq = "".join(raw_fasta.split()[1:])
                identity = calculate_sequence_identity(ref_seq, chain_seq)
                if identity >= 90.0 and identity > best_identity:
                    best_identity = identity
                    best_chain = chain

            if best_chain is None:
                log_message(
                    message_type="skip",
                    stage="process_structure_interface",
                    pdb_id=name,
                    reason="no chain with >=90% sequence identity to primary reference",
                )
                cmd.delete(name)
                return False, -1.0, "", {}

            # ── Step 2: align entire object using primary chain as anchor ─────
            # cmd.align transforms ALL atoms of the object that contains the
            # mobile selection, so auxiliary chains are automatically moved
            # into the reference coordinate frame.
            aln_result = cmd.align(f"{name} and chain {best_chain}", ref_obj_name)
            rmsd = float(aln_result[0])

            # ── Step 3: collect per-chain sequences and Cα centroids ──────────
            # Both are now in the reference coordinate frame.
            chain_sequences: Dict[str, str] = {}
            chain_ca_centroids: Dict[str, np.ndarray] = {}

            for chain in chains:
                raw_fasta = cmd.get_fastastr(selection=f"{name} and chain {chain}")
                chain_sequences[chain] = "".join(raw_fasta.split()[1:])

                ca_xyz = cmd.get_coords(
                    f"{name} and chain {chain} and name CA", state=1
                )
                if ca_xyz is not None and len(ca_xyz) > 0:
                    chain_ca_centroids[chain] = np.mean(np.array(ca_xyz), axis=0)

            # ── Step 4: find each auxiliary chain in the target ───────────────
            aux_mapping: Dict[str, str] = {}  # ref_chain_id → target_chain_id

            for ref_chain_id, ref_aux_seq in ref_aux_seqs.items():
                ref_aux_centroid = ref_aux_centroids.get(ref_chain_id)
                if ref_aux_centroid is None:
                    log_message(
                        message_type="warning",
                        stage="process_structure_interface",
                        pdb_id=name,
                        ref_chain=ref_chain_id,
                        warning="reference auxiliary centroid not available; "
                                "falling back to sequence-only matching",
                    )
                    ref_aux_centroid = np.zeros(3)

                matched = find_auxiliary_chain(
                    chain_sequences=chain_sequences,
                    chain_ca_centroids=chain_ca_centroids,
                    ref_aux_seq=ref_aux_seq,
                    ref_aux_centroid=ref_aux_centroid,
                    primary_chain=best_chain,
                    min_identity=min_auxiliary_identity,
                    max_centroid_deviation=max_auxiliary_centroid_deviation,
                )

                if matched is None:
                    log_message(
                        message_type="auxiliary_chain_not_found",
                        stage="process_structure_interface",
                        pdb_id=name,
                        ref_chain=ref_chain_id,
                        min_identity=min_auxiliary_identity,
                    )
                    if require_all_chains:
                        cmd.delete(name)
                        return False, -1.0, "", {}
                    # require_all_chains=False: continue with the chains we have
                else:
                    aux_mapping[ref_chain_id] = matched
                    log_message(
                        message_type="auxiliary_chain_matched",
                        pdb_id=name,
                        ref_chain=ref_chain_id,
                        target_chain=matched,
                    )

            # ── Step 5: build output object with primary + auxiliary chains ───
            chains_to_keep = [best_chain] + list(aux_mapping.values())
            chain_sel = " or ".join(f"chain {c}" for c in chains_to_keep)
            output_obj = f"{name}_interface"
            cmd.create(output_obj, f"{name} and ({chain_sel})")
            cmd.delete(name)

            # ── Step 6: renumber chains to UniProt positions ──────────────────
            if renumber:
                if uniprot_id:
                    _apply_uniprot_renumbering(
                        obj_name=output_obj,
                        chain_id=best_chain,
                        uniprot_id=uniprot_id,
                        pdb_id=name,
                    )

                if aux_mapping and auxiliary_uniprot_ids:
                    for ref_chain_id, target_chain_id in aux_mapping.items():
                        aux_uniprot = (auxiliary_uniprot_ids or {}).get(ref_chain_id)
                        if aux_uniprot:
                            _apply_uniprot_renumbering(
                                obj_name=output_obj,
                                chain_id=target_chain_id,
                                uniprot_id=aux_uniprot,
                                pdb_id=name,
                            )

            cmd.save(str(output_path), output_obj)
            cmd.delete(output_obj)

            log_message(
                message_type="alignment_stats",
                pdb_id=name,
                primary_chain=best_chain,
                primary_identity=best_identity,
                rmsd=rmsd,
                aux_mapping=aux_mapping,
            )
            return True, rmsd, str(output_path), aux_mapping

        except Exception as e:
            log_message(
                message_type="error",
                stage="process_structure_interface",
                pdb_id=name,
                error=str(e),
            )
            cmd.delete(name)
            cmd.delete(f"{name}_interface")
            return False, -1.0, "", {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_uniprot_renumbering(
    obj_name: str,
    chain_id: str,
    uniprot_id: str,
    pdb_id: str,
) -> None:
    """Renumbers residues of *chain_id* in a PyMOL object to UniProt positions.

    Fetches the canonical sequence for *uniprot_id*, aligns it against the
    PDB sequence extracted from PyMOL, and calls ``cmd.alter`` to update each
    residue's sequence number in-place.

    Args:
        obj_name: Name of the PyMOL object to modify.
        chain_id: Chain whose residues are to be renumbered.
        uniprot_id: UniProt accession supplying the canonical numbering.
        pdb_id: Used for logging only.
    """
    model: Dict = {"resis": [], "seq": []}
    cmd.iterate(
        f"{obj_name} and chain {chain_id} and name CA",
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

    if not pdb_resis:
        return

    uni_seq = fetch_uniprot_sequence(uniprot_id)
    if not uni_seq:
        return

    relative_map = generate_renumbering_map(pdb_seq_str, uni_seq, pdb_start_resi=0)
    for seq_idx, new_uni_res in relative_map.items():
        if seq_idx < len(pdb_resis):
            old_res_id = pdb_resis[seq_idx]
            cmd.alter(
                f"{obj_name} and chain {chain_id} and resi {old_res_id}",
                f"resi='{new_uni_res}'",
            )
    cmd.sort(obj_name)

    log_message(
        message_type="renumbering_applied",
        pdb_id=pdb_id,
        chain=chain_id,
        n_mapped=len(relative_map),
    )
