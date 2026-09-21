"""Structure alignment against a reference, and UniProt-based renumbering.

Uses :mod:`qpocket_pipeline.processing.structure` (Biopython) rather than PyMOL,
which has no pip wheel for most cluster operating systems. The superposition
reproduces ``cmd.align``; on the 20 PARG structures every atom lands within
0.0016 A of where PyMOL placed it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from Bio import Align
from Bio.PDB.Residue import Residue
from eliot import log_message, start_action

from qpocket_pipeline.processing import structure as st
from qpocket_pipeline.utils import calculate_sequence_identity, find_auxiliary_chain
from qpocket_pipeline.global_vars import FASTA_UNIPROT_TEMPLATE

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
    reference: Sequence[Residue],
    ref_seq: str,
    uniprot_id: Optional[str] = None,
    renumber: bool = True,
) -> Tuple[bool, float, str]:
    """Aligns a target structure to the reference and optionally renumbers residues.

    Steps:

    1. Load the target, strip water.
    2. Select the chain with >= 20 % sequence identity to the reference (the
       highest, if several).
    3. Superimpose that chain onto *reference*, as ``cmd.align`` did.
    4. If *renumber* and *uniprot_id* are given, remap residue numbers to
       UniProt positions.
    5. Save the aligned chain, ligands included, as a PDB file.

    Args:
        target_path: Path to the input CIF or PDB file.
        output_dir: Directory to write the output PDB.
        reference: Amino-acid residues of the reference chain, already in the
            reference frame.
        ref_seq: Single-letter sequence of the reference chain.
        uniprot_id: UniProt accession used for renumbering (optional).
        renumber: Whether to apply UniProt-based renumbering.

    Returns:
        ``(success, rmsd, output_file_path)`` -- on failure rmsd is ``-1.0``
        and path is an empty string.
    """
    target_path = Path(target_path)
    output_dir = Path(output_dir)
    name = target_path.stem
    output_path = output_dir / f"{name}_aligned.pdb"

    with start_action(action_type="process_structure", pdb_id=name):
        try:
            full = st.load(target_path)
            st.remove_water(full)

            best_chain: Optional[str] = None
            best_identity = 0.0
            for chain in full[0]:
                identity = calculate_sequence_identity(ref_seq, st.chain_sequence(chain))
                if identity >= 20.0 and identity > best_identity:
                    best_identity = identity
                    best_chain = chain.id

            if best_chain is None:
                log_message(
                    message_type="skip",
                    stage="process_structure",
                    pdb_id=name,
                    reason="no chain with >=20% sequence identity",
                )
                return False, -1.0, ""

            mobile = st.keep_chains(full, [best_chain])
            fit = st.superpose(st.polymer_residues(mobile[0][best_chain]), reference)
            st.apply(mobile, fit)

            log_message(
                message_type="alignment_stats",
                pdb_id=name,
                chain=best_chain,
                identity=best_identity,
                rmsd=fit.rmsd,
                n_atoms=fit.n_atoms,
            )

            if renumber and uniprot_id:
                _apply_uniprot_renumbering(
                    chain=mobile[0][best_chain], uniprot_id=uniprot_id, pdb_id=name,
                )

            st.save(mobile, output_path, st.read_links(target_path))
            return True, fit.rmsd, str(output_path)

        except Exception as e:
            log_message(
                message_type="error",
                stage="process_structure",
                pdb_id=name,
                error=str(e),
            )
            return False, -1.0, ""


def process_structure_interface(
    target_path: str | Path,
    output_dir: str | Path,
    reference: Sequence[Residue],
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
    interface between two or more chains:

    1. Load the full target structure.
    2. Identify the primary chain by sequence identity (>= 90 %) to *ref_seq*.
    3. Fit on the primary chain and move the **whole structure**, so every
       auxiliary chain arrives in the reference frame.
    4. For each reference auxiliary chain, find the matching target chain via
       :func:`~qpocket_pipeline.utils.find_auxiliary_chain` (sequence filter,
       then C-alpha centroid proximity to separate homodimer copies).
    5. Skip the structure if *require_all_chains* and any auxiliary chain is
       unmatched.
    6. Save the primary chain and all matched auxiliary chains.
    7. Optionally renumber each chain to canonical UniProt positions.

    Args:
        target_path: Path to the input CIF or PDB file.
        output_dir: Directory to write the output PDB.
        reference: Amino-acid residues of the reference primary chain.
        ref_seq: Single-letter sequence of the reference primary chain.
        ref_aux_seqs: ``{ref_chain_id: sequence}`` for each auxiliary chain.
        ref_aux_centroids: ``{ref_chain_id: centroid_xyz}`` of each reference
            auxiliary chain's C-alpha atoms, in the reference frame.
        uniprot_id: UniProt accession for the primary chain (renumbering).
        auxiliary_uniprot_ids: Per-chain UniProt accessions keyed by reference
            chain ID. Chains absent from it are not renumbered.
        renumber: Whether to apply UniProt-based renumbering.
        min_auxiliary_identity: Minimum identity fraction for an auxiliary match.
        max_auxiliary_centroid_deviation: Centroid deviation (A) above which a
            match is reported as suspicious.
        require_all_chains: Fail the structure if any auxiliary chain is missing.

    Returns:
        ``(success, rmsd, output_file_path, aux_mapping)``, where *aux_mapping*
        maps each reference auxiliary chain to its matched target chain. On
        failure: ``-1.0``, ``""`` and ``{}``.
    """
    target_path = Path(target_path)
    output_dir = Path(output_dir)
    name = target_path.stem
    output_path = output_dir / f"{name}_aligned.pdb"

    with start_action(action_type="process_structure_interface", pdb_id=name):
        try:
            full = st.load(target_path)
            st.remove_water(full)
            chains = [c.id for c in full[0]]

            # -- Step 1: identify primary chain --------------------------------
            best_chain: Optional[str] = None
            best_identity = 0.0
            for chain_id in chains:
                identity = calculate_sequence_identity(
                    ref_seq, st.chain_sequence(full[0][chain_id])
                )
                if identity >= 90.0 and identity > best_identity:
                    best_identity = identity
                    best_chain = chain_id

            if best_chain is None:
                log_message(
                    message_type="skip",
                    stage="process_structure_interface",
                    pdb_id=name,
                    reason="no chain with >=90% sequence identity to primary reference",
                )
                return False, -1.0, "", {}

            # -- Step 2: fit on the primary chain, move everything --------------
            fit = st.superpose(st.polymer_residues(full[0][best_chain]), reference)
            st.apply(full, fit)
            rmsd = fit.rmsd

            # -- Step 3: per-chain sequences and C-alpha centroids --------------
            chain_sequences: Dict[str, str] = {}
            chain_ca_centroids: Dict[str, np.ndarray] = {}
            for chain_id in chains:
                chain = full[0][chain_id]
                chain_sequences[chain_id] = st.chain_sequence(chain)
                ca = st.ca_coords(chain)
                if len(ca):
                    chain_ca_centroids[chain_id] = ca.mean(axis=0)

            # -- Step 4: find each auxiliary chain -----------------------------
            aux_mapping: Dict[str, str] = {}
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
                        return False, -1.0, "", {}
                else:
                    aux_mapping[ref_chain_id] = matched
                    log_message(
                        message_type="auxiliary_chain_matched",
                        pdb_id=name,
                        ref_chain=ref_chain_id,
                        target_chain=matched,
                    )

            # -- Step 5: primary + auxiliary chains -----------------------------
            output = st.keep_chains(full, [best_chain] + list(aux_mapping.values()))

            # -- Step 6: renumber to UniProt positions ------------------------
            if renumber:
                if uniprot_id:
                    _apply_uniprot_renumbering(
                        chain=output[0][best_chain], uniprot_id=uniprot_id, pdb_id=name,
                    )
                for ref_chain_id, target_chain_id in aux_mapping.items():
                    aux_uniprot = (auxiliary_uniprot_ids or {}).get(ref_chain_id)
                    if aux_uniprot:
                        _apply_uniprot_renumbering(
                            chain=output[0][target_chain_id], uniprot_id=aux_uniprot,
                            pdb_id=name,
                        )

            st.save(output, output_path, st.read_links(target_path))

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
            return False, -1.0, "", {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_uniprot_renumbering(chain, uniprot_id: str, pdb_id: str) -> None:
    """Renumbers a chain's residues, in place, to UniProt positions.

    Fetches the canonical sequence for *uniprot_id*, aligns the chain's sequence
    against it, and moves every residue carrying a mapped number to its UniProt
    position.

    The chain sequence has one letter per amino acid. PyMOL built it from
    "every atom named CA", which also caught calcium ions and counted a residue
    twice when it had alternate locations -- both of which shift the sequence
    being aligned.

    Args:
        chain: Biopython chain to renumber.
        uniprot_id: UniProt accession supplying the canonical numbering.
        pdb_id: Used for logging only.
    """
    residues = st.polymer_residues(chain)
    if not residues:
        return

    pdb_seq = "".join(st.one_letter(r) for r in residues)
    pdb_resis = [r.id[1] for r in residues]

    uni_seq = fetch_uniprot_sequence(uniprot_id)
    if not uni_seq:
        return

    relative_map = generate_renumbering_map(pdb_seq, uni_seq, pdb_start_resi=0)
    old_to_new = {
        pdb_resis[idx]: new for idx, new in relative_map.items() if idx < len(pdb_resis)
    }
    clashes = st.renumber(chain, old_to_new)

    log_message(
        message_type="renumbering_applied",
        pdb_id=pdb_id,
        chain=chain.id,
        n_mapped=len(relative_map),
        n_clashes=clashes,
    )
