"""Structure alignment against a reference, and UniProt-based renumbering."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB.Residue import Residue
from eliot import log_message, start_action

from qpocket.core import structure as st
from qpocket.core.sequence import (
    calculate_sequence_identity,
    generate_renumbering_map,
    match_auxiliary_chain,
)
from qpocket.io import rcsb
from qpocket.io import structures as sio

def process_structure(
    target_path: str | Path,
    output_dir: str | Path,
    reference: Sequence[Residue],
    ref_seq: str,
    uniprot_id: Optional[str] = None,
    renumber: bool = True,
) -> Tuple[bool, float, str]:
    """Aligns a target structure to the reference and optionally renumbers residues.

    Args:
        target_path: Path to the input CIF or PDB file.
        output_dir: Directory to write the output PDB.
        reference: Amino-acid residues of the reference chain, already in the reference frame.
        ref_seq: Single-letter sequence of the reference chain.
        uniprot_id: UniProt accession used for renumbering (optional).
        renumber: Whether to apply UniProt-based renumbering.

    Returns:
        ``(success, rmsd, output_file_path)`` -- on failure rmsd is ``-1.0`` and path is an empty string.
    """
    target_path = Path(target_path)
    output_dir = Path(output_dir)
    name = target_path.stem
    output_path = output_dir / f"{name}_aligned.pdb"

    with start_action(action_type="process_structure", pdb_id=name):
        try:
            full, _ = st.without_water(sio.load(target_path))

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
            mobile = st.moved(mobile, fit)

            log_message(
                message_type="alignment_stats",
                pdb_id=name,
                chain=best_chain,
                identity=best_identity,
                rmsd=fit.rmsd,
                n_atoms=fit.n_atoms,
            )

            if renumber and uniprot_id:
                mobile = _apply_uniprot_renumbering(mobile, best_chain, uniprot_id, name)

            sio.save(mobile, output_path, sio.read_links(target_path))
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

    Args:
        target_path: Path to the input CIF or PDB file.
        output_dir: Directory to write the output PDB.
        reference: Amino-acid residues of the reference primary chain.
        ref_seq: Single-letter sequence of the reference primary chain.
        ref_aux_seqs: ``{ref_chain_id: sequence}`` for each auxiliary chain.
        ref_aux_centroids: ``{ref_chain_id: centroid_xyz}`` of each reference auxiliary chain's C-alpha atoms, in the reference frame.
        uniprot_id: UniProt accession for the primary chain (renumbering).
        auxiliary_uniprot_ids: Per-chain UniProt accessions keyed by reference chain ID.
        renumber: Whether to apply UniProt-based renumbering.
        min_auxiliary_identity: Minimum identity fraction for an auxiliary match.
        max_auxiliary_centroid_deviation: Centroid deviation (A) above which a match is reported as suspicious.
        require_all_chains: Fail the structure if any auxiliary chain is missing.

    Returns:
        ``(success, rmsd, output_file_path, aux_mapping)``, where *aux_mapping* maps each reference auxiliary chain to its matched target chain.
    """
    target_path = Path(target_path)
    output_dir = Path(output_dir)
    name = target_path.stem
    output_path = output_dir / f"{name}_aligned.pdb"

    with start_action(action_type="process_structure_interface", pdb_id=name):
        try:
            full, _ = st.without_water(sio.load(target_path))
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
            full = st.moved(full, fit)
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

                match = match_auxiliary_chain(
                    chain_sequences=chain_sequences,
                    chain_ca_centroids=chain_ca_centroids,
                    ref_aux_seq=ref_aux_seq,
                    ref_aux_centroid=ref_aux_centroid,
                    primary_chain=best_chain,
                    min_identity=min_auxiliary_identity,
                )
                _log_aux_match(match, max_auxiliary_centroid_deviation)
                matched = match.chain

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
                    output = _apply_uniprot_renumbering(output, best_chain, uniprot_id, name)
                for ref_chain_id, target_chain_id in aux_mapping.items():
                    aux_uniprot = (auxiliary_uniprot_ids or {}).get(ref_chain_id)
                    if aux_uniprot:
                        output = _apply_uniprot_renumbering(
                            output, target_chain_id, aux_uniprot, name
                        )

            sio.save(output, output_path, sio.read_links(target_path))

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

def _apply_uniprot_renumbering(structure, chain_id: str, uniprot_id: str, pdb_id: str):
    """Copy of a structure with one chain renumbered to UniProt positions.

    Args:
        structure: Structure holding the chain (not modified).
        chain_id: Chain to renumber.
        uniprot_id: UniProt accession supplying the numbering.
        pdb_id: Used for logging only.

    Returns:
        The renumbered structure, or the input when nothing could be mapped.
    """
    chain = structure[0][chain_id]
    residues = st.polymer_residues(chain)
    if not residues:
        return structure

    pdb_seq = "".join(st.one_letter(r) for r in residues)
    pdb_resis = [r.id[1] for r in residues]

    uni_seq = rcsb.fetch_uniprot_sequence(uniprot_id)
    if not uni_seq:
        return structure

    relative_map = generate_renumbering_map(pdb_seq, uni_seq, pdb_start_resi=0)
    if not relative_map:
        log_message(
            message_type="warning",
            stage="generate_renumbering_map",
            warning="no alignment produced",
        )
    old_to_new = {
        pdb_resis[idx]: new for idx, new in relative_map.items() if idx < len(pdb_resis)
    }
    new_chain, clashes = st.renumbered(chain, old_to_new)

    log_message(
        message_type="renumbering_applied",
        pdb_id=pdb_id,
        chain=chain.id,
        n_mapped=len(relative_map),
        n_clashes=clashes,
    )
    return st.with_chain(structure, new_chain)


def _log_aux_match(match, max_centroid_deviation: float) -> None:
    """Emit the auxiliary-chain warnings find_auxiliary_chain used to log."""
    if match.chain is None or match.n_candidates < 2:
        return
    if match.fallback:
        log_message(
            message_type="warning",
            stage="find_auxiliary_chain",
            warning="Cα centroids unavailable for spatial disambiguation; "
                    "falling back to highest sequence identity",
            chosen_chain=match.chain,
        )
        return
    if match.distance > max_centroid_deviation:
        log_message(
            message_type="warning",
            stage="find_auxiliary_chain",
            warning=(
                f"best auxiliary chain {match.chain!r} centroid is {match.distance:.1f} Å "
                f"from reference auxiliary (threshold {max_centroid_deviation:.1f} Å); "
                "possible crystal contact — verify manually"
            ),
            chosen_chain=match.chain,
            centroid_distance=match.distance,
        )
    log_message(
        message_type="auxiliary_chain_resolved",
        chosen_chain=match.chain,
        n_candidates=match.n_candidates,
        centroid_distance=match.distance,
    )
