"""Sequence identity, UniProt renumbering maps, chain matching and RCSB queries (pure)."""
from __future__ import annotations

from io import StringIO
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from Bio import Align, SeqIO


def calculate_sequence_identity(seq1: str, seq2: str) -> float:
    """Percent identity of two sequences from a global alignment.

    Args:
        seq1: First single-letter sequence.
        seq2: Second single-letter sequence.

    Returns:
        Identity in [0, 100].
    """
    if not seq1 or not seq2:
        return 0.0

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0

    try:
        best = next(iter(aligner.align(seq1, seq2)))
    except StopIteration:
        return 0.0

    # With zero mismatch/gap scores, best.score equals the number of identical positions.
    n_matches = int(best.score)
    # Global alignment length equals the longer sequence (no gap penalty → no insertions).
    alignment_length = max(len(seq1), len(seq2))

    if alignment_length == 0:
        return 0.0
    return (n_matches / alignment_length) * 100.0


def generate_renumbering_map(
    pdb_seq: str,
    uniprot_seq: str,
    pdb_start_resi: int = 0,
) -> Dict[int, int]:
    """Map sequence positions of a chain onto 1-based UniProt numbers.

    Args:
        pdb_seq: Chain sequence.
        uniprot_seq: Canonical UniProt sequence.
        pdb_start_resi: Offset added to every key.

    Returns:
        {chain position: UniProt number}; empty if nothing aligns.
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


class AuxMatch(NamedTuple):
    """How an auxiliary chain was matched.

    Attributes:
        chain: Matched chain ID, or None.
        n_candidates: Chains passing the identity filter.
        distance: Centroid distance to the reference auxiliary chain (A), if used.
        fallback: True when centroids were missing and identity decided.
    """

    chain: Optional[str]
    n_candidates: int
    distance: Optional[float]
    fallback: bool


def match_auxiliary_chain(
    chain_sequences: Dict[str, str],
    chain_ca_centroids: Dict[str, np.ndarray],
    ref_aux_seq: str,
    ref_aux_centroid: np.ndarray,
    primary_chain: str,
    min_identity: float = 0.70,
) -> AuxMatch:
    """Find the chain matching a reference auxiliary chain: sequence, then position.

    Args:
        chain_sequences: Chain ID to sequence, for the aligned target.
        chain_ca_centroids: Chain ID to CA centroid, in the reference frame.
        ref_aux_seq: Reference auxiliary chain sequence.
        ref_aux_centroid: Reference auxiliary CA centroid.
        primary_chain: Already-matched primary chain, never a candidate.
        min_identity: Minimum identity fraction.

    Returns:
        The match and how it was made.
    """
    candidates: List[Tuple[str, float]] = []
    for chain_id, seq in chain_sequences.items():
        if chain_id == primary_chain:
            continue
        if not seq:
            continue
        # calculate_sequence_identity returns a percentage [0, 100]
        identity_frac = calculate_sequence_identity(ref_aux_seq, seq) / 100.0
        if identity_frac >= min_identity:
            candidates.append((chain_id, identity_frac))

    if not candidates:
        return AuxMatch(None, 0, None, False)
    if len(candidates) == 1:
        return AuxMatch(candidates[0][0], 1, None, False)

    # Several candidates (homomultimer): nearest CA centroid wins.
    best_chain: Optional[str] = None
    best_dist = float("inf")
    for chain_id, _ in candidates:
        centroid = chain_ca_centroids.get(chain_id)
        if centroid is None or len(centroid) != 3:
            continue
        dist = float(np.linalg.norm(centroid - ref_aux_centroid))
        if dist < best_dist:
            best_dist = dist
            best_chain = chain_id

    if best_chain is None:
        return AuxMatch(max(candidates, key=lambda x: x[1])[0], len(candidates), None, True)
    return AuxMatch(best_chain, len(candidates), best_dist, False)


def find_auxiliary_chain(
    chain_sequences: Dict[str, str],
    chain_ca_centroids: Dict[str, np.ndarray],
    ref_aux_seq: str,
    ref_aux_centroid: np.ndarray,
    primary_chain: str,
    min_identity: float = 0.70,
    max_centroid_deviation: float = 15.0,
) -> Optional[str]:
    """Chain ID matching a reference auxiliary chain, or None.

    Args:
        chain_sequences: Chain ID to sequence, for the aligned target.
        chain_ca_centroids: Chain ID to CA centroid, in the reference frame.
        ref_aux_seq: Reference auxiliary chain sequence.
        ref_aux_centroid: Reference auxiliary CA centroid.
        primary_chain: Already-matched primary chain, never a candidate.
        min_identity: Minimum identity fraction.
        max_centroid_deviation: Kept for callers; see match_auxiliary_chain.

    Returns:
        Matched chain ID, or None.
    """
    return match_auxiliary_chain(chain_sequences, chain_ca_centroids, ref_aux_seq,
                                 ref_aux_centroid, primary_chain, min_identity).chain


def chain_sequence_from_fasta(fasta_text: str, pdb_id: str, chain_id: str) -> Optional[str]:
    """Sequence of one chain from an RCSB entry FASTA.

    Args:
        fasta_text: FASTA as served by rcsb.org/fasta/entry.
        pdb_id: Entry ID.
        chain_id: Author chain ID.

    Returns:
        The sequence, or None if the chain is not listed.
    """
    for record in SeqIO.parse(StringIO(fasta_text), "fasta"):
        # Header format: >5R7Y_1|Chain A|...
        parts = record.description.split('|')
        if len(parts) >= 2:
            if pdb_id.upper() in parts[0].upper():
                # Clean "Chain A" or "Chains A, B"
                chains_clean = parts[1].replace("Chains", "").replace("Chain", "").strip()
                valid_chains = [c.strip() for c in chains_clean.split(',')]

                if chain_id in valid_chains:
                    return str(record.seq)
    return None


def rcsb_search_query(
    ref_sequence: str,
    sequence_identity: float = 1.0,
    resolution_limit: Optional[float] = 2.5,
    min_length: Optional[int] = 30,
    max_length: Optional[int] = None,
    experimental_method: Optional[str] = None,
    uniprot_id: Optional[str] = None,
) -> dict:
    """RCSB search request for entries similar to a sequence.

    Args:
        ref_sequence: Reference chain sequence.
        sequence_identity: Identity cutoff in (0, 1].
        resolution_limit: Maximum resolution (A), or None.
        min_length: Minimum chain length, or None.
        max_length: Maximum chain length, or None.
        experimental_method: Required method, or None.
        uniprot_id: Restrict to this UniProt accession, or None.

    Returns:
        Search API request body.
    """
    query_nodes = [
        {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 0.1,
                "identity_cutoff": sequence_identity,
                "target": "pdb_protein_sequence",
                "value": ref_sequence
            }
        }
    ]

    if resolution_limit:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": resolution_limit
            }
        })

    if min_length is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "greater_or_equal",
                "value": min_length
            }
        })

    if max_length is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "less_or_equal",
                "value": max_length
            }
        })

    if experimental_method:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": experimental_method.upper()
            }
        })

    if uniprot_id:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        })

    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": query_nodes},
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }
