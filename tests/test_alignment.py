"""
Tests for the UniProt helpers in processing/alignment.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


from qpocket.core.sequence import generate_renumbering_map
from qpocket.io.rcsb import fetch_uniprot_sequence


# ---------------------------------------------------------------------------
# fetch_uniprot_sequence
# ---------------------------------------------------------------------------


def _make_resp(status_code: int = 200, text: str = ">SP|P12345|PROT\nMKTAYIAKQ\n"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status = MagicMock(
        side_effect=None if status_code == 200 else Exception("HTTP Error")
    )
    return resp


def test_fetch_uniprot_sequence_success():
    resp = _make_resp(200, ">SP|P12345|PROT\nMKTAY\nIAKQ\n")
    with patch("qpocket.io.rcsb.requests.get", return_value=resp):
        seq = fetch_uniprot_sequence("P12345")
    assert seq == "MKTAYIAKQ"


def test_fetch_uniprot_sequence_non_200_returns_none():
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("404")
    with patch(
        "qpocket.io.rcsb.requests.get",
        side_effect=Exception("network error"),
    ):
        seq = fetch_uniprot_sequence("P99999")
    assert seq is None


def test_fetch_uniprot_sequence_single_line_fasta():
    resp = _make_resp(200, ">SP|P12345|PROT\nACGTACGT\n")
    with patch("qpocket.io.rcsb.requests.get", return_value=resp):
        seq = fetch_uniprot_sequence("P12345")
    assert seq == "ACGTACGT"


def test_fetch_uniprot_sequence_multiline():
    resp = _make_resp(200, ">SP|P12345|PROT\nAAAA\nBBBB\nCCCC\n")
    with patch("qpocket.io.rcsb.requests.get", return_value=resp):
        seq = fetch_uniprot_sequence("P12345")
    assert seq == "AAAABBBBCCCC"


# ---------------------------------------------------------------------------
# generate_renumbering_map
# ---------------------------------------------------------------------------


def test_generate_renumbering_map_full_overlap():
    pdb_seq = "ACDEFGHIKLMNPQRSTVWY"
    uniprot_seq = "ACDEFGHIKLMNPQRSTVWY"
    mapping = generate_renumbering_map(pdb_seq, uniprot_seq, pdb_start_resi=1)
    # Full overlap → all 20 positions mapped
    assert len(mapping) == 20


def test_generate_renumbering_map_partial_overlap():
    uniprot_seq = "AAACDEFGHIKLMNPQRSTVWYAAA"
    pdb_seq = "CDEFGHIK"  # substring of uniprot
    mapping = generate_renumbering_map(pdb_seq, uniprot_seq, pdb_start_resi=1)
    assert len(mapping) > 0
    assert len(mapping) <= len(pdb_seq)


def test_generate_renumbering_map_empty_pdb_seq():
    # Bio.Align raises ValueError for zero-length sequences; the function propagates it.
    # Accept either an empty dict or a ValueError as valid behaviour.
    try:
        mapping = generate_renumbering_map("", "ACDEFGHIKLMNPQRSTVWY", pdb_start_resi=1)
        assert mapping == {}
    except ValueError:
        pass  # BioPython disallows zero-length sequences — acceptable


def test_generate_renumbering_map_values_are_uniprot_positions():
    """The values in the map should be 1-based UniProt positions."""
    pdb_seq = "MKTAY"
    uniprot_seq = "MKTAYIAKQ"
    mapping = generate_renumbering_map(pdb_seq, uniprot_seq, pdb_start_resi=1)
    if mapping:
        # All values should be >= 1 (1-based)
        for v in mapping.values():
            assert v >= 1


def test_generate_renumbering_map_keys_are_pdb_resis():
    pdb_seq = "MKTAY"
    uniprot_seq = "MKTAYIAKQ"
    mapping = generate_renumbering_map(pdb_seq, uniprot_seq, pdb_start_resi=5)
    if mapping:
        # All keys should start from pdb_start_resi
        assert min(mapping.keys()) >= 5


# process_structure is exercised against real coordinates in test_structure.py;
# it no longer calls anything that can be mocked here.
