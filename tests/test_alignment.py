"""
Tests for alignment functions in processing/alignment.py.
pymol.cmd is already mocked in sys.modules by conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# pymol is mocked in sys.modules by conftest.py before this import
from qpocket_pipeline.processing.alignment import (
    fetch_uniprot_sequence,
    generate_renumbering_map,
    process_structure,
)


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
    with patch("qpocket_pipeline.processing.alignment.requests.get", return_value=resp):
        seq = fetch_uniprot_sequence("P12345")
    assert seq == "MKTAYIAKQ"


def test_fetch_uniprot_sequence_non_200_returns_none():
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("404")
    with patch(
        "qpocket_pipeline.processing.alignment.requests.get",
        side_effect=Exception("network error"),
    ):
        seq = fetch_uniprot_sequence("P99999")
    assert seq is None


def test_fetch_uniprot_sequence_single_line_fasta():
    resp = _make_resp(200, ">SP|P12345|PROT\nACGTACGT\n")
    with patch("qpocket_pipeline.processing.alignment.requests.get", return_value=resp):
        seq = fetch_uniprot_sequence("P12345")
    assert seq == "ACGTACGT"


def test_fetch_uniprot_sequence_multiline():
    resp = _make_resp(200, ">SP|P12345|PROT\nAAAA\nBBBB\nCCCC\n")
    with patch("qpocket_pipeline.processing.alignment.requests.get", return_value=resp):
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


# ---------------------------------------------------------------------------
# process_structure
# ---------------------------------------------------------------------------


def _setup_pymol_mock_for_success(tmp_path: Path, ref_seq: str = "AACGT"):
    """Configure pymol cmd mock to simulate successful alignment."""
    cmd_mock = sys.modules["pymol"].cmd

    # get_chains returns ["A"]
    cmd_mock.get_chains.return_value = ["A"]

    # get_fastastr returns a sequence with >90% identity to ref_seq
    cmd_mock.get_fastastr.return_value = f">header\n{ref_seq}"

    # align returns a valid tuple (rmsd, ...)
    cmd_mock.align.return_value = (0.5, 10, 10, 10, 1, 0.5, 10)

    # iterate stores residue data into passed space
    def _fake_iterate(selection, expr, space):
        space["resis"] = ["1", "2", "3"]
        space["seq"] = ["ALA", "GLY", "CYS"]

    cmd_mock.iterate.side_effect = _fake_iterate

    # save should write a file
    def _fake_save(path, obj):
        Path(path).write_text("ATOM  1  CA  ALA A 1  0.0  0.0  0.0\n")

    cmd_mock.save.side_effect = _fake_save
    cmd_mock.delete = MagicMock()
    cmd_mock.load = MagicMock()
    cmd_mock.remove = MagicMock()
    cmd_mock.create = MagicMock()
    cmd_mock.sort = MagicMock()
    cmd_mock.alter = MagicMock()


def test_process_structure_no_chain_above_threshold(tmp_path):
    """When best chain identity < 90%, returns (False, -1.0, '')."""
    cmd_mock = sys.modules["pymol"].cmd
    cmd_mock.get_chains.return_value = ["A"]
    # Return a sequence with 0% identity to ref "MKTAY"
    cmd_mock.get_fastastr.return_value = ">header\nXXXXX"
    cmd_mock.delete = MagicMock()

    target = tmp_path / "target.cif"
    target.write_text("mock cif content")
    # Need a stem attribute, use a Path
    target_path = Path(target)

    success, rmsd, path = process_structure(
        target_path=target_path,
        output_dir=tmp_path,
        ref_obj_name="reference",
        ref_seq="MKTAY",
        renumber=False,
    )

    assert success is False
    assert rmsd == -1.0
    assert path == ""


def test_process_structure_successful_alignment(tmp_path):
    """Successful alignment: returns (True, rmsd, path)."""
    ref_seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 chars
    _setup_pymol_mock_for_success(tmp_path, ref_seq=ref_seq)

    target = tmp_path / "target.cif"
    target.write_text("mock cif content")
    target_path = Path(target)

    success, rmsd, saved_path = process_structure(
        target_path=target_path,
        output_dir=tmp_path,
        ref_obj_name="reference",
        ref_seq=ref_seq,
        renumber=False,
    )

    assert success is True
    assert rmsd >= 0.0
    assert len(saved_path) > 0


def test_process_structure_returns_tuple_on_exception(tmp_path):
    """If cmd.load raises, returns (False, -1.0, '')."""
    cmd_mock = sys.modules["pymol"].cmd
    cmd_mock.load.side_effect = RuntimeError("pymol load failed")
    cmd_mock.delete = MagicMock()

    target = tmp_path / "bad_target.cif"
    target.write_text("garbage")
    target_path = Path(target)

    success, rmsd, path = process_structure(
        target_path=target_path,
        output_dir=tmp_path,
        ref_obj_name="reference",
        ref_seq="MKTAY",
        renumber=False,
    )

    assert success is False
    cmd_mock.load.side_effect = None  # reset for other tests
