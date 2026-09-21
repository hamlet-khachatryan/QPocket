"""
Tests for PDB/UniProt query functions in pdb_query.py.
All HTTP calls are mocked via pytest-mock / unittest.mock.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from qpocket.io.rcsb import (
    download_structures,
    fetch_structure_ids,
    get_uniprot_accession,
)


# ---------------------------------------------------------------------------
# get_uniprot_accession
# ---------------------------------------------------------------------------


def _make_rcsb_search_response(entity_identifier: str = "1ABC_1"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "result_set": [{"identifier": entity_identifier}]
    }
    return resp


def _make_entity_response(chain_id: str = "A", accession: str = "P12345"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "rcsb_polymer_entity_container_identifiers": {
            "auth_asym_ids": [chain_id],
            "reference_sequence_identifiers": [
                {"database_name": "UniProt", "database_accession": accession}
            ],
        }
    }
    return resp


def test_get_uniprot_accession_success():
    search_resp = _make_rcsb_search_response("1ABC_1")
    entity_resp = _make_entity_response("A", "P12345")

    with patch("qpocket.io.rcsb.requests.post", return_value=search_resp):
        with patch("qpocket.io.rcsb.requests.get", return_value=entity_resp):
            result = get_uniprot_accession("1ABC", "A")

    assert result == "P12345"


def test_get_uniprot_accession_http_error():
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

    with patch("qpocket.io.rcsb.requests.post", return_value=resp):
        result = get_uniprot_accession("ZZZZ", "A")

    assert result is None


def test_get_uniprot_accession_network_timeout():
    with patch(
        "qpocket.io.rcsb.requests.post",
        side_effect=requests.exceptions.Timeout("timed out"),
    ):
        result = get_uniprot_accession("1ABC", "A")

    assert result is None


def test_get_uniprot_accession_chain_mismatch_returns_none():
    """Entity belongs to chain B but we asked for chain A → should return None."""
    search_resp = _make_rcsb_search_response("1ABC_1")
    entity_resp = _make_entity_response(chain_id="B", accession="P99999")

    with patch("qpocket.io.rcsb.requests.post", return_value=search_resp):
        with patch("qpocket.io.rcsb.requests.get", return_value=entity_resp):
            result = get_uniprot_accession("1ABC", "A")

    assert result is None


def test_get_uniprot_accession_empty_result_set():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"result_set": []}

    with patch("qpocket.io.rcsb.requests.post", return_value=resp):
        result = get_uniprot_accession("1ABC", "A")

    assert result is None


# ---------------------------------------------------------------------------
# fetch_structure_ids
# ---------------------------------------------------------------------------

MOCK_FASTA = """\
>1ABC_1|Chain A|Protein XYZ
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
"""

MOCK_SEARCH_RESPONSE = {
    "result_set": [
        {"identifier": "1ABC"},
        {"identifier": "2DEF"},
        {"identifier": "3GHI"},
    ]
}


def _mock_fasta_get(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = MOCK_FASTA
    return resp


def _mock_search_post(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = MOCK_SEARCH_RESPONSE
    return resp


def test_fetch_structure_ids_success():
    with patch("qpocket.io.rcsb.requests.get", side_effect=_mock_fasta_get):
        with patch("qpocket.io.rcsb.requests.post", side_effect=_mock_search_post):
            ids = fetch_structure_ids("1ABC", "A", sequence_identity=0.9)

    assert "1ABC" in ids
    assert len(ids) >= 3


def test_fetch_structure_ids_includes_ref_pdb():
    """The reference PDB ID must always be in the returned list."""
    with patch("qpocket.io.rcsb.requests.get", side_effect=_mock_fasta_get):
        with patch("qpocket.io.rcsb.requests.post", side_effect=_mock_search_post):
            ids = fetch_structure_ids("1ABC", "A")

    assert "1ABC" in ids


def test_fetch_structure_ids_empty_search_result():
    def _empty_post(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"result_set": []}
        return resp

    with patch("qpocket.io.rcsb.requests.get", side_effect=_mock_fasta_get):
        with patch("qpocket.io.rcsb.requests.post", side_effect=_empty_post):
            ids = fetch_structure_ids("1ABC", "A")

    # Should at least contain the reference ID itself
    assert "1ABC" in ids


def test_fetch_structure_ids_fasta_fetch_fails():
    def _fail_get(url, **kwargs):
        raise requests.exceptions.ConnectionError("network error")

    with patch("qpocket.io.rcsb.requests.get", side_effect=_fail_get):
        ids = fetch_structure_ids("1ABC", "A")

    assert ids == []


def test_fetch_structure_ids_chain_not_found_returns_empty():
    """FASTA doesn't contain the requested chain → empty list."""
    fasta_no_match = ">1ABC_1|Chain B|Protein XYZ\nMKTAY\n"

    def _get_no_chain(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.text = fasta_no_match
        return resp

    with patch("qpocket.io.rcsb.requests.get", side_effect=_get_no_chain):
        ids = fetch_structure_ids("1ABC", "A")

    assert ids == []


# ---------------------------------------------------------------------------
# download_structures
# ---------------------------------------------------------------------------


def test_download_structures_success(tmp_path):
    download_dir = str(tmp_path)
    cif_content = b"data_1ABC\n_entry.id 1ABC\n"

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_content = MagicMock(return_value=[cif_content])

    with patch("qpocket.io.rcsb.requests.get", return_value=response_mock):
        download_structures(["1ABC"], download_dir)

    assert (tmp_path / "1ABC.cif").exists()
    assert (tmp_path / "1ABC.cif").read_bytes() == cif_content


def test_download_structures_404_skips(tmp_path):
    download_dir = str(tmp_path)

    response_mock = MagicMock()
    response_mock.status_code = 404

    with patch("qpocket.io.rcsb.requests.get", return_value=response_mock):
        download_structures(["XXXX"], download_dir)

    assert not (tmp_path / "XXXX.cif").exists()


def test_download_structures_existing_file_not_redownloaded(tmp_path):
    existing = tmp_path / "1ABC.cif"
    existing.write_bytes(b"existing content")

    with patch("qpocket.io.rcsb.requests.get") as mock_get:
        download_structures(["1ABC"], str(tmp_path))
        mock_get.assert_not_called()


def test_download_structures_network_error_continues(tmp_path):
    """Network error on one ID should not crash; subsequent IDs still attempted."""
    call_count = 0
    cif_content = b"data_2DEF\n"

    def _side_effect(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if "1ABC" in url:
            raise requests.exceptions.ConnectionError("fail")
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_content = MagicMock(return_value=[cif_content])
        return resp

    with patch("qpocket.io.rcsb.requests.get", side_effect=_side_effect):
        download_structures(["1ABC", "2DEF"], str(tmp_path))

    assert (tmp_path / "2DEF.cif").exists()
    assert not (tmp_path / "1ABC.cif").exists()
