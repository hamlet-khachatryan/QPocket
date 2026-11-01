"""
Tests for PDB/UniProt query functions in pdb_query.py.
All HTTP calls are mocked via pytest-mock / unittest.mock.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


from qpocket.io.rcsb import (
    fetch_structure_ids,
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


# ---------------------------------------------------------------------------
# download_structures
# ---------------------------------------------------------------------------


