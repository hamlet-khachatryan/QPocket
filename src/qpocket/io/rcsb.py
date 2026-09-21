"""Network access: RCSB search and downloads, UniProt accessions and sequences."""
import os
from typing import List, Optional

import requests
from eliot import log_message, start_action

from qpocket.constants import FASTA_UNIPROT_TEMPLATE
from qpocket.core.sequence import chain_sequence_from_fasta, rcsb_search_query

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

def get_uniprot_accession(pdb_id: str, chain_id: str) -> Optional[str]:
    """Retrieves the UniProt accession ID for a specific PDB chain."""

    with start_action(action_type="fetch_uniprot", pdb_id=pdb_id, chain_id=chain_id):
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.entry_id",
                    "operator": "exact_match",
                    "value": pdb_id.upper()
                }
            },
            "return_type": "polymer_entity",
            "request_options": {"return_all_hits": True}
        }

        try:
            resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for result in data.get("result_set", []):
                entity_id = result["identifier"].split("_")[1]

                entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
                ent_resp = requests.get(entity_url, timeout=10)

                if ent_resp.status_code == 200:
                    ent_data = ent_resp.json()
                    identifiers = ent_data.get("rcsb_polymer_entity_container_identifiers", {})
                    auth_chains = identifiers.get("auth_asym_ids", [])

                    if chain_id in auth_chains:
                        refs = identifiers.get("reference_sequence_identifiers", [])
                        for r in refs:
                            if r["database_name"] == "UniProt":
                                return r["database_accession"]
            return None
        except Exception as e:
            log_message(message_type="uniprot_fetch_error", error=str(e))
            return None

def fetch_structure_ids(
        ref_pdb_id: str,
        ref_chain_id: str,
        sequence_identity: float = 1.0,
        resolution_limit: Optional[float] = 2.5,
        min_length: Optional[int] = 30,
        max_length: Optional[int] = None,
        experimental_method: Optional[str] = None,
        enforce_same_uniprot: bool = False
) -> List[str]:
    """
    Fetches PDB IDs based on sequence similarity and metadata.
    """
    full_ref = f"{ref_pdb_id}_{ref_chain_id}"

    with start_action(action_type="fetch_structures", ref=full_ref):
        fasta_url = f"{RCSB_FASTA_URL}{ref_pdb_id}"

        try:
            r = requests.get(fasta_url, timeout=30)
            r.raise_for_status()
            ref_sequence = chain_sequence_from_fasta(r.text, ref_pdb_id, ref_chain_id)

            if not ref_sequence:
                log_message(message_type="error", msg=f"Sequence not found for {full_ref}")
                return []

        except Exception as e:
            log_message(message_type="error", msg=f"FASTA fetch failed: {e}")
            return []

        uid = get_uniprot_accession(ref_pdb_id, ref_chain_id) if enforce_same_uniprot else None
        query = rcsb_search_query(
            ref_sequence,
            sequence_identity=sequence_identity,
            resolution_limit=resolution_limit,
            min_length=min_length,
            max_length=max_length,
            experimental_method=experimental_method,
            uniprot_id=uid,
        )

        try:
            resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=30)
            resp.raise_for_status()
            pdb_ids = [item["identifier"] for item in resp.json().get("result_set", [])]
            if ref_pdb_id.upper() not in pdb_ids:
                pdb_ids.append(ref_pdb_id.upper())
            return pdb_ids
        except Exception as e:
            log_message(message_type="search_error", error=str(e))
            return []

def download_structures(pdb_ids: List[str], download_dir: str):
    """Downloads .cif files for the given PDB IDs."""
    with start_action(action_type="download_batch", count=len(pdb_ids)):
        for pdb_id in pdb_ids:
            file_path = os.path.join(download_dir, f"{pdb_id}.cif")
            if os.path.exists(file_path):
                continue

            url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
            try:
                r = requests.get(url, stream=True, timeout=30)
                if r.status_code == 200:
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    log_message(message_type="download_fail", pdb_id=pdb_id, status=r.status_code)
            except Exception as e:
                log_message(message_type="download_error", pdb_id=pdb_id, error=str(e))


def fetch_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """Canonical sequence of a UniProt accession.

    Args:
        uniprot_id: UniProt accession (e.g. ``"P0DTC2"``).

    Returns:
        Single-letter sequence, or None on failure.
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
