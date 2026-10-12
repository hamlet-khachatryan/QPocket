"""Structures similar to a reference entry, searched on and downloaded from RCSB."""
from dataclasses import dataclass
from typing import Optional

from eliot import log_message

from qpocket.core.ports import SOURCES
from qpocket.core.types import RawStructure, Sources
from qpocket.io import layout, rcsb


@dataclass(frozen=True)
class PdbQueryParams:
    """What to search for.

    Attributes:
        pdb_id: Reference entry.
        chain_id: Reference chain.
        sequence_identity: Minimum identity to the reference chain.
        min_length: Shortest chain accepted.
        max_length: Longest chain accepted.
        resolution_limit: Worst resolution accepted.
        experimental_method: Method filter.
        enforce_same_uniprot: Keep only entries on the reference's UniProt accession.
    """

    pdb_id: str
    chain_id: str
    sequence_identity: float
    min_length: Optional[int]
    max_length: Optional[int]
    resolution_limit: Optional[float]
    experimental_method: Optional[str]
    enforce_same_uniprot: bool


@SOURCES.register("pdb_query", params=PdbQueryParams)
def pdb_query(params: PdbQueryParams, ws: layout.Workspace) -> Sources:
    """Search RCSB, download the hits and the reference.

    Args:
        params: The search.
        ws: The run.

    Returns:
        The ingested structures.

    Raises:
        ValueError: If the search finds nothing.
    """
    pdb_ids = rcsb.fetch_structure_ids(
        ref_pdb_id=params.pdb_id, ref_chain_id=params.chain_id,
        sequence_identity=params.sequence_identity, min_length=params.min_length,
        max_length=params.max_length, resolution_limit=params.resolution_limit,
        experimental_method=params.experimental_method,
        enforce_same_uniprot=params.enforce_same_uniprot,
    )
    if not pdb_ids:
        log_message(message_type="pipeline_abort", reason="no structures found from PDB query")
        raise ValueError(f"the PDB query around {params.pdb_id} found no structures")
    log_message(message_type="pdb_query_complete", n_structures=len(pdb_ids))
    if params.pdb_id not in pdb_ids:
        pdb_ids.append(params.pdb_id)
    rcsb.download_structures(pdb_ids, ws.original)
    structures = tuple(RawStructure(p.stem, p) for p in layout.structure_files(ws))
    return Sources(RawStructure(params.pdb_id, ws.original / f"{params.pdb_id}.cif"), structures)
