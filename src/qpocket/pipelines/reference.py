"""The reference: aligned and cleaned once, its pocket located once."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser
from eliot import log_message, start_action

from qpocket.config import QPocketConfig, should_renumber
from qpocket.core import pocket
from qpocket.core import structure as st
from qpocket.core.geometry import compute_pocket_bounding_box
from qpocket.core.pocket import select_cavity_by_centroid
from qpocket.core.types import AlignmentContext, RawStructure, Reference, Structure
from qpocket.io import layout
from qpocket.io import rcsb as pdb_query
from qpocket.io import structures as sio
from qpocket.io.tables import write_points
from qpocket.io.tools import run_cavfind


@lru_cache(maxsize=4)
def reference_residues(path: str, chain: str):
    """Amino-acid residues of the aligned reference chain, cached per process.

    Args:
        path: Aligned reference PDB.
        chain: Chain to fit onto.

    Returns:
        Residues in order.
    """
    # Read back from the saved file, so targets are fitted to exactly the coordinates on disk.
    return st.polymer_residues(
        st.keep_chains(sio.load(Path(path)), [chain], protein_only=True)[0][chain]
    )


def alignment_context(cfg: QPocketConfig, ws: layout.Workspace,
                      source: RawStructure) -> Tuple[AlignmentContext, Structure]:
    """Save the aligned reference and everything a target needs to be aligned to it.

    Args:
        cfg: QPocket configuration.
        ws: The run.
        source: The ingested reference.

    Returns:
        (context, the reference's aligned record).

    Raises:
        FileNotFoundError: If the reference file is missing.
        ValueError: If the reference chain is not in it.
    """
    interface_mode = bool(cfg.reference.auxiliary_chain_ids)
    ref_id = source.pdb_id

    reference_path = Path(source.path)
    if not reference_path.exists():
        log_message(message_type="error", stage="align_structures",
                    error="reference structure not found", reference_id=ref_id,
                    searched=[str(reference_path)])
        raise FileNotFoundError(f"Reference structure for {ref_id!r} not found in {ws.original}")

    # ── Load reference; extract primary chain sequence ────────────────────
    ref_structure, _ = st.without_water(sio.load(reference_path))
    ref_model = ref_structure[0]
    if cfg.reference.chain_id not in ref_model:
        raise ValueError(
            f"reference chain {cfg.reference.chain_id!r} not found in {reference_path.name}; "
            f"it has {[c.id for c in ref_model]}"
        )
    ref_chain_seq = st.chain_sequence(ref_model[cfg.reference.chain_id])

    # ── Collect auxiliary chain data from the reference structure ─────────
    ref_aux_seqs: Dict[str, str] = {}
    ref_aux_centroids: Dict[str, np.ndarray] = {}

    if interface_mode:
        for aux_chain in cfg.reference.auxiliary_chain_ids:
            if aux_chain not in ref_model:
                log_message(
                    message_type="warning",
                    stage="align_structures",
                    warning=f"reference auxiliary chain {aux_chain!r} not in structure",
                )
                continue
            ref_aux_seqs[aux_chain] = st.chain_sequence(ref_model[aux_chain])
            ca_xyz = st.ca_coords(ref_model[aux_chain])
            if len(ca_xyz):
                ref_aux_centroids[aux_chain] = ca_xyz.mean(axis=0)
            else:
                log_message(
                    message_type="warning",
                    stage="align_structures",
                    warning=f"no Cα atoms found for reference auxiliary chain {aux_chain!r}",
                )

    # ── Save aligned reference (primary + auxiliary if interface mode) ─────
    keep = [cfg.reference.chain_id]
    if interface_mode:
        keep += list(cfg.reference.auxiliary_chain_ids)

    aligned_reference_path = ws.aligned / f"{ref_id}_aligned.pdb"
    sio.save(st.keep_chains(ref_structure, keep), aligned_reference_path,
             sio.read_links(reference_path))

    log_message(
        message_type="reference_processed",
        pdb_id=ref_id,
        seq_length=len(ref_chain_seq),
        interface_mode=interface_mode,
        aux_chains=list(ref_aux_seqs.keys()) if interface_mode else [],
    )

    # ── Resolve UniProt IDs ───────────────────────────────────────────────
    # Renumbering needs a UniProt accession resolved from a real PDB entry.
    # In fragment-screen mode there is none, and the lookup is two HTTP round
    # trips that return None and then silently no-op.
    renumber = should_renumber(cfg)
    uniprot_id = cfg.alignment.manual_uniprot_id
    if uniprot_id is None and renumber:
        uniprot_id = pdb_query.get_uniprot_accession(
            cfg.reference.pdb_id, cfg.reference.chain_id
        )

    resolved_aux_uniprots: Dict[str, str] = {}
    if interface_mode:
        manual_aux = dict(cfg.alignment.auxiliary_uniprot_ids or {})
        for aux_chain in cfg.reference.auxiliary_chain_ids:
            if aux_chain in manual_aux:
                resolved_aux_uniprots[aux_chain] = manual_aux[aux_chain]
            else:
                uid = (
                    pdb_query.get_uniprot_accession(cfg.reference.pdb_id, aux_chain)
                    if renumber
                    else None
                )
                if uid:
                    resolved_aux_uniprots[aux_chain] = uid


    record = Structure(pdb_id=ref_id, source=str(reference_path), rmsd_to_ref=0.0,
                       aligned_file=str(aligned_reference_path),
                       aux_chain_mapping=(str({c: c for c in cfg.reference.auxiliary_chain_ids})
                                          if interface_mode else None))

    context = AlignmentContext(
        reference_path=str(aligned_reference_path),
        reference_chain=cfg.reference.chain_id,
        reference_sequence=ref_chain_seq,
        uniprot_id=uniprot_id,
        aux_sequences=dict(ref_aux_seqs),
        aux_centroids={k: tuple(float(x) for x in v) for k, v in ref_aux_centroids.items()},
        aux_uniprots=dict(resolved_aux_uniprots),
        renumber=renumber,
        interface_mode=interface_mode,
    )
    return context, record


def ligand_centroid(ref_pdb_path: Path, ligand_resname: str,
                    search_chains: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """Centroid of the reference ligand in a PDB file.

    Args:
        ref_pdb_path: Aligned reference PDB.
        ligand_resname: Ligand residue name.
        search_chains: Chains to search, or None for all.

    Returns:
        (3,) centroid, or None if absent or unreadable.
    """
    try:
        structure = PDBParser(QUIET=True).get_structure("reference", ref_pdb_path)
        centroid = pocket.ligand_centroid(structure, ligand_resname, search_chains)
        if centroid is None:
            searched = search_chains if search_chains is not None else "all chains"
            log_message(
                message_type="warning",
                stage="get_ref_ligand_centroid",
                warning=(
                    f"ligand {ligand_resname!r} not found in {searched} "
                    f"of {ref_pdb_path}"
                ),
            )
        return centroid
    except Exception as e:
        log_message(
            message_type="error",
            stage="get_ref_ligand_centroid",
            error=str(e),
        )
        return None


def reference_pocket_box(pdb_path: Path, pdb_id: str, ref_centroid: np.ndarray,
                         pocket_save_dir: Path, save_intermediate: bool = False,
                         bbox_padding: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find the reference pocket and derive its bounding box.

    Args:
        pdb_path: Cleaned reference PDB.
        pdb_id: Its identifier.
        ref_centroid: Reference ligand centroid.
        pocket_save_dir: Where binding_site.csv goes.
        save_intermediate: Also write binding_site_initial.csv.
        bbox_padding: Box expansion in Å.

    Returns:
        (lower, upper), or None when cavity detection fails.
    """
    with start_action(
        action_type="extract_reference_pocket_bbox",
        pdb_id=pdb_id,
        bbox_padding=bbox_padding,
    ):
        pocket_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            found_cavities = run_cavfind(pdb_path, pdb_id)
        except Exception as e:
            log_message(
                message_type="error",
                stage="extract_reference_pocket_bbox",
                pdb_id=pdb_id,
                error=str(e),
            )
            return None

        if not found_cavities:
            log_message(
                message_type="warning",
                stage="extract_reference_pocket_bbox",
                pdb_id=pdb_id,
                warning="no cavities found in reference structure",
            )
            return None

        coords = select_cavity_by_centroid(found_cavities, ref_centroid)
        if coords is None:
            log_message(
                message_type="warning",
                stage="extract_reference_pocket_bbox",
                pdb_id=pdb_id,
                warning="reference centroid does not overlap any cavity",
            )
            return None

        n_points = len(coords)

        pocket_csv = pocket_save_dir / "binding_site.csv"
        write_points(coords, pocket_csv)

        if save_intermediate:
            write_points(coords, pocket_save_dir / "binding_site_initial.csv")

        bbox_min, bbox_max = compute_pocket_bounding_box(coords, padding=bbox_padding)

        log_message(
            message_type="reference_bbox_computed",
            pdb_id=pdb_id,
            n_points=n_points,
            bbox_min=bbox_min.tolist(),
            bbox_max=bbox_max.tolist(),
        )
        return bbox_min, bbox_max


def _reference_pocket(cfg: QPocketConfig, ws: layout.Workspace,
                      record: Structure) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    centroid: Optional[np.ndarray] = None
    box = None
    if centroid is not None:
        box = reference_pocket_box(Path(record.cleaned_file), record.pdb_id, centroid,
                                   ws.pockets / record.pdb_id, cfg.io.save_intermediate, 1.0)
        if box is not None:
            log_message(message_type="reference_bbox_ready", pdb_id=record.pdb_id)
    return centroid, box


def build_reference(cfg: QPocketConfig, ws: layout.Workspace, source: RawStructure) -> Reference:
    """Align and clean the reference, locate its pocket, and persist it.

    Args:
        cfg: QPocket configuration.
        ws: The run.
        source: The ingested reference.

    Returns:
        The reference.

    Raises:
        ValueError: If the reference cannot be cleaned or its ligand is missing.
    """
    from qpocket.pipelines.structure import clean_file

    with start_action(action_type="build_reference", pdb_id=source.pdb_id):
        context, record = alignment_context(cfg, ws, source)
        cleaned = clean_file(record, ws, cfg.cleaning.allowed_cofactors)
        if cleaned is None:
            log_message(message_type="error", stage="prepare_and_featurize",
                        error="the reference structure could not be cleaned")
            raise ValueError(f"the reference structure {source.pdb_id} could not be cleaned")
        record = record._replace(cleaned_file=cleaned)
        centroid, box = _reference_pocket(cfg, ws, record)
        ref = Reference(source.pdb_id, context, record, centroid, box)
        layout.write_reference(ws, ref.to_json())
        return ref


def load_reference(ws: layout.Workspace) -> Reference:
    """The reference written by the prepare stage.

    Args:
        ws: The run.

    Returns:
        The reference.

    Raises:
        FileNotFoundError: If prepare has not run.
    """
    return Reference.from_json(layout.read_reference(ws))
