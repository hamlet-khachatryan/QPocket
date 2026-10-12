"""Conformational pipeline stages and config loading; each stage takes (cfg, dirs)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from eliot import log_message, start_action

from qpocket.config import (STRUCTURE_SUFFIXES, QPocketConfig, run_id, sanitise_id,
                            should_renumber)
from qpocket.io import hdf5, layout
from qpocket.io import logging as qlog
from qpocket.pipelines import collect, parallel
from qpocket.pipelines.distance import compute_geodesic
from qpocket.pipelines.distance import compute_spectral_distance_matrix  # noqa: F401 (a stage)
from qpocket.io import rcsb as pdb_query
from qpocket.io import structures as sio
from qpocket.pipelines import align as alignment
from qpocket.pipelines import clean as cleaning
from qpocket.pipelines import pockets as extraction
from qpocket.core import pocket
from qpocket.core import structure as st

def inputs_table(dirs: Dict[str, Path]) -> Optional[pd.DataFrame]:
    """The ``pdb_id -> source file`` record written when structures are ingested."""
    path = dirs["metadata"] / "inputs.csv"
    return pd.read_csv(path) if path.exists() else None


def reference_id(cfg: QPocketConfig, dirs: Optional[Dict[str, Path]] = None) -> str:
    """The ``pdb_id`` of the reference structure, whatever the mode.

    Args:
        cfg: QPocket configuration.
        dirs: Directory map.

    Returns:
        The identifier the reference structure was ingested under.
    """
    if cfg.mode != "fragment_screen":
        return cfg.reference.pdb_id

    if dirs is not None:
        table = inputs_table(dirs)
        if table is not None and "is_reference" in table.columns:
            flagged = table[table["is_reference"].astype(bool)]
            if not flagged.empty:
                return str(flagged.iloc[0]["pdb_id"])

    if cfg.fragment_screen.reference_path:
        return sanitise_id(Path(cfg.fragment_screen.reference_path).stem)

    # No explicit reference file: fall back to the configured index over the
    # ingest record, then to the first structure.
    if dirs is not None:
        table = inputs_table(dirs)
        if table is not None and len(table):
            idx = min(int(cfg.fragment_screen.reference_index), len(table) - 1)
            return str(table.iloc[idx]["pdb_id"])

    return cfg.reference.pdb_id


def execute(cfg: QPocketConfig, pipeline_stages: Sequence[Callable]) -> Path:
    """Set up the run directory and logging, then run stages in order.

    Args:
        cfg: Validated config.
        pipeline_stages: Callables taking (cfg, dirs).

    Returns:
        Run root directory.
    """
    dirs = layout.create(cfg.paths.workdir, run_id(cfg))
    log_path = dirs["root"] / "qpocket_pipeline.log"
    qlog.configure(cfg, log_path)

    with start_action(
        action_type="pipeline_execution",
        reference_pdb=run_id(cfg),
        mode=cfg.mode,
        workdir=str(dirs["root"]),
    ):
        log_message(
            message_type="workspace_ready",
            root_path=str(dirs["root"]),
            log_path=str(log_path),
        )
        for stage in pipeline_stages:
            with start_action(action_type=f"stage_{stage.__name__}"):
                stage(cfg, dirs)
    return dirs["root"]


def download_structures(cfg: QPocketConfig, dirs: Dict[str, Path]) -> List[str]:
    """Ingest stage for PDB query: search RCSB and download the hits.

    Args:
        cfg: Config in pdb_query mode.
        dirs: Run directories.

    Returns:
        Downloaded PDB IDs; empty when the search finds nothing.
    """
    output_dir = dirs["original"]
    pdb_ids = pdb_query.fetch_structure_ids(
        ref_pdb_id=cfg.reference.pdb_id,
        ref_chain_id=cfg.reference.chain_id,
        sequence_identity=cfg.pdb_query.sequence_identity,
        min_length=cfg.pdb_query.min_length,
        max_length=cfg.pdb_query.max_length,
        resolution_limit=cfg.pdb_query.resolution_limit,
        experimental_method=cfg.pdb_query.experimental_method,
        enforce_same_uniprot=cfg.pdb_query.enforce_same_uniprot,
    )

    if pdb_ids:
        log_message(
            message_type="pdb_query_complete",
            n_structures=len(pdb_ids),
        )
        if cfg.reference.pdb_id not in pdb_ids:
            pdb_ids.append(cfg.reference.pdb_id)
        pdb_query.download_structures(pdb_ids, output_dir)
        return pdb_ids

    log_message(
        message_type="pipeline_abort",
        reason="no structures found from PDB query",
    )
    return []


def ingest_local_structures(cfg: QPocketConfig, dirs: Dict[str, Path]) -> List[str]:
    """Ingest stage for fragment screens: copy the listed local files in.

    Args:
        cfg: Config in fragment_screen mode.
        dirs: Run directories.

    Returns:
        Structure IDs, reference first.

    Raises:
        ValueError: If nothing could be ingested.
    """
    structure_ids = _load_fragment_screen_structures(cfg, dirs["original"])
    if not structure_ids:
        # Returning [] here used to fall through into alignment, which then
        # failed somewhere much less informative.
        raise ValueError(
            "fragment_screen ingested no structures — check "
            "fragment_screen.structure_paths / csv_path and reference_path"
        )
    return structure_ids


def fetch_structures(cfg: QPocketConfig, output_dir: Union[str, Path]) -> List[str]:
    """Run the ingest stage for the config's mode into one directory.

    Args:
        cfg: Validated config.
        output_dir: Where structures go.

    Returns:
        Structure identifiers.
    """
    dirs = {"original": Path(output_dir)}
    if cfg.mode == "fragment_screen":
        return ingest_local_structures(cfg, dirs)
    return download_structures(cfg, dirs)


class AlignmentContext(NamedTuple):
    """What a worker needs to align one structure onto the reference.

    Attributes:
        reference_path: Aligned reference PDB; workers reload it themselves.
        reference_chain: Chain every target is fitted onto.
        reference_sequence: That chain's sequence.
        uniprot_id: Accession used for renumbering, or None.
        aux_sequences: Auxiliary chain sequences, interface mode only.
        aux_centroids: Auxiliary chain CA centroids, interface mode only.
        aux_uniprots: Auxiliary chain accessions, interface mode only.
        renumber: Whether to renumber onto UniProt positions.
        interface_mode: Whether auxiliary chains are matched too.
    """

    reference_path: str
    reference_chain: str
    reference_sequence: str
    uniprot_id: Optional[str]
    aux_sequences: Dict[str, str]
    aux_centroids: Dict[str, Tuple[float, float, float]]
    aux_uniprots: Dict[str, str]
    renumber: bool
    interface_mode: bool


@lru_cache(maxsize=4)
def reference_residues(path: str, chain: str):
    """Amino-acid residues of the aligned reference chain, cached per process.

    Args:
        path: Aligned reference PDB.
        chain: Chain to fit onto.

    Returns:
        Residues in order.
    """
    # Read back from the saved file, as PyMOL did, so targets are fitted to
    # exactly the coordinates on disk.
    return st.polymer_residues(
        st.keep_chains(sio.load(Path(path)), [chain], protein_only=True)[0][chain]
    )


def alignment_context(cfg: QPocketConfig, dirs: Dict[str, Path]) -> Tuple[AlignmentContext, Dict]:
    """Prepare the reference and everything a target needs to be aligned to it.

    Args:
        cfg: QPocket configuration.
        dirs: Run directories.

    Returns:
        (context, the reference's own master row).

    Raises:
        FileNotFoundError: If the reference structure is missing.
        ValueError: If the reference chain is not in it.
    """
    interface_mode = bool(cfg.reference.auxiliary_chain_ids)
    ref_id = reference_id(cfg, dirs)

    # A PDB query always downloads CIF; fragment screening copies whatever
    # the user had, so search the accepted suffixes rather than assuming one.
    candidates = [dirs["original"] / f"{ref_id}.cif"]
    if cfg.mode == "fragment_screen":
        candidates = [dirs["original"] / f"{ref_id}{sfx}" for sfx in STRUCTURE_SUFFIXES]

    reference_path = next((c for c in candidates if c.exists()), None)
    if reference_path is None:
        log_message(
            message_type="error",
            stage="align_structures",
            error="reference structure not found",
            reference_id=ref_id,
            searched=[str(c) for c in candidates],
        )
        raise FileNotFoundError(
            f"Reference structure for {ref_id!r} not found in {dirs['original']}"
        )

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

    aligned_reference_path = dirs["aligned"] / f"{ref_id}_aligned.pdb"
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


    row: Dict = {
        "pdb_id": ref_id,
        "rmsd_to_ref": 0.0,
        "aligned_file": str(aligned_reference_path),
    }
    if interface_mode:
        row["aux_chain_mapping"] = str({c: c for c in cfg.reference.auxiliary_chain_ids})

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
    return context, row


def align_one(cfg: QPocketConfig, context: AlignmentContext, pid_path: Path,
              output_dir: Path) -> Dict:
    """Align one structure onto the reference.

    Args:
        cfg: QPocket configuration.
        context: Reference paths and sequences.
        pid_path: Structure to align.
        output_dir: Where the aligned PDB goes.

    Returns:
        Its master.csv row; aligned_file is absent when alignment failed.
    """
    pid = pid_path.stem
    reference = reference_residues(context.reference_path, context.reference_chain)
    row: Dict = {"pdb_id": pid}
    if context.interface_mode:
        success, rmsd, aligned_file, aux_mapping = (
            alignment.process_structure_interface(
                target_path=pid_path,
                output_dir=output_dir,
                reference=reference,
                ref_seq=context.reference_sequence,
                ref_aux_seqs=context.aux_sequences,
                ref_aux_centroids=context.aux_centroids,
                uniprot_id=context.uniprot_id,
                auxiliary_uniprot_ids=context.aux_uniprots,
                renumber=context.renumber,
                min_auxiliary_identity=cfg.alignment.min_auxiliary_identity,
                max_auxiliary_centroid_deviation=(
                    cfg.alignment.max_auxiliary_centroid_deviation
                ),
                require_all_chains=cfg.alignment.require_all_chains,
            )
        )
    else:
        success, rmsd, aligned_file = alignment.process_structure(
            target_path=pid_path,
            output_dir=output_dir,
            reference=reference,
            ref_seq=context.reference_sequence,
            uniprot_id=context.uniprot_id,
            renumber=context.renumber,
        )
        aux_mapping = {}

    if success:
        row = {
            "pdb_id": pid,
            "rmsd_to_ref": rmsd,
            "aligned_file": aligned_file,
        }
        if context.interface_mode:
            row["aux_chain_mapping"] = str(aux_mapping)
        log_message(
            message_type="alignment_ok",
            pdb_id=pid,
            rmsd=rmsd,
        )
    else:
        log_message(
            message_type="alignment_failed",
            pdb_id=pid,
        )
    return row


def align_structures(cfg: QPocketConfig, dirs: Dict[str, Path]) -> None:
    """Align every downloaded structure onto the reference, on compute.workers workers.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
    """
    interface_mode = bool(cfg.reference.auxiliary_chain_ids)
    with start_action(action_type="align_structures", interface_mode=interface_mode):
        context, reference_row = alignment_context(cfg, dirs)
        targets = sorted(
            path
            for suffix in STRUCTURE_SUFFIXES
            for path in dirs["original"].glob(f"*{suffix}")
        )
        others = [p for p in targets if p.stem != reference_row["pdb_id"]]
        rows = parallel.run_map(
            lambda path: align_one(cfg, context, path, dirs["aligned"]),
            others,
            workers=cfg.compute.workers,
            backend=cfg.compute.backend,
            stage="align",
            log_dir=Path(dirs["root"]) / parallel.WORKER_LOG_DIR,
            label=lambda path: path.stem,
        )
        # Rows follow the sorted input order, reference included: master.csv order
        # decides HDF5 insertion order and the frequency tables' tie-breaks.
        by_id = {path.stem: row for path, row in zip(others, rows)}
        by_id[reference_row["pdb_id"]] = reference_row
        rmsd_data = [by_id[p.stem] for p in targets
                     if p.stem in by_id and by_id[p.stem].get("aligned_file")]

        if rmsd_data:
            df = pd.DataFrame(rmsd_data)
            matrix_path = dirs["metadata"] / "master.csv"
            df.to_csv(str(matrix_path), index=False)
            log_message(
                message_type="master_csv_saved",
                path=str(matrix_path),
                n_structures=len(rmsd_data),
            )

        log_message(message_type="stage_complete", stage="align_structures")


def clean_one(cfg: QPocketConfig, dirs: Dict[str, Path], row) -> Optional[str]:
    """Clean one aligned structure.

    Args:
        cfg: QPocket configuration.
        dirs: Run directories.
        row: Its master.csv row.

    Returns:
        The cleaned file path, or None when cleaning failed or there was nothing to clean.
    """
    aligned_filename = row["aligned_file"]
    if pd.isna(aligned_filename):
        return None

    clean_filename = Path(aligned_filename).stem.replace("_aligned", "_clean") + ".pdb"
    output_path = dirs["cleaned"] / clean_filename

    success = cleaning.clean_structure(
        input_pdb_path=aligned_filename,
        output_pdb_path=str(output_path),
        allowed_cofactors=cfg.cleaning.allowed_cofactors,
    )
    if success:
        log_message(
            message_type="cleaning_ok",
            pdb_id=row.get("pdb_id", "unknown"),
            output=str(output_path),
        )
        return str(output_path)
    log_message(message_type="cleaning_failed", pdb_id=row.get("pdb_id", "unknown"))
    return None


def clean_structures(cfg: QPocketConfig, dirs: Dict[str, Path]) -> None:
    """Removes non-standard residues from aligned structures.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
    """
    master_csv = dirs["metadata"] / "master.csv"

    with start_action(action_type="clean_structures"):

        if not master_csv.exists():
            log_message(
                message_type="error",
                stage="clean_structures",
                error="master.csv not found",
                path=str(master_csv),
            )
            raise FileNotFoundError("master.csv not found — run alignment first.")

        df = pd.read_csv(master_csv)

        if "aligned_file" not in df.columns:
            raise FileNotFoundError("master.csv is missing 'aligned_file' column.")

        log_message(
            message_type="cleaning_start",
            n_structures=len(df),
        )

        cleaned_files = parallel.run_map(
            lambda row: clean_one(cfg, dirs, row),
            [row for _, row in df.iterrows()],
            workers=cfg.compute.workers,
            backend=cfg.compute.backend,
            stage="clean",
            log_dir=Path(dirs["root"]) / parallel.WORKER_LOG_DIR,
            label=lambda row: str(row.get("pdb_id", "unknown")),
        )

        df["cleaned_file"] = cleaned_files
        df.to_csv(str(master_csv), index=False)
        log_message(
            message_type="stage_complete",
            stage="clean_structures",
            master_csv=str(master_csv),
        )


class ExtractionContext(NamedTuple):
    """Everything a single structure needs that had to be decided run-wide.

    Attributes:
        ref_centroid: Reference ligand centroid, or ``None`` outside cavity-detection mode.
        bbox_center: User-supplied box centre, or ``None``.
        bbox_dims: User-supplied box dimensions, or ``None``.
        ref_pocket_bbox: ``(lower, upper)`` corners of the reference pocket, or ``None`` when it could not be derived.
    """

    ref_centroid: Optional[np.ndarray]
    bbox_center: Optional[List[float]]
    bbox_dims: Optional[List[float]]
    ref_pocket_bbox: Optional[Tuple[np.ndarray, np.ndarray]]

    def to_json(self) -> Dict[str, object]:
        """Plain types, for ``metadata/extraction_context.json``."""
        def arr(value):
            return None if value is None else np.asarray(value).tolist()

        return {
            "ref_centroid": arr(self.ref_centroid),
            "bbox_center": None if self.bbox_center is None else list(self.bbox_center),
            "bbox_dims": None if self.bbox_dims is None else list(self.bbox_dims),
            "ref_pocket_bbox": (
                None if self.ref_pocket_bbox is None
                else [arr(self.ref_pocket_bbox[0]), arr(self.ref_pocket_bbox[1])]
            ),
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "ExtractionContext":
        """Rebuild a context written by :meth:`to_json`."""
        def arr(value):
            return None if value is None else np.asarray(value, dtype=float)

        box = payload.get("ref_pocket_bbox")
        return cls(
            ref_centroid=arr(payload.get("ref_centroid")),
            bbox_center=payload.get("bbox_center"),
            bbox_dims=payload.get("bbox_dims"),
            ref_pocket_bbox=None if box is None else (arr(box[0]), arr(box[1])),
        )


class StructureResult(NamedTuple):
    """What one structure contributes to the run-wide tables.

    Attributes:
        pdb_id: The structure.
        ligand_file: Path to the extracted ligand, or ``None``.
        ligand_id: Chemical-component ID of that ligand, or ``None``.
        protein_state: ``BOUND_STATE``, ``UNBOUND_STATE``, ``REFERENCE``, ``FAILED`` or ``NO_ANY_CAVITY_FOUND``.
        potential: Residues within reach of the pocket cloud.
        observed: ``RESIDUE|InteractionType`` entries from ProLIF.
        smiles: ``(smiles, ligand_name)`` when a ligand was extracted.
    """

    pdb_id: str
    ligand_file: Optional[str]
    ligand_id: Optional[str]
    protein_state: str
    potential: List[str]
    observed: List[str]
    smiles: Optional[Tuple[str, str]]

    def to_json(self) -> Dict[str, object]:
        """Plain-JSON form, for the sidecar files array tasks exchange.

        Returns:
            JSON-serialisable dict.
        """
        return {
            "pdb_id": self.pdb_id,
            "ligand_file": self.ligand_file,
            "ligand_id": self.ligand_id,
            "protein_state": self.protein_state,
            "potential": list(self.potential),
            "observed": list(self.observed),
            "smiles": None if self.smiles is None else list(self.smiles),
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "StructureResult":
        """Rebuild from the output of to_json.

        Args:
            payload: Dict written by to_json.

        Returns:
            The instance.
        """
        smiles = payload.get("smiles")
        return cls(
            pdb_id=payload["pdb_id"],
            ligand_file=payload.get("ligand_file"),
            ligand_id=payload.get("ligand_id"),
            protein_state=payload.get("protein_state", "FAILED"),
            potential=list(payload.get("potential") or []),
            observed=list(payload.get("observed") or []),
            smiles=None if smiles is None else (smiles[0], smiles[1]),
        )


def parts_dir(dirs: Dict[str, Path]) -> Path:
    """Directory holding one JSON sidecar per structure."""
    path = dirs["metadata"] / "parts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_extraction_context(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    df: pd.DataFrame,
) -> Optional[ExtractionContext]:
    """Locate the reference pocket, which every other structure is measured against.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        df: ``master.csv`` contents.

    Returns:
        The context, or ``None`` when the reference ligand centroid is required but cannot be computed.
    """
    save_intermediate = cfg.io.save_intermediate

    ref_centroid: Optional[np.ndarray] = None
    if cfg.reference.pocket_mode == "cavity_detection":
        ref_file = _resolve_reference_file(cfg, dirs)
        if cfg.reference.ligand_id and ref_file is not None:
            # Priority: an explicit ligand_chain_id wins; interface mode searches
            # every chain, because a homodimer ligand often sits in a third one;
            # otherwise search the primary chain.
            if cfg.reference.ligand_chain_id:
                search_chains: Optional[List[str]] = [cfg.reference.ligand_chain_id]
            elif cfg.reference.auxiliary_chain_ids:
                search_chains = None
            else:
                search_chains = [cfg.reference.chain_id]

            ref_centroid = extraction.get_ref_ligand_centroid(
                ref_file,
                cfg.reference.ligand_id,
                search_chains=search_chains,
            )
            if ref_centroid is None:
                log_message(
                    message_type="error",
                    stage="build_extraction_context",
                    error="could not compute reference ligand centroid",
                )
                return None

    bbox_center = None
    bbox_dims = None
    if cfg.reference.pocket_mode == "bounding_box":
        if cfg.reference.bounding_box.center and cfg.reference.bounding_box.dimensions:
            bbox_center = cfg.reference.bounding_box.center
            bbox_dims = cfg.reference.bounding_box.dimensions

    ref_pocket_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if cfg.reference.pocket_mode == "cavity_detection" and ref_centroid is not None:
        ref_pdb_id = reference_id(cfg, dirs)
        ref_row_mask = df["pdb_id"] == ref_pdb_id
        if ref_row_mask.any():
            ref_row = df[ref_row_mask].iloc[0]
            ref_cleaned_file = ref_row.get("cleaned_file")
            if ref_cleaned_file and not pd.isna(ref_cleaned_file):
                ref_cleaned_path = Path(ref_cleaned_file)
                if ref_cleaned_path.exists():
                    ref_pocket_bbox = extraction.extract_reference_pocket_bbox(
                        pdb_path=ref_cleaned_path,
                        pdb_id=ref_pdb_id,
                        ref_centroid=ref_centroid,
                        pocket_save_dir=dirs["pockets"] / ref_pdb_id,
                        save_intermediate=save_intermediate,
                        bbox_padding=1.0,
                    )
                    if ref_pocket_bbox is not None:
                        log_message(
                            message_type="reference_bbox_ready",
                            pdb_id=ref_pdb_id,
                        )

    return ExtractionContext(ref_centroid, bbox_center, bbox_dims, ref_pocket_bbox)


def save_extraction_context(dirs: Dict[str, Path], context: ExtractionContext) -> Path:
    """Persist the context so array tasks can reuse it."""
    path = dirs["metadata"] / "extraction_context.json"
    path.write_text(json.dumps(context.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


def load_extraction_context(dirs: Dict[str, Path]) -> ExtractionContext:
    """Read back a context written by :func:`save_extraction_context`.

    Raises:
        FileNotFoundError: If the prepare stage has not run.
    """
    path = dirs["metadata"] / "extraction_context.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the prepare stage before per-structure tasks"
        )
    return ExtractionContext.from_json(json.loads(path.read_text(encoding="utf-8")))


def protonate_one(cfg: QPocketConfig, dirs: Dict[str, Path], row) -> Optional[str]:
    """Write one docking-ready structure: protonated protein, protonated ligand.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        row: The structure's master row, carrying ligand_file and ligand_id.

    Returns:
        Path to the protonated PDB, or None when protonation failed.
    """
    from qpocket.core.hydroxyl import optimise_hydroxyls
    from qpocket.io.interactions import (load_molecule, protonated_ligand,
                                         write_complex, write_molecule)
    from qpocket.io.tools import protonate_for_interactions

    pdb_id = row["pdb_id"]
    cleaned = row.get("cleaned_file")
    if not cleaned or not Path(cleaned).exists():
        return None

    out_dir = Path(dirs["cleaned"]).parent / "protonated"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{pdb_id}.pdb"
    if destination.exists():
        return str(destination)

    try:
        pqr = protonate_for_interactions(Path(cleaned), out_dir / f"{pdb_id}.pqr")
        ligand_mol = None
        ligand_id = row.get("ligand_id")
        ligand_file = row.get("ligand_file")
        if ligand_id and ligand_file and Path(ligand_file).exists():
            ligand_mol = protonated_ligand(Path(ligand_file))
        write_complex(destination, pqr, ligand_mol, ligand_id)
        # pdb2pqr placed the hydroxyl hydrogens before the ligand existed, so
        # their rotamers are re-chosen now that it is present.
        write_molecule(destination, optimise_hydroxyls(load_molecule(destination)))
    except Exception as error:
        log_message(message_type="error", stage="protonation",
                    pdb_id=pdb_id, error=str(error))
        return None
    return str(destination)


def _featurize_reference(cfg: QPocketConfig, dirs: Dict[str, Path], row) -> None:
    """Featurise the reference's own pocket, which its branch does not.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        row: The reference's master row.
    """
    # Every other structure is featurised inside extract_pocket_from_structure.
    # The reference takes a shorter branch that only lifts its ligand out, so
    # without this it is missing from its own distance matrix and clustering.
    pdb_id = str(row["pdb_id"])
    cloud = dirs["pockets"] / pdb_id / "binding_site.csv"
    featurized = dirs["pockets"] / pdb_id / "featurized.csv"
    cleaned = row.get("cleaned_file")
    if featurized.exists() or not cloud.exists() or not cleaned:
        return
    extraction.featurize(cloud, Path(cleaned), cfg.features.enabled)


def _extract_reference_ligand(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    row: pd.Series,
) -> Tuple[Optional[str], str, Optional[str]]:
    """Pull the reference structure's own ligand out of its aligned model."""
    pdb_id = row["pdb_id"]
    ref_pocket_dir = dirs["pockets"] / pdb_id
    ref_aligned = Path(row["aligned_file"])

    if not cfg.reference.ligand_id:
        return None, "REFERENCE", None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(ref_aligned))
        choice = None
        final_pocket_csv = ref_pocket_dir / "binding_site.csv"
        if final_pocket_csv.exists():
            final_pocket_coords = pd.read_csv(final_pocket_csv)[["x", "y", "z"]].to_numpy()
            choice = pocket.pick_ligand(
                structure, final_pocket_coords, (cfg.cleaning.allowed_cofactors or "").split("|")
            )
        if choice is not None:
            res_name = choice.residue.get_resname().strip()
            block = pocket.extract_pdb_block(choice.residue, choice.chain_id)
            lig_out = dirs["extracted_ligands"] / f"{pdb_id}_{res_name}.pdb"
            with open(lig_out, "w", encoding="utf-8") as f:
                f.write(block)
            return str(lig_out), "BOUND_STATE", res_name
    except Exception as e:
        log_message(
            message_type="warning",
            pdb_id=pdb_id,
            warning=f"reference ligand extraction failed: {e}",
        )

    return None, "REFERENCE", None


def process_one_structure(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    row: pd.Series,
    context: ExtractionContext,
    write_part: bool = True,
) -> StructureResult:
    """Extract, featurise and analyse a single structure.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        row: One row of ``master.csv``.
        context: Run-wide reference geometry.
        write_part: Write the JSON sidecar.

    Returns:
        This structure's contribution to the run-wide tables.
    """
    pdb_id = row["pdb_id"]
    is_reference = pdb_id == reference_id(cfg, dirs)
    smiles: Optional[Tuple[str, str]] = None

    with start_action(action_type="process_structure", pdb_id=pdb_id):
        if is_reference and context.ref_pocket_bbox is not None:
            lf, form, lig_id = _extract_reference_ligand(cfg, dirs, row)
            _featurize_reference(cfg, dirs, row)
        else:
            lf, form, lig_id = extraction.extract_pocket_from_structure(
                row,
                dirs,
                ref_centroid=context.ref_centroid if is_reference else None,
                allowed_cofactors=cfg.cleaning.allowed_cofactors,
                pocket_mode=cfg.reference.pocket_mode,
                bounding_box_center=context.bbox_center,
                bounding_box_dims=context.bbox_dims,
                pocket_definition="Solvent",
                interaction_padding=cfg.reference.interaction_padding,
                ref_pocket_bbox=None if is_reference else context.ref_pocket_bbox,
                save_intermediate=cfg.io.save_intermediate,
                providers=cfg.features.enabled,
            )

        enriched = row.copy()
        enriched["ligand_file"] = lf
        enriched["ligand_id"] = lig_id
        enriched["protein_state"] = form
        # The docking-ready structure is written before interactions are read
        # from it, and an apo structure still gets its protonated protein.
        enriched["protonated_file"] = protonate_one(cfg, dirs, enriched)

        analysis = extraction.run_analysis_step(
            enriched, dirs,
            enabled=cfg.interactions.enabled,
            parameters=cfg.interactions.parameters,
        )

        # Second pass: crop to the observed contacts and re-featurise, so
        # binding_site.csv and featurized.csv always describe the same cloud.
        if (
            cfg.reference.pocket_definition == "Interactions"
            and analysis["observed"]
            and form not in ("FAILED", "NO_ANY_CAVITY_FOUND")
        ):
            lf2, form2, lig_id2 = extraction.extract_pocket_from_structure(
                enriched,
                dirs,
                ref_centroid=context.ref_centroid if is_reference else None,
                allowed_cofactors=cfg.cleaning.allowed_cofactors,
                pocket_mode=cfg.reference.pocket_mode,
                bounding_box_center=context.bbox_center,
                bounding_box_dims=context.bbox_dims,
                pocket_definition="Interactions",
                interaction_padding=cfg.reference.interaction_padding,
                observed_interactions=analysis["observed"],
                ref_pocket_bbox=None if is_reference else context.ref_pocket_bbox,
                save_intermediate=cfg.io.save_intermediate,
                force_featurize=True,  # B-3: stale featurized.csv must be regenerated
                providers=cfg.features.enabled,
            )
            if form2 not in ("FAILED",):
                lf, form, lig_id = lf2, form2, lig_id2

        # Every extracted ligand contributes to actives.smi, not just the
        # reference, and its bond orders come from the component definition.
        if lf and lig_id and Path(lf).exists():
            from qpocket.io.interactions import ligand_smiles

            _smi = ligand_smiles(Path(lf))
            smiles = (_smi, str(lig_id)) if _smi else None

        result = StructureResult(
            pdb_id=pdb_id,
            ligand_file=lf,
            ligand_id=lig_id,
            protein_state=form,
            potential=analysis["potential"],
            observed=analysis["observed"],
            smiles=smiles,
        )

        if write_part:
            (parts_dir(dirs) / f"{pdb_id}.json").write_text(
                json.dumps(result.to_json(), indent=2) + "\n"
            , encoding="utf-8")

        log_message(message_type="structure_processed", pdb_id=pdb_id, status=form)
        return result


def featurize_structures(cfg: QPocketConfig, dirs: Dict[str, Path], df, context) -> None:
    """Identify and featurise every structure's pocket, on compute.workers workers.

    Args:
        cfg: QPocket configuration.
        dirs: Run directories.
        df: master.csv rows.
        context: Reference pocket context.
    """
    rows = [row for _, row in df.iterrows()]
    parallel.run_map(
        lambda row: process_one_structure(cfg, dirs, row, context),
        rows,
        workers=cfg.compute.workers,
        backend=cfg.compute.backend,
        stage="featurize",
        log_dir=Path(dirs["root"]) / parallel.WORKER_LOG_DIR,
        label=lambda row: str(row["pdb_id"]),
    )


def write_feature_record(meta_dir: Path, enabled: Sequence[str]) -> Path:
    """Record the feature set a run used, for readers of its pockets.

    Args:
        meta_dir: The run's metadata directory.
        enabled: Provider names, in config order.

    Returns:
        Path of features.json.
    """
    from qpocket.core.features import columns_for
    from qpocket.core.ports import FEATURES

    payload = {"enabled": list(enabled),
               "columns": list(columns_for(FEATURES.resolve(list(enabled))))}
    path = Path(meta_dir) / "features.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def collect_structure_results(cfg: QPocketConfig, dirs: Dict[str, Path]) -> None:
    """Merge the per-structure sidecars into the run-wide tables.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
    """
    with start_action(action_type="collect_structure_results"):
        csv_path = dirs["metadata"] / "master.csv"
        if not csv_path.exists():
            log_message(
                message_type="error",
                stage="collect_structure_results",
                error="master.csv missing",
            )
            return

        df = pd.read_csv(csv_path)
        results: Dict[str, StructureResult] = {}
        for part in sorted(parts_dir(dirs).glob("*.json")):
            try:
                result = StructureResult.from_json(json.loads(part.read_text(encoding="utf-8")))
            except Exception as e:
                log_message(
                    message_type="warning",
                    stage="collect_structure_results",
                    warning=f"unreadable sidecar {part.name}: {e}",
                )
                continue
            results[result.pdb_id] = result

        missing = [pid for pid in df["pdb_id"] if pid not in results]
        if missing:
            log_message(
                message_type="warning",
                stage="collect_structure_results",
                warning="structures with no sidecar",
                pdb_ids=missing[:20],
                n_missing=len(missing),
            )

        df["ligand_file"] = [
            results[pid].ligand_file if pid in results else None for pid in df["pdb_id"]
        ]
        df["ligand_id"] = [
            results[pid].ligand_id if pid in results else None for pid in df["pdb_id"]
        ]
        df["protein_state"] = [
            results[pid].protein_state if pid in results else "FAILED"
            for pid in df["pdb_id"]
        ]
        df.to_csv(csv_path, index=False)

        # Written in master order rather than task-completion order, so the file
        # is identical whichever way the structures were processed.
        smi_path = dirs["screening"] / "actives.smi"
        with open(smi_path, "w", encoding="utf-8") as f:
            f.write("SMILES\tPDB_ID\tLigand_Name\n")
            for pid in df["pdb_id"]:
                result = results.get(pid)
                if result is not None and result.smiles:
                    f.write(f"{result.smiles[0]}\t{pid}\t{result.smiles[1]}\n")

        write_feature_record(dirs["metadata"], cfg.features.enabled)

        # master.csv order, not sidecar-glob order: HDF5 preserves insertion
        # order, and the frequency tables break ties on it, so iterating the
        # glob would make the output depend on how the work was scheduled.
        ordered = [pid for pid in df["pdb_id"] if pid in results]
        potential_data = {pid: results[pid].potential for pid in ordered}
        observed_data = {pid: results[pid].observed for pid in ordered}

        obs_h5 = dirs["metadata"] / "observed_interactions.h5"
        pot_h5 = dirs["metadata"] / "potential_interactions.h5"
        hdf5.save_interactions(potential_data, pot_h5, compression=cfg.io.compression)
        hdf5.save_interactions(observed_data, obs_h5, compression=cfg.io.compression)

        log_message(message_type="interaction_data_saved", n_structures=len(results))

        try:
            with start_action(action_type="frequency_analysis"):
                collect.calculate_observed_frequencies(str(obs_h5))
                collect.calculate_potential_frequencies(str(pot_h5))
                collect.calculate_binding_site_utilization(str(obs_h5), str(pot_h5))
                log_message(message_type="frequency_analysis_complete")
        except Exception as e:
            log_message(
                message_type="error",
                stage="frequency_analysis",
                error=str(e),
            )


def structure_chain(cfg: QPocketConfig, dirs: Dict[str, Path], align_context: AlignmentContext,
                    extraction_context, pid_path: Path) -> Dict:
    """Take one structure from input file to geodesic cache.

    Args:
        cfg: QPocket configuration.
        dirs: Run directories.
        align_context: Reference paths and sequences.
        extraction_context: Reference pocket context.
        pid_path: Input structure.

    Returns:
        Its master.csv row; incomplete when a step failed.
    """
    row = align_one(cfg, align_context, pid_path, dirs["aligned"])
    if not row.get("aligned_file"):
        return row
    row["cleaned_file"] = clean_one(cfg, dirs, pd.Series(row))
    if row["cleaned_file"] is None:
        return row
    process_one_structure(cfg, dirs, pd.Series(row), extraction_context)
    # Build the geodesic here too: the worker already has the pocket warm, and the
    # cache is keyed on featurized.csv, so the distance stage reuses it untouched.
    featurized = dirs["pockets"] / str(row["pdb_id"]) / "featurized.csv"
    if featurized.exists():
        compute_geodesic(featurized, cfg.distance.geo_graph_neighbor_radius)
    return row


def prepare_and_featurize(cfg: QPocketConfig, dirs: Dict[str, Path]) -> None:
    """Align, clean, identify and featurise every structure, then collect the run.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
    """
    # The reference goes first: its pocket defines the box every other structure
    # is matched against. The rest fan out across compute.workers.
    with start_action(action_type="prepare_and_featurize"):
        align_context, reference_row = alignment_context(cfg, dirs)
        reference_row["cleaned_file"] = clean_one(cfg, dirs, pd.Series(reference_row))
        if reference_row["cleaned_file"] is None:
            log_message(message_type="error", stage="prepare_and_featurize",
                        error="the reference structure could not be cleaned")
            return

        extraction_context = build_extraction_context(cfg, dirs, pd.DataFrame([reference_row]))
        if extraction_context is None:
            return
        save_extraction_context(dirs, extraction_context)

        targets = sorted(
            path
            for suffix in STRUCTURE_SUFFIXES
            for path in dirs["original"].glob(f"*{suffix}")
        )
        others = [p for p in targets if p.stem != reference_row["pdb_id"]]
        log_message(message_type="extraction_start", n_structures=len(targets))

        process_one_structure(cfg, dirs, pd.Series(reference_row), extraction_context)
        reference_featurized = dirs["pockets"] / str(reference_row["pdb_id"]) / "featurized.csv"
        if reference_featurized.exists():
            compute_geodesic(reference_featurized, cfg.distance.geo_graph_neighbor_radius)

        rows = parallel.run_map(
            lambda path: structure_chain(cfg, dirs, align_context, extraction_context, path),
            others,
            workers=cfg.compute.workers,
            backend=cfg.compute.backend,
            stage="structure",
            log_dir=Path(dirs["root"]) / parallel.WORKER_LOG_DIR,
            label=lambda path: path.stem,
        )

        # Sorted input order, reference in its own place: master.csv order decides
        # HDF5 insertion order and the frequency tables' tie-breaks.
        by_id = {path.stem: row for path, row in zip(others, rows)}
        by_id[reference_row["pdb_id"]] = reference_row
        complete = [by_id[p.stem] for p in targets
                    if p.stem in by_id and by_id[p.stem].get("cleaned_file")]

        master_csv = dirs["metadata"] / "master.csv"
        pd.DataFrame(complete).to_csv(str(master_csv), index=False)
        log_message(message_type="master_csv_saved", path=str(master_csv),
                    n_structures=len(complete))

        collect_structure_results(cfg, dirs)
        log_message(message_type="stage_complete", stage="prepare_and_featurize")


def extract_pockets_and_interactions(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
) -> None:
    """Extracts binding pockets and analyses interactions for all structures.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
    """
    with start_action(action_type="extract_pockets_and_interactions"):

        csv_path = dirs["metadata"] / "master.csv"
        if not csv_path.exists():
            log_message(
                message_type="error",
                stage="extract_pockets_and_interactions",
                error="master.csv missing",
            )
            return

        df = pd.read_csv(csv_path)

        context = build_extraction_context(cfg, dirs, df)
        if context is None:
            return
        save_extraction_context(dirs, context)

        log_message(message_type="extraction_start", n_structures=len(df))

        featurize_structures(cfg, dirs, df, context)
        collect_structure_results(cfg, dirs)

        log_message(
            message_type="stage_complete",
            stage="extract_pockets_and_interactions",
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_reference_file(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
) -> Optional[Path]:
    """Returns the aligned PDB path of the reference structure."""
    # One source of truth for "which structure is the reference". This used to
    # index master.csv positionally in fragment mode, which disagreed with the
    # name align_structures actually wrote.
    ref_pdb_id = reference_id(cfg, dirs)
    ref_file = dirs["aligned"] / f"{ref_pdb_id}_aligned.pdb"
    return ref_file if ref_file.exists() else None


def _fragment_input_paths(cfg: QPocketConfig) -> List[Path]:
    """Structure paths named by the config, reference first.

    Args:
        cfg: QPocket configuration.

    Returns:
        Absolute paths in ingest order.
    """
    listed: List[str] = []

    if cfg.fragment_screen.input_type == "csv":
        if not cfg.fragment_screen.csv_path:
            log_message(message_type="error", stage="load_fragment_screen",
                        error="csv_path not provided")
            return []
        csv_p = Path(cfg.fragment_screen.csv_path)
        if not csv_p.exists():
            log_message(message_type="error", stage="load_fragment_screen",
                        error="CSV file not found", path=str(csv_p))
            return []
        df = pd.read_csv(csv_p)
        if "pdb_path" not in df.columns:
            log_message(message_type="error", stage="load_fragment_screen",
                        error="CSV missing 'pdb_path' column")
            return []
        listed = [str(v) for v in df["pdb_path"].tolist()]

    elif cfg.fragment_screen.input_type == "paths":
        if not cfg.fragment_screen.structure_paths:
            log_message(message_type="error", stage="load_fragment_screen",
                        error="structure_paths not provided")
            return []
        listed = [str(v) for v in cfg.fragment_screen.structure_paths]

    ordered: List[Path] = []
    seen: set = set()

    if cfg.fragment_screen.reference_path:
        resolved = Path(cfg.fragment_screen.reference_path).expanduser().resolve()
        ordered.append(resolved)
        seen.add(resolved)

    for raw in listed:
        resolved = Path(raw).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)

    return ordered


def _load_fragment_screen_structures(
    cfg: QPocketConfig,
    output_dir: Union[str, Path],
) -> List[str]:
    """Copy local structures into the run and record where each came from.

    Args:
        cfg: QPocket configuration.
        output_dir: ``structures/original`` for this run.

    Returns:
        Identifiers, in ingest order, reference first.
    """
    import shutil

    output_dir = Path(output_dir)
    paths = _fragment_input_paths(cfg)
    if not paths:
        return []

    ref_path = (
        Path(cfg.fragment_screen.reference_path).expanduser().resolve()
        if cfg.fragment_screen.reference_path
        else None
    )

    structure_ids: List[str] = []
    records: List[Dict[str, object]] = []
    used: Dict[str, int] = {}

    for src in paths:
        if not src.exists():
            log_message(message_type="warning", stage="load_fragment_screen",
                        warning=f"structure not found: {src}")
            continue
        if src.suffix.lower() not in STRUCTURE_SUFFIXES:
            log_message(message_type="warning", stage="load_fragment_screen",
                        warning=f"unrecognised structure format, skipping: {src.name}")
            continue

        # Two directories can hold the same filename; disambiguate rather than
        # letting one silently overwrite the other.
        base = sanitise_id(src.stem)
        count = used.get(base, 0)
        used[base] = count + 1
        struct_id = base if count == 0 else f"{base}_{count + 1}"

        dest = output_dir / f"{struct_id}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        structure_ids.append(struct_id)
        records.append({
            "pdb_id": struct_id,
            "source_path": str(src),
            "is_reference": bool(ref_path is not None and src == ref_path),
        })
        log_message(message_type="fragment_loaded", source=src.name, id=struct_id)

    if records and not any(r["is_reference"] for r in records):
        # No explicit reference file: fall back to the configured index so some
        # structure is always the reference.
        idx = min(int(cfg.fragment_screen.reference_index), len(records) - 1)
        records[idx]["is_reference"] = True

    if records:
        meta_dir = output_dir.parent.parent / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(meta_dir / "inputs.csv", index=False)

    log_message(
        message_type="fragment_screen_loaded",
        n_structures=len(structure_ids),
        reference=next((r["pdb_id"] for r in records if r["is_reference"]), None),
    )
    return structure_ids
