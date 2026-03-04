from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from eliot import add_destinations, log_message, start_action, to_file, MemoryLogger
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pymol import cmd

from qpocket_pipeline.config import QPocketConfig
from qpocket_pipeline import utils
from qpocket_pipeline.distance_logic.spectral_distance import compute_spectral_distance_matrix
from qpocket_pipeline import pdb_query
from qpocket_pipeline.processing import alignment
from qpocket_pipeline.processing import cleaning
from qpocket_pipeline.processing import extraction

CONF_PATH = Path(__file__).parent.parent / "conf"

def load_qpocket_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> QPocketConfig:
    """Loads, merges, and validates QPocket configuration.
    Configuration priority: CLI overrides > external file > built-in defaults.

    Args:
        config_path: Path to an external YAML configuration file.
        overrides: Dot-notation override strings
            (e.g. ``["mode=fragment_screen", "reference.pdb_id=6HP0"]``).

    Returns:
        Fully validated :class:`~qpocket_pipeline.config.QPocketConfig`.
    """
    GlobalHydra.instance().clear()

    with initialize_config_module(
        version_base=None,
        config_module="conf",
    ):
        cfg = compose(config_name="config", overrides=overrides)

    if config_path:
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, user_cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    final_cfg = QPocketConfig(
        mode=cfg.mode,
        paths=cfg.paths,
        reference=cfg.reference,
        fragment_screen=cfg.fragment_screen,
        pdb_query=cfg.pdb_query,
        alignment=cfg.alignment,
        cleaning=cfg.cleaning,
        io=cfg.io,
        spectral_distance_matrix=cfg.spectral_distance_matrix,
    )
    return final_cfg

def run_qpocket_pipeline(
    qpocket_config_path: Optional[Union[str, Path]] = None,
    pdb_id: Optional[str] = None,
    chain_id: Optional[str] = None,
    ligand_id: Optional[str] = None,
    workdir: Optional[Union[str, Path]] = None,
    sequence_identity: Optional[float] = None,
    resolution_limit: Optional[float] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    experimental_method: Optional[str] = None,
    enforce_same_uniprot: Optional[bool] = None,
    renumber_residues: bool = True,
    manual_uniprot_id: Optional[str] = None,
    allowed_cofactors: Optional[str] = None,
    geo_graph_neighbor_radius: Optional[float] = None,
    paralell_jobs: Optional[int] = None,
    global_k: Optional[int] = None,
    sigma: Optional[float] = None,
    max_points: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    affinity_damping: Optional[float] = None,
    affinity_max_iter: Optional[int] = None,
    log_format: Optional[str] = None,
    save_intermediate: Optional[bool] = None,
    compression: Optional[str] = None,
) -> None:
    """Execute the complete QPocket pipeline.

    Args:
        qpocket_config_path: Path to a YAML configuration file.
        pdb_id: Reference PDB identifier.
        chain_id: Chain in the reference structure.
        ligand_id: Ligand residue name in the reference structure.
        workdir: Working directory for all output files.
        sequence_identity: Minimum sequence identity for PDB query.
        resolution_limit: Maximum resolution limit (Å) for PDB query.
        min_length: Minimum protein sequence length for PDB query.
        max_length: Maximum protein sequence length for PDB query.
        experimental_method: Experimental method filter (e.g. ``"X-RAY DIFFRACTION"``).
        enforce_same_uniprot: Restrict PDB query to the same UniProt entry.
        renumber_residues: Renumber residues to match UniProt positions.
        manual_uniprot_id: Override the UniProt ID used for renumbering.
        allowed_cofactors: ``"|"``-separated cofactor names to retain after cleaning.
        geo_graph_neighbor_radius: Geodesic graph neighbourhood radius (Å).
        paralell_jobs: ``joblib`` parallel worker count (``-1`` = all cores).
        global_k: Spectral fingerprint length.
        sigma: Gaussian heat kernel σ.
        max_points: Maximum pocket size for spectral embedding.
        alpha: Geodesic weight in the combined distance metric.
        beta: Feature cost scaling factor.
        affinity_damping: Affinity Propagation damping factor.
        affinity_max_iter: Maximum Affinity Propagation iterations.
        log_format: Eliot log format — ``"json"`` or ``"text"``.
        save_intermediate: Persist intermediate pocket files.
        compression: HDF5 compression — ``"gzip"`` or ``"none"``.
    """
    overrides: List[str] = []

    if pdb_id:
        overrides.append(f"reference.pdb_id={pdb_id.upper()}")
    if chain_id:
        overrides.append(f"reference.chain_id={chain_id.upper()}")
    if ligand_id:
        overrides.append(f"reference.ligand_id={ligand_id.upper()}")
    if workdir:
        overrides.append(f"paths.workdir={workdir}")
    if sequence_identity is not None:
        overrides.append(f"pdb_query.sequence_identity={sequence_identity}")
    if resolution_limit is not None:
        overrides.append(f"pdb_query.resolution_limit={resolution_limit}")
    if min_length is not None:
        overrides.append(f"pdb_query.min_length={min_length}")
    if max_length is not None:
        overrides.append(f"pdb_query.max_length={max_length}")
    if experimental_method:
        overrides.append(f"pdb_query.experimental_method={experimental_method.upper()}")
    if enforce_same_uniprot is not None:
        overrides.append(f"pdb_query.enforce_same_uniprot={enforce_same_uniprot}")
    if renumber_residues:
        overrides.append(f"alignment.renumber_residues={renumber_residues}")
    if manual_uniprot_id:
        overrides.append(f"alignment.manual_uniprot_id={manual_uniprot_id}")
    if allowed_cofactors:
        overrides.append(f"cleaning.allowed_cofactors={allowed_cofactors}")
    if geo_graph_neighbor_radius is not None:
        overrides.append(
            f"spectral_distance_matrix.geo_graph_neighbor_radius={geo_graph_neighbor_radius}"
        )
    if paralell_jobs is not None:
        overrides.append(f"spectral_distance_matrix.paralell_jobs={paralell_jobs}")
    if global_k is not None:
        overrides.append(f"spectral_distance_matrix.global_k={global_k}")
    if sigma is not None:
        overrides.append(f"spectral_distance_matrix.sigma={sigma}")
    if max_points is not None:
        overrides.append(f"spectral_distance_matrix.max_points={max_points}")
    if alpha is not None:
        overrides.append(f"spectral_distance_matrix.alpha={alpha}")
    if beta is not None:
        overrides.append(f"spectral_distance_matrix.beta={beta}")
    if affinity_damping is not None:
        overrides.append(f"spectral_distance_matrix.affinity_damping={affinity_damping}")
    if affinity_max_iter is not None:
        overrides.append(f"spectral_distance_matrix.affinity_max_iter={affinity_max_iter}")
    if log_format:
        overrides.append(f"io.log_format={log_format}")
    if save_intermediate is not None:
        overrides.append(f"io.save_intermediate={save_intermediate}")
    if compression:
        overrides.append(f"io.compression={compression}")

    cfg = load_qpocket_config(config_path=qpocket_config_path, overrides=overrides)
    dirs = utils.setup_directory_hierarchy(cfg.paths.workdir, cfg.reference.pdb_id)

    log_path = dirs["root"] / "qpocket_pipeline.log"
    _configure_logging(cfg, log_path)

    with start_action(
        action_type="pipeline_execution",
        reference_pdb=cfg.reference.pdb_id,
        mode=cfg.mode,
        workdir=str(dirs["root"]),
    ):
        log_message(
            message_type="workspace_ready",
            root_path=str(dirs["root"]),
            log_path=str(log_path),
        )

        with start_action(action_type="stage_data_retrieval"):
            fetch_structures(cfg, dirs["original"])

        with start_action(action_type="stage_preprocessing"):
            align_structures(cfg, dirs)
            clean_structures(cfg, dirs)
            extract_pockets_and_interactions(cfg, dirs)
            compute_spectral_distance_matrix(cfg, dirs)


def fetch_structures(cfg: QPocketConfig, output_dir: Union[str, Path]) -> List[str]:
    """Fetches or loads structures based on pipeline mode.

    Args:
        cfg: QPocket configuration.
        output_dir: Directory in which to place downloaded / copied structures.

    Returns:
        List of structure identifiers.
    """
    if cfg.mode == "fragment_screen":
        return _load_fragment_screen_structures(cfg, output_dir)

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


def align_structures(cfg: QPocketConfig, dirs: Dict[str, Path]) -> None:
    """Aligns all downloaded structures to the reference using PyMOL.

    The aligned reference is saved first; then every other structure is
    aligned and optionally residue-renumbered to match UniProt positions.
    Results are written to ``metadata/master.csv``.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
    """
    with start_action(action_type="align_structures"):

        cmd.delete("all")
        reference_path = dirs["original"] / f"{cfg.reference.pdb_id}.cif"
        if not reference_path.exists():
            log_message(
                message_type="error",
                stage="align_structures",
                error="reference CIF not found",
                path=str(reference_path),
            )
            raise FileNotFoundError(f"Reference file not found: {reference_path}")

        cmd.load(str(reference_path), "static")
        cmd.remove("resn hoh")
        cmd.select("keep", f"static and chain {cfg.reference.chain_id}")
        aligned_reference_path = dirs["aligned"] / f"{cfg.reference.pdb_id}_aligned.pdb"
        cmd.save(str(aligned_reference_path), "keep")

        ref_chain_seq = cmd.get_fastastr(
            selection=f"static and chain {cfg.reference.chain_id}"
        )
        ref_chain_seq = "".join(ref_chain_seq.split()[1:])
        cmd.delete("all")

        log_message(
            message_type="reference_processed",
            pdb_id=cfg.reference.pdb_id,
            seq_length=len(ref_chain_seq),
        )

        uniprot_id = (
            cfg.alignment.manual_uniprot_id
            or pdb_query.get_uniprot_accession(cfg.reference.pdb_id, cfg.reference.chain_id)
        )

        cmd.load(aligned_reference_path, "ref_full")
        cmd.remove("ref_full and (solvent or resn HOH)")
        cmd.create(
            "reference_obj",
            f"ref_full and chain {cfg.reference.chain_id} and polymer.protein",
        )
        cmd.delete("ref_full")

        rmsd_data = []
        for pid_path in dirs["original"].glob("*.cif"):
            pid = pid_path.stem
            if pid == cfg.reference.pdb_id:
                rmsd_data.append({
                    "pdb_id": pid,
                    "rmsd_to_ref": 0.0,
                    "aligned_file": str(aligned_reference_path),
                })
                continue

            success, rmsd, aligned_file = alignment.process_structure(
                target_path=pid_path,
                output_dir=dirs["aligned"],
                ref_obj_name="reference_obj",
                ref_seq=ref_chain_seq,
                uniprot_id=uniprot_id,
                renumber=cfg.alignment.renumber_residues,
            )

            if success:
                rmsd_data.append({
                    "pdb_id": pid,
                    "rmsd_to_ref": rmsd,
                    "aligned_file": aligned_file,
                })
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

        if rmsd_data:
            df = pd.DataFrame(rmsd_data)
            matrix_path = dirs["metadata"] / "master.csv"
            df.to_csv(str(matrix_path), index=False)
            log_message(
                message_type="master_csv_saved",
                path=str(matrix_path),
                n_structures=len(rmsd_data),
            )

        cmd.delete("all")
        log_message(message_type="stage_complete", stage="align_structures")


def clean_structures(cfg: QPocketConfig, dirs: Dict[str, Path]) -> None:
    """Removes non-standard residues from aligned structures.

    Reads ``metadata/master.csv``, cleans each aligned structure, and updates
    the CSV with ``cleaned_file`` paths.

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

        cleaned_files = []
        log_message(
            message_type="cleaning_start",
            n_structures=len(df),
        )

        for _, row in df.iterrows():
            aligned_filename = row["aligned_file"]
            if pd.isna(aligned_filename):
                cleaned_files.append(None)
                continue

            clean_filename = (
                Path(aligned_filename).stem.replace("_aligned", "_clean") + ".pdb"
            )
            output_path = dirs["cleaned"] / clean_filename

            success = cleaning.clean_structure(
                input_pdb_path=aligned_filename,
                output_pdb_path=str(output_path),
                allowed_cofactors=cfg.cleaning.allowed_cofactors,
            )

            if success:
                cleaned_files.append(str(output_path))
                log_message(
                    message_type="cleaning_ok",
                    pdb_id=row.get("pdb_id", "unknown"),
                    output=str(output_path),
                )
            else:
                cleaned_files.append(None)
                log_message(
                    message_type="cleaning_failed",
                    pdb_id=row.get("pdb_id", "unknown"),
                )

        df["cleaned_file"] = cleaned_files
        df.to_csv(str(master_csv), index=False)
        log_message(
            message_type="stage_complete",
            stage="clean_structures",
            master_csv=str(master_csv),
        )


def extract_pockets_and_interactions(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
) -> None:
    """Extracts binding pockets and analyses interactions for all structures.

    Two-phase pocket extraction:

    1. **Reference structure** — the pocket is identified by centroid overlap.
       Its point cloud is used to derive a reference bounding box (1 Å
       padding).
    2. **All other structures** — cavities are ranked by volumetric overlap
       with the reference bounding box; the cavity with the highest overlap
       is selected.

    After pocket extraction, ligand files are generated, PLIP interactions
    are analysed, and frequency statistics are written to the metadata
    directory.

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
        save_intermediate = cfg.io.save_intermediate
        compression = cfg.io.compression

        smi_path = dirs["screening"] / "actives.smi"
        if not smi_path.exists():
            with open(smi_path, "w") as f:
                f.write("SMILES\tPDB_ID\tLigand_Name\n")

        ref_centroid: Optional[np.ndarray] = None
        if cfg.reference.pocket_mode == "cavity_detection":
            ref_file = _resolve_reference_file(cfg, df, dirs)
            if cfg.reference.ligand_id and ref_file is not None:
                ref_centroid = utils.get_ref_ligand_centroid(
                    ref_file,
                    cfg.reference.ligand_id,
                    cfg.reference.chain_id,
                )
                if ref_centroid is None:
                    log_message(
                        message_type="error",
                        stage="extract_pockets_and_interactions",
                        error="could not compute reference ligand centroid",
                    )
                    return

        bbox_center = None
        bbox_dims = None
        if cfg.reference.pocket_mode == "bounding_box":
            if cfg.reference.bounding_box.center and cfg.reference.bounding_box.dimensions:
                bbox_center = cfg.reference.bounding_box.center
                bbox_dims = cfg.reference.bounding_box.dimensions

        ref_pocket_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None

        if cfg.reference.pocket_mode == "cavity_detection" and ref_centroid is not None:
            ref_pdb_id = cfg.reference.pdb_id
            ref_row_mask = df["pdb_id"] == ref_pdb_id
            if ref_row_mask.any():
                ref_row = df[ref_row_mask].iloc[0]
                ref_cleaned_file = ref_row.get("cleaned_file")
                if ref_cleaned_file and not pd.isna(ref_cleaned_file):
                    ref_cleaned_path = dirs["cleaned"] / ref_cleaned_file
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

        # ── Phase 2: Process all structures ───────────────────────────────────
        log_message(
            message_type="extraction_start",
            n_structures=len(df),
        )

        lig_files: List[Optional[str]] = []
        forms: List[str] = []
        potential_data: Dict[str, List[str]] = {}
        observed_data: Dict[str, List[str]] = {}

        for idx, row in df.iterrows():
            pdb_id = row["pdb_id"]
            is_reference = pdb_id == cfg.reference.pdb_id

            if is_reference and ref_pocket_bbox is not None:
                # Treat reference as BOUND if ligand_id was provided
                lf = None
                form = "REFERENCE"
                ref_pocket_dir = dirs["pockets"] / pdb_id
                ref_aligned = dirs["aligned"] / f"{pdb_id}_aligned.pdb"

                if cfg.reference.ligand_id:
                    # Extract reference ligand directly
                    from Bio.PDB import PDBParser
                    parser = PDBParser(QUIET=True)
                    try:
                        structure = parser.get_structure(pdb_id, str(ref_aligned))
                        candidates = []
                        final_pocket_csv = ref_pocket_dir / "binding_site.csv"
                        if final_pocket_csv.exists():
                            import pandas as _pd
                            final_pocket_coords = _pd.read_csv(final_pocket_csv)[["x", "y", "z"]].to_numpy()
                            for model in structure:
                                for chain in model:
                                    for res in chain:
                                        if not utils.is_residue_ligand(
                                            res,
                                            (cfg.cleaning.allowed_cofactors or "").split("|"),
                                        ):
                                            continue
                                        atoms = np.array([a.get_coord() for a in res])
                                        if len(atoms) == 0:
                                            continue
                                        if utils.check_overlap(
                                            atoms, final_pocket_coords, use_majority=True
                                        ):
                                            candidates.append((res, chain.id, len(atoms)))
                        if candidates:
                            best_res, best_chain_id, n_atoms = max(
                                candidates, key=lambda x: x[2]
                            )
                            res_name = best_res.get_resname().strip()
                            fname = f"{pdb_id}_{res_name}.pdb"
                            block = utils.extract_pdb_block(best_res, best_chain_id)
                            lig_out = dirs["extracted_ligands"] / fname
                            with open(lig_out, "w") as f:
                                f.write(block)
                            smi = utils.get_smiles_from_pdb_block(block)
                            if smi:
                                with open(smi_path, "a") as f:
                                    f.write(f"{smi}\t{pdb_id}\t{res_name}\n")
                            lf = str(lig_out)
                            form = "BOUND_STATE"
                    except Exception as e:
                        log_message(
                            message_type="warning",
                            pdb_id=pdb_id,
                            warning=f"reference ligand extraction failed: {e}",
                        )
                lig_files.append(lf)
                forms.append(form)
            else:
                # Non-reference (or reference without bbox): full extraction
                lf, form = extraction.extract_pocket_from_structure(
                    row,
                    dirs,
                    ref_centroid=ref_centroid if is_reference else None,
                    allowed_cofactors=cfg.cleaning.allowed_cofactors,
                    pocket_mode=cfg.reference.pocket_mode,
                    bounding_box_center=bbox_center,
                    bounding_box_dims=bbox_dims,
                    pocket_definition="Solvent",
                    interaction_padding=cfg.fragment_screen.interaction_padding,
                    ref_pocket_bbox=None if is_reference else ref_pocket_bbox,
                    save_intermediate=save_intermediate,
                )
                lig_files.append(lf)
                forms.append(form)

            row = row.copy()
            row["ligand_file"] = lf
            row["protein_state"] = form

            # Interaction analysis
            analysis = extraction.run_analysis_step(row, dirs)
            potential_data[pdb_id] = analysis["potential"]
            observed_data[pdb_id] = analysis["observed"]

            # Second pass: Interactions-mode pocket cropping
            if (
                cfg.reference.pocket_definition == "Interactions"
                and analysis["observed"]
                and form not in ("FAILED", "NO_ANY_CAVITY_FOUND")
            ):
                lf2, form2 = extraction.extract_pocket_from_structure(
                    row,
                    dirs,
                    ref_centroid=ref_centroid if is_reference else None,
                    allowed_cofactors=cfg.cleaning.allowed_cofactors,
                    pocket_mode=cfg.reference.pocket_mode,
                    bounding_box_center=bbox_center,
                    bounding_box_dims=bbox_dims,
                    pocket_definition="Interactions",
                    interaction_padding=cfg.fragment_screen.interaction_padding,
                    observed_interactions=analysis["observed"],
                    ref_pocket_bbox=None if is_reference else ref_pocket_bbox,
                    save_intermediate=save_intermediate,
                )
                if form2 not in ("FAILED",):
                    lig_files[-1] = lf2
                    forms[-1] = form2

            log_message(
                message_type="structure_processed",
                pdb_id=pdb_id,
                status=forms[-1],
            )

        df["ligand_file"] = lig_files
        df["protein_state"] = forms
        df.to_csv(csv_path, index=False)

        # Persist interaction data
        obs_h5 = dirs["metadata"] / "observed_interactions.h5"
        pot_h5 = dirs["metadata"] / "potential_interactions.h5"
        utils.save_to_hdf5_ascii(potential_data, pot_h5, compression=compression)
        utils.save_to_hdf5_ascii(observed_data, obs_h5, compression=compression)

        log_message(message_type="interaction_data_saved")

        try:
            with start_action(action_type="frequency_analysis"):
                utils.calculate_observed_frequencies(str(obs_h5))
                utils.calculate_potential_frequencies(str(pot_h5))
                utils.calculate_binding_site_utilization(str(obs_h5), str(pot_h5))
                log_message(message_type="frequency_analysis_complete")
        except Exception as e:
            log_message(
                message_type="error",
                stage="frequency_analysis",
                error=str(e),
            )

        log_message(
            message_type="stage_complete",
            stage="extract_pockets_and_interactions",
        )


def _configure_logging(cfg: QPocketConfig, log_path: Path) -> None:
    """Sets up Eliot structured logging to file (and optionally stdout).

    Args:
        cfg: QPocket configuration (reads ``cfg.io.log_format``).
        log_path: Path to the log output file.
    """
    log_file = open(str(log_path), "w")

    if cfg.io.log_format == "text":
        from eliot import to_file as _to_file
        from eliot.prettyprint import pretty_print

        # Write JSON to file; pretty-print to stdout
        to_file(log_file)
        add_destinations(pretty_print)
    else:
        to_file(log_file)


def _resolve_reference_file(
    cfg: QPocketConfig,
    df: pd.DataFrame,
    dirs: Dict[str, Path],
) -> Optional[Path]:
    """Returns the aligned PDB path of the reference structure."""
    if cfg.mode == "fragment_screen":
        ref_idx = cfg.fragment_screen.reference_index
        ref_pdb_id = (
            df.iloc[ref_idx]["pdb_id"]
            if ref_idx < len(df)
            else df.iloc[0]["pdb_id"]
        )
    else:
        ref_pdb_id = cfg.reference.pdb_id

    ref_file = dirs["aligned"] / f"{ref_pdb_id}_aligned.pdb"
    return ref_file if ref_file.exists() else None


def _load_fragment_screen_structures(
    cfg: QPocketConfig,
    output_dir: Union[str, Path],
) -> List[str]:
    """Copies/links fragment screening structures into the pipeline directory."""
    import shutil

    output_dir = Path(output_dir)
    structure_paths: List[str] = []

    if cfg.fragment_screen.input_type == "csv":
        if not cfg.fragment_screen.csv_path:
            log_message(
                message_type="error",
                stage="load_fragment_screen",
                error="csv_path not provided",
            )
            return []
        csv_p = Path(cfg.fragment_screen.csv_path)
        if not csv_p.exists():
            log_message(
                message_type="error",
                stage="load_fragment_screen",
                error="CSV file not found",
                path=str(csv_p),
            )
            return []
        df = pd.read_csv(csv_p)
        if "pdb_path" not in df.columns:
            log_message(
                message_type="error",
                stage="load_fragment_screen",
                error="CSV missing 'pdb_path' column",
            )
            return []
        structure_paths = df["pdb_path"].tolist()

    elif cfg.fragment_screen.input_type == "paths":
        if not cfg.fragment_screen.structure_paths:
            log_message(
                message_type="error",
                stage="load_fragment_screen",
                error="structure_paths not provided",
            )
            return []
        structure_paths = cfg.fragment_screen.structure_paths

    structure_ids: List[str] = []
    for i, struct_path in enumerate(structure_paths):
        src = Path(struct_path)
        if not src.exists():
            log_message(
                message_type="warning",
                stage="load_fragment_screen",
                warning=f"structure not found: {src}",
            )
            continue
        struct_id = f"fragment_{i:04d}"
        dest = output_dir / f"{struct_id}{src.suffix}"
        shutil.copy2(src, dest)
        structure_ids.append(struct_id)
        log_message(
            message_type="fragment_loaded",
            index=i,
            source=src.name,
            id=struct_id,
        )

    log_message(
        message_type="fragment_screen_loaded",
        n_structures=len(structure_ids),
    )
    return structure_ids


import numpy as np  # noqa: E402 — needed inside functions above