"""Entry point for a full run: options to config, config to the mode's orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from qpocket.config import QPocketConfig
from qpocket.pipelines import fragment_screen, pdb_query
from qpocket.config import load_qpocket_config

#: The orchestrator for each ``mode``.
ORCHESTRATORS = {"pdb_query": pdb_query, "fragment_screen": fragment_screen}


def run(cfg: QPocketConfig) -> Path:
    """Run the pipeline the config's mode names.

    Args:
        cfg: Validated config.

    Returns:
        Run root directory.
    """
    return ORCHESTRATORS[cfg.mode].run(cfg)


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
    renumber_residues: Optional[bool] = None,
    manual_uniprot_id: Optional[str] = None,
    allowed_cofactors: Optional[str] = None,
    geo_graph_neighbor_radius: Optional[float] = None,
    parallel_jobs: Optional[int] = None,
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
    interaction_padding: Optional[float] = None,
    extra_overrides: Optional[List[str]] = None,
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
        parallel_jobs: ``joblib`` parallel worker count (``-1`` = all cores).
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
        interaction_padding: Crop radius (Å) around interacting residues for ``pocket_definition == "Interactions"`` mode.
        extra_overrides: Dot-list overrides applied last.
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
    if renumber_residues is not None:
        overrides.append(f"alignment.renumber_residues={renumber_residues}")
    if manual_uniprot_id:
        overrides.append(f"alignment.manual_uniprot_id={manual_uniprot_id}")
    if allowed_cofactors:
        overrides.append(f"cleaning.allowed_cofactors={allowed_cofactors}")
    if geo_graph_neighbor_radius is not None:
        overrides.append(
            f"spectral_distance_matrix.geo_graph_neighbor_radius={geo_graph_neighbor_radius}"
        )
    if parallel_jobs is not None:
        overrides.append(f"distance.parallel_jobs={parallel_jobs}")
    if global_k is not None:
        overrides.append(f"distance.global_k={global_k}")
    if sigma is not None:
        overrides.append(f"distance.sigma={sigma}")
    if max_points is not None:
        overrides.append(f"distance.max_points={max_points}")
    if alpha is not None:
        overrides.append(f"distance.shape_weight={alpha}")
    if beta is not None:
        overrides.append(f"distance.chem_scale={beta}")
    if affinity_damping is not None:
        overrides.append(f"distance.clustering.single.damping={affinity_damping}")
    if affinity_max_iter is not None:
        overrides.append(f"distance.clustering.single.max_iter={affinity_max_iter}")
    if log_format:
        overrides.append(f"io.log_format={log_format}")
    if save_intermediate is not None:
        overrides.append(f"io.save_intermediate={save_intermediate}")
    if compression:
        overrides.append(f"io.compression={compression}")
    if interaction_padding is not None:
        overrides.append(f"reference.interaction_padding={interaction_padding}")

    if extra_overrides:
        # Applied last: a caller that names something explicitly outranks a
        # keyword-derived default.
        overrides.extend(extra_overrides)

    cfg = load_qpocket_config(config_path=qpocket_config_path, overrides=overrides)
    run(cfg)
