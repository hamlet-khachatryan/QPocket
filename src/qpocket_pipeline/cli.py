"""
QPocket Pipeline CLI Module

Provides command-line interface for the QPocket pipeline.
"""
from __future__ import annotations

import click
from pathlib import Path
from typing import Optional, List

from qpocket_pipeline.api import run_qpocket_pipeline


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    QPocket - Protein Binding Pocket Analysis Pipeline

    A framework for analysis and benchmarking of experimentally observed
    conformational space of protein binding pockets for drug design applications.
    """
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file"
)
@click.option(
    "--mode",
    type=click.Choice(["pdb_query", "fragment_screen"]),
    default="pdb_query",
    help="Pipeline mode: pdb_query or fragment_screen"
)
@click.option(
    "--pdb-id",
    help="Reference PDB ID (e.g., 6HP0)"
)
@click.option(
    "--chain-id",
    default="A",
    help="Chain identifier for reference structure"
)
@click.option(
    "--ligand-id",
    help="Ligand identifier in reference structure"
)
@click.option(
    "--workdir",
    type=click.Path(),
    help="Working directory for output files"
)
@click.option(
    "--pocket-mode",
    type=click.Choice(["cavity_detection", "bounding_box"]),
    default="cavity_detection",
    help="Pocket identification mode"
)
@click.option(
    "--pocket-definition",
    type=click.Choice(["Interactions", "Solvent"]),
    default="Solvent",
    help="Pocket representation mode"
)
@click.option(
    "--bbox-center",
    type=float,
    multiple=True,
    help="Bounding box center coordinates (x y z)"
)
@click.option(
    "--bbox-dimensions",
    type=float,
    multiple=True,
    help="Bounding box dimensions (dx dy dz)"
)
@click.option(
    "--csv-path",
    type=click.Path(exists=True),
    help="CSV file with structure paths (fragment_screen mode)"
)
@click.option(
    "--structure-paths",
    multiple=True,
    help="List of structure file paths (fragment_screen mode)"
)
@click.option(
    "--reference-index",
    type=int,
    default=0,
    help="Index of reference structure in fragment_screen mode"
)
@click.option(
    "--interaction-padding",
    type=float,
    default=5.0,
    help="Padding around interacting residues (Angstroms)"
)
@click.option(
    "--sequence-identity",
    type=float,
    help="Minimum sequence identity for PDB query"
)
@click.option(
    "--resolution-limit",
    type=float,
    help="Maximum resolution limit for PDB query"
)
@click.option(
    "--min-length",
    type=int,
    help="Minimum sequence length for PDB query"
)
@click.option(
    "--max-length",
    type=int,
    help="Maximum sequence length for PDB query"
)
@click.option(
    "--experimental-method",
    help="Experimental method filter (e.g., X-RAY DIFFRACTION)"
)
@click.option(
    "--enforce-same-uniprot",
    is_flag=True,
    help="Fetch only structures with same UniProt ID"
)
@click.option(
    "--renumber-residues/--no-renumber-residues",
    default=True,
    help="Renumber residues to match UniProt"
)
@click.option(
    "--manual-uniprot-id",
    help="Force specific UniProt ID for alignment"
)
@click.option(
    "--allowed-cofactors",
    help="Pipe-separated list of allowed cofactor names"
)
@click.option(
    "--geo-graph-neighbor-radius",
    type=float,
    help="Neighbor radius for geometric graph"
)
@click.option(
    "--parallel-jobs",
    type=int,
    help="Number of parallel jobs (-1 for all cores)"
)
@click.option(
    "--affinity-damping",
    type=float,
    help="Affinity Propagation damping factor [0.5, 1.0)"
)
@click.option(
    "--affinity-max-iter",
    type=int,
    help="Maximum Affinity Propagation iterations"
)
@click.option(
    "--log-format",
    type=click.Choice(["json", "text"]),
    help="Eliot log format"
)
@click.option(
    "--save-intermediate",
    is_flag=True,
    help="Save intermediate (pre-crop) pocket point clouds"
)
@click.option(
    "--compression",
    type=click.Choice(["gzip", "none"]),
    help="HDF5 compression for interaction data"
)
@click.option(
    "--global-k",
    type=int,
    help="Global length of spectral fingerprint"
)
@click.option(
    "--sigma",
    type=float,
    help="Standard deviation of Heat Gaussian kernel"
)
@click.option(
    "--max-points",
    type=int,
    help="Maximum number of points for spectral clustering"
)
@click.option(
    "--alpha",
    type=float,
    help="Weight of geometric cost in distance matrix"
)
@click.option(
    "--beta",
    type=float,
    help="Scaling factor of feature cost in distance matrix"
)
def run(
    config: Optional[str],
    mode: str,
    pdb_id: Optional[str],
    chain_id: str,
    ligand_id: Optional[str],
    workdir: Optional[str],
    pocket_mode: str,
    pocket_definition: str,
    bbox_center: tuple,
    bbox_dimensions: tuple,
    csv_path: Optional[str],
    structure_paths: tuple,
    reference_index: int,
    interaction_padding: float,
    sequence_identity: Optional[float],
    resolution_limit: Optional[float],
    min_length: Optional[int],
    max_length: Optional[int],
    experimental_method: Optional[str],
    enforce_same_uniprot: bool,
    renumber_residues: bool,
    manual_uniprot_id: Optional[str],
    allowed_cofactors: Optional[str],
    geo_graph_neighbor_radius: Optional[float],
    parallel_jobs: Optional[int],
    affinity_damping: Optional[float],
    affinity_max_iter: Optional[int],
    global_k: Optional[int],
    sigma: Optional[float],
    max_points: Optional[int],
    alpha: Optional[float],
    beta: Optional[float],
    log_format: Optional[str],
    save_intermediate: bool,
    compression: Optional[str],
):
    """
    Run the QPocket pipeline with specified parameters.

    Examples:

    \b
    # PDB query mode
    qpocket run --pdb-id 6HP0 --chain-id A --ligand-id LKQ --workdir ./results

    \b
    # Fragment screen mode with CSV
    qpocket run --mode fragment_screen --csv-path structures.csv --workdir ./results

    \b
    # With bounding box pocket definition
    qpocket run --pdb-id 6HP0 --pocket-mode bounding_box --bbox-center 10.0 20.0 30.0 --bbox-dimensions 15.0 15.0 15.0

    \b
    # With interaction-based pocket representation
    qpocket run --pdb-id 6HP0 --pocket-definition Interactions --interaction-padding 7.0
    """
    # Build overrides list
    overrides = []

    if mode:
        overrides.append(f"mode={mode}")
    if pdb_id:
        overrides.append(f"reference.pdb_id={pdb_id}")
    if chain_id:
        overrides.append(f"reference.chain_id={chain_id}")
    if ligand_id:
        overrides.append(f"reference.ligand_id={ligand_id}")
    if pocket_mode:
        overrides.append(f"reference.pocket_mode={pocket_mode}")
    if pocket_definition:
        overrides.append(f"reference.pocket_definition={pocket_definition}")

    if len(bbox_center) == 3:
        overrides.append(f"reference.bounding_box.center=[{bbox_center[0]},{bbox_center[1]},{bbox_center[2]}]")
    if len(bbox_dimensions) == 3:
        overrides.append(f"reference.bounding_box.dimensions=[{bbox_dimensions[0]},{bbox_dimensions[1]},{bbox_dimensions[2]}]")

    if csv_path:
        overrides.append(f"fragment_screen.csv_path={csv_path}")
    if structure_paths:
        paths_str = "[" + ",".join(f'"{p}"' for p in structure_paths) + "]"
        overrides.append(f"fragment_screen.structure_paths={paths_str}")
    if reference_index is not None:
        overrides.append(f"fragment_screen.reference_index={reference_index}")
    if interaction_padding:
        overrides.append(f"fragment_screen.interaction_padding={interaction_padding}")

    if workdir:
        overrides.append(f"paths.workdir={workdir}")

    # PDB query parameters
    if sequence_identity is not None:
        overrides.append(f"pdb_query.sequence_identity={sequence_identity}")
    if resolution_limit is not None:
        overrides.append(f"pdb_query.resolution_limit={resolution_limit}")
    if min_length is not None:
        overrides.append(f"pdb_query.min_length={min_length}")
    if max_length is not None:
        overrides.append(f"pdb_query.max_length={max_length}")
    if experimental_method:
        overrides.append(f"pdb_query.experimental_method={experimental_method}")
    if enforce_same_uniprot:
        overrides.append(f"pdb_query.enforce_same_uniprot=true")

    # Alignment parameters
    overrides.append(f"alignment.renumber_residues={renumber_residues}")
    if manual_uniprot_id:
        overrides.append(f"alignment.manual_uniprot_id={manual_uniprot_id}")

    # Cleaning parameters
    if allowed_cofactors:
        overrides.append(f"cleaning.allowed_cofactors={allowed_cofactors}")

    # Distance matrix parameters
    if geo_graph_neighbor_radius is not None:
        overrides.append(f"spectral_distance_matrix.geo_graph_neighbor_radius={geo_graph_neighbor_radius}")
    if parallel_jobs is not None:
        overrides.append(f"spectral_distance_matrix.paralell_jobs={parallel_jobs}")
    if affinity_damping is not None:
        overrides.append(f"spectral_distance_matrix.affinity_damping={affinity_damping}")
    if affinity_max_iter is not None:
        overrides.append(f"spectral_distance_matrix.affinity_max_iter={affinity_max_iter}")
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
    if log_format:
        overrides.append(f"io.log_format={log_format}")
    if save_intermediate:
        overrides.append("io.save_intermediate=true")
    if compression:
        overrides.append(f"io.compression={compression}")

    # Run pipeline via API
    click.echo("Starting QPocket pipeline...")
    click.echo(f"Mode: {mode}")

    run_qpocket_pipeline(
        qpocket_config_path=config,
        pdb_id=pdb_id,
        chain_id=chain_id,
        ligand_id=ligand_id,
        workdir=workdir,
        sequence_identity=sequence_identity,
        resolution_limit=resolution_limit,
        min_length=min_length,
        max_length=max_length,
        experimental_method=experimental_method,
        enforce_same_uniprot=enforce_same_uniprot,
        renumber_residues=renumber_residues,
        manual_uniprot_id=manual_uniprot_id,
        allowed_cofactors=allowed_cofactors,
        geo_graph_neighbor_radius=geo_graph_neighbor_radius,
        paralell_jobs=parallel_jobs,
        affinity_damping=affinity_damping,
        affinity_max_iter=affinity_max_iter,
        global_k=global_k,
        sigma=sigma,
        max_points=max_points,
        alpha=alpha,
        beta=beta,
        log_format=log_format,
        save_intermediate=save_intermediate or None,
        compression=compression,
    )

    click.echo(" Pipeline completed successfully!")


@cli.command()
@click.argument("config_file", type=click.Path())
def validate(config_file: str):
    """
    Validate a configuration file.

    Example:

    \b
    qpocket validate config.yaml
    """
    from qpocket_pipeline.api import load_qpocket_config

    click.echo(f"Validating configuration file: {config_file}")

    try:
        cfg = load_qpocket_config(config_path=config_file)
        click.echo(" Configuration is valid!")
        click.echo("\nConfiguration summary:")
        click.echo(str(cfg))
    except Exception as e:
        click.echo(f" Configuration validation failed: {e}", err=True)
        raise click.Abort()


@cli.command()
def init():
    """
    Create a template configuration file in the current directory.

    Example:

    \b
    qpocket init
    """
    template = """# QPocket Configuration File
# Generated by: qpocket init

mode: "pdb_query"  # "pdb_query" or "fragment_screen"

reference:
  pdb_id: "6HP0"
  chain_id: "A"
  ligand_id: "LKQ"
  pocket_definition: "Solvent"   # "Interactions" or "Solvent"
  pocket_mode: "cavity_detection"  # "cavity_detection" or "bounding_box"
  bounding_box:
    center: null    # [x, y, z]
    dimensions: null  # [dx, dy, dz]

fragment_screen:
  input_type: "csv"   # "csv" or "paths"
  csv_path: null
  structure_paths: null
  reference_index: 0
  interaction_padding: 5.0

paths:
  workdir: "./results"

pdb_query:
  sequence_identity: 1.0
  resolution_limit: 2.5
  min_length: 20
  max_length: null
  experimental_method: null
  enforce_same_uniprot: false

alignment:
  renumber_residues: true
  manual_uniprot_id: null

cleaning:
  allowed_cofactors: null  # e.g., "HEM|MG|CL"

io:
  log_format: "json"        # "json" or "text"
  save_intermediate: false  # save pre-crop pocket CSV
  compression: "gzip"       # "gzip" or "none"

spectral_distance_matrix:
  geo_graph_neighbor_radius: 2.0
  paralell_jobs: -1
  global_k: 100
  sigma: 10.0
  max_points: 5000
  alpha: 0.85
  beta: 12.5
  affinity_damping: 0.9
  affinity_max_iter: 300
"""

    config_path = Path("qpocket_config.yaml")
    if config_path.exists():
        click.confirm(f"{config_path} already exists. Overwrite?", abort=True)

    config_path.write_text(template)
    click.echo(f" Created configuration template: {config_path}")


if __name__ == "__main__":
    cli()
