"""Running the pipeline, and checking or creating its configuration."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import click

from qpocket.pipelines.run import run_qpocket_pipeline

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration YAML file"
)
@click.option(
    "--mode",
    type=click.Choice(["pdb_query", "fragment_screen"]),
    default=None,
    help="Pipeline mode: pdb_query or fragment_screen"
)
@click.option(
    "--pdb-id",
    help="Reference PDB ID (e.g., 6HP0)"
)
@click.option(
    "--chain-id",
    default=None,
    help="Chain identifier for reference structure"
)
@click.option(
    "--workdir",
    type=click.Path(),
    help="Working directory for output files"
)
@click.option(
    "--pocket-definition",
    type=click.Choice(["Interactions", "Solvent"]),
    default=None,
    help="Pocket representation mode"
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
    "--input-type",
    type=click.Choice(["csv", "paths"]),
    default=None,
    help="Fragment-screen input: a CSV of paths, or an explicit path list.",
)
@click.option(
    "--reference-path",
    type=click.Path(exists=True),
    default=None,
    help="Fragment-screen reference structure (names the run).",
)
@click.option(
    "--reference-index",
    type=int,
    default=None,
    help="Index of reference structure in fragment_screen mode"
)
@click.option(
    "--interaction-padding",
    type=float,
    default=None,
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
    default=None,
    help="Renumber residues to match UniProt (default: on for pdb_query, "
         "off for fragment_screen)"
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
    help="Deprecated alias of --workers for the distance stage only."
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Parallel workers for per-structure work: 0 = half the cores, -1 = all."
)
@click.option(
    "--features",
    "features_list",
    default=None,
    help="Comma-separated feature providers, e.g. apbs,cp_lp,hbond,aromatic."
)
@click.option(
    "--no-apbs",
    is_flag=True,
    default=False,
    help="Drop the apbs provider (skips APBS, and pdb2pqr when nothing else needs it)."
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
    "--shape-weight",
    "--alpha",
    "alpha",
    type=float,
    help="Weight of geometric cost in distance matrix"
)
@click.option(
    "--chem-scale",
    "--beta",
    "beta",
    type=float,
    help="Scaling factor of feature cost in distance matrix"
)
def run(
    config: Optional[str],
    mode: str,
    pdb_id: Optional[str],
    chain_id: str,
    workdir: Optional[str],
    pocket_definition: str,
    csv_path: Optional[str],
    structure_paths: tuple,
    input_type: Optional[str],
    reference_path: Optional[str],
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
    workers: Optional[int],
    features_list: Optional[str],
    no_apbs: bool,
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
    """Run the QPocket pipeline with specified parameters.

    Examples:
    \b
    # PDB query mode
    qpocket run --pdb-id 6HP0 --chain-id A --workdir ./results

    \b
    # Fragment screen mode with CSV
    qpocket run --mode fragment_screen --csv-path structures.csv --workdir ./results

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
    if pocket_definition:
        overrides.append(f"reference.pocket_definition={pocket_definition}")

    if csv_path:
        overrides.append(f"fragment_screen.csv_path={csv_path}")
    if structure_paths:
        paths_str = "[" + ",".join(f'"{p}"' for p in structure_paths) + "]"
        overrides.append(f"fragment_screen.structure_paths={paths_str}")
    if reference_index is not None:
        overrides.append(f"fragment_screen.reference_index={reference_index}")
    if input_type:
        overrides.append(f"fragment_screen.input_type={input_type}")
    if reference_path:
        overrides.append(f"fragment_screen.reference_path={reference_path}")
    if interaction_padding is not None:
        overrides.append(f"reference.interaction_padding={interaction_padding}")

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
    if renumber_residues is not None:
        overrides.append(f"alignment.renumber_residues={renumber_residues}")
    if manual_uniprot_id:
        overrides.append(f"alignment.manual_uniprot_id={manual_uniprot_id}")

    # Cleaning parameters
    if allowed_cofactors:
        overrides.append(f"cleaning.allowed_cofactors={allowed_cofactors}")

    # Distance matrix parameters
    if geo_graph_neighbor_radius is not None:
        overrides.append(f"distance.geo_graph_neighbor_radius={geo_graph_neighbor_radius}")
    if parallel_jobs is not None:
        overrides.append(f"distance.parallel_jobs={parallel_jobs}")
    if workers is not None:
        overrides.append(f"compute.workers={workers}")
    if features_list or no_apbs:
        from qpocket.core.features import DEFAULT_PROVIDERS

        chosen = ([f.strip() for f in features_list.split(",") if f.strip()]
                  if features_list else list(DEFAULT_PROVIDERS))
        if no_apbs:
            chosen = [f for f in chosen if f != "apbs"]
        overrides.append(f"features.enabled=[{','.join(chosen)}]")
    if affinity_damping is not None:
        overrides.append(f"distance.clustering.single.damping={affinity_damping}")
    if affinity_max_iter is not None:
        overrides.append(f"distance.clustering.single.max_iter={affinity_max_iter}")
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
    if log_format:
        overrides.append(f"io.log_format={log_format}")
    if save_intermediate:
        overrides.append("io.save_intermediate=true")
    if compression:
        overrides.append(f"io.compression={compression}")

    # Run pipeline via API
    click.echo("Starting QPocket pipeline...")
    click.echo(f"Mode: {mode or 'from config'}")

    run_qpocket_pipeline(
        qpocket_config_path=config,
        # These were built above and then discarded, which is why --mode and
        # every fragment-screen flag silently did nothing.
        extra_overrides=overrides,
        pdb_id=pdb_id,
        chain_id=chain_id,
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
        parallel_jobs=parallel_jobs,
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


@click.command()
@click.argument("config_file", type=click.Path())
def validate(config_file: str):
    """Validate a configuration file.

    Example:
    \b
    qpocket validate config.yaml
    """
    from qpocket.config import load_qpocket_config

    click.echo(f"Validating configuration file: {config_file}")

    try:
        cfg = load_qpocket_config(config_path=config_file)
        click.echo(" Configuration is valid!")
        click.echo("\nConfiguration summary:")
        click.echo(str(cfg))
    except Exception as e:
        click.echo(f" Configuration validation failed: {e}", err=True)
        raise click.Abort()


@click.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="qpocket_config.yaml",
    help="Where to write the configuration file.",
)
def init(output: str):
    """Write a configuration file containing every setting and its default.

    The file is a copy of the defaults the pipeline actually loads, so it can
    never fall behind them. Edit it and pass it with --config.

    Example:
    \b
    qpocket init
    qpocket run --config qpocket_config.yaml
    """
    from qpocket.config import CONF_PATH

    source = CONF_PATH / "config.yaml"
    if not source.exists():
        click.echo(f"Packaged defaults not found at {source}", err=True)
        raise click.Abort()

    target = Path(output)
    if target.exists():
        click.confirm(f"{target} already exists. Overwrite?", abort=True)

    header = (
        "# QPocket configuration\n"
        "#\n"
        "# Written by `qpocket init` as a copy of the packaged defaults, so every\n"
        "# setting below is the one the pipeline would use anyway. Delete the lines\n"
        "# you do not want to change.\n"
        "#\n"
        "# Required before a run: reference.pdb_id (pdb_query mode) or\n"
        "# fragment_screen.reference_path (fragment_screen mode).\n"
        "\n"
    )
    body = source.read_text(encoding="utf-8")
    # `defaults:` is Hydra's own composition list, meaningless in a user file.
    body = re.sub(r"^defaults:\n(  - .*\n)+\n?", "", body)
    target.write_text(header + body, encoding="utf-8")

    click.echo(f"Created {target}")
    click.echo("Edit it, then run:  qpocket run --config " + str(target))


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

def _load_run(config: Optional[str], workdir: Optional[str], overrides: List[str]):
    """Load a config and its directory map, the way every slurm command needs."""
    from qpocket.config import load_qpocket_config, run_id as api_run_id
    from qpocket.io import layout

    if workdir:
        overrides = list(overrides) + [f"paths.workdir={Path(workdir).resolve()}"]
    cfg = load_qpocket_config(config_path=config, overrides=overrides or None)
    dirs = layout.create(cfg.paths.workdir, api_run_id(cfg))
    return cfg, dirs
