"""QPocket Pipeline CLI Module."""
from __future__ import annotations

import re
import sys

import click
from pathlib import Path
from typing import Optional, List

from qpocket.pipelines.run import run_qpocket_pipeline


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """QPocket - Protein Binding Pocket Analysis Pipeline.

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
    default=None,
    help="Pocket identification mode"
)
@click.option(
    "--pocket-definition",
    type=click.Choice(["Interactions", "Solvent"]),
    default=None,
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
    ligand_id: Optional[str],
    workdir: Optional[str],
    pocket_mode: str,
    pocket_definition: str,
    bbox_center: tuple,
    bbox_dimensions: tuple,
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


@cli.command()
@click.argument("config_file", type=click.Path())
def validate(config_file: str):
    """Validate a configuration file.

    Example:
    \b
    qpocket validate config.yaml
    """
    from qpocket.pipelines.stages import load_qpocket_config

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
    from qpocket.pipelines.stages import CONF_PATH

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
        "# fragment_screen.reference_path (fragment_screen mode), and\n"
        "# reference.ligand_id when pocket_definition is \"Interactions\".\n"
        "\n"
    )
    body = source.read_text()
    # `defaults:` is Hydra's own composition list, meaningless in a user file.
    body = re.sub(r"^defaults:\n(  - .*\n)+\n?", "", body)
    target.write_text(header + body)

    click.echo(f"Created {target}")
    click.echo("Edit it, then run:  qpocket run --config " + str(target))


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

def _load_run(config: Optional[str], workdir: Optional[str], overrides: List[str]):
    """Load a config and its directory map, the way every slurm command needs."""
    from qpocket.pipelines.stages import load_qpocket_config, run_id as api_run_id
    from qpocket.io import workspace

    if workdir:
        overrides = list(overrides) + [f"paths.workdir={Path(workdir).resolve()}"]
    cfg = load_qpocket_config(config_path=config, overrides=overrides or None)
    dirs = workspace.setup_directory_hierarchy(cfg.paths.workdir, api_run_id(cfg))
    return cfg, dirs


@cli.group()
def slurm():
    """Generate and submit the pipeline as SLURM job arrays.

    APBS dominates the runtime — 216s of a 366s 21-structure run — and every
    structure is independent, so featurisation runs as one array task each.
    """
    pass


@slurm.command("generate")
@click.option("--config", "-c", type=click.Path(exists=True), required=True,
              help="Pipeline configuration file.")
@click.option("--workdir", type=click.Path(), default=None,
              help="Override paths.workdir.")
@click.option("--stages", default=None,
              help="Comma-separated subset of stages to generate.")
@click.option("--submit", is_flag=True, default=False,
              help="Also run sbatch and print the job IDs.")
def slurm_generate(config: str, workdir: Optional[str], stages: Optional[str],
                   submit: bool):
    """Write sbatch scripts for every stage, plus a dependency-chained driver.

    Example:
    \b
    qpocket slurm generate --config experiments/PARG.yaml
    bash <workdir>/<PDB>/slurm/submit_all.sh
    """
    from qpocket.pipelines import slurm as slurm_mod
    from qpocket.pipelines import tasks

    cfg, dirs = _load_run(config, workdir, [])

    if getattr(cfg.slurm, "split_apbs_stage", False):
        click.echo(
            "slurm.split_apbs_stage is not implemented yet (it needs a per-structure "
            "APBS grid cache). Set it to false — APBS already runs as one task per "
            "structure inside the featurize stage.",
            err=True,
        )
        raise click.Abort()

    wanted = [s.strip() for s in stages.split(",")] if stages else None
    pdb_ids = tasks.manifest_ids(cfg, dirs, "featurize")
    if not pdb_ids:
        click.echo(
            "No structures found. The array stages are sized from master.csv, so "
            "run the prepare stage first:\n"
            f"  qpocket task prepare --config {config}",
            err=True,
        )

    scripts = slurm_mod.write_scripts(cfg, dirs, Path(config), pdb_ids, stages=wanted)
    driver = slurm_mod.write_submit_all(cfg, dirs, scripts)

    click.echo(f"Wrote {sum(len(v) for v in scripts.values())} scripts to "
               f"{slurm_mod.slurm_dir(dirs)}")
    for name, paths in scripts.items():
        for path in paths:
            click.echo(f"  {path.name}")
    click.echo(f"\n{len(pdb_ids)} structures in the array stages.")

    if not submit:
        click.echo(f"\nReview them, then submit:\n  bash {driver}")
        return

    try:
        submitted = slurm_mod.submit(dirs, scripts, dependency=cfg.slurm.dependency)
    except RuntimeError as e:
        click.echo(f"Submission failed: {e}", err=True)
        raise click.Abort()
    for name, ids in submitted.items():
        click.echo(f"  {name}: {', '.join(ids) if ids else '(nothing to submit)'}")


@slurm.command("status")
@click.option("--config", "-c", type=click.Path(exists=True), required=True,
              help="Pipeline configuration file.")
@click.option("--workdir", type=click.Path(), default=None,
              help="Override paths.workdir.")
def slurm_status(config: str, workdir: Optional[str]):
    """Report how far each stage has got, judged by the files on disk.

    Artifacts rather than sacct, so it keeps answering after the accounting
    database has rotated the jobs away.
    """
    from qpocket.pipelines import slurm as slurm_mod

    cfg, dirs = _load_run(config, workdir, [])
    rows = slurm_mod.stage_progress(cfg, dirs)

    click.echo(f"{'stage':<12}{'done':>12}   missing")
    for row in rows:
        done = f"{row['complete']}/{row['expected']}"
        missing = ", ".join(str(m) for m in row["missing"]) or "-"
        click.echo(f"{row['stage']:<12}{done:>12}   {missing}")


@slurm.command("check")
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Pipeline configuration file.")
@click.pass_context
def slurm_check(ctx, config: Optional[str]):
    """Verify the toolchain on a compute node. An alias for `qpocket doctor`."""
    ctx.invoke(doctor, config=config, smoke=True)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Also check that this configuration loads.")
@click.option("--smoke/--no-smoke", default=True,
              help="Run APBS on a tiny synthetic system (a few seconds).")
def doctor(config: Optional[str], smoke: bool):
    """Check that this machine can actually run the pipeline.

    Reports the platform, every import, the external tools, and -- unless
    --no-smoke -- runs APBS for real, which is the check that matters on Linux:
    apbs-binary only patches its library path on macOS, so a wheel that imports
    is not proof of a binary that runs.

    Run it on a compute node, not the login node; they are often different.

    Example:
    \b
    qpocket doctor
    srun -p compute qpocket doctor      # where the work will actually happen
    """
    import importlib
    import platform
    import shutil

    failures: List[str] = []

    def line(status: str, label: str, detail: str = "") -> None:
        click.echo(f"  {status:<6} {label:<22} {detail}")

    def ok(label, detail=""):
        line("ok", label, detail)

    def bad(label, detail=""):
        line("FAIL", label, detail)
        failures.append(label)

    click.echo("Q-Pocket doctor\n")

    # --- platform ---------------------------------------------------------
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from qpocket_doctor import fmt, glibc_version  # type: ignore

        libc = f", glibc {fmt(glibc_version())}" if platform.system() == "Linux" else ""
    except Exception:
        libc = ""
    ok("platform", f"{platform.system()} {platform.machine()}{libc}")
    ok("python", f"{platform.python_version()} at {sys.executable}")

    # --- imports ----------------------------------------------------------
    for module, why in (
        ("numpy", ""), ("scipy", ""), ("pandas", ""), ("sklearn", ""),
        ("h5py", ""), ("joblib", ""), ("hydra", ""), ("omegaconf", ""),
        ("eliot", ""), ("Bio", "biopython"), ("biopandas", ""), ("rdkit", ""),
        ("openbabel", ""), ("pdb2pqr", ""),
        ("apbs_binary", "featurisation"),
    ):
        try:
            importlib.import_module(module)
            ok(f"import {module}", why)
        except Exception as e:
            bad(f"import {module}", f"{type(e).__name__}: {e}")

    # --- external tools ---------------------------------------------------
    for tool, required in (("pdb2pqr", True), ("sbatch", False)):
        found = shutil.which(tool)
        if found:
            ok(f"{tool} on PATH", found)
        elif required:
            bad(f"{tool} on PATH", "not found")
        else:
            line("warn", f"{tool} on PATH", "not found (only needed for `qpocket slurm`)")

    # --- APBS actually running -------------------------------------------
    if smoke:
        from qpocket import doctor as checks

        for label, probe in (
            ("pdb2pqr runs", checks.run_pdb2pqr_smoke_test),
            ("apbs runs", checks.run_apbs_smoke_test),
        ):
            try:
                seconds, detail = probe()
                ok(label, f"{detail} in {seconds:.1f}s")
            except Exception as e:
                bad(label, f"{type(e).__name__}: {e}")

    # --- the configuration ------------------------------------------------
    if config:
        try:
            from qpocket.pipelines.stages import load_qpocket_config

            load_qpocket_config(config_path=config)
            ok("config loads", config)
        except Exception as e:
            bad("config loads", f"{type(e).__name__}: {e}")

    click.echo()
    if failures:
        click.echo(f"{len(failures)} check(s) failed: {', '.join(failures)}", err=True)
        raise click.Abort()
    click.echo("This node can run the pipeline.")


# ---------------------------------------------------------------------------
# Array-task entry point (invoked by the generated sbatch scripts)
# ---------------------------------------------------------------------------

@cli.group(hidden=True)
def task():
    """Run one pipeline stage. Invoked by the generated sbatch scripts."""
    pass


def _make_task_command(name: str):
    @click.option("--config", "-c", type=click.Path(exists=True), default=None,
                  help="Pipeline configuration file.")
    @click.option("--workdir", type=click.Path(), default=None,
                  help="Override paths.workdir.")
    @click.option("--manifest", type=click.Path(), default=None,
                  help="Manifest of PDB IDs, for a per-structure stage.")
    @click.option("--index", type=int, default=None,
                  help="Row of the manifest, normally SLURM_ARRAY_TASK_ID.")
    @click.option("--force", is_flag=True, default=False,
                  help="Recompute even if the output already exists.")
    def command(config, workdir, manifest, index, force):
        from qpocket.pipelines import tasks
        from qpocket.pipelines.stages import _configure_logging

        cfg, dirs = _load_run(config, workdir, [])

        # One log per task: the pipeline log is opened truncating, so a shared
        # path would have every array task erase the others.
        suffix = f"_{index}" if index is not None else ""
        log_dir = Path(dirs["root"]) / "slurm" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _configure_logging(cfg, log_dir / f"{name}{suffix}.log")

        try:
            tasks.run_task(name, cfg, dirs,
                           manifest=Path(manifest) if manifest else None,
                           index=index, force=force)
        except tasks.TaskError as e:
            click.echo(f"[{name}] {e}", err=True)
            raise SystemExit(1)
        click.echo(f"[{name}] done")

    command.__name__ = f"task_{name}"
    command.__doc__ = f"Run the {name} stage."
    return command


for _name in ("prepare", "apbs", "featurize", "collect", "geodesic", "distance"):
    task.command(_name)(_make_task_command(_name))


from qpocket.cli.compare import (  # noqa: E402  (registered after the group exists)
    compare_command,
    compare_structures_command,
    featurize_command,
)

from qpocket.cli.screen import screen_cli  # noqa: E402


@cli.command("recluster")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--mode", type=click.Choice(["two_stage", "single"]), default=None)
@click.option("--min-cluster-size", type=int, default=None, help="Shape clusters smaller than this are not split.")
@click.option("--shape-method", type=click.Choice(["hierarchical", "kmedoids", "affinity"]), default=None)
@click.option("--shape-linkage", type=click.Choice(["average", "complete", "single", "weighted"]), default=None)
@click.option("--shape-k", type=int, default=None, help="Pin the number of shape clusters.")
@click.option("--shape-threshold", type=float, default=None, help="Cut the shape tree at this height.")
@click.option("--chem-method", type=click.Choice(["hierarchical", "kmedoids", "affinity"]), default=None)
@click.option("--chem-linkage", type=click.Choice(["average", "complete", "single", "weighted"]), default=None)
@click.option("--chem-k", type=int, default=None, help="Pin the number of chemistry sub-clusters.")
@click.option("--chem-threshold", type=float, default=None, help="Cut each chemistry tree at this height.")
def recluster_command(run_dir, mode, min_cluster_size, shape_method, shape_linkage, shape_k, shape_threshold,
                      chem_method, chem_linkage, chem_k, chem_threshold):
    """Recluster a finished run, starting from its own settings; the old clustering is archived.

    \b
    qpocket recluster runs/7KG6 --shape-threshold 0.012 --chem-threshold 40
    """
    from dataclasses import replace

    import pandas as pd

    from qpocket.pipelines import recluster

    settings = recluster.load_settings(Path(run_dir))

    def stage(current, method, linkage, k, threshold):
        changes = {}
        if method:
            changes["method"] = method
        if linkage:
            changes["linkage"] = linkage
        if k is not None:
            changes.update(k=k, criterion="count", threshold=None)
        if threshold is not None:
            changes.update(threshold=threshold, criterion="threshold", method="hierarchical")
        return replace(current, **changes)

    settings = replace(
        settings,
        mode=mode or settings.mode,
        min_cluster_size=min_cluster_size or settings.min_cluster_size,
        shape=stage(settings.shape, shape_method, shape_linkage, shape_k, shape_threshold),
        chem=stage(settings.chem, chem_method, chem_linkage, chem_k, chem_threshold),
    )
    try:
        out = recluster.run(Path(run_dir), settings)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    labels = pd.read_csv(out)
    click.echo(f"{labels['cluster_id'].nunique()} clusters "
               f"({labels['shape_cluster_id'].nunique()} shape groups) -> {out}")

for _command in (featurize_command, compare_command, compare_structures_command):
    cli.add_command(_command)
cli.add_command(screen_cli, name="screen")
