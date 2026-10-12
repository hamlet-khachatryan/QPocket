"""Working on a finished run: reclustering it and exploring it."""
from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command("recluster")
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

    from qpocket.arms import recluster

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

@click.command("explore")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False), required=False)
@click.option("--port", type=int, default=None, help="Port to serve on.")
@click.option("--headless", is_flag=True, help="Do not open a browser; just print the URL.")
@click.option("--edit", is_flag=True, help="Open the notebook for editing rather than running it.")
def explore_command(run_dir, port, headless, edit):
    """Open the explorer notebook on a run directory, or on a folder of runs."""
    import subprocess

    from qpocket.viz import explorer_path

    command = [sys.executable, "-m", "marimo", "edit" if edit else "run",
               str(explorer_path())]
    if port is not None:
        command += ["-p", str(port)]
    if headless:
        command.append("--headless")
    if run_dir:
        command += ["--", "--runs", str(Path(run_dir).resolve())]

    try:
        raise SystemExit(subprocess.call(command))
    except FileNotFoundError:
        raise click.ClickException(
            "marimo is not installed; install the viz extra with `uv sync --extra viz`"
        )
