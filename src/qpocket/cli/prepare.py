"""`qpocket prepare`: turn a dataset download into the inputs of a run."""
from __future__ import annotations

from pathlib import Path

import click


@click.group("prepare")
def prepare_cli() -> None:
    """Turn a dataset download into the inputs of a QPocket run."""


@prepare_cli.command("fragalysis")
@click.argument("download", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--path-prefix", type=click.Path(path_type=Path), default=None,
              help="Where OUT_DIR will live on the machine that runs QPocket (default: OUT_DIR).")
@click.option("--workdir", default=None, help="paths.workdir of the run (default: <prefix>/runs).")
@click.option("--run-name", default=None, help="reference.pdb_id, naming the run (default: the target).")
@click.option("--chain", default="A", show_default=True, help="Chain ID every structure is written under.")
@click.option("--workers", type=int, default=0, show_default=True,
              help="compute.workers: 0 = half the cores, -1 = all.")
@click.option("--cofactors", default=None, help="cleaning.allowed_cofactors, '|'-separated.")
def fragalysis_command(download, out_dir, path_prefix, workdir, run_name, chain, workers, cofactors) -> None:
    """Prepare a Fragalysis DOWNLOAD as a fragment-screen run in OUT_DIR.

    Each aligned observation becomes one structure: the chain its ligand binds, with
    the ligand renamed to a code of its own whose bonds come from its SDF.  OUT_DIR gets
    structures/, ligand_definitions/, inputs.csv, report.csv, qpocket.yaml and run.sh;
    run.sh sets QPOCKET_CCD_CACHE, which every later qpocket command on the run needs.
    """
    from qpocket.io import fragalysis

    opts = fragalysis.Options(path_prefix=path_prefix, workdir=workdir, run_name=run_name,
                              chain=chain, workers=workers, cofactors=cofactors)
    try:
        summary = fragalysis.prepare(download, out_dir, opts)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    for reason, n in summary.skipped.items():
        click.echo(f"skipped {n}: {reason}")
    if not summary.prepared:
        raise click.ClickException(f"no observation could be prepared; see {out_dir / 'report.csv'}")
    resolution = f" ({summary.reference_resolution} Å)" if summary.reference_resolution else ""
    click.echo(f"prepared {summary.prepared} observations, {summary.compounds} compounds; "
               f"reference {summary.reference}{resolution}")
    click.echo(f"run it with {out_dir / 'run.sh'}")
