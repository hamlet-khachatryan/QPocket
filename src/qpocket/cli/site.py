"""`qpocket site`: per-binding-site steps on a finished run."""
from __future__ import annotations

import functools

import click
import pandas as pd

from qpocket.arms import sites as steps

_RUN_DIR = click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
_SITE = click.option("--site", "site_id", required=True, help="Site id, e.g. site_01.")


def _reported(func):
    """Turn the step's own SiteError into a one-line click error; other exceptions keep their traceback."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except steps.SiteError as exc:
            raise click.ClickException(str(exc))
    return wrapper


@click.group("site")
def site_cli() -> None:
    """Per-binding-site steps on a finished run."""


@site_cli.command("list")
@_RUN_DIR
@_reported
def list_command(run_dir: str) -> None:
    """Print the run's sites and which steps each has done."""
    table = steps.site_table(run_dir)
    status = pd.DataFrame([steps.site_status(run_dir, s) for s in table["site_id"]], index=table.index)
    click.echo(pd.concat([table, status], axis=1).to_string(index=False))


@site_cli.command("featurize")
@_RUN_DIR
@_SITE
@click.option("--workers", type=int, default=None, help="Parallel workers (default: the run's config).")
@_reported
def featurize_command(run_dir: str, site_id: str, workers) -> None:
    """Featurise every pocket of one site."""
    done = steps.featurize_site(run_dir, site_id, workers)
    click.echo(f"{site_id}: featurised {len(done)} pockets")


@site_cli.command("shape")
@_RUN_DIR
@_SITE
@_reported
def shape_command(run_dir: str, site_id: str) -> None:
    """Shape-only clustering of one site."""
    click.echo(f"{site_id}: wrote {steps.shape_cluster_site(run_dir, site_id)}")


@site_cli.command("cluster")
@_RUN_DIR
@_SITE
@_reported
def cluster_command(run_dir: str, site_id: str) -> None:
    """Two-stage (shape then chemistry) clustering of one site."""
    click.echo(f"{site_id}: wrote {steps.two_stage_site(run_dir, site_id)}")


@site_cli.command("regroup")
@_RUN_DIR
@click.option("--cutoff", "link_cutoff", type=float, required=True,
              help="Single-linkage heavy-atom distance in Å for joining ligands into a site.")
@_reported
def regroup_command(run_dir: str, link_cutoff: float) -> None:
    """Regroup ligands into sites at a new cutoff; unchanged sites keep their work."""
    click.echo(steps.regroup_sites(run_dir, link_cutoff).to_string(index=False))
