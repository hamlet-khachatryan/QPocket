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
@click.option("--cavity-overlap", type=float, help="Box IoU two cavities (or two sites) need to be grouped.")
@click.option("--ligand-fraction", type=float, help="Share of a ligand's heavy atoms inside a site box.")
@click.option("--box-padding", type=float, help="Å added to every face of a site box when assigning ligands.")
@click.option("--min-cavity-points", type=int, help="Smaller CavFind cavities never form or join a site.")
@click.option("--min-structure-fraction", type=float, help="Share of structures a site without ligands needs.")
@_reported
def regroup_command(run_dir: str, **params) -> None:
    """Rebuild the sites with new parameters; sites whose cavities are unchanged keep their work."""
    given = {key: value for key, value in params.items() if value is not None}
    if not given:
        raise click.UsageError("give at least one of --cavity-overlap, --ligand-fraction, --box-padding, "
                               "--min-cavity-points, --min-structure-fraction")
    click.echo(steps.regroup_sites(run_dir, **given).to_string(index=False))
