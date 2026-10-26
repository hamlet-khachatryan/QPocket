"""Generating, submitting and following SLURM array jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from qpocket.cli.doctor import doctor
from qpocket.cli.run import _load_run

@click.group()
def slurm():
    """Generate and submit the pipeline as SLURM job arrays.

    Every structure is independent, so the run processes one per array task, and
    a site featurises one cloud per array task — APBS, the costliest step, runs there.
    """
    pass


_SITE_HELP = "A binding site of the run, such as site_01: its stages instead of the run's."


def _check_site(dirs, site: Optional[str]) -> None:
    """Abort unless the run has found this site."""
    if not site:
        return
    from qpocket.arms.sites import SiteError, site_table

    try:
        known = list(site_table(dirs.root)["site_id"])
    except SiteError as e:
        click.echo(f"{e}; run the run stages first", err=True)
        raise click.Abort()
    if site not in known:
        click.echo(f"unknown site {site!r}; known: {', '.join(known) or 'none'}", err=True)
        raise click.Abort()


@slurm.command("generate")
@click.option("--config", "-c", type=click.Path(exists=True), required=True,
              help="Pipeline configuration file.")
@click.option("--workdir", type=click.Path(), default=None,
              help="Override paths.workdir.")
@click.option("--stages", default=None,
              help="Comma-separated subset of stages to generate.")
@click.option("--submit", is_flag=True, default=False,
              help="Also run sbatch and print the job IDs.")
@click.option("--site", default=None, help=_SITE_HELP)
def slurm_generate(config: str, workdir: Optional[str], stages: Optional[str],
                   submit: bool, site: Optional[str]):
    """Write sbatch scripts for every stage, plus a dependency-chained driver.

    Without --site: the run's stages (prepare, structure, sites). With --site,
    once the run has finished: that site's stages (site_featurize, site_cluster).

    Example:
    \b
    qpocket slurm generate --config experiments/PARG.yaml
    bash <workdir>/<PDB>/slurm/submit_all.sh
    qpocket slurm generate --config experiments/PARG.yaml --site site_01
    bash <workdir>/<PDB>/slurm/submit_site_01.sh
    """
    from qpocket.pipelines import slurm as slurm_mod
    from qpocket.pipelines import tasks

    cfg, dirs = _load_run(config, workdir, [])
    _check_site(dirs, site)

    wanted = [s.strip() for s in stages.split(",")] if stages else None
    if site:
        pdb_ids = tasks.manifest_ids(cfg, dirs, "site_featurize", site=site)
        if not pdb_ids:
            click.echo(f"{site} has no clouds to featurise.", err=True)
    else:
        pdb_ids = tasks.manifest_ids(cfg, dirs, "structure")
        if not pdb_ids:
            click.echo(
                "No structures found. The array stage is sized from the ingested "
                "structures, so run the prepare stage first:\n"
                f"  qpocket task prepare --config {config}",
                err=True,
            )

    scripts = slurm_mod.write_scripts(cfg, dirs, Path(config), pdb_ids, stages=wanted, site=site)
    driver = slurm_mod.write_submit_all(cfg, dirs, scripts, site=site)

    click.echo(f"Wrote {sum(len(v) for v in scripts.values())} scripts to "
               f"{slurm_mod.slurm_dir(dirs)}")
    for name, paths in scripts.items():
        for path in paths:
            click.echo(f"  {path.name}")
    click.echo(f"\n{len(pdb_ids)} structures in the array stage.")

    if not submit:
        click.echo(f"\nReview them, then submit:\n  bash {driver}")
        return

    try:
        submitted = slurm_mod.submit(dirs, scripts, dependency=cfg.slurm.dependency, site=site)
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
@click.option("--site", default=None, help=_SITE_HELP)
def slurm_status(config: str, workdir: Optional[str], site: Optional[str]):
    """Report how far each stage has got, judged by the files on disk.

    Artifacts rather than sacct, so it keeps answering after the accounting
    database has rotated the jobs away.
    """
    from qpocket.pipelines import slurm as slurm_mod

    cfg, dirs = _load_run(config, workdir, [])
    _check_site(dirs, site)
    rows = slurm_mod.stage_progress(cfg, dirs, site=site)

    click.echo(f"{'stage':<16}{'done':>12}   missing")
    for row in rows:
        done = f"{row['complete']}/{row['expected']}"
        missing = ", ".join(str(m) for m in row["missing"]) or "-"
        click.echo(f"{row['stage']:<16}{done:>12}   {missing}")


@slurm.command("check")
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Pipeline configuration file.")
@click.pass_context
def slurm_check(ctx, config: Optional[str]):
    """Verify the toolchain on a compute node. An alias for `qpocket doctor`."""
    ctx.invoke(doctor, config=config, smoke=True)
