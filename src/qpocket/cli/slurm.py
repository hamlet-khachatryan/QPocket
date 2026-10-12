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
