"""CLI entry point for QPocket-Screen.

Usage:
    qpocket screen prepare --workdir <run> [--config screen.yaml] [--submit slurm|local]
    qpocket screen run-local <run>/screening/docking_jobs.csv --workers 4
    qpocket screen analyze  --workdir <run> [--config screen.yaml]

``qpocket-screen`` is the same group under its old name.
"""
from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="QPocket")
def screen_cli() -> None:
    """QPocket virtual screening workflow."""


@screen_cli.command("prepare")
@click.option("--workdir", required=True, help="Pipeline output directory.")
@click.option("--config", "config_path", default=None, help="Path to screen_config.yaml.")
@click.option(
    "--run-mode",
    default="cross_docking",
    type=click.Choice(["cross_docking", "virtual_screening"]),
    show_default=True,
    help="Screening mode.",
)
@click.option(
    "--engine",
    default="vina",
    type=click.Choice(["vina", "quickvina"]),
    show_default=True,
    help="Docking engine.",
)
@click.option("--engine-path", default=None, help="Path/name of the docking engine binary.")
@click.option("--exhaustiveness", default=None, type=int, help="Vina exhaustiveness.")
@click.option("--num-modes", default=None, type=int, help="Number of binding modes per run.")
@click.option("--cpus", default=None, type=int, help="CPUs per docking job.")
@click.option("--use-slurm", is_flag=True, default=False, help="Same as --submit slurm.")
@click.option("--submit", type=click.Choice(["slurm", "local"]), default=None,
              help="Write SLURM array scripts or a local run command (default: slurm.enabled).")
@click.option("--slurm-partition", default=None, help="SLURM partition name.")
@click.option("--slurm-time", default=None, help="SLURM time limit (e.g. 02:00:00).")
@click.option("--box-mode", default=None, type=click.Choice(["ligand_based", "cavity_based"]))
@click.option("--box-padding", default=None, type=float, help="Box padding in Å.")
def prepare(
    workdir: str,
    config_path: str | None,
    run_mode: str,
    engine: str,
    engine_path: str | None,
    exhaustiveness: int | None,
    num_modes: int | None,
    cpus: int | None,
    use_slurm: bool,
    submit: str | None,
    slurm_partition: str | None,
    slurm_time: str | None,
    box_mode: str | None,
    box_padding: float | None,
) -> None:
    """Prepare proteins, ligands, and docking job scripts."""
    from qpocket.pipelines.screening import prepare as prepare_campaign

    overrides: dict = {"run_mode": run_mode, "docking": {"engine": engine}}

    if engine_path:
        overrides.setdefault("docking", {})["engine_path"] = engine_path
    if exhaustiveness is not None:
        overrides.setdefault("docking", {})["exhaustiveness"] = exhaustiveness
    if num_modes is not None:
        overrides.setdefault("docking", {})["num_modes"] = num_modes
    if cpus is not None:
        overrides.setdefault("docking", {})["cpus"] = cpus
    if use_slurm:
        overrides.setdefault("slurm", {})["enabled"] = True
    if slurm_partition:
        overrides.setdefault("slurm", {})["partition"] = slurm_partition
    if slurm_time:
        overrides.setdefault("slurm", {})["time"] = slurm_time
    if box_mode:
        overrides.setdefault("box", {})["mode"] = box_mode
    if box_padding is not None:
        overrides.setdefault("box", {})["padding"] = box_padding

    # Known issue (spec §6): these defaults always override the config file.
    plan = prepare_campaign(workdir, config_path, submit=submit or ("slurm" if use_slurm else None), **overrides)
    click.echo(f"\nTask list: {plan.jobs_csv}")
    for script in plan.scripts:
        click.echo(f"Script:    {script}")
    click.echo(f"\nNext:\n  {plan.command.replace(chr(10), chr(10) + '  ')}")


@screen_cli.command("analyze")
@click.option("--workdir", required=True, help="Pipeline output directory.")
@click.option("--config", "config_path", default=None, help="Path to screen_config.yaml.")
@click.option(
    "--run-mode",
    default=None,
    type=click.Choice(["cross_docking", "virtual_screening"]),
    help="Screening mode (overrides config).",
)
@click.option(
    "--rmsd-threshold",
    default=None,
    type=float,
    help="RMSD threshold for docking success (Å).",
)
@click.option("--bedroc-alpha", default=None, type=float, help="BEDROC alpha parameter.")
def analyze(
    workdir: str,
    config_path: str | None,
    run_mode: str | None,
    rmsd_threshold: float | None,
    bedroc_alpha: float | None,
) -> None:
    """Analyse docking results and write CSV reports."""
    from qpocket.pipelines.screening import analyze as analyze_campaign

    overrides: dict = {}
    if run_mode:
        overrides["run_mode"] = run_mode
    if rmsd_threshold is not None:
        overrides.setdefault("analysis", {})["rmsd_threshold"] = rmsd_threshold
    if bedroc_alpha is not None:
        overrides.setdefault("analysis", {})["bedroc_alpha"] = bedroc_alpha

    click.echo(f"Reports written to: {analyze_campaign(workdir, config_path, **overrides)}")


@screen_cli.command("run-task")
@click.argument("jobs_csv", type=click.Path(exists=True))
@click.option("--index", type=int, required=True, help="0-based row of the task list.")
@click.option("--cpus", type=int, default=1, show_default=True, help="Threads for the engine.")
@click.option("--engine", default=None, help="Docking executable (default: config engine_path).")
def run_task(jobs_csv: str, index: int, cpus: int, engine: str | None) -> None:
    """Run one docking job; used by the SLURM array script."""
    from pathlib import Path

    from qpocket.pipelines.screening import run_job

    code = run_job(Path(jobs_csv), index, cpus, engine)
    if code:
        raise SystemExit(code)


@screen_cli.command("run-local")
@click.argument("jobs_csv", type=click.Path(exists=True))
@click.option("--workers", type=int, default=1, show_default=True, help="Jobs run at once.")
@click.option("--cpus", type=int, default=1, show_default=True, help="Threads per job.")
@click.option("--engine", default=None, help="Docking executable (default: config engine_path).")
def run_local(jobs_csv: str, workers: int, cpus: int, engine: str | None) -> None:
    """Run every docking job on this machine (any OS)."""
    from pathlib import Path

    from qpocket.pipelines.screening import run_local as run_all

    failed = run_all(Path(jobs_csv), workers, cpus, engine)
    click.echo(f"{failed} job(s) failed" if failed else "All docking jobs completed.")
    if failed:
        raise SystemExit(1)
