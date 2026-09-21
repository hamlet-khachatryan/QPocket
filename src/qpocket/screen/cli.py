"""CLI entry point for QPocket-Screen.

Usage:
    qpocket-screen prepare --workdir <pipeline_outdir> [--config screen.yaml]
    qpocket-screen analyze  --workdir <pipeline_outdir> [--config screen.yaml]
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
@click.option("--use-slurm", is_flag=True, default=False, help="Enable SLURM job array script.")
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
    slurm_partition: str | None,
    slurm_time: str | None,
    box_mode: str | None,
    box_padding: float | None,
) -> None:
    """Prepare proteins, ligands, and docking job scripts."""
    from qpocket.screen.api import run_qpocket_screen_prepare

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

    run_qpocket_screen_prepare(workdir, config_path, **overrides)


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
    from qpocket.screen.api import run_qpocket_screen_analyze

    overrides: dict = {}
    if run_mode:
        overrides["run_mode"] = run_mode
    if rmsd_threshold is not None:
        overrides.setdefault("analysis", {})["rmsd_threshold"] = rmsd_threshold
    if bedroc_alpha is not None:
        overrides.setdefault("analysis", {})["bedroc_alpha"] = bedroc_alpha

    run_qpocket_screen_analyze(workdir, config_path, **overrides)
