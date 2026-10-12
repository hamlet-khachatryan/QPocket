"""One pipeline stage per call; the entry point of every generated sbatch script."""
from __future__ import annotations

from pathlib import Path

import click

from qpocket.cli.run import _load_run

# ---------------------------------------------------------------------------
# Array-task entry point (invoked by the generated sbatch scripts)
# ---------------------------------------------------------------------------

@click.group(hidden=True)
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
        from qpocket.io import logging as qlog
        from qpocket.pipelines import tasks

        cfg, dirs = _load_run(config, workdir, [])

        # One log per task: the pipeline log is opened truncating, so a shared
        # path would have every array task erase the others.
        suffix = f"_{index}" if index is not None else ""
        log_dir = Path(dirs["root"]) / "slurm" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        qlog.configure(cfg, log_dir / f"{name}{suffix}.log")

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
