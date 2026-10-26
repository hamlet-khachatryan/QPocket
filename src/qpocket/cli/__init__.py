"""Command-line interface: the `qpocket` group; each command lives in its own module."""
import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """QPocket - Protein Binding Pocket Analysis Pipeline.

    A framework for analysis and benchmarking of experimentally observed
    conformational space of protein binding pockets for drug design applications.
    """
    pass


# Modules, not their commands, are imported: `qpocket.cli.run` stays the module.
from qpocket.cli import compare, doctor, run, runs, screen, site, slurm, task  # noqa: E402

for _command in (run.run, run.validate, run.init, slurm.slurm, doctor.doctor, task.task,
                 runs.recluster_command, runs.explore_command, compare.featurize_command,
                 compare.compare_command, compare.compare_structures_command):
    cli.add_command(_command)
cli.add_command(screen.screen_cli, name="screen")
cli.add_command(site.site_cli, name="site")
