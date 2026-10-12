"""AutoDock Vina and QuickVina: the same command line, different executables."""
from typing import List, Mapping

from qpocket.core.ports import DOCKING
from qpocket.core.screening import vina_argv


@DOCKING.register("vina", tools=("vina",))
def vina(job: Mapping, executable: str, cpus: int) -> List[str]:
    """Command line for one Vina run.

    Args:
        job: One docking_jobs.csv row.
        executable: The engine binary.
        cpus: Threads.

    Returns:
        argv, no shell involved.
    """
    return vina_argv(job, executable, cpus)


@DOCKING.register("quickvina", tools=("qvina",))
def quickvina(job: Mapping, executable: str, cpus: int) -> List[str]:
    """Command line for one QuickVina run; it takes Vina's arguments.

    Args:
        job: One docking_jobs.csv row.
        executable: The engine binary.
        cpus: Threads.

    Returns:
        argv, no shell involved.
    """
    return vina_argv(job, executable, cpus)
