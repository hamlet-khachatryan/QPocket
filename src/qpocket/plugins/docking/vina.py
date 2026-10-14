"""AutoDock Vina and QuickVina 2: the same command line and PDBQT output."""
from qpocket.core.ports import DOCKING
from qpocket.core.screening import Engine, pdbqt_poses, vina_argv


@DOCKING.register("quickvina", tools=("qvina2",))
def quickvina() -> Engine:
    """QuickVina 2: Vina's search, faster.

    Returns:
        The engine.
    """
    return Engine(vina_argv, pdbqt_poses, ".pdbqt", "qvina2")


@DOCKING.register("vina", tools=("vina",))
def vina() -> Engine:
    """AutoDock Vina.

    Returns:
        The engine.
    """
    return Engine(vina_argv, pdbqt_poses, ".pdbqt", "vina")
