"""AutoDock Vina and QuickVina 2: the same command line and PDBQT output."""
from pathlib import Path
from typing import List

from qpocket.core.ports import DOCKING
from qpocket.core.screening import Engine, Pose, vina_argv, vina_poses


def _read(path: Path) -> List[Pose]:
    path = Path(path)
    return vina_poses(path.read_text(encoding="utf-8")) if path.exists() else []


@DOCKING.register("quickvina", tools=("qvina2",))
def quickvina() -> Engine:
    """QuickVina 2: Vina's search, faster.

    Returns:
        The engine.
    """
    return Engine(vina_argv, _read, ".pdbqt", "qvina2")


@DOCKING.register("vina", tools=("vina",))
def vina() -> Engine:
    """AutoDock Vina.

    Returns:
        The engine.
    """
    return Engine(vina_argv, _read, ".pdbqt", "vina")
