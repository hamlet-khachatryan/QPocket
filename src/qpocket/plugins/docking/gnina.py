"""gnina: Vina-style search with CNN rescoring; SDF output."""
from pathlib import Path
from typing import List, Mapping

from qpocket.core.ports import DOCKING
from qpocket.core.screening import Engine, Pose, _seed_args


def _argv(job: Mapping, executable: str, cpus: int) -> List[str]:
    return [executable, "-r", str(job["protein_pdbqt"]), "-l", str(job["ligand_pdbqt"]),
            "-o", str(job["output_pdbqt"]),
            "--center_x", str(job["center_x"]), "--center_y", str(job["center_y"]),
            "--center_z", str(job["center_z"]),
            "--size_x", str(job["size_x"]), "--size_y", str(job["size_y"]), "--size_z", str(job["size_z"]),
            "--exhaustiveness", str(job["exhaustiveness"]), "--num_modes", str(job["num_modes"]),
            "--cpu", str(cpus), *_seed_args(job)]


def _read(path: Path) -> List[Pose]:
    from rdkit import Chem

    if not Path(path).exists():
        return []
    poses = []
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is None:
            continue
        score = float(mol.GetProp("minimizedAffinity")) if mol.HasProp("minimizedAffinity") else float("nan")
        poses.append(Pose(len(poses) + 1, score, mol))
    return poses


@DOCKING.register("gnina", tools=("gnina",))
def gnina() -> Engine:
    """gnina.

    Returns:
        The engine.
    """
    return Engine(_argv, _read, ".sdf", "gnina")
