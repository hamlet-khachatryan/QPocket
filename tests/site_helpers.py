"""Ligand, protein and cavity files for site tests, shared by the site test modules."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from qpocket.core.types import Ligand, Structure


def _lig(tmp_path, pdb_id, resname, resseq, atoms):
    path = tmp_path / f"{pdb_id}_{resname}{resseq}.pdb"
    path.write_text("".join(
        f"HETATM{i:5d} {n:<4s} {resname} A{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2s}\n"
        for i, (n, e, x, y, z) in enumerate(atoms, 1)) + "END\n")
    return Ligand(pdb_id, "A", resname, resseq, " ", len(atoms), str(path))


def _protein(tmp_path, name):
    path = tmp_path / f"{name}.pdb"
    path.write_text("ATOM      1  CA  TYR A   1       6.000   0.000   0.000  1.00  0.00           C\n"
                    "ATOM      2  CA  ALA A   2      -3.000   3.000   0.000  1.00  0.00           C\n"
                    "ATOM      3  CA  GLY A   3      40.000  40.000  40.000  1.00  0.00           C\n"
                    "END\n")
    return str(path)


# Each given point becomes a 2x2x2 voxel block (0.35 Å), so a handful of points
# makes a cavity above the 50-point minimum; points within 3.5 Å form one cavity.
_BLOCK = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * 0.35


def _cavities(tmp_path, name, points):
    path = tmp_path / f"{name}_cav.csv"
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts):
        pairs = cKDTree(pts).query_pairs(3.5, output_type="ndarray")
        graph = np.zeros((len(pts), len(pts)))
        graph[pairs[:, 0], pairs[:, 1]] = 1
        labels = connected_components(graph, directed=False)[1]
    else:
        labels = np.zeros(0, int)
    rows = [(*(p + o), int(label)) for p, label in zip(pts, labels) for o in _BLOCK]
    pd.DataFrame(rows, columns=["x", "y", "z", "cavity"]).to_csv(path, index=False)
    return str(path)


# the cavity around ligands drawn between x = 0 and 6 on the y = 0..2 band
INSIDE = [(float(x), float(y), float(z)) for x in range(-3, 7) for y in (-1, 2) for z in (-1, 1)]
FAR = [(30.0, 30.0, 30.0)]


def _structure(tmp_path, pdb_id, ligs=(), points=INSIDE + FAR, state="BOUND_STATE", contacts=("TYR_1_A|HBDonor",)):
    prot = _protein(tmp_path, pdb_id)
    return Structure(pdb_id, "x", 0.5, prot, cleaned_file=prot, protonated_file=prot, protein_state=state,
                     ligands=tuple(ligs), contacts=tuple((lig.key, tuple(contacts)) for lig in ligs),
                     cavities_file=_cavities(tmp_path, pdb_id, points) if points else None)


def site_ids(root):
    """The run's site ids, in rank order."""
    from pathlib import Path

    return pd.read_csv(Path(root) / "sites" / "sites.csv")["site_id"].tolist()
