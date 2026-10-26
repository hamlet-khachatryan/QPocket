"""Ligand, protein and cavity files for site tests, shared by the site test modules."""
import pandas as pd

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


def _cavities(tmp_path, name, points):
    path = tmp_path / f"{name}_cav.csv"
    pd.DataFrame([(x, y, z, i % 2) for i, (x, y, z) in enumerate(points)],
                 columns=["x", "y", "z", "cavity"]).to_csv(path, index=False)
    return str(path)


INSIDE = [(float(x), 2.0, 0.0) for x in range(-3, 7)]  # ten points along the site
FAR = [(30.0, 30.0, 30.0)]


def _structure(tmp_path, pdb_id, ligs=(), points=INSIDE + FAR, state="BOUND_STATE", contacts=("TYR_1_A|HBDonor",)):
    prot = _protein(tmp_path, pdb_id)
    return Structure(pdb_id, "x", 0.5, prot, cleaned_file=prot, protonated_file=prot, protein_state=state,
                     ligands=tuple(ligs), contacts=tuple((lig.key, tuple(contacts)) for lig in ligs),
                     cavities_file=_cavities(tmp_path, pdb_id, points) if points else None)
