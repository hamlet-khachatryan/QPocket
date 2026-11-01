"""Reading a complex, splitting it, and writing a docking-ready structure."""
from pathlib import Path

import pytest
from rdkit import Chem

from qpocket.io import interactions as interactions_io

TWO_GLYCINES = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00  0.00           O
END
"""


GLYCINE_PQR = "ATOM      1  N   GLY     1       0.000   0.000   0.000  0.0000 1.8240\n"


FIXTURES = Path(__file__).resolve().parent / "fixtures/complexes"


@pytest.mark.skipif(not (FIXTURES / "1STP.pqr").exists(), reason="run scripts/make_complex_fixtures.py first")
def test_two_same_name_ligands_keep_separate_contacts(tmp_path):
    from rdkit.Geometry import Point3D

    mol_a = interactions_io.protonated_ligand(FIXTURES / "1STP_ligand.pdb")
    mol_b = Chem.Mol(mol_a)
    conf = mol_b.GetConformer()
    for i in range(mol_b.GetNumAtoms()):  # far from the protein: no contacts
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + 80.0, p.y, p.z))
    out = interactions_io.write_complex(tmp_path / "c.pdb", FIXTURES / "1STP.pqr",
                                        [(mol_a, "BTN"), (mol_b, "BTN")])
    a = interactions_io.interactions_for(out, "BTN", ligand_resseq=900)
    b = interactions_io.interactions_for(out, "BTN", ligand_resseq=901)
    assert a != b
    assert {e.split("|")[0].rsplit("_", 1)[0] for e in a} >= {"ASN_23", "ASP_128"}  # hydroxyls unoptimised here
    assert b == ["no_ligand_found"]


