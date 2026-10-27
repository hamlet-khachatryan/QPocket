"""Reading a complex, splitting it, and writing a docking-ready structure."""
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qpocket.io import interactions as interactions_io

TWO_GLYCINES = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00  0.00           O
END
"""


def test_pdb_line_uses_the_exact_columns():
    line = interactions_io.pdb_line("ATOM", 1, "CA", "GLY", "A", 16,
                                    1.234, 5.678, 9.012, "C")
    assert line[:6] == "ATOM  "
    assert line[12:16] == " CA "
    assert line[17:20] == "GLY"
    assert line[21] == "A"
    assert int(line[22:26]) == 16
    assert float(line[30:38]) == 1.234
    assert line[76:78] == " C"


def test_four_character_atom_names_start_in_column_13():
    line = interactions_io.pdb_line("ATOM", 1, "HD11", "LEU", "A", 5, 0.0, 0.0, 0.0, "H")
    assert line[12:16] == "HD11"


def test_round_trip_through_pdb_line_is_parseable():
    from qpocket.core.perception import parse_pdb_atoms
    line = interactions_io.pdb_line("HETATM", 7, "N1", "BEN", "A", 900,
                                    -2.797, 14.235, 14.491, "N")
    atom = parse_pdb_atoms(line)[0]
    assert atom.name == "N1" and atom.res_name == "BEN" and atom.res_seq == 900
    assert atom.element == "N" and atom.is_hetatm


def test_missing_ligand_reports_the_sentinel(tmp_path):
    path = tmp_path / "apo.pdb"
    path.write_text(TWO_GLYCINES, encoding="utf-8")
    assert interactions_io.interactions_for(path, None) == ["no_ligand_found"]
    assert interactions_io.interactions_for(path, "") == ["no_ligand_found"]


def test_absent_file_reports_the_sentinel(tmp_path):
    assert interactions_io.interactions_for(tmp_path / "nope.pdb", "LIG") == ["file_not_found"]


def test_ligand_absent_from_the_file_reports_the_sentinel(tmp_path):
    path = tmp_path / "apo.pdb"
    path.write_text(TWO_GLYCINES, encoding="utf-8")
    assert interactions_io.interactions_for(path, "LIG") == ["no_ligand_found"]


GLYCINE_PQR = "ATOM      1  N   GLY     1       0.000   0.000   0.000  0.0000 1.8240\n"


def test_each_ligand_gets_its_own_residue_number(tmp_path):
    pqr = tmp_path / "p.pqr"
    pqr.write_text(GLYCINE_PQR, encoding="utf-8")
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    out = interactions_io.write_complex(tmp_path / "c.pdb", pqr, [(mol, "lig"), (None, "BAD"), (mol, "LIG")])
    het = [l for l in out.read_text().splitlines() if l.startswith("HETATM")]
    assert {(l[17:20], l[21], int(l[22:26])) for l in het} == {("LIG", "A", 900), ("LIG", "A", 902)}


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


def test_a_two_letter_metal_survives_the_protonated_complex(tmp_path):
    """pdb2pqr writes zinc as 'ZN ION'; its element must be Zn, not Z."""
    from qpocket.io.interactions import load_molecule, write_complex

    pqr = tmp_path / "protein.pqr"
    pqr.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000 -0.3000 1.8240\n"
        "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  0.0000 1.9080\n"
        "ATOM      3  C   GLY A   1       2.009   1.420   0.000  0.5000 1.9080\n"
        "ATOM      4  O   GLY A   1       1.251   2.390   0.000 -0.5000 1.6612\n"
        "HETATM 9999 ZN ION     1      10.256   -3.894    5.661  2.0000  1.1000\n")
    out = tmp_path / "complex.pdb"
    write_complex(out, pqr, [])
    zinc = [l for l in out.read_text().splitlines() if "ZN" in l]
    assert zinc and zinc[0][76:78].strip() == "ZN"
    mol = load_molecule(out)
    assert "Zn" in {a.GetSymbol() for a in mol.GetAtoms()}
