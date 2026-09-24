"""Reading a complex, splitting it, and writing a docking-ready structure."""
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
