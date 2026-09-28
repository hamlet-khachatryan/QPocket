"""Tests for qpocket.viz.pdb_text.

The point of these is that a browser viewer parses PDB by fixed columns, so a
record that is one character off is silently misread rather than rejected. Every
test here asserts on column positions or on a round-trip through an independent
parser, never on substrings.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpocket.viz.pdb_text import (
    BACKBONE,
    atom_line,
    bbox_corners,
    blocks_to_text,
    ligand_pdb,
    parse_atom,
    pocket_context_models,
    pseudo_atom_pdb,
    read_protein_atoms,
    residue_models,
)

# PDB fixed columns (1-indexed, inclusive) as 0-indexed slices.
FIELDS = {
    "record": (0, 6),
    "serial": (6, 11),
    "name": (12, 16),
    "altloc": (16, 17),
    "resname": (17, 20),
    "chain": (21, 22),
    "resseq": (22, 26),
    "icode": (26, 27),
    "x": (30, 38),
    "y": (38, 46),
    "z": (46, 54),
    "occupancy": (54, 60),
    "bfactor": (60, 66),
    "element": (76, 78),
}


def field(line, key):
    lo, hi = FIELDS[key]
    return line[lo:hi]


def make_atom(name="C1", element="C", resname="LIG", chain="A", resseq=100, xyz=(1.0, 2.0, 3.0)):
    return dict(
        name=name,
        altloc=" ",
        resname=resname,
        chain=chain,
        resseq=resseq,
        icode=" ",
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        element=element,
    )


# ---------------------------------------------------------------------------
# atom_line: fixed columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,element,expected",
    [
        ("C1", "C", " C1 "),  # 1-char element gets a leading space
        ("CA", "C", " CA "),
        ("H011", "H", "H011"),  # 4-char name fills the field, no leading space
        ("FE", "FE", "FE  "),  # 2-char element starts at column 13
        ("CL21", "CL", "CL21"),
    ],
)
def test_atom_name_occupies_columns_13_to_16(name, element, expected):
    line = atom_line(1, make_atom(name=name, element=element))
    assert field(line, "name") == expected


@pytest.mark.parametrize("name,element", [("C1", "C"), ("H011", "H"), ("FE", "FE")])
def test_every_field_stays_in_its_columns(name, element):
    line = atom_line(7, make_atom(name=name, element=element, chain="B", xyz=(29.118, -0.578, 4.478)))
    assert len(line) == 80
    assert field(line, "record") == "ATOM  "
    assert int(field(line, "serial")) == 7
    assert field(line, "altloc") == " "
    assert field(line, "resname") == "LIG"
    assert field(line, "chain") == "B"
    assert int(field(line, "resseq")) == 100
    assert float(field(line, "x")) == pytest.approx(29.118)
    assert float(field(line, "y")) == pytest.approx(-0.578)
    assert float(field(line, "z")) == pytest.approx(4.478)
    assert float(field(line, "occupancy")) == pytest.approx(1.0)
    assert field(line, "element").strip() == element


def test_bfactor_column_carries_the_value():
    line = atom_line(1, make_atom(), b=73.42)
    assert float(field(line, "bfactor")) == pytest.approx(73.42)
    assert len(line) == 80


def test_insertion_code_is_preserved():
    atom = make_atom()
    atom["icode"] = "A"
    line = atom_line(1, atom)
    assert field(line, "icode") == "A"
    assert int(field(line, "resseq")) == 100
    assert len(line) == 80


def test_hetatm_record_name():
    line = atom_line(1, make_atom(), record="HETATM")
    assert field(line, "record") == "HETATM"
    assert len(line) == 80


@pytest.mark.parametrize("resname,expected", [("LIG", "LIG"), ("A1B2C", "A1B")])
def test_long_residue_name_is_truncated_not_shifted(resname, expected):
    line = atom_line(1, make_atom(resname=resname))
    assert field(line, "resname") == expected
    assert len(line) == 80


def test_serial_and_resseq_wrap_rather_than_widen():
    line = atom_line(100001, make_atom(resseq=10007))
    assert len(line) == 80
    assert int(field(line, "serial")) == 1
    assert int(field(line, "resseq")) == 7


# ---------------------------------------------------------------------------
# parse_atom: well-formed and shifted input
# ---------------------------------------------------------------------------


def test_parse_well_formed_record():
    line = "HETATM    1  PA  A3R A1001      29.118   0.578  -4.478  1.00  0.00           P  "
    atom = parse_atom(line)
    assert atom["name"] == "PA"
    assert atom["resname"] == "A3R"
    assert atom["chain"] == "A"
    assert atom["resseq"] == 1001
    assert (atom["x"], atom["y"], atom["z"]) == pytest.approx((29.118, 0.578, -4.478))
    assert atom["element"] == "P"


def test_parse_recovers_a_one_column_shift():
    """The layout older extracted-ligand files were written with."""
    shifted = "HETATM   24  H011 WD7 A1001      22.773   0.394   1.435  1.00  0.00           H"
    atom = parse_atom(shifted)
    assert atom is not None
    assert atom["name"] == "H011"
    assert atom["resname"] == "WD7"
    assert atom["resseq"] == 1001
    assert (atom["x"], atom["y"], atom["z"]) == pytest.approx((22.773, 0.394, 1.435))
    assert atom["element"] == "H"


def test_reemitting_a_shifted_record_puts_it_back_in_column():
    shifted = "HETATM   24  H011 WD7 A1001      22.773   0.394   1.435  1.00  0.00           H"
    line = atom_line(24, parse_atom(shifted), record="HETATM")
    assert len(line) == 80
    assert field(line, "name") == "H011"
    assert field(line, "resname") == "WD7"
    assert int(field(line, "resseq")) == 1001


def test_parse_infers_missing_element_from_the_atom_name():
    line = "ATOM      1  CB  ALA A  10      1.000   2.000   3.000  1.00  0.00"
    assert parse_atom(line)["element"] == "C"


def test_parse_returns_none_for_unreadable_records():
    assert parse_atom("ATOM      1  CB  ALA A  10   junk") is None


# ---------------------------------------------------------------------------
# round-trip through an independent parser
# ---------------------------------------------------------------------------


def parse_with_biopython(text, tmp_path, name="x.pdb"):
    from Bio.PDB import PDBParser

    path = tmp_path / name
    path.write_text(text)
    return list(PDBParser(QUIET=True).get_structure("x", str(path)).get_atoms())


def test_emitted_block_round_trips_through_biopython(tmp_path):
    """A 4-character atom name used to shift the line and break PDBParser."""
    text = blocks_to_text(
        [
            atom_line(1, make_atom(name="PA", element="P", xyz=(29.118, 0.578, -4.478)), record="HETATM"),
            atom_line(2, make_atom(name="H011", element="H", xyz=(22.773, 0.394, 1.435)), record="HETATM"),
        ]
    )
    atoms = parse_with_biopython(text, tmp_path)
    assert [a.get_name() for a in atoms] == ["PA", "H011"]
    assert atoms[1].get_coord()[0] == pytest.approx(22.773)


def test_ligand_pdb_repairs_a_shifted_file(tmp_path):
    raw = "\n".join(
        [
            "HETATM    1  PA  WD7 A1001      29.118   0.578  -4.478  1.00  0.00           P  ",
            "HETATM   24  H011 WD7 A1001      22.773   0.394   1.435  1.00  0.00           H",
        ]
    )
    src = tmp_path / "lig.pdb"
    src.write_text(raw + "\n")

    from Bio.PDB import PDBParser

    with pytest.raises(ValueError):
        PDBParser(QUIET=True).get_structure("raw", str(src))

    atoms = parse_with_biopython(ligand_pdb(str(src)), tmp_path, "fixed.pdb")
    assert len(atoms) == 2
    assert {a.element for a in atoms} == {"P", "H"}


# ---------------------------------------------------------------------------
# pseudo_atom_pdb
# ---------------------------------------------------------------------------


def test_pseudo_atoms_get_one_residue_number_each(tmp_path):
    """Sharing a residue number makes parsers collapse the cloud to one atom."""
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    text = pseudo_atom_pdb(coords, b=[0.0, 50.0, 100.0])
    assert len(parse_with_biopython(text, tmp_path)) == 3
    resseqs = [int(field(line, "resseq")) for line in text.splitlines()]
    assert resseqs == [1, 2, 3]


def test_pseudo_atom_bfactors_are_written(tmp_path):
    text = pseudo_atom_pdb([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], b=[12.5, 99.5])
    atoms = parse_with_biopython(text, tmp_path)
    assert sorted(a.get_bfactor() for a in atoms) == pytest.approx([12.5, 99.5])


def test_pseudo_atom_bfactor_length_is_checked():
    with pytest.raises(ValueError, match="b-factors"):
        pseudo_atom_pdb([(0.0, 0.0, 0.0)], b=[1.0, 2.0])


def test_pseudo_atom_pdb_of_nothing_is_empty():
    assert pseudo_atom_pdb([]) == ""


# ---------------------------------------------------------------------------
# bbox_corners
# ---------------------------------------------------------------------------


def test_bbox_corners_span_the_padded_extent():
    pts = np.array([[0.0, 0.0, 0.0], [4.0, 6.0, 8.0]])
    corners = bbox_corners(pts, pad=1.0)
    assert corners.shape == (8, 3)
    assert corners.min(axis=0) == pytest.approx([-1.0, -1.0, -1.0])
    assert corners.max(axis=0) == pytest.approx([5.0, 7.0, 9.0])


# ---------------------------------------------------------------------------
# read_protein_atoms and pocket_context_models
# ---------------------------------------------------------------------------


def write_protein(tmp_path, lines, name="prot.pdb"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def residue_block(resseq, chain="A", resname="ALA", base=0.0, altloc=" "):
    """Backbone plus one side-chain atom, all at ``base`` along x."""
    out = []
    for i, (name, element) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]):
        out.append(
            f"ATOM  {resseq * 10 + i:5d} {' ' + name:<4}{altloc}{resname:>3} "
            f"{chain}{resseq:4d}    {base + i * 0.1:8.3f}{0.0:8.3f}{0.0:8.3f}"
            f"  1.00  0.00          {element:>2}  "
        )
    return out


def test_read_protein_atoms_drops_water_hydrogen_and_hetero(tmp_path):
    path = write_protein(
        tmp_path,
        residue_block(1)
        + [
            "ATOM     99  H   ALA A   1       0.000   0.000   0.000  1.00  0.00           H  ",
            "ATOM     98  O   HOH A 500       0.000   0.000   0.000  1.00  0.00           O  ",
            "HETATM   97  ZN   ZN A 600       0.000   0.000   0.000  1.00  0.00          ZN  ",
        ],
    )
    names = [a["name"] for a in read_protein_atoms(path)]
    assert names == ["N", "CA", "C", "O", "CB"]


def test_alternate_conformations_are_collapsed(tmp_path):
    """Emitted records carry a blank altLoc, so duplicates would draw twice."""
    path = write_protein(
        tmp_path,
        residue_block(1, altloc="A") + residue_block(1, altloc="B", base=5.0),
    )
    atoms = read_protein_atoms(path)
    assert len(atoms) == 5
    keys = [(a["chain"], a["resseq"], a["icode"], a["name"]) for a in atoms]
    assert len(set(keys)) == len(keys)


def test_pocket_context_models_splits_cartoon_from_lining_side_chains(tmp_path):
    # Residue 1 sits on the cloud, residue 2 is nearby, residue 3 is far away.
    path = write_protein(
        tmp_path,
        residue_block(1, base=0.0) + residue_block(2, base=8.0) + residue_block(3, base=40.0),
    )
    cloud = np.array([[0.0, 0.0, 0.0]])
    models = pocket_context_models(path, cloud, context_radius=12.0, contact_radius=4.0)

    assert models.n_context == 2  # residues 1 and 2
    assert models.n_contact == 1  # residue 1 only

    cartoon_names = {line[12:16].strip() for line in models.cartoon.splitlines()}
    assert cartoon_names <= set(BACKBONE)
    side_names = {line[12:16].strip() for line in models.sidechains.splitlines()}
    assert side_names == {"CB"}
    side_res = {int(line[22:26]) for line in models.sidechains.splitlines()}
    assert side_res == {1}


def test_pocket_context_models_handles_an_empty_cloud(tmp_path):
    path = write_protein(tmp_path, residue_block(1))
    models = pocket_context_models(path, np.empty((0, 3)), context_radius=12.0)
    assert models == ("", "", 0, 0, 0)


def write_two_residue_pdb(path):
    """One residue on the cloud, one far away, each with backbone and a side chain."""
    lines = []
    for resseq, offset in ((10, 0.0), (99, 40.0)):
        for name, element in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")):
            lines.append(atom_line(len(lines) + 1, make_atom(
                name=name, element=element, resname="TYR", chain="A",
                resseq=resseq, xyz=(offset, 0.0, 0.0))))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_a_forced_residue_gets_its_side_chain_however_far_it_is(tmp_path):
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    cloud = np.zeros((1, 3))

    without = pocket_context_models(pdb, cloud, context_radius=12.0, contact_radius=4.0)
    assert "  99 " not in without.sidechains

    with_forced = pocket_context_models(pdb, cloud, context_radius=12.0,
                                        contact_radius=4.0, always=["TYR_99_A"])
    forced = [line for line in with_forced.sidechains.splitlines()
              if int(field(line, "resseq")) == 99]
    assert [field(line, "name").strip() for line in forced] == ["CB"]


def test_a_forced_residue_brings_its_backbone_into_the_cartoon(tmp_path):
    # Without this the stick floats unattached, 40 A from any cartoon.
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    models = pocket_context_models(pdb, np.zeros((1, 3)), context_radius=12.0,
                                   contact_radius=4.0, always=["TYR_99_A"])
    backbone = {field(line, "name").strip() for line in models.cartoon.splitlines()
                if int(field(line, "resseq")) == 99}
    assert backbone == set(BACKBONE)


def test_n_forced_counts_only_the_residues_actually_present(tmp_path):
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    models = pocket_context_models(pdb, np.zeros((1, 3)), context_radius=12.0,
                                   contact_radius=4.0,
                                   always=["TYR_99_A", "TRP_500_B", "not a key"])
    assert models.n_forced == 1


def test_forcing_a_residue_already_in_contact_changes_nothing(tmp_path):
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    plain = pocket_context_models(pdb, np.zeros((1, 3)), context_radius=12.0, contact_radius=4.0)
    forced = pocket_context_models(pdb, np.zeros((1, 3)), context_radius=12.0,
                                   contact_radius=4.0, always=["TYR_10_A"])
    assert forced.sidechains == plain.sidechains
    assert forced.cartoon == plain.cartoon
    assert forced.n_forced == 1


def test_the_default_keeps_the_old_behaviour(tmp_path):
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    models = pocket_context_models(pdb, np.zeros((1, 3)), context_radius=12.0, contact_radius=4.0)
    assert models.n_forced == 0
    assert models.n_contact == 1


def test_residue_models_returns_one_block_per_class(tmp_path):
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    blocks = residue_models(pdb, {"observed": ["TYR_10_A"], "potential": ["TYR_99_A"]})
    assert set(blocks) == {"observed", "potential"}
    assert {int(field(line, "resseq")) for line in blocks["observed"].splitlines()} == {10}
    assert {int(field(line, "resseq")) for line in blocks["potential"].splitlines()} == {99}


def test_no_atom_lands_in_two_blocks(tmp_path):
    # A residue named in both classes belongs to the first, so 3Dmol never draws
    # one atom twice in two colours.
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    blocks = residue_models(pdb, {"observed": ["TYR_10_A"], "potential": ["TYR_10_A"]})
    assert blocks["observed"]
    assert "potential" not in blocks


def test_an_empty_class_is_left_out(tmp_path):
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    assert set(residue_models(pdb, {"observed": [], "potential": ["TYR_99_A"]})) == {"potential"}


def test_a_class_naming_no_present_residue_is_left_out(tmp_path):
    pdb = write_two_residue_pdb(tmp_path / "s.pdb")
    assert residue_models(pdb, {"observed": ["TRP_500_B"]}) == {}
