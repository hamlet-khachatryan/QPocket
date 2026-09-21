"""
Tests for pure utility functions in qpocket_pipeline/utils.py.
No external process mocking needed; all functions are pure Python / numpy.
"""
from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest

from qpocket_pipeline.utils import (
    calculate_sequence_identity,
    check_overlap,
    create_bounding_box_pocket,
    crop_pocket_by_interactions,
    extract_pdb_block,
    is_residue_ligand,
    save_to_hdf5_ascii,
    setup_directory_hierarchy,
)


# ---------------------------------------------------------------------------
# calculate_sequence_identity
# ---------------------------------------------------------------------------


def test_sequence_identity_identical():
    assert calculate_sequence_identity("ACGT", "ACGT") == pytest.approx(100.0)


def test_sequence_identity_empty():
    result = calculate_sequence_identity("", "")
    assert result == pytest.approx(0.0)


def test_sequence_identity_partial():
    # "AAAA" vs "AABB" — 2 matches out of 4 aligned positions = 50%
    result = calculate_sequence_identity("AAAA", "AABB")
    assert 0.0 < result < 100.0


def test_sequence_identity_no_overlap():
    # Completely different characters; pairwise2 still aligns but matches = 0
    result = calculate_sequence_identity("AAAA", "TTTT")
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_overlap
# ---------------------------------------------------------------------------


def test_check_overlap_all_inside():
    box_points = np.array([[0, 0, 0], [10, 10, 10]], dtype=float)
    pts = np.array([[1, 1, 1], [5, 5, 5], [9, 9, 9]], dtype=float)
    assert check_overlap(pts, box_points, tolerance=0.0) == True  # noqa: E712


def test_check_overlap_all_outside():
    box_points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    pts = np.array([[50, 50, 50], [60, 60, 60]], dtype=float)
    assert check_overlap(pts, box_points, tolerance=0.0) == False  # noqa: E712


def test_check_overlap_majority_inside():
    box_points = np.array([[0, 0, 0], [10, 10, 10]], dtype=float)
    # 3 inside, 1 outside
    pts = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [100, 100, 100]], dtype=float)
    assert check_overlap(pts, box_points, tolerance=0.0, use_majority=True) == True  # noqa: E712


def test_check_overlap_majority_outside():
    box_points = np.array([[0, 0, 0], [2, 2, 2]], dtype=float)
    # 1 inside, 3 outside
    pts = np.array([[1, 1, 1], [50, 50, 50], [60, 60, 60], [70, 70, 70]], dtype=float)
    assert check_overlap(pts, box_points, tolerance=0.0, use_majority=True) == False  # noqa: E712


def test_check_overlap_single_point_inside():
    box_points = np.array([[0, 0, 0], [10, 10, 10]], dtype=float)
    pt = np.array([5.0, 5.0, 5.0])
    assert check_overlap(pt, box_points, tolerance=0.0) == True  # noqa: E712


def test_check_overlap_single_point_outside():
    box_points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    pt = np.array([100.0, 100.0, 100.0])
    assert check_overlap(pt, box_points, tolerance=0.0) == False  # noqa: E712


def test_check_overlap_empty_box():
    box_points = np.array([], dtype=float).reshape(0, 3)
    pts = np.array([[1, 2, 3]], dtype=float)
    assert check_overlap(pts, box_points, tolerance=0.0) == False  # noqa: E712


# ---------------------------------------------------------------------------
# create_bounding_box_pocket
# ---------------------------------------------------------------------------


def test_create_bounding_box_pocket_shape():
    corners = create_bounding_box_pocket([0, 0, 0], [4, 4, 4], tolerance=0.0)
    assert corners.shape == (8, 3)


def test_create_bounding_box_pocket_center():
    center = [5.0, 5.0, 5.0]
    dims = [4.0, 4.0, 4.0]
    corners = create_bounding_box_pocket(center, dims, tolerance=0.0)
    # All corners are ±2 from center
    assert np.all(np.abs(corners - np.array(center)) <= 2.0 + 1e-9)


def test_create_bounding_box_pocket_with_tolerance():
    corners = create_bounding_box_pocket([0, 0, 0], [2, 2, 2], tolerance=1.0)
    assert corners.shape == (8, 3)
    # Half-dim = 1.0, tolerance = 1.0 → max extent = 2.0
    assert np.max(np.abs(corners)) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# crop_pocket_by_interactions
# ---------------------------------------------------------------------------


def test_crop_pocket_keeps_nearby_points():
    pocket = np.array([[0, 0, 0], [1, 0, 0], [100, 100, 100]], dtype=float)
    interacting = np.array([[0, 0, 0]], dtype=float)
    result = crop_pocket_by_interactions(pocket, interacting, padding=5.0)
    # Points at 0 and 1 should be kept; 100,100,100 removed
    assert result.shape[0] == 2


def test_crop_pocket_removes_far_points():
    pocket = np.array([[50, 50, 50], [60, 60, 60]], dtype=float)
    interacting = np.array([[0, 0, 0]], dtype=float)
    result = crop_pocket_by_interactions(pocket, interacting, padding=5.0)
    assert result.shape[0] == 0


def test_crop_pocket_all_within_padding():
    pocket = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    interacting = np.array([[1, 1, 1]], dtype=float)
    result = crop_pocket_by_interactions(pocket, interacting, padding=10.0)
    assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# is_residue_ligand
# ---------------------------------------------------------------------------


class _MockResidue:
    """Minimal mock of a BioPython Residue."""
    def __init__(self, het_flag: str, resname: str):
        self.id = (het_flag, 1, " ")
        self._resname = resname

    def get_resname(self):
        return self._resname


def test_is_residue_ligand_standard_aa():
    res = _MockResidue(" ", "ALA")
    assert is_residue_ligand(res, allowed_cofactors=[]) is False


def test_is_residue_ligand_water():
    res = _MockResidue("W", "HOH")
    assert is_residue_ligand(res, allowed_cofactors=[]) is False


def test_is_residue_ligand_real_ligand():
    res = _MockResidue("H_LIG", "LIG")
    assert is_residue_ligand(res, allowed_cofactors=[]) is True


def test_is_residue_ligand_excluded_cofactor():
    res = _MockResidue("H_ATP", "ATP")
    # ATP is in HET_TO_REMOVE
    assert is_residue_ligand(res, allowed_cofactors=[]) is False


def test_is_residue_ligand_allowed_cofactor_excluded():
    res = _MockResidue("H_NAD", "NAD")
    # NAD is in HET_TO_REMOVE list
    assert is_residue_ligand(res, allowed_cofactors=[]) is False


# ---------------------------------------------------------------------------
# extract_pdb_block
# ---------------------------------------------------------------------------


class _MockAtom:
    def __init__(self, name, coord, element="C"):
        self.name = name
        self._coord = coord
        self.element = element

    def get_coord(self):
        return self._coord


class _MockResidueForBlock:
    def __init__(self):
        self.resname = "LIG"
        self.id = ("H_LIG", 100, " ")
        self._atoms = [
            _MockAtom("C1", (1.0, 2.0, 3.0), "C"),
            _MockAtom("O1", (4.0, 5.0, 6.0), "O"),
        ]

    def __iter__(self):
        return iter(self._atoms)


def test_extract_pdb_block_contains_hetatm():
    res = _MockResidueForBlock()
    block = extract_pdb_block(res, "A")
    assert "HETATM" in block


def test_extract_pdb_block_has_coords():
    res = _MockResidueForBlock()
    block = extract_pdb_block(res, "A")
    assert "1.000" in block


def test_extract_pdb_block_multiple_lines():
    res = _MockResidueForBlock()
    block = extract_pdb_block(res, "A")
    lines = [l for l in block.splitlines() if l.strip()]
    assert len(lines) == 2


class _MockResidueNamed:
    """Residue whose atom names / resname exercise the fixed-column edge cases."""

    def __init__(self, atoms, resname="LIG", res_seq=100):
        self.resname = resname
        self.id = ("H_LIG", res_seq, " ")
        self._atoms = atoms

    def __iter__(self):
        return iter(self._atoms)


# PDB fixed columns (1-indexed, inclusive) -> 0-indexed slice.
_FIELDS = {
    "record": (0, 6),
    "serial": (6, 11),
    "name": (12, 16),
    "altloc": (16, 17),
    "resname": (17, 20),
    "chain": (21, 22),
    "resseq": (22, 26),
    "x": (30, 38),
    "y": (38, 46),
    "z": (46, 54),
    "element": (76, 78),
}


def _field(line, key):
    lo, hi = _FIELDS[key]
    return line[lo:hi]


@pytest.mark.parametrize(
    "atom_name,element,expected_name_field",
    [
        ("C1", "C", " C1 "),      # 1-char element -> leading space
        ("PA", "P", " PA "),
        ("H011", "H", "H011"),    # 4-char name fills cols 13-16, no leading space
        ("FE", "FE", "FE  "),     # 2-char element starts at col 13
        ("CL21", "CL", "CL21"),
    ],
)
def test_extract_pdb_block_atom_name_column(atom_name, element, expected_name_field):
    res = _MockResidueNamed([_MockAtom(atom_name, (1.0, 2.0, 3.0), element)])
    line = extract_pdb_block(res, "A").splitlines()[0]
    assert _field(line, "name") == expected_name_field


@pytest.mark.parametrize("atom_name,element", [("C1", "C"), ("H011", "H"), ("FE", "FE")])
def test_extract_pdb_block_fields_stay_in_their_columns(atom_name, element):
    """Every field must land in its own columns regardless of atom-name length."""
    res = _MockResidueNamed([_MockAtom(atom_name, (29.118, -0.578, 4.478), element)])
    line = extract_pdb_block(res, "B").splitlines()[0]

    assert len(line) == 80
    assert _field(line, "record") == "HETATM"
    assert int(_field(line, "serial")) == 1
    assert _field(line, "altloc") == " "
    assert _field(line, "resname") == "LIG"
    assert _field(line, "chain") == "B"
    assert int(_field(line, "resseq")) == 100
    assert float(_field(line, "x")) == pytest.approx(29.118)
    assert float(_field(line, "y")) == pytest.approx(-0.578)
    assert float(_field(line, "z")) == pytest.approx(4.478)
    assert _field(line, "element").strip() == element


def test_extract_pdb_block_parses_with_biopython(tmp_path):
    """A 4-character atom name used to shift the line and break PDBParser."""
    from Bio.PDB import PDBParser

    res = _MockResidueNamed(
        [
            _MockAtom("PA", (29.118, 0.578, -4.478), "P"),
            _MockAtom("H011", (22.773, 0.394, 1.435), "H"),
        ]
    )
    path = tmp_path / "lig.pdb"
    path.write_text(extract_pdb_block(res, "A"))

    structure = PDBParser(QUIET=True).get_structure("lig", str(path))
    atoms = list(structure.get_atoms())
    assert [a.get_name() for a in atoms] == ["PA", "H011"]
    assert atoms[1].get_coord()[0] == pytest.approx(22.773)


def test_extract_pdb_block_truncates_long_resname():
    res = _MockResidueNamed([_MockAtom("C1", (1.0, 2.0, 3.0), "C")], resname="A1B2C")
    line = extract_pdb_block(res, "A").splitlines()[0]
    assert _field(line, "resname") == "A1B"
    assert len(line) == 80


# ---------------------------------------------------------------------------
# save_to_hdf5_ascii + roundtrip
# ---------------------------------------------------------------------------


def test_save_to_hdf5_ascii_and_read(tmp_path):
    out = tmp_path / "test.h5"
    data = {
        "structure1": ["ALA_10|HBond", "GLU_20|SaltBridge"],
        "structure2": ["PHE_5|PiStack"],
    }
    save_to_hdf5_ascii(data, out)

    assert out.exists()
    with h5py.File(out, "r") as hf:
        assert "structure1" in hf
        assert "structure2" in hf
        entries = [e.decode("ascii") for e in hf["structure1"][:]]
        # Sorted: ALA_10|HBond comes before GLU_20|SaltBridge
        assert "ALA_10|HBond" in entries
        assert "GLU_20|SaltBridge" in entries


def test_save_to_hdf5_ascii_overwrites_existing(tmp_path):
    out = tmp_path / "test.h5"
    save_to_hdf5_ascii({"key": ["old_value"]}, out)
    save_to_hdf5_ascii({"key": ["new_value"]}, out)

    with h5py.File(out, "r") as hf:
        entries = [e.decode("ascii") for e in hf["key"][:]]
        assert entries == ["new_value"]


# ---------------------------------------------------------------------------
# setup_directory_hierarchy
# ---------------------------------------------------------------------------


def test_setup_directory_hierarchy_creates_dirs(tmp_path):
    dirs = setup_directory_hierarchy(str(tmp_path), "1ABC")
    expected_keys = {
        "root", "structures", "original", "aligned", "cleaned",
        "extracted_ligands", "pockets", "metadata", "screening",
        "decoys", "actives",
    }
    assert expected_keys == set(dirs.keys())
    for key, path in dirs.items():
        assert path.exists(), f"Directory for key '{key}' was not created: {path}"


def test_setup_directory_hierarchy_uses_pdb_id(tmp_path):
    dirs = setup_directory_hierarchy(str(tmp_path), "2XYZ")
    assert "2XYZ" in str(dirs["root"])
