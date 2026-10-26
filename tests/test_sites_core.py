"""Ligand finding picks every drug-like residue, once per residue."""
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from qpocket.constants import BUFFER_RESNAMES
from qpocket.core.sites import (box_iou, cavity_cores, find_ligands, grid_spacing, ligand_sites, median_box,
                                merge_sites, site_members, site_name, voxel_box)
from qpocket.core.types import Ligand

# the synthetic residues deliberately carry unidentifiable elements
pytestmark = pytest.mark.filterwarnings("ignore::Bio.PDB.PDBExceptions.PDBConstructionWarning")
FIXTURES = Path(__file__).resolve().parent / "fixtures/complexes"
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "1STP.pdb").exists() or not (FIXTURES / "3PTB.pdb").exists(),
    reason="run scripts/make_complex_fixtures.py first")


def _parse(pdb_id):
    return PDBParser(QUIET=True).get_structure(pdb_id, FIXTURES / f"{pdb_id}.pdb")


def _two_copies(pdb_id):
    """The structure with its BTN copied into another chain under the next residue number."""
    s = _parse(pdb_id)
    chain = next(iter(s[0]))
    btn = next(r for r in chain if r.get_resname() == "BTN")
    other = next((c for c in s[0] if c.id != chain.id), chain)
    twin = btn.copy()
    twin.id = (btn.id[0], btn.id[1] + 1, btn.id[2])
    other.add(twin)
    return s


@needs_fixtures
def test_finds_biotin():
    found = find_ligands(_parse("1STP"), "1STP", [], 6)
    assert [l.resname for l, _ in found] == ["BTN"]
    assert found[0][0].key.startswith("1STP:A:BTN:")


@needs_fixtures
def test_size_threshold_and_cofactors():
    s = _parse("3PTB")
    assert [l.resname for l, _ in find_ligands(s, "3PTB", [], 9)] == ["BEN"]
    assert find_ligands(s, "3PTB", [], 10) == []          # BEN has 9 heavy atoms
    assert find_ligands(s, "3PTB", ["BEN"], 6) == []


@needs_fixtures
def test_water_and_standard_residues_are_never_reported():
    found = find_ligands(_parse("3PTB"), "3PTB", [], 1)   # threshold 1 would admit a water
    assert all(r.id[0] not in (" ", "W") for _, r in found)
    assert not {"HOH", "SER", "ASP"} & {l.resname for l, _ in found}


@needs_fixtures
def test_same_resname_twice_gives_two_ligands():
    found = find_ligands(_two_copies("1STP"), "1STP", [], 6)
    assert len({l.key for l, _ in found}) == 2


def test_ligand_json_round_trip():
    lig = Ligand("1ABC", "A", "LIG", 7, "A", 12, "x.pdb", "CC")
    assert Ligand.from_json(lig.to_json()) == lig
    assert lig.key == "1ABC:A:LIG:7A"


def _inorganic_structure():
    """A chain holding Na, Cl and sulfate residues next to a 6-carbon ligand."""
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    def residue(name, seq, elements, names=None):
        res = Residue(("H_" + name, seq, " "), name, " ")
        for i, el in enumerate(elements):
            atom_name = names[i] if names else f"{el}{i}"
            res.add(Atom(atom_name, [float(seq), float(i), 0.0], 1.0, 1.0, " ", atom_name, i, el))
        return res

    chain = Chain("A")
    for res in (residue("NA", 1, ["NA"]), residue("CL", 2, ["CL"]), residue("SO4", 3, ["S", "O", "O", "O", "O"]),
                residue("BNZ", 4, ["C"] * 6),
                residue("UNK", 5, ["X"] * 6, [f"C{i}" for i in range(1, 7)]),
                residue("BLK", 6, [""] * 6, [f"C{i}" for i in range(1, 7)])):
        chain.add(res)
    model = Model(0)
    model.add(chain)
    structure = Structure("x")
    structure.add(model)
    return structure


def test_carbon_free_residues_are_never_ligands():
    found = find_ligands(_inorganic_structure(), "X", [], 1)
    assert [l.resname for l, _ in found if l.resname in ("NA", "CL", "SO4", "BNZ")] == ["BNZ"]


def test_unidentified_elements_fall_back_to_the_atom_name():
    found = {l.resname: l for l, _ in find_ligands(_inorganic_structure(), "X", [], 6)}
    assert {"UNK", "BLK"} <= set(found) and found["UNK"].n_heavy == found["BLK"].n_heavy == 6


def test_cofactor_names_match_case_insensitively():
    assert "BNZ" not in [l.resname for l, _ in find_ligands(_inorganic_structure(), "X", ["bnz"], 1)]


def _buffer_structure():
    """A chain holding glycerol and citrate, both carbon-bearing and big enough to be ligands."""
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    chain = Chain("A")
    for seq, (name, elements) in enumerate((("GOL", "COCOCO"), ("CIT", "CCCCCCOOOOOOO")), start=1):
        res = Residue(("H_" + name, seq, " "), name, " ")
        for i, el in enumerate(elements):
            res.add(Atom(f"{el}{i}", [float(seq), float(i), 0.0], 1.0, 1.0, " ", f"{el}{i}", i, el))
        chain.add(res)
    model = Model(0)
    model.add(chain)
    structure = Structure("x")
    structure.add(model)
    return structure


def test_buffers_are_never_ligands():
    assert [l.resname for l, _ in find_ligands(_buffer_structure(), "X", [], 1)] == []


def test_keep_list_turns_a_buffer_back_into_a_ligand():
    found = find_ligands(_buffer_structure(), "X", [], 1, keep=["cit"])
    assert [l.resname for l, _ in found] == ["CIT"]


def test_buffer_list_holds_the_spec_names():
    assert {"GOL", "EDO", "PEG", "DMS", "EPE", "TRS", "MES", "CIT", "ACT"} <= BUFFER_RESNAMES
    assert "ZN" not in BUFFER_RESNAMES and "LIG" not in BUFFER_RESNAMES


@needs_fixtures
def test_biotin_still_found_at_threshold_one():
    assert [l.resname for l, _ in find_ligands(_parse("1STP"), "1STP", [], 1)] == ["BTN"]


def _x_boxes(*spans):
    """Boxes spanning ``(lo, hi)`` along x and the unit interval along y and z."""
    return np.array([[lo, 0, 0, hi, 1, 1] for lo, hi in spans], float)


def test_voxel_box_gives_a_flat_cavity_volume():
    b = voxel_box(np.array([[0, 0, 0], [1, 0, 0]], float), half=0.35)
    assert np.allclose(b, [-0.35, -0.35, -0.35, 1.35, 0.35, 0.35])


def test_iou_identical_disjoint_and_third():
    a = _x_boxes((0, 2))
    assert box_iou(a, a)[0, 0] == 1.0
    assert box_iou(a, _x_boxes((5, 7)))[0, 0] == 0.0
    assert box_iou(a, _x_boxes((1, 3)))[0, 0] == pytest.approx(1 / 3)


def test_cores_never_hold_a_pair_below_threshold():
    rng = np.random.default_rng(0)
    lo = rng.uniform(0, 6, size=(60, 3))
    boxes = np.hstack([lo, lo + rng.uniform(2, 5, size=(60, 3))])
    groups = cavity_cores(boxes, 0.5)
    assert sorted(i for g in groups for i in g) == list(range(60))
    assert all(box_iou(boxes[g], boxes[g]).min() >= 0.5 for g in groups if len(g) > 1)


def test_cores_of_zero_and_one_box():
    assert cavity_cores(np.zeros((0, 6)), 0.5) == []
    assert cavity_cores(_x_boxes((0, 1)), 0.5) == [[0]]


def test_a_b_c_gives_two_sites_with_b_in_both():
    boxes = _x_boxes((0, 3), (1, 4), (2, 5))     # IoU(A,B) = IoU(B,C) = 0.5, IoU(A,C) = 0.2
    sites = site_members(np.vstack([median_box(boxes[[0, 1]]), median_box(boxes[[1, 2]])]), boxes, 0.5)
    assert sorted(map(sorted, sites)) == [[0, 1], [1, 2]]


def test_split_pocket_cores_merge_into_one_site():
    boxes = _x_boxes((0, 10), (0, 10), (2, 12), (-2, 8))   # (2,12) and (-2,8) overlap at 0.43
    cores = cavity_cores(boxes, 0.5)
    assert len(cores) == 2
    assert [sorted(g) for g in merge_sites(cores, boxes, 0.5)] == [[0, 1, 2, 3]]


def test_distant_sites_stay_apart():
    boxes = _x_boxes((0, 10), (30, 40))
    assert len(merge_sites(cavity_cores(boxes, 0.5), boxes, 0.5)) == 2


def test_ligand_fraction_boundary_inclusive():
    site_boxes = np.array([[0, 0, 0, 10, 10, 10]], float)
    coords = np.array([[x, 5, 5] for x in (1, 2, 3, 4, 5, 11, 12, 13, 14, 15)], float)
    assert ligand_sites([coords], site_boxes, 0.5, 0.0) == [[0]]
    assert ligand_sites([coords], site_boxes, 0.6, 0.0) == [[]]


def test_padding_widens_the_box():
    site_boxes = np.array([[0, 0, 0, 10, 10, 10]], float)
    outside = np.array([[11, 5, 5], [11, 6, 5]], float)
    assert ligand_sites([outside], site_boxes, 0.5, 0.0) == [[]]
    assert ligand_sites([outside], site_boxes, 0.5, 1.5) == [[0]]


def test_names_unique_when_residue_repeats():
    assert [site_name(1, "H21"), site_name(4, "H21"), site_name(2, "")] == ["S1-H21", "S4-H21", "S2"]


def test_grid_spacing_of_a_regular_grid():
    grid = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)], float) * 0.7
    assert grid_spacing([grid]) == pytest.approx(0.7)
    assert grid_spacing([np.zeros((1, 3))]) == 0.0
