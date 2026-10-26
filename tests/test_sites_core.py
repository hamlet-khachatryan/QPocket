"""Ligand finding picks every drug-like residue, once per residue."""
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from qpocket.constants import BUFFER_RESNAMES
from qpocket.core.sites import find_ligands, group_ligands, points_in_box, site_box, site_ids
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


A = np.array([[0, 0, 0], [1, 0, 0]]); B = A + [4.5, 0, 0]; C = A + [9.0, 0, 0]; FAR = A + [50, 0, 0]


def test_overlapping_join_far_split():
    labels = group_ligands([A, A + 0.5, FAR], cutoff=4.0)
    assert labels[0] == labels[1] != labels[2]


def test_bridge_chains_sites():  # accepted per spec decision 13
    labels = group_ligands([A, B, C], cutoff=4.0)
    assert len(set(labels)) == 1


def test_group_ligands_edge_sizes():
    assert group_ligands([], cutoff=4.0).tolist() == []
    assert group_ligands([A], cutoff=4.0).tolist() == [0]


def test_site_ids_by_holo_count():
    ids = site_ids(np.array([0, 1, 1, 1]), ["P1", "P1", "P2", "P3"])
    assert ids == ["site_02", "site_01", "site_01", "site_01"]


def test_site_ids_ties_by_appearance_and_wide_padding():
    assert site_ids(np.array([5, 3]), ["P1", "P2"]) == ["site_01", "site_02"]
    assert site_ids(np.array([], dtype=int), []) == []
    ids = site_ids(np.arange(100), [f"P{i}" for i in range(100)])
    assert ids[0] == "site_001" and ids[99] == "site_100"


def test_points_in_box():
    lo, hi = site_box([A], padding=1.0)
    pts = np.array([[0.5, 0, 0], [5, 0, 0]])
    assert points_in_box(pts, lo, hi).tolist() == [[0.5, 0, 0]]
    assert points_in_box(np.array([[2.0, 0, 0]]), lo, hi).shape == (1, 3)  # inclusive bound
