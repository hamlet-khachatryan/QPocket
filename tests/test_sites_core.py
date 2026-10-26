"""Ligand finding picks every drug-like residue, once per residue."""
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

from qpocket.core.sites import find_ligands
from qpocket.core.types import Ligand

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

    def residue(name, seq, elements):
        res = Residue(("H_" + name, seq, " "), name, " ")
        for i, el in enumerate(elements):
            res.add(Atom(f"{el}{i}", [float(seq), float(i), 0.0], 1.0, 1.0, " ", f"{el}{i}", i, el))
        return res

    chain = Chain("A")
    for res in (residue("NA", 1, ["NA"]), residue("CL", 2, ["CL"]), residue("SO4", 3, ["S", "O", "O", "O", "O"]),
                residue("BNZ", 4, ["C"] * 6)):
        chain.add(res)
    model = Model(0)
    model.add(chain)
    structure = Structure("x")
    structure.add(model)
    return structure


def test_carbon_free_residues_are_never_ligands():
    found = find_ligands(_inorganic_structure(), "X", [], 1)
    assert [l.resname for l, _ in found] == ["BNZ"]


@needs_fixtures
def test_biotin_still_found_at_threshold_one():
    assert [l.resname for l, _ in find_ligands(_parse("1STP"), "1STP", [], 1)] == ["BTN"]
