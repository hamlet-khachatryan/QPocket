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
def test_builtin_list_hides_biotin_by_default():
    assert find_ligands(_parse("1STP"), "1STP", [], 6) == []


@needs_fixtures
def test_finds_biotin():
    found = find_ligands(_parse("1STP"), "1STP", [], 6, include=["BTN"])
    assert [l.resname for l, _ in found] == ["BTN"]
    assert found[0][0].key.startswith("1STP:A:BTN:")


@needs_fixtures
def test_min_heavy_atoms_and_lists():
    s = _parse("3PTB")
    assert find_ligands(s, "3PTB", [], 10, include=["BEN"]) == []   # BEN has 9 heavy atoms
    assert [l.resname for l, _ in find_ligands(s, "3PTB", [], 9, include=["BEN"])] == ["BEN"]
    assert find_ligands(s, "3PTB", [], 6, include=["BEN"], exclude=["BEN"]) == []   # exclude beats include
    assert find_ligands(s, "3PTB", ["BEN"], 6, include=["BEN"]) == []   # cofactor wins


@needs_fixtures
def test_same_resname_twice_gives_two_ligands():
    found = find_ligands(_two_copies("1STP"), "1STP", [], 6, include=["BTN"])
    assert len({l.key for l, _ in found}) == 2


def test_ligand_json_round_trip():
    lig = Ligand("1ABC", "A", "LIG", 7, "A", 12, "x.pdb", "CC")
    assert Ligand.from_json(lig.to_json()) == lig
    assert lig.key == "1ABC:A:LIG:7A"
