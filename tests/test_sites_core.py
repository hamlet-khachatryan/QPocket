"""Ligand finding picks every drug-like residue, once per residue."""
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from qpocket.core.sites import (find_ligands, median_box,
                                site_members)

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


def _x_boxes(*spans):
    """Boxes spanning ``(lo, hi)`` along x and the unit interval along y and z."""
    return np.array([[lo, 0, 0, hi, 1, 1] for lo, hi in spans], float)


def test_a_b_c_gives_two_sites_with_b_in_both():
    boxes = _x_boxes((0, 3), (1, 4), (2, 5))     # IoU(A,B) = IoU(B,C) = 0.5, IoU(A,C) = 0.2
    sites = site_members(np.vstack([median_box(boxes[[0, 1]]), median_box(boxes[[1, 2]])]), boxes, 0.5)
    assert sorted(map(sorted, sites)) == [[0, 1], [1, 2]]


# --- manual overrides re-applied to the automatic grouping --------------------

# candidates 0-3 form site A (cores {0,1} and {2,3}), 4-5 site B, 6 site C
_KEYS = [("P1", 0), ("P2", 0), ("P3", 0), ("P4", 0), ("P1", 1), ("P2", 1), ("P5", 0)]
_CORES = [[0, 1], [2, 3], [4, 5], [6]]
_GROUPS = [[0, 1, 2, 3], [4, 5], [6]]


def _cavs(*idx):
    return [list(_KEYS[i]) for i in idx]


def _apply(overrides, groups=_GROUPS):
    from qpocket.core.sites import apply_overrides

    done = apply_overrides([list(g) for g in groups], _CORES, [list(g) for g in groups], _KEYS, overrides)
    return done.groups, done.dropped, done.report


def test_merge_override_joins_two_groups():
    groups, dropped, report = _apply([{"op": "merge", "sites": [_cavs(0, 1, 2, 3), _cavs(4, 5)]}])
    assert sorted(map(sorted, groups)) == [[0, 1, 2, 3, 4, 5], [6]]
    assert dropped == [] and report == [{"op": "merge", "status": "applied", "parts": 2,
                                         "groups": [[0, 1, 2, 3, 4, 5]]}]


def test_split_override_returns_its_cores():
    groups, _, report = _apply([{"op": "split", "cavities": _cavs(0, 1, 2, 3)}])
    assert sorted(map(sorted, groups)) == [[0, 1], [2, 3], [4, 5], [6]]
    assert report[0] == {"op": "split", "status": "applied", "parts": 2, "groups": [[0, 1], [2, 3]]}


def test_override_follows_a_slightly_changed_site():
    groups, dropped, _ = _apply([{"op": "drop", "cavities": _cavs(0, 1, 2)}])  # 3 of the site's 4
    assert dropped == [[0, 1, 2, 3]]


def test_split_parts_keep_overlap_members_but_not_each_others_cavities():
    from qpocket.core.sites import apply_overrides

    def members_of(pieces):  # every part's box also overlaps candidate 6 (site C) and candidate 2 (its sibling)
        return [set(p) | {2, 6} for p in pieces]

    done = apply_overrides([list(g) for g in _GROUPS], _CORES, [list(g) for g in _GROUPS], _KEYS,
                           [{"op": "split", "cavities": _cavs(0, 1, 2, 3)}], members_of=members_of)
    by_group = {tuple(g): m for g, m in zip(done.groups, done.members)}
    assert by_group[(0, 1)] == [0, 1, 6] and by_group[(2, 3)] == [2, 3, 6]
