"""Rotatable hydroxyl hydrogens are aimed at their acceptors."""
import numpy as np
from rdkit import Chem

from qpocket.core.hydroxyl import optimise_hydroxyls, rotatable_hydroxyls
from qpocket.core.perception import parse_pdb_atoms, perceive

# A serine hydroxyl with its hydrogen pointing away from a nearby carbonyl oxygen.
SER_AND_ACCEPTOR = """\
ATOM      1  CB  SER A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  OG  SER A   1       1.430   0.000   0.000  1.00  0.00           O
ATOM      3  HG  SER A   1       1.760  -0.900   0.000  1.00  0.00           H
HETATM    4  O1  LIG A 900       1.900   2.600   0.000  1.00  0.00           O
END
"""
TEMPLATES = {"SER": {frozenset({"CB", "OG"}): "SING"}, "LIG": {}}


def _built():
    return perceive(parse_pdb_atoms(SER_AND_ACCEPTOR), TEMPLATES)


def _angle(mol, oxygen=1, hydrogen=2, acceptor=3):
    pos = mol.GetConformer().GetPositions()
    to_donor, to_acceptor = pos[oxygen] - pos[hydrogen], pos[acceptor] - pos[hydrogen]
    cosine = np.dot(to_donor, to_acceptor) / (
        np.linalg.norm(to_donor) * np.linalg.norm(to_acceptor))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def test_a_hydroxyl_is_recognised_as_rotatable():
    assert rotatable_hydroxyls(_built()) == [(0, 1, 2)]


def test_a_carbonyl_oxygen_is_not_rotatable():
    """It has no hydrogen, so there is nothing to turn."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC=O"))
    assert rotatable_hydroxyls(mol) == []


def test_the_hydrogen_turns_towards_the_acceptor():
    before = _angle(_built())
    after = _angle(optimise_hydroxyls(_built()))
    assert before < 120.0, "the fixture must start with the hydrogen pointing away"
    assert after > 120.0, f"expected a hydrogen bond, got {after:.0f} degrees"


def test_the_oxygen_and_heavy_atoms_do_not_move():
    original = _built()
    moved = optimise_hydroxyls(original)
    before = original.GetConformer().GetPositions()
    after = moved.GetConformer().GetPositions()
    assert np.allclose(before[[0, 1, 3]], after[[0, 1, 3]])


def test_the_hydrogen_keeps_its_bond_length():
    original = _built()
    moved = optimise_hydroxyls(original)
    def length(mol):
        pos = mol.GetConformer().GetPositions()
        return np.linalg.norm(pos[2] - pos[1])
    assert np.isclose(length(original), length(moved), atol=1e-6)


def test_optimisation_is_deterministic():
    runs = [optimise_hydroxyls(_built()).GetConformer().GetPositions()[2]
            for _ in range(3)]
    assert np.allclose(runs[0], runs[1]) and np.allclose(runs[1], runs[2])


def test_a_hydroxyl_with_no_acceptor_is_left_alone():
    lonely = SER_AND_ACCEPTOR.replace(
        "HETATM    4  O1  LIG A 900       1.900   2.600   0.000  1.00  0.00           O\n", "")
    mol = perceive(parse_pdb_atoms(lonely), TEMPLATES)
    before = mol.GetConformer().GetPositions()[2].copy()
    after = optimise_hydroxyls(mol).GetConformer().GetPositions()[2]
    assert np.allclose(before, after)
