"""Docked-pose RMSD is measured in place and is symmetry-aware."""


import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qpocket.arms.screen.analysis import compute_rmsd_mcs


def _mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return Chem.RemoveHs(mol)


def _translated(mol, shift):
    moved = Chem.Mol(mol)
    conf = moved.GetConformer()
    for i in range(moved.GetNumAtoms()):
        conf.SetAtomPosition(i, np.array(conf.GetAtomPosition(i)) + shift)
    return moved


def test_partial_mcs_scores_only_the_mapped_atoms_in_place():
    crystal = _mol("CC(=O)Nc1ccccc1CCCl")
    part = Chem.RWMol(crystal)
    for idx in (12, 11, 10):  # drop CCCl: the acetanilide left keeps the crystal's coordinates
        part.RemoveAtom(idx)
    docked = part.GetMol()
    Chem.SanitizeMol(docked)
    assert Chem.MolToSmiles(docked) == Chem.MolToSmiles(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"))
    assert compute_rmsd_mcs(docked, crystal) == pytest.approx(0.0, abs=1e-6)
    assert compute_rmsd_mcs(_translated(docked, (5.0, 0, 0)), crystal) == pytest.approx(5.0, abs=1e-3)


