"""Docked-pose RMSD is measured in place and is symmetry-aware."""

import random

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


def test_rmsd_is_measured_in_place():
    crystal = _mol("CC(=O)Nc1ccccc1")
    moved = _translated(crystal, (5.0, 0, 0))
    assert compute_rmsd_mcs(moved, crystal) == pytest.approx(5.0, abs=1e-3)


def test_rmsd_of_identical_pose_is_zero():
    crystal = _mol("CC(=O)Nc1ccccc1")
    assert compute_rmsd_mcs(Chem.Mol(crystal), crystal) == pytest.approx(0.0, abs=1e-6)


def test_symmetric_ligand_is_not_penalised_for_atom_order():
    crystal = _mol("c1ccc2ccccc2c1")
    order = list(range(crystal.GetNumAtoms()))
    random.Random(0).shuffle(order)
    assert compute_rmsd_mcs(Chem.RenumberAtoms(crystal, order), crystal) == pytest.approx(0.0, abs=1e-3)
