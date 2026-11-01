"""Ligand protonation states follow pKa at the requested pH, not the CCD's drawing."""
from rdkit import Chem

from qpocket.core.protonation import assign_protonation


def _charges(mol):
    return sorted(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetFormalCharge())


def test_a_real_ligand_is_charged_at_its_non_aromatic_sites():
    """The PARG ligand is a zwitterion: piperidinium and carboxylate."""
    mol = Chem.MolFromSmiles("Cn1c(=O)c2[nH]c(SCCN3CCC(C(=O)O)CC3)nc2n(C)c1=O")
    assert _charges(assign_protonation(mol, ph=7.4)) == [-1, 1]


