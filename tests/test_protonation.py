"""Ligand protonation states follow pKa at the requested pH, not the CCD's drawing."""
from rdkit import Chem

from qpocket.core.protonation import assign_protonation, dominant_state


def _charges(mol):
    return sorted(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetFormalCharge())


def test_amidine_is_protonated_at_physiological_ph():
    """Benzamidine's amidine has pKa about 11.6, so it is cationic at pH 7.4."""
    assert "+" in dominant_state("NC(=N)c1ccccc1", ph=7.4)


def test_carboxylic_acid_is_deprotonated():
    assert "[O-]" in dominant_state("OC(=O)CCCCC1SCC2NC(=O)NC12", ph=7.4)


def test_assign_protonation_charges_the_amidine():
    mol = Chem.MolFromSmiles("NC(=N)c1ccccc1")
    assert _charges(assign_protonation(mol, ph=7.4)) == [1]


def test_assign_protonation_charges_a_zwitterion_both_ways():
    """Glycine carries both charges at physiological pH."""
    mol = Chem.MolFromSmiles("NCC(=O)O")
    assert _charges(assign_protonation(mol, ph=7.4)) == [-1, 1]


def test_a_real_ligand_is_charged_at_its_non_aromatic_sites():
    """The PARG ligand is a zwitterion: piperidinium and carboxylate."""
    mol = Chem.MolFromSmiles("Cn1c(=O)c2[nH]c(SCCN3CCC(C(=O)O)CC3)nc2n(C)c1=O")
    assert _charges(assign_protonation(mol, ph=7.4)) == [-1, 1]


def test_an_aromatic_ring_keeps_its_hydrogen_and_its_aromaticity():
    """Deprotonating a xanthine N-H would cost the ring every pi-stacking hit."""
    mol = Chem.MolFromSmiles("Cn1c(=O)c2[nH]c(SCCN3CCC(C(=O)O)CC3)nc2n(C)c1=O")
    charged = assign_protonation(mol, ph=7.4)
    assert not any(a.GetIsAromatic() and a.GetFormalCharge() for a in charged.GetAtoms())
    rings = charged.GetRingInfo().AtomRings()
    aromatic = [r for r in rings
                if all(charged.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    assert len(aromatic) == 2


def test_neutral_molecule_is_left_alone():
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert _charges(assign_protonation(mol, ph=7.4)) == []


def test_assign_protonation_does_not_mutate_its_argument():
    mol = Chem.MolFromSmiles("NC(=N)c1ccccc1")
    assign_protonation(mol, ph=7.4)
    assert _charges(mol) == []
