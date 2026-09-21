"""Atom hydrophobicity: real elements and canonical residue names."""
import pytest

from qpocket.constants import atomic_hydrophobicity as hp
from qpocket.constants import canonical_residue, element_of


@pytest.mark.parametrize("atom,res,element", [
    ("ZN", "ZN", "ZN"), ("MG", "MG", "MG"), ("FE", "HEM", "FE"), ("CA", "CME", "C"), ("SE", "MSE", "SE"),
    ("CA", "ALA", "C"), ("CA", "CA", "CA"), ("1HB", "ALA", "H"), ("OG1", "THR", "O"),
])
def test_element_of(atom, res, element):
    assert element_of(atom, res) == element


def test_metals_and_selenium_now_match():
    assert hp("ZN", "ZN") == -2.0 and hp("MG", "MG") == -2.0 and hp("FE", "HEM") == -2.0
    assert hp("SE", "MSE") == 0.6


def test_calcium_ion_is_not_an_alpha_carbon():
    assert hp("CA", "CA") == -2.0 and hp("CA", "ALA") == 1.0


@pytest.mark.parametrize("amber,std,atom", [
    ("ASH", "ASP", "OD1"), ("GLH", "GLU", "OE2"), ("LYN", "LYS", "NZ"),
    ("HIE", "HIS", "CE1"), ("HIP", "HIS", "CG"), ("CYX", "CYS", "SG"),
])
def test_amber_names_score_like_standard_ones(amber, std, atom):
    assert canonical_residue(amber) == std and hp(atom, amber) == hp(atom, std)


def test_standard_atoms_unchanged():
    # Guards the common case against collateral changes.
    assert (hp("CB", "ALA"), hp("C", "ALA"), hp("N", "GLY"), hp("O", "GLY"), hp("OG", "SER"),
            hp("NZ", "LYS"), hp("NH1", "ARG"), hp("CZ", "PHE"), hp("OD1", "ASP"), hp("", "ALA")) == \
           (1.0, 0.0, -0.5, -1.0, -0.5, -2.0, -2.0, 0.5, -2.0, 0.0)
