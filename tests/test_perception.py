"""Molecules are built from component templates, never from bond-order guessing."""
import numpy as np
from rdkit import Chem

from qpocket.core import perception

# A glycine dipeptide whose peptide bond is 1.13 A, not 1.33. Distance-based
# perception reads that as a triple bond; template perception cannot.
SHORT_BOND_PDB = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.139   1.420   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.890   2.660   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.348   2.660   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       5.960   3.720   0.000  1.00  0.00           O
END
"""

GLY_BONDS = {frozenset({"N", "CA"}): "SING", frozenset({"CA", "C"}): "SING",
             frozenset({"C", "O"}): "DOUB"}


def test_parse_reads_fixed_columns():
    atoms = perception.parse_pdb_atoms(SHORT_BOND_PDB)
    assert len(atoms) == 8
    assert atoms[0].name == "N" and atoms[0].res_name == "GLY"
    assert atoms[0].chain_id == "A" and atoms[0].res_seq == 1
    assert atoms[0].element == "N"
    assert np.allclose(atoms[2].xyz, [2.009, 1.420, 0.000])


def test_two_letter_elements_are_not_split():
    line = "HETATM    1 ZN    ZN A 201       0.000   0.000   0.000  1.00  0.00          ZN\n"
    assert perception.parse_pdb_atoms(line)[0].element == "Zn"


def test_short_peptide_bond_is_single_not_triple():
    atoms = perception.parse_pdb_atoms(SHORT_BOND_PDB)
    mol = perception.perceive(atoms, {"GLY": GLY_BONDS})
    bond = mol.GetBondBetweenAtoms(2, 4)      # C of residue 1, N of residue 2
    assert bond is not None, "the peptide bond must still be made"
    assert bond.GetBondType() == Chem.BondType.SINGLE


def test_carbonyl_keeps_the_double_bond_from_its_template():
    mol = perception.perceive(perception.parse_pdb_atoms(SHORT_BOND_PDB),
                              {"GLY": GLY_BONDS})
    assert mol.GetBondBetweenAtoms(2, 3).GetBondType() == Chem.BondType.DOUBLE


def test_hydrogen_goes_to_the_nearest_heavy_atom_with_room():
    pdb = (
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  H   GLY A   1      -0.350   0.930   0.000  1.00  0.00           H\n"
        "END\n"
    )
    mol = perception.perceive(perception.parse_pdb_atoms(pdb),
                              {"GLY": {frozenset({"N", "CA"}): "SING"}})
    assert [n.GetIdx() for n in mol.GetAtomWithIdx(2).GetNeighbors()] == [0]


def test_no_implicit_hydrogens_are_added():
    """ProLIF's SMARTS count hydrogens; implicit ones make every match fail."""
    mol = perception.perceive(perception.parse_pdb_atoms(SHORT_BOND_PDB),
                              {"GLY": GLY_BONDS})
    assert all(a.GetNoImplicit() for a in mol.GetAtoms())
    assert sum(a.GetNumImplicitHs() for a in mol.GetAtoms()) == 0


def test_every_atom_carries_residue_information():
    """ProLIF splits the protein into residues from this."""
    mol = perception.perceive(perception.parse_pdb_atoms(SHORT_BOND_PDB),
                              {"GLY": GLY_BONDS})
    info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
    assert info is not None
    assert info.GetResidueName() == "GLY" and info.GetChainId() == "A"


def test_component_ids_lists_what_must_be_fetched():
    assert perception.component_ids(perception.parse_pdb_atoms(SHORT_BOND_PDB)) == {"GLY"}


def test_perceive_does_not_mutate_its_arguments():
    atoms = perception.parse_pdb_atoms(SHORT_BOND_PDB)
    before = list(atoms)
    perception.perceive(atoms, {"GLY": GLY_BONDS})
    assert atoms == before
