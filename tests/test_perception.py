"""Molecules are built from component templates, never from bond-order guessing."""
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


def test_carbonyl_keeps_the_double_bond_from_its_template():
    mol = perception.perceive(perception.parse_pdb_atoms(SHORT_BOND_PDB),
                              {"GLY": GLY_BONDS})
    assert mol.GetBondBetweenAtoms(2, 3).GetBondType() == Chem.BondType.DOUBLE


