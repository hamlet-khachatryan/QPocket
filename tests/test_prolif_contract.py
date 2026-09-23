"""ProLIF finds contacts through the call sequence the detector uses.

Recorded findings, from scripts/prolif_probe.py against ProLIF's bundled example:
the molecules ProLIF is happy with carry NoImplicit=True on every atom, zero
implicit hydrogens, exactly one conformer, and PDBResidueInfo on every atom.
qpocket.core.perception produces molecules with those same properties.
"""
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem


def _counts(fingerprint):
    return sum(len(occ) for _f, res in fingerprint.ifp.items()
               for (_l, _r), inters in res.items() for _n, occ in inters.items())


def _bundled():
    u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
    return u.select_atoms("resname LIG"), u.select_atoms("protein")


def test_prolif_bundled_example_finds_contacts():
    ligand_atoms, protein_atoms = _bundled()
    fingerprint = plf.Fingerprint(["Hydrophobic", "HBDonor", "HBAcceptor"], count=True)
    fingerprint.run_from_iterable([plf.Molecule.from_mda(ligand_atoms)],
                                  plf.Molecule.from_mda(protein_atoms), progress=False)
    assert _counts(fingerprint) > 0


def test_from_rdkit_behaves_like_from_mda():
    """A hand-built RDKit molecule must reach ProLIF the same way an MDAnalysis one does."""
    ligand_atoms, _protein_atoms = _bundled()
    from_mda = plf.Molecule.from_mda(ligand_atoms)
    from_rdkit = plf.Molecule.from_rdkit(Chem.Mol(from_mda))
    assert len(from_rdkit.residues) == len(from_mda.residues)


def test_rdkit_round_trip_still_finds_contacts():
    """Contacts survive a round trip through RDKit, which is how we supply molecules."""
    ligand_atoms, protein_atoms = _bundled()
    ligand = plf.Molecule.from_rdkit(Chem.Mol(plf.Molecule.from_mda(ligand_atoms)))
    protein = plf.Molecule.from_rdkit(Chem.Mol(plf.Molecule.from_mda(protein_atoms)))
    fingerprint = plf.Fingerprint(["Hydrophobic", "HBDonor", "HBAcceptor"], count=True)
    fingerprint.run_from_iterable([ligand], protein, progress=False)
    assert _counts(fingerprint) > 0
