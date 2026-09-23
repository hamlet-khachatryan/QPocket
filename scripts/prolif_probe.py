"""Find the ProLIF call sequence that works, starting from its own example data."""
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem


def counts(fingerprint):
    """Total number of detected contacts in a fingerprint.

    Args:
        fingerprint: A prolif Fingerprint that has been run.

    Returns:
        Number of contacts found.
    """
    return sum(len(occ) for _f, res in fingerprint.ifp.items()
               for (_l, _r), inters in res.items() for _n, occ in inters.items())


def describe(mol, label):
    """Print the molecule properties ProLIF's matching depends on.

    Args:
        mol: A prolif Molecule.
        label: Name for the printout.
    """
    rd = Chem.Mol(mol)
    print(f"{label}: {rd.GetNumAtoms()} atoms, {len(mol.residues)} residues, "
          f"noImplicit={rd.GetAtomWithIdx(0).GetNoImplicit()}, "
          f"implicitHs={sum(a.GetNumImplicitHs() for a in rd.GetAtoms())}, "
          f"aromatic={sum(1 for a in rd.GetAtoms() if a.GetIsAromatic())}, "
          f"conformers={rd.GetNumConformers()}, "
          f"monomerInfo={rd.GetAtomWithIdx(0).GetPDBResidueInfo() is not None}")


def baseline():
    """Run ProLIF on its bundled example, which is known to produce contacts.

    Returns:
        Number of contacts found.
    """
    u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
    lig = plf.Molecule.from_mda(u.select_atoms("resname LIG"))
    prot = plf.Molecule.from_mda(u.select_atoms("protein"))
    fp = plf.Fingerprint(["Hydrophobic", "HBDonor", "HBAcceptor"], count=True)
    fp.run_from_iterable([lig], prot, progress=False)
    describe(lig, "bundled ligand")
    describe(prot, "bundled protein")
    return counts(fp)


if __name__ == "__main__":
    print("bundled example contacts:", baseline())
