"""Physical constants, residue tables and feature parameters."""
import numpy as np
from typing import Set

FP_DTYPE = np.float32

###
# Featurisation parameters and functional
###
SIGMA_HBOND = 2.2
SIGMA_AROM = 2.7
SIGMA_HYDRO = 1.5

ION_PARAMS = {
    "MG": (2.0, 1.4),
    "ZN": (2.0, 1.1),
    "CA": (2.0, 1.7),
    "FE": (3.0, 1.0),
    "NA": (1.0, 1.9),
    "K": (1.0, 2.3),
    "CL": (-1.0, 1.8),
    "MN": (2.0, 1.4),
    "CU": (2.0, 1.4),
    "NI": (2.0, 1.4),
}

DONOR_RESIDUES: Set[str] = {
    "ARG",
    "LYS",
    "HIS",
    "TRP",
    "ASN",
    "GLN",
    "TYR",
    "SER",
    "THR",
    "HIP",
    "HIE",
    "HID",
    "LYN",
    "ARN",
}

ACCEPTOR_RESIDUES: Set[str] = {
    "ASP",
    "GLU",
    "ASN",
    "GLN",
    "TYR",
    "SER",
    "THR",
    "HIS",
    "MET",
    "ASH",
    "GLH",
    "HIP",
    "HIE",
    "HID",
}

AROMATIC_RESIDUES: Set[str] = {"PHE", "TYR", "TRP", "HIS", "HIP", "HIE", "HID"}


#: Metal ions scored as charged (-2) by the lipophilic potential.
METALS = frozenset({"MG", "ZN", "CA", "FE", "MN", "CO", "CU", "NI"})

# pdb2pqr's AMBER names encode protonation; the scores are per standard residue.
_AMBER_TO_STANDARD = {"ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "CYX": "CYS", "CYM": "CYS",
                      "HID": "HIS", "HIE": "HIS", "HIP": "HIS"}


def canonical_residue(res_name: str) -> str:
    """Standard residue name for an AMBER protonation variant.

    Args:
        res_name: Residue name, e.g. HIE.

    Returns:
        Standard three-letter name, e.g. HIS.
    """
    name = res_name.strip().upper()
    return _AMBER_TO_STANDARD.get(name, name)


def element_of(atom_name: str, res_name: str) -> str:
    """Element of an atom from its PDB name; PQR files carry no element column.

    Args:
        atom_name: PDB atom name.
        res_name: Residue name.

    Returns:
        Upper-case element symbol.
    """
    name, res = atom_name.strip().upper(), res_name.strip().upper()
    if name in METALS and (name != "CA" or res in ("CA", "ION")):
        return name  # CA is calcium only as an ion residue; elsewhere it is an alpha carbon
    if name.startswith("SE") and res == "MSE":
        return "SE"
    return name.lstrip("0123456789")[:1]


def atomic_hydrophobicity(atom_name: str, res_name: str) -> float:
    """Hydrophobicity score of a PDB atom, used by the lipophilic potential.

    Args:
        atom_name: PDB atom name.
        res_name: Residue name; AMBER variants are accepted.

    Returns:
        Score from -2 (charged) to 1 (aliphatic carbon).
    """
    if not atom_name:
        return 0.0
    element, res_name = element_of(atom_name, res_name), canonical_residue(res_name)

    if element in METALS:
        return -2.0

    if element == "C":
        if atom_name == "C":
            return 0.0
        if res_name in {"PHE", "TYR", "TRP", "HIS"}:
            if atom_name in {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "CZ2", "CZ3", "CH2"}:
                return 0.5
        return 1.0

    if element == "N":
        if atom_name == "N":
            return -0.5
        if res_name == "LYS" and atom_name == "NZ":
            return -2.0
        if res_name == "ARG" and atom_name in {"NH1", "NH2"}:
            return -2.0
        return -1.0

    if element == "O":
        if res_name in {"ASP", "GLU"} and atom_name in {"OD1", "OD2", "OE1", "OE2"}:
            return -2.0
        if atom_name == "O":
            return -1.0
        return -0.5

    if element in ("S", "SE"):
        return 0.6

    return 0.0


###
# Cleaning and alignmenr global variables
###

FASTA_UNIPROT_TEMPLATE = "https://rest.uniprot.org/uniprotkb/{}.fasta"

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

# Heavy atoms of each standard amino acid: a modified residue turned into its parent keeps only these.
_BACKBONE = ("N", "CA", "C", "O", "OXT")
STANDARD_AA_ATOMS = {name: frozenset(_BACKBONE + side) for name, side in {
    'ALA': ("CB",), 'ARG': ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), 'ASN': ("CB", "CG", "OD1", "ND2"),
    'ASP': ("CB", "CG", "OD1", "OD2"), 'CYS': ("CB", "SG"), 'GLN': ("CB", "CG", "CD", "OE1", "NE2"),
    'GLU': ("CB", "CG", "CD", "OE1", "OE2"), 'GLY': (), 'HIS': ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    'ILE': ("CB", "CG1", "CG2", "CD1"), 'LEU': ("CB", "CG", "CD1", "CD2"),
    'LYS': ("CB", "CG", "CD", "CE", "NZ"), 'MET': ("CB", "CG", "SD", "CE"),
    'PHE': ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"), 'PRO': ("CB", "CG", "CD"), 'SER': ("CB", "OG"),
    'THR': ("CB", "OG1", "CG2"), 'TRP': ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    'TYR': ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"), 'VAL': ("CB", "CG1", "CG2"),
}.items()}

THREE_TO_ONE_MAPPING = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}


# Crystallisation additives: never ligands, so they never form a site or enter
# the protonated complex. Cleaning strips them as non-amino acids; keep them out
# of cleaning.allowed_cofactors, which would parametrise them with the protein.
# A run that needs one as a ligand lists it in sites.keep_ligands.
BUFFER_RESNAMES = frozenset({
    # cryoprotectants and solvents
    'GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', 'P6G', '2PE', '12P', '15P',
    'P33', 'PE4', 'PE8', 'XPE', '7PE', 'MPD', 'MRD', 'PGO', 'PGR', 'IPA',
    'EOH', 'MOH', 'DMS', 'DMF', 'BU3',
    # buffers and buffer ions
    'TRS', 'EPE', 'MES', 'MPO', 'BTB', 'B3P', 'CXS', 'NHE', 'PIN', 'BCN',
    'IMD', 'CIT', 'FLC', 'TLA', 'TAR', 'MLI', 'SIN', 'ACT', 'ACY', 'FMT',
})
