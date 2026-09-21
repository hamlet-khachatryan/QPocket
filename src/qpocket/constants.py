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


def GET_ATOMIC_HYDROPHOBICITY(atom_name: str, res_name: str) -> float:
    """Hydrophobicity score of a PDB atom, used by the lipophilic potential.

    Args:
        atom_name: PDB atom name.
        res_name: Residue name.

    Returns:
        Score from -2 (charged) to 1 (aliphatic carbon).
    """
    if not atom_name:
        return 0.0
    element = atom_name[0]
    if element.isdigit() and len(atom_name) > 1:
        element = atom_name[1]


    if element in {"Mg", "Zn", "Ca", "Fe", "Mn", "Co", "Cu", "Ni"}:
        return -2.0

    if element == "C":
        if atom_name == "C":
            return 0.0
        if res_name in {"PHE", "TYR", "TRP", "HIS", "HIP", "HIE", "HID"}:
            if atom_name in {
                "CG",
                "CD1",
                "CD2",
                "CE1",
                "CE2",
                "CZ",
                "CZ2",
                "CZ3",
                "CH2",
            }:
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

    if element == "S" or element == "SE":
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

HET_TO_REMOVE = [
    "A", "C", "G", "K", "T", "U", "BR", "CA", "CD", "CL", "CO", "CS",
    "CU", "FE", "HG", "MG", "MN", "NA", "NI", "NO", "O2", "OH", "PI", "PT",
    "SR", "TL", "XE", "ZN", "1CP", "1MA", "1PE", "1PG", "2CP", "2GP", "2GP", "2MG",
    "3CP", "3DR", "3HC", "4CA", "4CO", "5GP", "5MC", "5MU", "6FA", "7MG", "8OG", "A3P",
    "ABA", "ACE", "ACO", "ACO", "ACP", "ACT", "ACY", "ADE", "ADJ", "ADN", "ADP", "ADP",
    "AF3", "AIB", "AKG", "AMP", "AMP", "AMX", "ANP", "ANP", "APC", "APR", "ATP", "ATP",
    "AZI", "B12", "BCA", "BCB", "BCL", "BCL", "BEN", "BEZ", "BGC", "BH4", "BLA", "BLV",
    "BMA", "BMA", "BME", "BOG", "BOG", "BPB", "BPH", "BPH", "BRO", "BTN", "C8E", "C8E",
    "CA3", "CA5", "CAA", "CAC", "CAN", "CAO", "CCH", "CDL", "CDP", "CEA", "CGU", "CH2",
    "CH3", "CIC", "CIT", "CIT", "CL1", "CL2", "CLA", "CLN", "CMC", "CME", "CMO", "CMP",
    "CMP", "CMX", "CNC", "CND", "CO3", "CO8", "COA", "COA", "COB", "COD", "COF", "COH",
    "COJ", "CON", "COS", "COT", "COY", "CP3", "CPS", "CRO", "CRY", "CRY", "CS8", "CSD",
    "CSO", "CSS", "CSW", "CTP", "CU1", "CXM", "CYN", "DAK", "DAL", "DAN", "DCA", "DCC",
    "DDH", "DEU", "DHE", "DHE", "DLE", "DMS", "DND", "DOX", "DPN", "DTT", "DTT", "DVA",
    "EDO", "EGL", "EOH", "EPE", "EPE", "ETA", "F3S", "F43", "F6P", "F6P", "FAA", "FAB",
    "FAD", "FAD", "FAE", "FAM", "FAS", "FCX", "FDA", "FE2", "FEC", "FES", "FLC", "FMA",
    "FME", "FMN", "FMN", "FMT", "FNE", "FNS", "FOR", "FPP", "FS3", "FS4", "FUC", "FUC",
    "GAL", "GAL", "GDP", "GDP", "GLC", "GLC", "GLU", "GLY", "GMP", "GNP", "GNP", "GOL",
    "GOL", "GOL", "GSH", "GTP", "GTP", "GTT", "H2U", "H4B", "HAS", "HAX", "HC4", "HDD",
    "HDM", "HE6", "HEA", "HEB", "HEC", "HEC", "HED", "HED", "HEG", "HEM", "HEM", "HEO",
    "HES", "HEV", "HIF", "HMG", "HNI", "HPA", "HXC", "HYP", "IMD", "IMP", "IOD", "IOH",
    "IPA", "IPH", "IUM", "KCX", "LAT", "LDA", "LDA", "LI1", "LI1", "LLP", "LYX", "M2G",
    "MAL", "MAL", "MAN", "MAN", "MCA", "MCD", "MDE", "MES", "MES", "MET", "MGD", "MGD",
    "MHM", "MK1", "MLC", "MLY", "MMP", "MO2", "MO3", "MO4", "MO5", "MO6", "MOH", "MP1",
    "MPD", "MPD", "MPO", "MSE", "MTE", "MTO", "MVA", "MYA", "MYR", "MYR", "NAC", "NAD",
    "NAD", "NAE", "NAG", "NAG", "NAH", "NAI", "NAI", "NAJ", "NAP", "NAP", "NAQ", "NAX",
    "NBD", "NBP", "NCO", "NDA", "NDC", "NDO", "NDP", "NDP", "NGA", "NGA", "NH2", "NH4",
    "NHD", "NHM", "NHO", "NLE", "NMO", "NMX", "NO3", "OCS", "ODP", "OMC", "OMG", "ORN",
    "OXY", "P6G", "PAD", "PC3", "PCA", "PCU", "PEG", "PEG", "PEO", "PEP", "PG4", "PG4",
    "PGA", "PGD", "PGE", "PHB", "PLM", "PLP", "PNI", "PO4", "POP", "POP", "POR", "PP9",
    "PSU", "PSU", "PTR", "PXZ", "PYR", "PYR", "RET", "RFL", "SAH", "SAM", "SAR", "SCA",
    "SCD", "SCN", "SCO", "SEO", "SEP", "SFP", "SIA", "SIN", "SND", "SO4", "SPM", "SPM",
    "SRM", "SUC", "SUL", "TAP", "TAR", "TDP", "TDP", "TMN", "TMP", "TMP", "TPO", "TPQ",
    "TRS", "TRS", "TTP", "TTP", "TYS", "U10", "UDP", "UDP", "UMP", "UMP", "UNL", "UNX",
    "UPG", "UTP", "VO4", "WO4", "XYP", "XYS", "XYS", "ZEM", "ZID", "ZN2", "ZN3", "ZNH"
]

THREE_TO_ONE_MAPPING = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

