"""Receptor PDBQT from the run's PQR: PQR charges, AutoDock types, united-atom hydrogens."""
import numpy as np

from qpocket.core.screening import receptor_pdbqt
from qpocket.io.tables import read_pqr

# A tautomer-aware histidine (HIE: proton on NE2) and a serine, as pdb2pqr --ffout=AMBER writes them.
PQR = """\
ATOM      1  N   HIE A  12       0.000   0.000   0.000 -0.4157 1.8240
ATOM      2  H   HIE A  12      -0.900   0.300   0.000  0.2719 0.6000
ATOM      3  CA  HIE A  12       1.450   0.000   0.000  0.0188 1.9080
ATOM      4  HA  HIE A  12       1.800  -1.000   0.000  0.0881 1.1000
ATOM      5  CB  HIE A  12       2.000   1.400   0.000 -0.0074 1.9080
ATOM      6  CG  HIE A  12       3.500   1.400   0.000  0.1868 1.9080
ATOM      7  ND1 HIE A  12       4.200   2.500   0.000 -0.5432 1.8240
ATOM      8  CE1 HIE A  12       5.400   2.100   0.000  0.1635 1.9080
ATOM      9  NE2 HIE A  12       5.500   0.800   0.000 -0.2795 1.8240
ATOM     10  HE2 HIE A  12       6.300   0.200   0.000  0.3339 0.6000
ATOM     11  CD2 HIE A  12       4.300   0.300   0.000 -0.2207 1.9080
ATOM     12  OG  SER A  13       8.000   8.000   8.000 -0.6546 1.7210
ATOM     13  HG  SER A  13       8.900   8.000   8.000  0.4275 0.0000
HETATM 9999 ZN ION     1      10.000  10.000  10.000  2.0000 1.3900
"""


def _records(text):
    return {line[12:16].strip(): (line[77:79].strip(), float(line[70:76])) for line in text.splitlines()
            if line.startswith(("ATOM", "HETATM"))}


def test_types_follow_protonation_and_aromaticity(tmp_path):
    (tmp_path / "r.pqr").write_text(PQR)
    rec = _records(receptor_pdbqt(read_pqr(tmp_path / "r.pqr")))
    assert rec["ND1"][0] == "NA" and rec["NE2"][0] == "N"         # HIE: ND1 is the acceptor
    assert rec["CG"][0] == "A" and rec["CE1"][0] == "A" and rec["CB"][0] == "C"
    assert rec["H"][0] == "HD" and rec["HE2"][0] == "HD" and rec["HG"][0] == "HD"
    assert rec["OG"][0] == "OA" and rec["ZN"][0] == "Zn"


def test_nonpolar_hydrogens_are_merged_into_their_carbon(tmp_path):
    (tmp_path / "r.pqr").write_text(PQR)
    atoms = read_pqr(tmp_path / "r.pqr")
    text = receptor_pdbqt(atoms)
    rec = _records(text)
    assert "HA" not in rec
    assert np.isclose(rec["CA"][1], 0.0188 + 0.0881, atol=1e-3)
    total = sum(float(line[70:76]) for line in text.splitlines() if line.startswith(("ATOM", "HETATM")))
    assert np.isclose(total, float(atoms.charges.sum()), atol=0.01)   # charge is conserved


def test_lines_are_fixed_column_pdbqt(tmp_path):
    (tmp_path / "r.pqr").write_text(PQR)
    line = receptor_pdbqt(read_pqr(tmp_path / "r.pqr")).splitlines()[0]
    assert line.startswith("ATOM      1  N   HIE A  12") and len(line) >= 79
    assert float(line[30:38]) == 0.0 and line[77:79].strip() == "N"
