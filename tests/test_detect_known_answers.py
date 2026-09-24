"""The detector reproduces binding modes described in the literature."""
from pathlib import Path

import pytest

from qpocket.io import interactions as interactions_io

FIXTURES = Path(__file__).resolve().parent / "fixtures/complexes"
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "3PTB_complex.pdb").exists(),
    reason="run scripts/make_complex_fixtures.py first")


def residues(entries):
    """Residue keys without the chain suffix, for readable assertions."""
    return {entry.split("|")[0].rsplit("_", 1)[0] for entry in entries}


@needs_fixtures
def test_trypsin_benzamidine_salt_bridge():
    """Benzamidine's amidine pairs with Asp189 at the bottom of the S1 pocket."""
    found = interactions_io.interactions_for(FIXTURES / "3PTB_complex.pdb", "BEN")
    assert "ASP_189" in residues(found)
    assert any(entry.startswith("ASP_189") and "Saltbridge" in entry for entry in found)
    assert {"SER_190", "GLY_219"} & residues(found)


@needs_fixtures
def test_detection_is_deterministic():
    runs = [interactions_io.interactions_for(FIXTURES / "3PTB_complex.pdb", "BEN")
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.skipif(not (FIXTURES / "1STP_complex.pdb").exists(),
                    reason="run scripts/make_complex_fixtures.py first")
def test_streptavidin_biotin_hydrogen_bond_network():
    """Biotin's ureido group hydrogen-bonds Asn23, Ser27 and Asp128."""
    found = interactions_io.interactions_for(FIXTURES / "1STP_complex.pdb", "BTN")
    assert {"ASN_23", "SER_27", "ASP_128"} <= residues(found)


@pytest.mark.skipif(not (FIXTURES / "7KG6_complex.pdb").exists(),
                    reason="run scripts/make_complex_fixtures.py first")
def test_parg_adenine_stacks_on_phe902():
    """The adenine of the PARG ligand stacks on Phe902."""
    found = interactions_io.interactions_for(FIXTURES / "7KG6_complex.pdb", "WDG")
    assert any(entry.startswith("PHE_902") and "Pi-stacking" in entry for entry in found)
