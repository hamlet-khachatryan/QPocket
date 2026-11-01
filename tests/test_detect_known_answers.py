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


@pytest.mark.skipif(not (FIXTURES / "1STP_complex.pdb").exists(),
                    reason="run scripts/make_complex_fixtures.py first")
def test_streptavidin_biotin_hydrogen_bond_network():
    """Biotin's ureido group hydrogen-bonds Asn23, Ser27 and Asp128."""
    found = interactions_io.interactions_for(FIXTURES / "1STP_complex.pdb", "BTN")
    assert {"ASN_23", "SER_27", "ASP_128"} <= residues(found)


