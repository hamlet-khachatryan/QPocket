"""One pdb2pqr run per structure; the ion-repaired PQR feeds every feature."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qpocket.constants import atomic_hydrophobicity
from qpocket.io import tables

PQR = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000 -0.4000 1.5000
ATOM      2  CA  ALA A   1       1.538   1.000   1.000  0.1000 2.0000
HETATM 9999 ZN ION     1       5.000   5.000   5.000  2.0000 1.3900
"""


def test_read_pqr_keeps_radius_chain_and_resseq(tmp_path):
    (tmp_path / "x.pqr").write_text(PQR)
    atoms = tables.read_pqr(tmp_path / "x.pqr")
    assert np.allclose(atoms.radii, [1.5, 2.0, 1.39])
    assert atoms.chain_ids == ["A", "A", ""] and atoms.res_seqs == [1, 1, 1]
    assert atoms.charges[-1] == 2.0 and atoms.res_names[-1] == "ION"


def test_repaired_ions_are_metals():
    assert atomic_hydrophobicity("ZN", "ION") == -2.0 and atomic_hydrophobicity("CA", "ION") == -2.0


def _fake_pdb2pqr(calls):
    def run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_text(PQR)

        class Done:
            returncode = 0

            def communicate(self):
                return b"", b""
        return Done()
    return run


def test_featurize_runs_pdb2pqr_once_and_keeps_the_pqr(tmp_path, minimal_pdb):
    from qpocket.pipelines.structure import featurize

    pocket = tmp_path / "pockets" / "1ABC"
    pocket.mkdir(parents=True)
    pd.DataFrame(np.ones((3, 3)) * 2, columns=list("xyz")).to_csv(pocket / "binding_site.csv", index=False)
    calls, seen = [], {}

    def fake_apbs(pqr_path, points):
        seen["pqr"] = Path(pqr_path)
        return np.zeros(len(points))

    with patch("qpocket.io.tools.Popen", side_effect=_fake_pdb2pqr(calls)), \
         patch("qpocket.io.tools.apbs_potential", side_effect=fake_apbs):
        featurize(pocket / "binding_site.csv", minimal_pdb)
    assert len(calls) == 1 and "--ffout=AMBER" in calls[0]
    assert (pocket / "protein.pqr").exists() and seen["pqr"] == pocket / "protein.pqr"
    assert (pocket / "featurized.csv").exists()


def test_cp_sees_the_repaired_ion(tmp_path, minimal_pdb):
    from qpocket.io import tools

    (tmp_path / "p.pqr").write_text(PQR)
    data = tools.structure_data(minimal_pdb, tables.read_pqr(tmp_path / "p.pqr"))
    assert 2.0 in data.charges.tolist() and -2.0 in data.hydrophobicity.tolist()


def test_ion_repair_adds_real_ions_only(tmp_path):
    """A cofactor's alpha carbon named CA must not come back as a calcium ion."""
    from qpocket.io.tools import repair_pqr_with_missing_ions

    pdb = tmp_path / "p.pdb"
    pdb.write_text(
        "HETATM  100  CA  CME A 123      10.000  10.000  10.000  1.00  0.00           C\n"
        "HETATM  101 ZN    ZN A 900      20.000  20.000  20.000  1.00  0.00          ZN\n"
        "HETATM  102 CA    CA A 901      30.000  30.000  30.000  1.00  0.00          CA\n"
    )
    pqr = tmp_path / "p.pqr"
    pqr.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000 -0.4000 1.5000\n")
    repair_pqr_with_missing_ions(pdb, pqr)
    added = [line.split() for line in pqr.read_text().splitlines()[1:]]
    assert sorted(a[2] for a in added) == ["CA", "ZN"]
    assert all(float(a[5]) in (20.0, 30.0) for a in added)   # nothing at the CME carbon
