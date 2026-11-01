"""One pdb2pqr run per structure; the ion-repaired PQR feeds every feature."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PQR = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000 -0.4000 1.5000
ATOM      2  CA  ALA A   1       1.538   1.000   1.000  0.1000 2.0000
HETATM 9999 ZN ION     1       5.000   5.000   5.000  2.0000 1.3900
"""


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


