"""The reference structure belongs in its own run.

With pocket_definition "Interactions" the second pass happens to featurise the
reference. With the default "Solvent" nothing did, so the structure a run is
named after was silently missing from its own distance matrix.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiments/PARG/7KG6/structures/original"


@pytest.fixture(scope="module")
def solvent_run(tmp_path_factory):
    """A small fragment screen with the default pocket definition."""
    if not SOURCE.exists() or shutil.which("uv") is None:
        pytest.skip("needs the bundled PARG structures and uv")
    workdir = tmp_path_factory.mktemp("solvent")
    done = subprocess.run(
        [str(Path(sys.executable).parent / "qpocket"), "run",
         "--mode", "fragment_screen", "--input-type", "paths",
         "--reference-path", str(SOURCE / "7KG6.cif"),
         "--structure-paths", str(SOURCE / "7KG6.cif"),
         "--structure-paths", str(SOURCE / "4B1H.cif"),
         "--ligand-id", "WDG", "--chain-id", "A",
         "--pocket-definition", "Solvent",
         "--workdir", str(workdir)],
        cwd=REPO, capture_output=True, text=True, timeout=2400,
    )
    run = workdir / "7KG6"
    if not (run / "metadata" / "master.csv").exists():
        pytest.fail(f"the run did not finish: {done.stderr[-2000:]}")
    return run


@pytest.mark.golden
def test_the_reference_gets_a_featurised_pocket(solvent_run):
    """It took a branch that extracted its ligand and nothing else."""
    master = pd.read_csv(solvent_run / "metadata/master.csv")
    featurised = {p.parent.name for p in (solvent_run / "pockets").glob("*/featurized.csv")}
    missing = sorted(set(master["pdb_id"].astype(str)) - featurised)
    assert not missing, f"no featurized.csv for {missing}"


@pytest.mark.golden
def test_the_reference_reaches_the_clustering(solvent_run):
    master = pd.read_csv(solvent_run / "metadata/master.csv")
    clustered = pd.read_csv(solvent_run / "metadata/clustering_metadata.csv")
    missing = sorted(set(master["pdb_id"].astype(str))
                     - set(clustered["pdb_id"].astype(str)))
    assert not missing, f"absent from the clustering: {missing}"
