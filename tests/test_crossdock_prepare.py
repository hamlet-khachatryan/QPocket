"""prepare builds a self-contained, relocatable cross-docking folder."""
import json

import numpy as np
import pandas as pd
import pytest

from qpocket.io import layout


@pytest.fixture
def run(tmp_path, monkeypatch):
    from qpocket.arms.screen import crossdock

    ws = layout.create(str(tmp_path), "RUN")
    for rid, offset in (("R1", 0.0), ("R2", 10.0)):
        (ws.pockets / rid).mkdir(parents=True)
        pd.DataFrame({"x": [offset, offset + 1], "y": [0.0, 1.0], "z": [0.0, 1.0]}).to_csv(
            ws.pockets / rid / "binding_site.csv", index=False)
        (ws.cleaned / f"{rid}_clean.pdb").write_text("ATOM\n")
    pd.DataFrame({"pdb_id": ["R1", "R2"], "cleaned_file": [str(ws.cleaned / "R1_clean.pdb"),
                                                           str(ws.cleaned / "R2_clean.pdb")],
                  "aligned_file": ["a", "b"], "ligand_file": ["l1.pdb", "l2.pdb"]}
                 ).to_csv(ws.metadata / "master.csv", index=False)
    pd.DataFrame({"pdb_id": ["R1", "R2"]}).to_csv(ws.metadata / "selected_representatives.csv", index=False)
    (ws.screening / "actives.smi").write_text("SMILES\tPDB_ID\tLigand_Name\nCCO\tR1\tETH\nCCO\tR2\tETH\n")
    def fake_receptor(pdb, pocket, out, rid, cfg, sequences=None):
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.pdbqt").write_text("x")
        return out / f"{rid}.pdbqt"

    monkeypatch.setattr(crossdock, "prepare_receptor", fake_receptor)
    monkeypatch.setattr(crossdock, "prepare_ligands",
                        lambda actives, out, cfg: {a: out / f"{a}.pdbqt" for a in actives["active_id"]})
    return ws


def test_the_box_covers_every_representative_pocket(run):
    from qpocket.arms.screen.crossdock import shared_box

    box = shared_box(run, ["R1", "R2"], "representative_pockets", 5.0)
    assert box["min_x"] == pytest.approx(-5.0) and box["max_x"] == pytest.approx(16.0)


def test_prepare_writes_a_relocatable_folder(run):
    from qpocket.arms.screen import crossdock

    plan = crossdock.prepare(run.root)
    folder = run.screening / "crossdock"
    jobs = pd.read_csv(folder / "docking_jobs.csv")
    assert len(jobs) == 2 and set(jobs["representative"]) == {"R1", "R2"}
    assert all(not str(p).startswith("/") for col in ("protein_pdbqt", "ligand_pdbqt", "output_pdbqt")
               for p in jobs[col])
    assert set(jobs["engine"]) == {"quickvina"}
    assert (folder / "tasks.txt").read_text().count("qpocket screen run-task") == 2
    assert "SLURM_SUBMIT_DIR" in (folder / "run_docking.sbatch").read_text()
    assert json.loads((folder / "box.json").read_text())["mode"] == "representative_pockets"
    assert pd.read_csv(folder / "ligands.csv")["active_id"].tolist() == ["ETH"]
    assert plan.jobs_csv == folder / "docking_jobs.csv"
