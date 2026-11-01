"""prepare builds a self-contained, relocatable cross-docking folder."""
import json

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
    def fake_receptor(pdb, pocket, out, rid, cfg, sequences=None, failures=None):
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.pdbqt").write_text("x")
        return out / f"{rid}.pdbqt"

    monkeypatch.setattr(crossdock, "prepare_receptor", fake_receptor)
    monkeypatch.setattr(crossdock, "prepare_ligands",
                        lambda actives, out, cfg: {a: out / f"{a}.pdbqt" for a in actives["active_id"]})
    return ws


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A run with one site whose metadata, pockets and actives live under sites/site_01."""
    from qpocket.arms.screen import crossdock

    run_ws = layout.create(str(tmp_path), "RUN")
    ws = layout.site_workspace(run_ws, "site_01")
    for rid, offset in (("R1", 0.0), ("R2", 10.0)):
        (ws.pockets / rid).mkdir(parents=True)
        pd.DataFrame({"x": [offset, offset + 1], "y": [0.0, 1.0], "z": [0.0, 1.0]}).to_csv(
            ws.pockets / rid / "binding_site.csv", index=False)
        (ws.cleaned / f"{rid}_clean.pdb").write_text("ATOM\n")
        (ws.original / f"{rid}.pdb").write_text("ATOM\n")
    pd.DataFrame({"pdb_id": ["R1", "R2"], "cleaned_file": [str(ws.cleaned / "R1_clean.pdb"),
                                                           str(ws.cleaned / "R2_clean.pdb")],
                  "aligned_file": ["a", "b"], "ligand_file": ["l1.pdb", "l2.pdb"]}
                 ).to_csv(ws.metadata / "master.csv", index=False)
    pd.DataFrame({"pdb_id": ["R1", "R2"]}).to_csv(ws.metadata / "selected_representatives.csv", index=False)
    (ws.metadata / "clustering_params.json").write_text(json.dumps({"mode": "shape_only"}))
    (ws.screening / "actives.smi").write_text(
        "SMILES\tPDB_ID\tLigand_Name\tLigand_File\nCCO\tR1\tETH\tl1.pdb\nCCO\tR2\tETH\tl2.pdb\n")
    seen = {}

    def fake_receptor(pdb, pocket, out, rid, cfg, sequences=None, failures=None):
        seen[rid] = sequences
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.pdbqt").write_text("x")
        return out / f"{rid}.pdbqt"

    monkeypatch.setattr(crossdock, "prepare_receptor", fake_receptor)
    monkeypatch.setattr(crossdock, "prepare_ligands",
                        lambda actives, out, cfg: {a: out / f"{a}.pdbqt" for a in actives["active_id"]})
    monkeypatch.setattr("qpocket.io.repair.chain_sequences", lambda path: {"A": [path.name]})
    return ws, seen


def test_crossdock_prepare_on_site_dir(site):
    from qpocket.arms.screen import crossdock

    ws, seen = site
    crossdock.prepare(ws.root)
    folder = ws.root / "screening" / "crossdock"
    assert seen == {"R1": {"A": ["R1.pdb"]}, "R2": {"A": ["R2.pdb"]}}  # run's originals reached repair
    assert set(pd.read_csv(folder / "docking_jobs.csv")["representative"]) == {"R1", "R2"}
    assert json.loads((folder / "box.json").read_text())["clustering_mode"] == "shape_only"
    used = folder / "representatives_used"
    assert (used / "selected_representatives.csv").exists() and (used / "clustering_params.json").exists()


def _failing(reasons):
    """A prepare_receptor that fails with ``reasons[rid]`` and succeeds for the others."""
    def fake(pdb, pocket, out, rid, cfg, sequences=None, failures=None):
        if rid in reasons:
            failures[rid] = reasons[rid]
            return None
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.pdbqt").write_text("x")
        return out / f"{rid}.pdbqt"
    return fake


def test_every_receptor_failing_names_each_reason(run, monkeypatch):
    from qpocket.arms.screen import crossdock
    from qpocket.arms.screen.receptors import ReceptorPreparationError

    monkeypatch.setattr(crossdock, "prepare_receptor", _failing(
        {"R1": "ImportError: No module named 'openmm'", "R2": "RuntimeError: pdb2pqr exploded"}))
    with pytest.raises(ReceptorPreparationError) as e:
        crossdock.prepare(run.root)
    msg = str(e.value)
    assert "No module named 'openmm'" in msg and "uv sync --extra screen" in msg
    assert "pdb2pqr exploded" in msg and "no coordinates" not in msg


def test_the_cli_prints_the_failures_without_a_traceback(run, monkeypatch):
    from click.testing import CliRunner

    from qpocket.arms.screen import crossdock
    from qpocket.cli.screen import screen_cli

    monkeypatch.setattr(crossdock, "prepare_receptor", _failing({"R1": "RuntimeError: boom"}))
    partial = CliRunner().invoke(screen_cli, ["prepare", "--workdir", str(run.root)])
    assert partial.exit_code == 0 and "boom" in partial.output, partial.output

    monkeypatch.setattr(crossdock, "prepare_receptor", _failing(
        {"R1": "ImportError: No module named 'openmm'", "R2": "ImportError: No module named 'openmm'"}))
    failed = CliRunner().invoke(screen_cli, ["prepare", "--workdir", str(run.root)])
    assert failed.exit_code != 0 and "uv sync --extra screen" in failed.output
    assert "Traceback" not in failed.output and not isinstance(failed.exception, crossdock.ReceptorPreparationError)


# --- prepare prefers the final representatives and the saved box ---------------

def _finalise(ws, reps):
    """A minimal current final for the site fixture: these representatives, the applied clustering's hash."""
    from qpocket.io.tables import file_hash

    meta = ws.metadata
    pd.DataFrame({"pdb_id": ["R1", "R2"], "cluster_id": [0, 1], "is_representative": [True, True],
                  "dist_to_exemplar": [0.0, 0.0]}).to_csv(meta / "clustering_metadata.csv", index=False)
    folder = ws.root / "final"
    folder.mkdir(exist_ok=True)
    pd.DataFrame({"pdb_id": ["R1", "R2"], "cluster": ["S1", "S2"], "cluster_id": [0, 1],
                  "is_representative": [r in reps for r in ("R1", "R2")], "dist_to_representative": 0.0,
                  "holo": True}).to_csv(folder / "clusters.csv", index=False)
    for name in ("residue_profile.csv", "ligands.csv"):
        (folder / name).write_text("cluster\n")
    (folder / "settings.json").write_text(json.dumps({"clustering_hash": file_hash(meta / "clustering_metadata.csv")}))


def _saved(ws, final_hash, **bounds):
    box = {"mode": "representative_pockets", "padding": 1.0, "final_hash": final_hash,
           **{f"{s}_{a}": v for s, v in (("min", -1.0), ("max", 1.0)) for a in "xyz"},
           "center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "size_x": 2.0, "size_y": 2.0, "size_z": 2.0, **bounds}
    (ws.screening / "box.json").write_text(json.dumps(box))


def test_prepare_uses_the_final_representatives(site):
    from qpocket.arms.screen import crossdock

    ws, _ = site
    _finalise(ws, {"R2"})
    crossdock.prepare(ws.root)
    folder = ws.screening / "crossdock"
    assert set(pd.read_csv(folder / "docking_jobs.csv")["representative"]) == {"R2"}
    assert pd.read_csv(folder / "representatives_used" / "selected_representatives.csv")["pdb_id"].tolist() == ["R2"]


def test_a_box_saved_for_an_older_final_is_ignored(site):
    from qpocket.arms.screen import crossdock

    ws, _ = site
    _finalise(ws, {"R1", "R2"})
    _saved(ws, "an-older-final", max_x=42.0)
    plan = crossdock.prepare(ws.root)
    written = json.loads((ws.screening / "crossdock" / "box.json").read_text())
    assert written["source"] == "config" and written["max_x"] != 42.0
    assert any("predates the site's current final" in w for w in plan.warnings)


def test_prepare_refuses_while_a_local_run_is_alive(site):
    import subprocess
    import sys

    from qpocket.arms.screen import crossdock

    ws, _ = site
    crossdock.prepare(ws.root)
    folder = ws.screening / "crossdock"
    run = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)  # screen run-local"])
    try:
        (folder / "local_run.json").write_text(json.dumps({"pid": run.pid, "started_at": "t", "n_jobs": 2}))
        with pytest.raises(RuntimeError, match="still running"):
            crossdock.prepare(ws.root)
    finally:
        run.kill()
        run.wait()


