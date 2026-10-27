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
    def fake_receptor(pdb, pocket, out, rid, cfg, sequences=None, failures=None):
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


def test_reclustering_after_prepare_does_not_change_docking(site, monkeypatch):
    from qpocket.arms.screen import crossdock

    ws, _ = site
    crossdock.prepare(ws.root)
    folder = ws.screening / "crossdock"
    assert not list(folder.glob(".representatives_used*"))  # the staging folder is gone
    pd.DataFrame({"pdb_id": ["R2"]}).to_csv(ws.metadata / "selected_representatives.csv", index=False)
    (ws.metadata / "clustering_params.json").write_text(json.dumps({"mode": "two_stage"}))
    jobs = pd.read_csv(folder / "docking_jobs.csv")
    stray = jobs.iloc[[0]].assign(job_id="job_stray", representative="R9")
    pd.concat([jobs, stray]).to_csv(folder / "docking_jobs.csv", index=False)
    scored, logged = [], []

    def fake_pose_table(folder, jobs, actives):
        scored.extend(jobs["representative"])
        return pd.DataFrame(columns=["job_id", "representative", "active_id", "rank", "score", "rmsd"])

    def record(**fields):
        logged.append(fields)

    monkeypatch.setattr(crossdock, "pose_table", fake_pose_table)
    monkeypatch.setattr(crossdock, "log_message", record)
    crossdock.analyze(ws.root)
    assert sorted(scored) == ["R1", "R2"]  # the snapshot, not the re-clustered R2 alone; R9 dropped
    assert {"message_type": "crossdock_jobs_outside_snapshot", "dropped": 1} in logged
    used = folder / "representatives_used"
    assert pd.read_csv(used / "selected_representatives.csv")["pdb_id"].tolist() == ["R1", "R2"]
    assert json.loads((used / "clustering_params.json").read_text())["mode"] == "shape_only"
    assert json.loads((folder / "box.json").read_text())["clustering_mode"] == "shape_only"


def test_a_failed_prepare_leaves_the_previous_snapshot(site, monkeypatch):
    from qpocket.arms.screen import crossdock

    ws, _ = site
    crossdock.prepare(ws.root)
    used = ws.screening / "crossdock" / "representatives_used"
    pd.DataFrame({"pdb_id": ["R2"]}).to_csv(ws.metadata / "selected_representatives.csv", index=False)

    def broken(*args, **kwargs):
        raise RuntimeError("ligand preparation failed")

    monkeypatch.setattr(crossdock, "prepare_ligands", broken)
    with pytest.raises(RuntimeError):
        crossdock.prepare(ws.root)
    assert pd.read_csv(used / "selected_representatives.csv")["pdb_id"].tolist() == ["R1", "R2"]


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


def test_some_receptors_failing_still_prepares_the_rest(run, monkeypatch):
    from qpocket.arms.screen import crossdock

    logged = []
    monkeypatch.setattr(crossdock, "prepare_receptor", _failing({"R1": "RuntimeError: boom"}))
    monkeypatch.setattr(crossdock, "log_message", lambda **fields: logged.append(fields))
    crossdock.prepare(run.root)
    folder = run.screening / "crossdock"
    assert (folder / "box.json").exists()
    assert set(pd.read_csv(folder / "docking_jobs.csv")["representative"]) == {"R2"}
    warnings = [f for f in logged if f.get("stage") == "prepare_receptor"]
    assert warnings and "R1" in warnings[0]["failed"] and "boom" in warnings[0]["failed"]


def test_identical_reasons_are_grouped():
    from qpocket.arms.screen.receptors import describe_failures

    assert describe_failures({"A": "X", "B": "X"}) == "X (2 representatives: A, B)"
    many = describe_failures({f"R{i}": "X" for i in range(7)})
    assert many == "X (7 representatives: R0, R1, R2, R3, R4, …)"


def test_a_receptor_conversion_failure_is_reported(tmp_path, monkeypatch):
    from qpocket.arms.screen import receptors
    from qpocket.screen_config import ScreenConfig

    cfg = ScreenConfig()
    cfg.receptor.repair = False
    monkeypatch.setattr(receptors, "_protonate", lambda pdb, pqr, cfg: None)
    monkeypatch.setattr(receptors, "receptor_from_pqr", lambda pqr, pdbqt: False)
    cleaned = tmp_path / "R1_clean.pdb"
    cleaned.write_text("ATOM\n")
    failures = {}
    assert receptors.prepare_receptor(cleaned, None, tmp_path / "out", "R1", cfg, failures=failures) is None
    assert failures == {"R1": "PDBQT conversion failed"}


def test_no_receptors_leaves_the_previous_snapshot(site, monkeypatch):
    from qpocket.arms.screen import crossdock
    from qpocket.arms.screen.receptors import ReceptorPreparationError

    ws, _ = site
    crossdock.prepare(ws.root)
    used = ws.screening / "crossdock" / "representatives_used"
    pd.DataFrame({"pdb_id": ["R2"]}).to_csv(ws.metadata / "selected_representatives.csv", index=False)
    monkeypatch.setattr(crossdock, "prepare_receptor", _failing({"R2": "RuntimeError: boom"}))
    with pytest.raises(ReceptorPreparationError):
        crossdock.prepare(ws.root)
    assert pd.read_csv(used / "selected_representatives.csv")["pdb_id"].tolist() == ["R1", "R2"]


def test_a_partial_failure_is_on_the_plan_for_the_cli_to_print(run, monkeypatch):
    from qpocket.arms.screen import crossdock

    monkeypatch.setattr(crossdock, "prepare_receptor", _failing({"R1": "RuntimeError: boom"}))
    plan = crossdock.prepare(run.root)
    assert len(plan.warnings) == 1 and "R1" in plan.warnings[0] and "boom" in plan.warnings[0]


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
