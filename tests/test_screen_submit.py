"""Submitting a cross-docking: how many dockings, a background run here, or commands for a VM over SSH."""
import json
import subprocess
import sys

import pandas as pd
import pytest

from qpocket.arms.screen import submit
from qpocket.screen_config import RemoteConfig, ScreenConfig


def _final(site, reps=("R1", "R2"), smiles=("CCO", "CCO", "CCN", "")):
    folder = site / "final"
    folder.mkdir(parents=True)
    pd.DataFrame({"pdb_id": ["R1", "R2", "R3"], "cluster": ["S1", "S2", "S2"], "cluster_id": [0, 1, 1],
                  "is_representative": [p in reps for p in ("R1", "R2", "R3")], "dist_to_representative": 0.0,
                  "holo": True}).to_csv(folder / "clusters.csv", index=False)
    pd.DataFrame({"cluster": "S1", "pdb_id": "R1", "ligand_key": [f"k{i}" for i in range(len(smiles))],
                  "resname": "LIG", "smiles": list(smiles)}).to_csv(folder / "ligands.csv", index=False)
    (folder / "residue_profile.csv").write_text("cluster,residue,contacted_pct,lining_pct\n")
    (folder / "settings.json").write_text("{}")


def test_pair_count_is_reps_times_distinct_compounds(tmp_path):
    site = tmp_path / "sites" / "S1"
    _final(site)
    assert submit.pair_count(tmp_path, "S1") == (2, 2, 4)


def test_engines_mark_missing_gnina(monkeypatch):
    def resolve(executable):
        if "gnina" in str(executable):
            raise FileNotFoundError("gnina not found")
        return f"/bin/{executable}"

    monkeypatch.setattr(submit, "resolve_executable", resolve)
    found = submit.engines(ScreenConfig())
    assert found["gnina"] is None and found["quickvina"] and found["vina"]


def _folder(tmp_path, n=3, done=1):
    folder = tmp_path / "crossdock"
    (folder / "poses").mkdir(parents=True)
    pd.DataFrame({"job_id": [f"j{i}" for i in range(n)],
                  "output_pdbqt": [f"poses/j{i}.pdbqt" for i in range(n)]}).to_csv(folder / "docking_jobs.csv",
                                                                                    index=False)
    for i in range(done):
        (folder / "poses" / f"j{i}.pdbqt").write_text("MODEL 1\n")
    return folder


def test_start_local_records_and_refuses_a_second_run(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    started = []

    class FakePopen:
        pid = 4242

        def __init__(self, argv, **kwargs):
            started.append((argv, kwargs.get("cwd")))

    monkeypatch.setattr(submit.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(submit, "_alive", lambda pid: pid == 4242)
    record = submit.start_local(folder, workers=2, cpus=4)
    assert record["pid"] == 4242 and record["n_jobs"] == 3
    assert json.loads((folder / "local_run.json").read_text())["pid"] == 4242
    argv, cwd = started[0]
    assert argv[-6:] == ["run-local", "docking_jobs.csv", "--workers", "2", "--cpus", "4"] and cwd == str(folder)
    with pytest.raises(RuntimeError, match="already running"):
        submit.start_local(folder, workers=2, cpus=4)


def test_progress_of_a_dead_run(tmp_path):
    folder = _folder(tmp_path, n=3, done=1)
    assert submit.local_progress(folder) == {"done": 1, "total": 3, "running": False, "started_at": None}
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (folder / "local_run.json").write_text(json.dumps({"pid": dead.pid, "started_at": "t", "n_jobs": 3}))
    progress = submit.local_progress(folder)
    assert progress["running"] is False and progress["done"] == 1 and progress["started_at"] == "t"


def test_remote_commands_quote_paths(tmp_path):
    import shlex

    remote = RemoteConfig(host="comp-bio-2", directory="/data/my runs", setup=["source ~/qp/.venv/bin/activate"])
    commands = submit.remote_commands(tmp_path / "crossdock", "S1-G'127", remote, workers=8, cpus=2)
    assert set(commands) == {"push", "start", "poll", "pull"}
    quoted = shlex.quote("/data/my runs/S1-G'127")
    start = commands["start"]
    assert start[:2] == ["ssh", "comp-bio-2"]
    tmux = shlex.split(start[2])
    assert tmux[:5] == ["tmux", "new", "-d", "-s", "qpocket_S1-G_127"]
    inner = tmux[5]
    assert "/data/my runs/S1-G'127" in shlex.split(inner.replace(";", " ; "))
    assert "qpocket screen run-local docking_jobs.csv --workers 8 --cpus 2" in inner
    assert any(quoted in part for part in commands["push"]) and any(quoted in part for part in commands["pull"])
    assert commands["push"][0] == "rsync" and "--exclude" in commands["push"]


def test_remote_status_sees_other_jobs():
    status = submit.remote_status("12\nqpocket_S1-G127: 1 windows\nsomeone_md: 2 windows\n", "S1-G127")
    assert status == {"poses": 12, "own_session": True, "other_jobs": ["someone_md"]}
    assert submit.remote_status("", "S1")["poses"] == 0


def test_vm_power_state_needs_az_and_a_vm(monkeypatch):
    assert submit.vm_power_state(RemoteConfig(host="h")) is None
    monkeypatch.setattr(submit.shutil, "which", lambda name: None)
    assert submit.vm_power_state(RemoteConfig(host="h", azure_vm="comp-bio-2", azure_resource_group="rg")) is None
