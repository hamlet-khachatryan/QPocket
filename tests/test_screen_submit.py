"""Submitting a cross-docking: how many dockings, a background run here, or commands for a VM over SSH."""
import json

import pandas as pd
import pytest

from qpocket.arms.screen import submit
from qpocket.screen_config import RemoteConfig


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


def test_remote_commands_expand_home_and_quote_paths(tmp_path):
    import shlex

    remote = RemoteConfig(host="comp-bio-2", directory="~/qpocket_crossdock",
                          setup=["source ~/qp/.venv/bin/activate"])
    commands = submit.remote_commands(tmp_path / "crossdock", "S1-G127", remote, workers=8, cpus=2)
    assert set(commands) == {"push", "start", "poll", "pull"}
    start = commands["start"]
    assert start[:4] == ["ssh", "-o", "BatchMode=yes", "comp-bio-2"]
    tmux = shlex.split(start[4])
    assert tmux[:5] == ["tmux", "new", "-d", "-s", "qpocket_S1-G127"]
    inner = tmux[5]
    assert "cd ~/qpocket_crossdock/S1-G127" in inner and "'~" not in inner     # ~ left for the remote shell
    assert "source ~/qp/.venv/bin/activate && cd" in inner                       # a failed setup stops the run
    assert "run-local docking_jobs.csv --workers 8 --cpus 2" in inner and "> run_local.log 2>&1" in inner
    push, pull = commands["push"], commands["pull"]
    assert push[-1] == "comp-bio-2:qpocket_crossdock/S1-G127/"                  # relative to the remote home
    assert pull[-2] == "comp-bio-2:qpocket_crossdock/S1-G127/poses/"
    assert "--delete" in push and "BatchMode=yes" in " ".join(push)


def test_remote_commands_refuse_paths_rsync_cannot_carry(tmp_path):
    with pytest.raises(ValueError, match="letters, digits"):
        submit.remote_commands(tmp_path, "S1", RemoteConfig(host="h", directory="/data/my runs"), 1, 1)
    with pytest.raises(ValueError, match="letters, digits"):
        submit.remote_commands(tmp_path, "S1-G'127", RemoteConfig(host="h", directory="/data"), 1, 1)


def test_a_reused_pid_is_not_our_run(tmp_path):
    import os

    folder = _folder(tmp_path)
    (folder / "local_run.json").write_text(json.dumps({"pid": os.getpid(), "started_at": "t", "n_jobs": 3}))
    assert submit.local_progress(folder)["running"] is False   # pytest holds the pid, not run-local


