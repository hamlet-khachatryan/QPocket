"""Docking engines share one contract: a command line and a pose reader."""
from pathlib import Path

import pandas as pd
import pytest

from qpocket.core import screening as scr
from qpocket.core.ports import DOCKING
from qpocket.screen_config import DockingConfig, ScreenConfig, validate_screen_config

VINA_OUT = """MODEL 1
REMARK VINA RESULT:    -8.1      0.000      0.000
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  O   UNL     1       1.200   0.000   0.000  1.00  0.00     0.000 OA
ENDMDL
MODEL 2
REMARK VINA RESULT:    -7.4      1.900      2.600
ATOM      1  C   UNL     1       5.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  O   UNL     1       6.200   0.000   0.000  1.00  0.00     0.000 OA
ENDMDL
"""


def _job(**extra):
    row = {c: "1" for c in scr.JOB_COLUMNS}
    row.update(protein_pdbqt="r.pdbqt", ligand_pdbqt="l.pdbqt", output_pdbqt="poses/j.pdbqt")
    row.update(extra)
    return row


def test_every_vina_model_becomes_a_pose_in_rank_order():
    poses = scr.vina_poses(VINA_OUT)
    assert [(p.rank, p.score) for p in poses] == [(1, -8.1), (2, -7.4)]
    assert poses[1].mol.GetConformer().GetAtomPosition(0).x == pytest.approx(5.0)


@pytest.mark.parametrize("name,exe,ext", [("quickvina", "qvina2", ".pdbqt"), ("vina", "vina", ".pdbqt"),
                                          ("gnina", "gnina", ".sdf")])
def test_each_engine_declares_its_binary_and_output(name, exe, ext):
    engine = DOCKING.get(name).fn()
    assert (engine.executable, engine.output_ext) == (exe, ext)


@pytest.mark.parametrize("name", ["quickvina", "vina"])
def test_vina_family_commands_are_todays(name):
    job = _job(engine=name)
    assert DOCKING.get(name).fn().argv(job, "/bin/x", 4) == scr.vina_argv(job, "/bin/x", 4)


def test_gnina_command_carries_the_box_and_output():
    argv = DOCKING.get("gnina").fn().argv(_job(output_pdbqt="poses/j.sdf"), "/bin/gnina", 2)
    assert argv[:5] == ["/bin/gnina", "-r", "r.pdbqt", "-l", "l.pdbqt"]
    assert argv[argv.index("-o") + 1] == "poses/j.sdf" and "--center_x" in argv and "--cpu" in argv


def test_jobs_record_the_engine(tmp_path):
    jobs = scr.docking_jobs([Path("p.pdbqt")], {"actives": [Path("l.pdbqt")]}, {}, tmp_path,
                            DockingConfig(engine="gnina"))
    assert list(jobs["engine"]) == ["gnina"]


def test_relative_paths_resolve_against_the_task_folder(tmp_path):
    from qpocket.arms.screen import campaign

    job = campaign.resolve_paths(_job(), tmp_path)
    assert job["protein_pdbqt"] == str(tmp_path / "r.pdbqt")
    assert campaign.resolve_paths(_job(protein_pdbqt="/abs/r.pdbqt"), tmp_path)["protein_pdbqt"] == "/abs/r.pdbqt"


def test_run_job_uses_the_rows_engine_and_folder(tmp_path, monkeypatch):
    from qpocket.arms.screen import campaign

    jobs_csv = tmp_path / "docking_jobs.csv"
    pd.DataFrame([_job(engine="quickvina")]).to_csv(jobs_csv, index=False)
    seen = {}
    monkeypatch.setattr(campaign, "docking_engine", lambda engine, path=None: f"/bin/{engine.executable}")
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda argv, **k: seen.setdefault("argv", argv) and type("R", (), {"returncode": 0})())
    assert campaign.run_job(jobs_csv, 0, 2) == 0
    assert seen["argv"][0] == "/bin/qvina2" and seen["argv"][2] == str(tmp_path / "r.pdbqt")


def test_an_unknown_engine_is_a_config_error():
    cfg = ScreenConfig()
    cfg.docking.engine = "teleport"
    with pytest.raises(ValueError, match="docking.engine"):
        validate_screen_config(cfg)
