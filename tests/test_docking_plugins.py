"""The docking engine is chosen by name and recorded in every job."""
import pandas as pd
import pytest

from qpocket.core import screening as scr
from qpocket.core.ports import DOCKING
from qpocket.screen_config import DockingConfig, ScreenConfig, validate_screen_config


def _job(tmp_path, **extra):
    row = {c: "1" for c in scr.JOB_COLUMNS}
    row.update(protein_pdbqt="p.pdbqt", ligand_pdbqt="l.pdbqt", output_pdbqt=str(tmp_path / "o.pdbqt"))
    row.update(extra)
    return row


@pytest.mark.parametrize("engine", ["vina", "quickvina"])
def test_each_engine_builds_todays_command(engine, tmp_path):
    job = _job(tmp_path, engine=engine)
    assert DOCKING.get(engine).fn(job, "/bin/x", 4) == scr.vina_argv(job, "/bin/x", 4)


def test_jobs_record_the_engine(tmp_path):
    from pathlib import Path

    jobs = scr.docking_jobs([Path("p.pdbqt")], {"actives": [Path("l.pdbqt")]}, {}, tmp_path,
                            DockingConfig(engine="quickvina"))
    assert list(jobs["engine"]) == ["quickvina"]


def test_an_old_job_table_without_the_column_means_vina(tmp_path, monkeypatch):
    from qpocket.arms.screen import campaign

    row = _job(tmp_path)
    row.pop("engine")
    jobs_csv = tmp_path / "jobs.csv"
    pd.DataFrame([row]).to_csv(jobs_csv, index=False)
    seen = {}
    monkeypatch.setattr(campaign, "docking_engine", lambda engine=None: "vina")
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda argv, **k: seen.setdefault("argv", argv) and type("R", (), {"returncode": 0})())
    assert campaign.run_job(jobs_csv, 0, 2) == 0
    assert seen["argv"][0] == "vina"


def test_an_unknown_engine_is_a_config_error():
    cfg = ScreenConfig()
    cfg.docking.engine = "teleport"
    with pytest.raises(ValueError, match="docking.engine"):
        validate_screen_config(cfg)
