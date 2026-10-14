"""Any docking binary can be declared in config and run from a task folder."""
import json

import pandas as pd
import pytest

from qpocket.core.screening import template_engine
from qpocket.screen_config import ScreenConfig, validate_screen_config

SPEC = {"executable": "/opt/smina", "output": "sdf", "score_property": "minimizedAffinity",
        "command": "{exe} -r {receptor} -l {ligand} -o {output} --center_x {center_x} --center_y {center_y} "
                   "--center_z {center_z} --size_x {size_x} --size_y {size_y} --size_z {size_z} "
                   "--exhaustiveness {exhaustiveness} --num_modes {num_modes} --cpu {cpus}"}


def _job():
    return {"protein_pdbqt": "r.pdbqt", "ligand_pdbqt": "l.pdbqt", "output_pdbqt": "poses/j.sdf",
            "center_x": 1, "center_y": 2, "center_z": 3, "size_x": 20, "size_y": 20, "size_z": 20,
            "exhaustiveness": 8, "num_modes": 9, "energy_range": 3.0, "engine": "smina"}


def test_the_template_becomes_a_command_line():
    engine = template_engine(SPEC)
    argv = engine.argv(_job(), engine.executable, 4)
    assert argv[:5] == ["/opt/smina", "-r", "r.pdbqt", "-l", "l.pdbqt"] and argv[-1] == "4"
    assert engine.output_ext == ".sdf"


@pytest.mark.parametrize("change,match", [
    ({"command": "{exe} {ligand} {output}"}, "receptor"),
    ({"command": "{exe} {receptor} {ligand} {output} {mystery}"}, "mystery"),
    ({"output": "mol2"}, "output"),
])
def test_bad_declarations_are_config_errors(change, match):
    cfg = ScreenConfig()
    cfg.docking.custom_engines = {"smina": {**SPEC, **change}}
    with pytest.raises(ValueError, match=match):
        validate_screen_config(cfg)


def test_a_custom_engine_can_be_the_engine():
    cfg = ScreenConfig()
    cfg.docking.custom_engines = {"smina": SPEC}
    cfg.docking.engine = "smina"
    validate_screen_config(cfg)


def test_a_custom_engine_may_not_shadow_a_builtin():
    cfg = ScreenConfig()
    cfg.docking.custom_engines = {"vina": SPEC}
    with pytest.raises(ValueError, match="vina"):
        validate_screen_config(cfg)


def test_a_task_folder_runs_its_custom_engine_without_config(tmp_path, monkeypatch):
    from qpocket.arms.screen import campaign

    (tmp_path / "engines.json").write_text(json.dumps({"smina": SPEC}))
    jobs_csv = tmp_path / "docking_jobs.csv"
    pd.DataFrame([_job()]).to_csv(jobs_csv, index=False)
    seen = {}
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda argv, **k: seen.setdefault("argv", argv) and type("R", (), {"returncode": 0})())
    monkeypatch.setattr(campaign, "docking_engine", lambda engine, path=None: engine.executable)
    assert campaign.run_job(jobs_csv, 0, 2) == 0
    assert seen["argv"][0] == "/opt/smina" and seen["argv"][2] == str(tmp_path / "r.pdbqt")
