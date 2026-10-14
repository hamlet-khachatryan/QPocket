"""Virtual screening: task list, SLURM-or-local submission, runners."""

import pandas as pd
import pytest

from qpocket.core import screening as scr
from qpocket.arms.screen import campaign as screening
from qpocket.screen_config import ScreenConfig

BOX = {"center_x": 1.0, "center_y": 2.0, "center_z": 3.0, "size_x": 20.0, "size_y": 21.0, "size_z": 22.0}


def _jobs(tmp_path, n_ligands=2):
    cfg = ScreenConfig()
    ligands = {"actives": [tmp_path / f"L{i}.pdbqt" for i in range(n_ligands)]}
    return scr.docking_jobs([tmp_path / "P1.pdbqt"], ligands, BOX, tmp_path / "results", cfg.docking)


def test_docking_jobs_one_row_per_pair(tmp_path):
    jobs = _jobs(tmp_path)
    assert list(jobs.columns) == list(scr.JOB_COLUMNS) and len(jobs) == 2
    assert jobs.loc[1, "job_id"] == "job_00001" and jobs.loc[1, "ligand_id"] == "L1"


def test_vina_argv_is_a_plain_argument_list(tmp_path):
    argv = scr.vina_argv(_jobs(tmp_path).iloc[0], "vina", 4)
    assert argv[0] == "vina" and argv[argv.index("--size_y") + 1] == "21.0" and argv[-2:] == ["--cpu", "4"]


def test_slurm_script_runs_python_per_task_and_parses_no_csv(tmp_path):
    text = scr.slurm_array_script(tmp_path / "jobs.csv", ScreenConfig().slurm, 1000, 7, tmp_path)
    assert "set -euo pipefail" in text and "#SBATCH --array=0-6%100" in text
    assert "qpocket screen run-task" in text and "SLURM_ARRAY_TASK_ID + 1000" in text
    assert "awk" not in text and "IFS" not in text


@pytest.mark.parametrize("n_jobs,max_array,expected", [(5, 2, 3), (5, 1000, 1)])
def test_slurm_submission_is_chunked_under_the_array_cap(tmp_path, n_jobs, max_array, expected):
    cfg = ScreenConfig()
    cfg.slurm.max_array_size = max_array
    plan = screening.write_submission(tmp_path / "docking_jobs.csv", n_jobs, cfg, "slurm")
    assert len(plan.scripts) == expected and len(plan.command.splitlines()) == expected
    if expected == 3:
        assert "SLURM_ARRAY_TASK_ID + 4" in plan.scripts[-1].read_text()


def test_local_submission_is_a_command_not_a_bash_parser(tmp_path):
    plan = screening.write_submission(tmp_path / "docking_jobs.csv", 3, ScreenConfig(), "local")
    assert plan.command.startswith("qpocket screen run-local")
    assert plan.scripts[0].name == "run_docking.sh" and plan.command in plan.scripts[0].read_text()


def _fake_inputs(monkeypatch):
    from qpocket.io import screening as io_screen
    from qpocket.arms.screen import receptors as screening_prep

    monkeypatch.setattr(io_screen, "extract_vina_box", lambda params, dirs: None)
    monkeypatch.setattr(screening_prep, "prepare_all_proteins", lambda *a, **k: ["P1", "P2"])
    monkeypatch.setattr(screening_prep, "prepare_all_ligands",
                        lambda *a, **k: {"actives": ["L1"], "decoys": []})


def test_prepare_writes_only_the_task_list_by_default(tmp_path, monkeypatch):
    _fake_inputs(monkeypatch)
    plan = screening.prepare(tmp_path)
    assert len(pd.read_csv(plan.jobs_csv)) == 2 and plan.scripts == []
    assert plan.command.startswith("qpocket screen run-local")
    assert not list((tmp_path / "screening").glob("run_docking*"))


@pytest.mark.parametrize("submit,suffix", [("slurm", ".sbatch"), ("local", ".sh")])
def test_scripts_are_written_on_request(tmp_path, monkeypatch, submit, suffix):
    _fake_inputs(monkeypatch)
    plan = screening.prepare(tmp_path, submit=submit)
    assert plan.scripts and plan.scripts[0].suffix == suffix


def test_run_local_defaults_to_half_the_cores(tmp_path, monkeypatch):
    import os

    jobs = _jobs(tmp_path)
    jobs.to_csv(tmp_path / "docking_jobs.csv", index=False)
    sizes = []

    class Pool:
        def __init__(self, max_workers):
            sizes.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items):
            return [0 for _ in items]

    monkeypatch.setattr(screening, "docking_engine", lambda engine, path=None: "vina")
    monkeypatch.setattr(screening, "ThreadPoolExecutor", Pool)
    assert screening.run_local(tmp_path / "docking_jobs.csv", engine="vina") == 0
    assert sizes == [max(1, (os.cpu_count() or 2) // 2)]


def test_run_local_runs_every_row_without_a_shell(tmp_path, monkeypatch):
    jobs = _jobs(tmp_path, n_ligands=3)
    jobs.to_csv(tmp_path / "docking_jobs.csv", index=False)
    seen = []

    class Done:
        returncode = 0

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        return Done()

    monkeypatch.setattr(screening, "docking_engine", lambda engine, path=None: "vina")
    monkeypatch.setattr(screening.subprocess, "run", fake_run)
    assert screening.run_local(tmp_path / "docking_jobs.csv", workers=2, engine="vina") == 0
    assert len(seen) == 3 and all(isinstance(a, list) and not k.get("shell") for a, k in seen)
    assert (tmp_path / "results").is_dir()


def test_screen_config_loads_and_validates():
    """OmegaConf cannot build Literal fields; loading used to fail outright."""
    from qpocket.screen_config import load_screen_config, validate_screen_config

    cfg = load_screen_config(run_mode="virtual_screening")
    validate_screen_config(cfg)
    assert cfg.run_mode == "virtual_screening"
    cfg.box.mode = "bogus"
    with pytest.raises(ValueError, match="box.mode"):
        validate_screen_config(cfg)


def test_a_missing_docking_engine_says_what_to_do(tmp_path, monkeypatch):
    from qpocket.io import tools

    monkeypatch.setenv("PATH", "")
    with pytest.raises(tools.ToolNotFoundError, match="engine_path"):
        screening.docking_engine(screening.engine_for({"engine": "gnina"}))  # vina is bundled
