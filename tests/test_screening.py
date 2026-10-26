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
    from qpocket.core.ports import DECOYS

    # Cross-docking has its own prepare; these cover the generic (virtual screening) path.
    monkeypatch.setitem(DECOYS._plugins, "lude", DECOYS.get("lude")._replace(fn=lambda a, o, p: None))


def test_prepare_writes_only_the_task_list_by_default(tmp_path, monkeypatch):
    _fake_inputs(monkeypatch)
    plan = screening.prepare(tmp_path, run_mode="virtual_screening")
    assert len(pd.read_csv(plan.jobs_csv)) == 2 and plan.scripts == []
    assert plan.command.startswith("qpocket screen run-local")
    assert not list((tmp_path / "screening").glob("run_docking*"))


@pytest.mark.parametrize("submit,suffix", [("slurm", ".sbatch"), ("local", ".sh")])
def test_scripts_are_written_on_request(tmp_path, monkeypatch, submit, suffix):
    _fake_inputs(monkeypatch)
    plan = screening.prepare(tmp_path, submit=submit, run_mode="virtual_screening")
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


def test_prepare_options_only_override_the_config_when_given(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from qpocket.arms.screen import campaign
    from qpocket.cli import cli

    seen = {}

    def fake_prepare(workdir, config_path, submit=None, **overrides):
        seen.update(overrides)
        jobs = tmp_path / "docking_jobs.csv"
        jobs.write_text("job_id\n")
        return campaign.ScreeningPlan(jobs, [], "true")

    monkeypatch.setattr(campaign, "prepare", fake_prepare)
    CliRunner().invoke(cli, ["screen", "prepare", "--workdir", str(tmp_path)])
    assert "run_mode" not in seen and "engine" not in seen.get("docking", {})
    CliRunner().invoke(cli, ["screen", "prepare", "--workdir", str(tmp_path), "--engine", "vina",
                             "--run-mode", "virtual_screening"])
    assert seen["run_mode"] == "virtual_screening" and seen["docking"]["engine"] == "vina"


def _ligand_pdb(path, shift, smiles="CCO"):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    for k in range(mol.GetNumAtoms()):
        q = conf.GetAtomPosition(k)
        conf.SetAtomPosition(k, (q.x + shift, q.y, q.z))
    Chem.MolToPDBFile(mol, str(path))
    return mol


def _two_site_run(tmp_path, here_dir=None):
    """P1 holds ethanol at site_01 (x≈0) and a bromide at site_02 (x≈30); site_01 lists only its own.

    here_dir moves site_01's file out of extracted_ligands, so a glob there can only find site_02's.
    """
    from qpocket.io import layout

    run = layout.create(str(tmp_path), "RUN")
    site = layout.site_workspace(run, "site_01")
    here = (here_dir or run.extracted_ligands) / "P1_EOH_A_101.pdb"
    there = run.extracted_ligands / "P1_EOH_B_201.pdb"
    _ligand_pdb(there, 30.0, "[Br-]")  # shares no atom with ethanol: no RMSD against it
    _ligand_pdb(here, 0.0)
    (site.screening / "actives.smi").write_text(f"SMILES\tPDB_ID\tLigand_Name\tLigand_File\nCCO\tP1\tEOH\t{here}\n")
    return run, site, here


def test_active_ligand_files_reads_the_site_column(tmp_path):
    from qpocket.io.screening import active_ligand_files

    _, site, here = _two_site_run(tmp_path)
    assert active_ligand_files(site.screening / "actives.smi") == {"P1_EOH": [here]}
    (site.screening / "actives.smi").write_text("SMILES\tPDB_ID\tLigand_Name\nCCO\tP1\tEOH\n")
    assert active_ligand_files(site.screening / "actives.smi") == {}


def test_campaign_prepare_on_site_dir(tmp_path, monkeypatch):
    from qpocket.io import screening as io_screen
    from qpocket.arms.screen import receptors as screening_prep

    real_box = io_screen.extract_vina_box
    _fake_inputs(monkeypatch)
    monkeypatch.setattr(io_screen, "extract_vina_box", real_box)
    seen = {}

    def fake_proteins(screening_dir, reps, master, cfg):
        seen["protein"] = (screening_dir, reps)
        return ["P1"]

    monkeypatch.setattr(screening_prep, "prepare_all_proteins", fake_proteins)
    run, site, _ = _two_site_run(tmp_path)
    plan = screening.prepare(site.root, run_mode="virtual_screening", box={"mode": "ligand_based"})
    box = io_screen.parse_vina_box(site.screening / "vina_box.txt")
    assert box["size_x"] < 20.0  # only this site's ethanol; both ligands would span over 30 Å
    assert seen["protein"] == (site.screening, site.metadata / "selected_representatives.csv")
    assert plan.jobs_csv == site.screening / "docking_jobs.csv"


def test_rmsd_uses_the_crystal_pose_at_this_site(tmp_path, monkeypatch):
    from qpocket.arms.screen import analysis

    run, site, here = _two_site_run(tmp_path, here_dir=tmp_path)
    docked = _ligand_pdb(tmp_path / "docked.pdb", 0.0)
    (site.screening / "results").mkdir()
    out = site.screening / "results" / "P1_P1_EOH.pdbqt"
    out.write_text("x")
    pd.DataFrame({"job_id": ["j0"], "output_pdbqt": [str(out)]}).to_csv(site.screening / "docking_jobs.csv",
                                                                         index=False)
    scores = pd.DataFrame({"job_id": ["j0"], "ligand_id": ["P1_EOH"], "pdb_id": ["P1"],
                           "molecule_type": ["actives"], "score": [-7.0], "output_exists": [True]})
    monkeypatch.setattr(analysis, "collect_docking_scores", lambda *a: scores)
    monkeypatch.setattr(analysis, "parse_vina_output", lambda path: (-7.0, docked))
    monkeypatch.setattr(analysis, "run_interaction_analysis", lambda *a: pd.DataFrame())
    analysis.run_full_analysis(site.screening, site.root, ScreenConfig())
    rmsd = pd.read_csv(site.screening / "analysis" / "rmsd_analysis.csv")
    assert rmsd["rmsd"].iloc[0] < 0.01 and bool(rmsd["success"].iloc[0])


def test_rmsd_falls_back_to_the_structure_glob_without_ligand_files(tmp_path, monkeypatch):
    from qpocket.arms.screen import analysis

    run, site, here = _two_site_run(tmp_path)
    (run.extracted_ligands / "P1_EOH_B_201.pdb").unlink()
    (site.screening / "actives.smi").write_text("SMILES\tPDB_ID\tLigand_Name\nCCO\tP1\tEOH\n")
    docked = _ligand_pdb(tmp_path / "docked.pdb", 0.0)
    pd.DataFrame({"job_id": ["j0"], "output_pdbqt": ["x.pdbqt"]}).to_csv(site.screening / "docking_jobs.csv",
                                                                       index=False)
    scores = pd.DataFrame({"job_id": ["j0"], "ligand_id": ["P1_EOH"], "pdb_id": ["P1"],
                           "molecule_type": ["actives"], "score": [-7.0], "output_exists": [True]})
    monkeypatch.setattr(analysis, "collect_docking_scores", lambda *a: scores)
    monkeypatch.setattr(analysis, "parse_vina_output", lambda path: (-7.0, docked))
    monkeypatch.setattr(analysis, "run_interaction_analysis", lambda *a: pd.DataFrame())
    analysis.run_full_analysis(site.screening, site.root, ScreenConfig())
    assert pd.read_csv(site.screening / "analysis" / "rmsd_analysis.csv")["rmsd"].iloc[0] < 0.01


def test_screen_site_option_resolves_to_the_site_dir(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from qpocket.arms.screen import campaign
    from qpocket.cli import cli
    from qpocket.io import layout

    run = layout.create(str(tmp_path), "RUN")
    layout.site_workspace(run, "site_01")
    pd.DataFrame({"site_id": ["site_01"]}).to_csv(run.sites / "sites.csv", index=False)
    seen = []
    monkeypatch.setattr(campaign, "analyze", lambda workdir, config_path, **o: seen.append(workdir) or workdir)
    ok = CliRunner().invoke(cli, ["screen", "analyze", "--workdir", str(run.root), "--site", "site_01"])
    assert ok.exit_code == 0 and seen == [str(run.sites / "site_01")]
    bad = CliRunner().invoke(cli, ["screen", "analyze", "--workdir", str(run.root), "--site", "site_09"])
    assert bad.exit_code != 0 and "unknown site 'site_09'; known: site_01" in bad.output and len(seen) == 1
    direct = CliRunner().invoke(cli, ["screen", "analyze", "--workdir", str(run.sites / "site_01")])
    assert direct.exit_code == 0 and seen[-1] == str(run.sites / "site_01")
