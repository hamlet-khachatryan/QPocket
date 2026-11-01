"""Virtual screening: task list, SLURM-or-local submission, runners."""

import pytest

from qpocket.core import screening as scr
from qpocket.arms.screen import campaign as screening
from qpocket.screen_config import ScreenConfig

BOX = {"center_x": 1.0, "center_y": 2.0, "center_z": 3.0, "size_x": 20.0, "size_y": 21.0, "size_z": 22.0}


def _jobs(tmp_path, n_ligands=2):
    cfg = ScreenConfig()
    ligands = {"actives": [tmp_path / f"L{i}.pdbqt" for i in range(n_ligands)]}
    return scr.docking_jobs([tmp_path / "P1.pdbqt"], ligands, BOX, tmp_path / "results", cfg.docking)


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


def test_a_missing_docking_engine_says_what_to_do(tmp_path, monkeypatch):
    from qpocket.io import tools

    monkeypatch.setenv("PATH", "")
    with pytest.raises(tools.ToolNotFoundError, match="engine_path"):
        screening.docking_engine(screening.engine_for({"engine": "gnina"}))  # vina is bundled


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


