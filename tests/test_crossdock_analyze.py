"""analyze scores every pose against every crystal copy and writes the success CSVs."""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from qpocket.core.screening import Pose
from qpocket.io import layout


def _ethanol(shift):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + shift, p.y, p.z))
    return mol


def test_analyze_takes_the_closest_crystal_copy(tmp_path, monkeypatch):
    from qpocket.arms.screen import crossdock
    from qpocket.core.ports import DOCKING

    ws = layout.create(str(tmp_path), "RUN")
    folder = ws.screening / "crossdock"
    (folder / "poses").mkdir(parents=True)
    far, near = tmp_path / "far.pdb", tmp_path / "near.pdb"
    Chem.MolToPDBFile(_ethanol(10.0), str(far))
    Chem.MolToPDBFile(_ethanol(0.5), str(near))
    pd.DataFrame({"active_id": ["ETH"], "smiles": ["CCO"], "canonical": ["CCO"],
                  "crystal_files": [f"{far};{near}"]}).to_csv(folder / "ligands.csv", index=False)
    pd.DataFrame({"job_id": ["job_00000"], "representative": ["R1"], "active_id": ["ETH"],
                  "engine": ["quickvina"], "output_pdbqt": ["poses/job_00000.pdbqt"]}
                 ).to_csv(folder / "docking_jobs.csv", index=False)
    (folder / "poses" / "job_00000.pdbqt").write_text("x")
    engine = DOCKING.get("quickvina").fn()
    fake = engine._replace(read_poses=lambda path: [Pose(1, -8.0, _ethanol(0.0)), Pose(2, -7.0, _ethanol(5.0))])
    monkeypatch.setattr(crossdock, "_engine", lambda name, folder=None: fake)

    result = crossdock.analyze(ws.root)
    assert result.poses["rmsd"].round(3).tolist() == [0.5, 4.5]
    assert result.pairs.iloc[0]["top1_success"] == True  # noqa: E712
    assert set(pd.read_csv(folder / "crossdock_success.csv")["level"]) == {"representative", "active", "overall"}
    assert (folder / "crossdock_poses.csv").exists() and (folder / "crossdock_pairs.csv").exists()


def test_a_missing_output_is_a_failed_pair(tmp_path):
    from qpocket.arms.screen import crossdock

    ws = layout.create(str(tmp_path), "RUN")
    folder = ws.screening / "crossdock"
    folder.mkdir(parents=True)
    pd.DataFrame({"active_id": ["ETH"], "smiles": ["CCO"], "canonical": ["CCO"], "crystal_files": [""]}
                 ).to_csv(folder / "ligands.csv", index=False)
    pd.DataFrame({"job_id": ["job_00000"], "representative": ["R1"], "active_id": ["ETH"],
                  "engine": ["quickvina"], "output_pdbqt": ["poses/job_00000.pdbqt"]}
                 ).to_csv(folder / "docking_jobs.csv", index=False)
    result = crossdock.analyze(ws.root)
    assert result.pairs.empty or not result.pairs["top1_success"].any()
    assert result.success.set_index("level").loc["overall", "n_pairs"] == 0
