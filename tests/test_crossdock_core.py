"""Pose RMSD in place, the job grid, and success rates."""
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qpocket.core import crossdock as cx


def _ethanol(shift=0.0):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + shift, p.y, p.z))
    return mol


def test_identical_poses_have_zero_rmsd():
    assert cx.pose_rmsd(_ethanol(), _ethanol(), "CCO") == pytest.approx(0.0, abs=1e-6)


def test_a_translated_pose_is_not_superposed_away():
    assert cx.pose_rmsd(_ethanol(3.0), _ethanol(), "CCO") == pytest.approx(3.0, abs=1e-6)


def test_symmetric_atoms_take_the_best_mapping():
    a = Chem.MolFromSmiles("OC(=O)C")
    AllChem.Compute2DCoords(a)
    b = Chem.Mol(a)
    conf = b.GetConformer()
    p0, p2 = conf.GetAtomPosition(0), conf.GetAtomPosition(2)
    conf.SetAtomPosition(0, p2)
    conf.SetAtomPosition(2, p0)          # swap the two carboxyl oxygens
    assert cx.pose_rmsd(a, b) == pytest.approx(0.0, abs=1e-6)


def test_the_job_grid_is_every_ligand_against_every_receptor():
    jobs = cx.crossdock_jobs({"R1": "receptors/R1.pdbqt", "R2": "receptors/R2.pdbqt"},
                             {"L1": "ligands/L1.pdbqt"}, {"center_x": 1.0, "center_y": 2.0, "center_z": 3.0,
                                                          "size_x": 20.0, "size_y": 20.0, "size_z": 20.0},
                             "quickvina", ".pdbqt", 8, 9, 3.0)
    assert list(zip(jobs["representative"], jobs["active_id"])) == [("R1", "L1"), ("R2", "L1")]
    assert jobs["output_pdbqt"].iloc[0] == "poses/job_00000.pdbqt"
    assert not any(str(p).startswith("/") for p in jobs["protein_pdbqt"])


def _poses():
    return pd.DataFrame({
        "job_id": ["j1", "j1", "j1", "j2", "j2", "j3"],
        "representative": ["R1", "R1", "R1", "R2", "R2", "R1"],
        "active_id": ["L1", "L1", "L1", "L1", "L1", "L2"],
        "rank": [1, 2, 3, 1, 2, 1],
        "score": [-9.0, -8.0, -7.0, -8.5, -8.4, -6.0],
        "rmsd": [3.0, 1.5, 0.9, 1.0, 4.0, np.nan],
    })


def test_pairs_score_top1_and_topk():
    pairs = cx.pair_table(_poses(), threshold=2.0, top_k=2).set_index("job_id")
    assert pairs.loc["j1", "top1_success"] == False and pairs.loc["j1", "topk_success"] == True  # noqa: E712
    assert pairs.loc["j1", "best_topk_rmsd"] == 1.5 and pairs.loc["j2", "top1_success"] == True  # noqa: E712
    assert pairs.loc["j3", "top1_success"] == False  # noqa: E712  (no RMSD is a failure)


def test_success_rates_by_representative_active_and_overall():
    s = cx.success_table(cx.pair_table(_poses(), 2.0, 2)).set_index(["level", "name"])
    assert s.loc[("representative", "R1"), "top1_success_rate"] == 0.0
    assert s.loc[("representative", "R1"), "topk_success_rate"] == 50.0
    assert s.loc[("active", "L1"), "top1_success_rate"] == 50.0
    assert s.loc[("overall", "all"), "n_pairs"] == 3


def test_the_seed_travels_in_the_job_and_reaches_vina():
    from qpocket.core.screening import vina_argv

    jobs = cx.crossdock_jobs({"R1": "receptors/R1.pdbqt"}, {"L1": "ligands/L1.pdbqt"},
                             {"center_x": 0, "center_y": 0, "center_z": 0, "size_x": 1, "size_y": 1, "size_z": 1},
                             "vina", ".pdbqt", 8, 9, 3.0, seed=7)
    argv = vina_argv(jobs.iloc[0], "vina", 2)
    assert argv[argv.index("--seed") + 1] == "7"


def test_older_job_tables_without_a_seed_still_run():
    from qpocket.core.screening import vina_argv

    job = {"protein_pdbqt": "r", "ligand_pdbqt": "l", "output_pdbqt": "o", "center_x": 0, "center_y": 0,
           "center_z": 0, "size_x": 1, "size_y": 1, "size_z": 1, "exhaustiveness": 8, "num_modes": 9,
           "energy_range": 3}
    assert "--seed" not in vina_argv(job, "vina", 2)
