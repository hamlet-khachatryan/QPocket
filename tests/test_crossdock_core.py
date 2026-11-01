"""Pose RMSD in place, the job grid, and success rates."""
import numpy as np
import pandas as pd
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


def test_pairs_record_the_best_of_every_sampled_pose():
    pairs = cx.pair_table(_poses(), threshold=2.0, top_k=2).set_index("job_id")
    assert pairs.loc["j1", "best_rmsd"] == 0.9 and pairs.loc["j1", "sampled_success"] == True  # noqa: E712  (rank 3)
    assert np.isnan(pairs.loc["j3", "best_rmsd"]) and pairs.loc["j3", "sampled_success"] == False  # noqa: E712


