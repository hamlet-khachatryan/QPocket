"""The Screening page's helpers: 3Dmol scenes, the 40-ligand picker, the success matrix and the pose viewer."""
import json
import re

import numpy as np
import pandas as pd

from qpocket.viz.pages import screening as page


def _payload(html):
    return json.loads(re.search(r'<script type="application/json" id="qp-scene">(.*?)</script>', html, re.S).group(1))


def _preview(tmp_path):
    from qpocket.arms.screen.box import BoxPreview
    from qpocket.core.screening import calculate_box, extremum_points

    frame = tmp_path / "frame.pdb"
    frame.write_text("ATOM      1  CA  TYR A   1       6.000   0.000   0.000  1.00  0.00           C\nEND\n")
    clouds = {"R1": np.array([[0.0, 0, 0], [1, 1, 1]]), "R2": np.array([[2.0, 0, 0]])}
    pts = np.vstack(list(clouds.values()))
    ext, labels = extremum_points(pts)
    return BoxPreview("representative_pockets", 2.0, calculate_box(pts, 0.0), calculate_box(pts, 2.0), pts, ext,
                      labels, ["TYR_1_A"], clouds, {"R1": "S1", "R2": "S2"}, frame, "h")


def test_success_matrix_takes_the_best_pose():
    pairs = pd.DataFrame({"representative": ["R1", "R1", "R2"], "active_id": ["A", "B", "A"],
                          "best_topk_rmsd": [1.5, 3.0, 0.9], "top1_rmsd": [2.5, 3.0, 0.9]})
    matrix = page.success_matrix(pairs)
    assert matrix.loc["R1", "A"] == 1.5 and np.isnan(matrix.loc["R2", "B"])
    fig = page.matrix_figure(matrix, threshold=2.0)
    assert fig.data[0].zmax == 4.0


def _task(tmp_path):
    folder = tmp_path / "crossdock"
    (folder / "receptors").mkdir(parents=True)
    (folder / "poses").mkdir()
    (folder / "receptors" / "R1.pdb").write_text("ATOM      1  CA  TYR A   1       6.000   0.000   0.000  1.00  0.00           C\nEND\n")
    crystal = tmp_path / "crystal.pdb"
    crystal.write_text("HETATM    1  C1  LIG A 101       1.000   1.000   1.000  1.00  0.00           C\nEND\n")
    pd.DataFrame({"active_id": ["A"], "smiles": ["C"], "canonical": ["C"], "crystal_files": [str(crystal)]}).to_csv(
        folder / "ligands.csv", index=False)
    job = pd.Series({"job_id": "j0", "representative": "R1", "active_id": "A", "engine": "vina",
                     "output_pdbqt": "poses/j0.pdbqt"})
    return folder, job


