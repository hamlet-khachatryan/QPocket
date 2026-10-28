"""The Screening page's helpers: 3Dmol scenes, the 40-ligand picker, the success matrix and the pose viewer."""
import json
import re

import numpy as np
import pandas as pd

from qpocket.viz.pages import screening as page


def _payload(html):
    return json.loads(re.search(r'<script type="application/json" id="qp-scene">(.*?)</script>', html, re.S).group(1))


def test_scene_html_carries_models_boxes_and_spheres():
    html = page.scene_html([{"pdb": "ATOM", "style": {"cartoon": {}}}],
                           [{"min": [0, 0, 0], "max": [1, 2, 3], "color": "#222", "dashed": False}],
                           [{"center": [0, 0, 0], "radius": 0.5, "color": "#000", "label": "-x"}], height=300)
    payload = _payload(html)
    assert len(payload["models"]) == 1 and payload["boxes"][0]["max"] == [1, 2, 3]
    assert payload["spheres"][0]["label"] == "-x" and "height:300px" in html.replace(" ", "")


def test_default_ligands_order_and_cap():
    clusters = pd.DataFrame({"pdb_id": ["A", "B", "C", "D"], "cluster": ["S1", "S1", "S2", "S3"],
                             "is_representative": [False, True, False, True]})
    ligands = pd.DataFrame({"cluster": ["S1", "S1", "S2", "S2", "S3"], "pdb_id": ["A", "B", "C", "C", "D"],
                            "ligand_key": ["a", "b", "c1", "c2", "d"], "resname": "L",
                            "smiles": ["X", "Y", "X", "Z", "W"]})
    assert page.default_ligands(ligands, clusters) == ["b", "d", "c1", "a", "c2"]
    assert page.default_ligands(ligands, clusters, limit=3) == ["b", "d", "c1"]


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


def test_cloud_scene_labels_six_extremes(tmp_path):
    payload = _payload(page.cloud_box_scene(_preview(tmp_path)))
    assert [s["label"] for s in payload["spheres"]] == ["-x", "+x", "-y", "+y", "-z", "+z"]
    assert len(payload["boxes"]) == 2 and payload["boxes"][1]["dashed"]
    clouds = [m for m in payload["models"] if "sphere" in m["style"]]
    assert len(clouds) == 2 and clouds[0]["style"]["sphere"]["color"] != clouds[1]["style"]["sphere"]["color"]


def test_ligand_scene_highlights_residues(tmp_path):
    lig = tmp_path / "lig.pdb"
    lig.write_text("HETATM    1  C1  LIG A 101       1.000   1.000   1.000  1.00  0.00           C\nEND\n")
    payload = _payload(page.ligand_box_scene(_preview(tmp_path), [str(lig)]))
    styles = [list(m["style"]) for m in payload["models"]]
    assert ["cartoon"] in styles and styles.count(["stick"]) == 2   # the residues, and the ligand


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


def test_pose_scene_without_poses(tmp_path):
    folder, job = _task(tmp_path)
    html, table = page.pose_scene(folder, job, top_n=3, threshold=2.0, residues=[])
    assert "No poses yet" in html and table.empty


def test_pose_scene_colours_by_rmsd(tmp_path, monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from qpocket.core.screening import Pose

    folder, job = _task(tmp_path)
    (folder / "poses" / "j0.pdbqt").write_text("MODEL 1\n")
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)

    class Engine:
        def read_poses(self, path):
            return [Pose(1, -7.0, mol), Pose(2, -6.5, mol), Pose(3, -6.0, mol)]

    monkeypatch.setattr(page, "engine_for", lambda job, folder=None: Engine())
    values = iter([1.0, 3.0, 9.0])
    monkeypatch.setattr(page, "pose_rmsd", lambda pose, crystal, smiles: next(values))
    html, table = page.pose_scene(folder, job, top_n=3, threshold=2.0, residues=["TYR_1_A"])
    assert table["rmsd"].tolist() == [1.0, 3.0, 9.0]
    colours = [m["style"]["stick"]["color"] for m in _payload(html)["models"] if m.get("role") == "pose"]
    assert colours == [page.GOOD, page.CLOSE, page.FAR]


def test_ligand_files_follow_their_keys(tmp_path):
    (tmp_path / "structures").mkdir()
    pd.DataFrame({"pdb_id": ["P1", "P2"], "chain": "A", "resname": ["AAA", "BBB"], "resseq": ["1", "2"], "icode": " ",
                  "n_heavy": "6", "file": ["/x/a.pdb", "/x/b.pdb"], "smiles": "", "site_ids": "S1"}).to_csv(
        tmp_path / "structures" / "ligands.csv", index=False)
    assert page.ligand_files(tmp_path, ["P2:A:BBB:2", "nope"]) == ["/x/b.pdb"]


def test_scene_message_is_escaped():
    html = page.scene_html([], [], [], message="<b>R1</b> & co")
    assert "&lt;b&gt;R1&lt;/b&gt; &amp; co" in html and "<b>R1" not in html


def test_ligand_scene_counts_only_ligands_it_draws(tmp_path):
    lig = tmp_path / "lig.pdb"
    lig.write_text("HETATM    1  C1  LIG A 101       1.000   1.000   1.000  1.00  0.00           C\nEND\n")
    html = page.ligand_box_scene(_preview(tmp_path), [str(lig), str(tmp_path / "gone.pdb")])
    assert "· 1 ligands (1 not found) ·" in html


def test_pose_scene_draws_every_crystal_pose_it_scores(tmp_path):
    folder, job = _task(tmp_path)
    second = tmp_path / "crystal2.pdb"
    second.write_text("HETATM    1  C1  LIG B 101       4.000   1.000   1.000  1.00  0.00           C\nEND\n")
    table = pd.read_csv(folder / "ligands.csv")
    table["crystal_files"] = table["crystal_files"] + ";" + str(second)
    table.to_csv(folder / "ligands.csv", index=False)
    (folder / "poses" / "j0.pdbqt").write_text("MODEL 1\n")

    class Engine:
        def read_poses(self, path):
            return []

    import qpocket.viz.pages.screening as screening

    original = screening.engine_for
    screening.engine_for = lambda job, folder=None: Engine()
    try:
        html, _ = page.pose_scene(folder, job, top_n=3, threshold=2.0, residues=[])
    finally:
        screening.engine_for = original
    assert [m["role"] for m in _payload(html)["models"]].count("crystal") == 2


def test_a_wide_success_matrix_gets_room_for_its_labels():
    matrix = pd.DataFrame(np.ones((3, 80)), index=["R1", "R2", "R3"], columns=[f"ligand_{i}" for i in range(80)])
    fig = page.matrix_figure(matrix, threshold=2.0)
    assert fig.layout.width >= 80 * 12 and fig.layout.xaxis.tickfont.size <= 9
    assert fig.layout.margin.b >= 100
