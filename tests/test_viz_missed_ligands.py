"""Ligands no site claimed: found from the run's tables and drawn brick red on the Site page."""
from pathlib import Path

import pandas as pd

from qpocket.viz import frame_structure, missed_ligands
from qpocket.viz.pages import site as site_page

LIGAND = ("HETATM    1  C1  Q00 A 805       1.000   2.000   3.000  1.00 20.00           C\n"
          "HETATM    2  O1  Q00 A 805       2.200   2.000   3.000  1.00 20.00           O\nEND\n")


def _run(root: Path) -> Path:
    """Three ligands, two of them missed (one sits in no site, one in a site without a cavity)."""
    run = root / "run"
    (run / "structures" / "extracted_ligands").mkdir(parents=True)
    (run / "sites").mkdir()
    (run / "metadata").mkdir()
    rows = []
    for pdb, resseq, icode, sites in (("x1a", "805", " ", "S1"), ("x2a", "701", " ", ""), ("x3a", "12", "B", "")):
        path = run / "structures" / "extracted_ligands" / f"{pdb}_A_Q00_{resseq}.pdb"
        path.write_text(LIGAND, encoding="utf-8")
        rows.append({"pdb_id": pdb, "chain": "A", "resname": "Q00", "resseq": resseq, "icode": icode,
                     "n_heavy": 2, "file": str(path), "smiles": "CO", "site_ids": sites})
    pd.DataFrame(rows).to_csv(run / "structures" / "ligands.csv", index=False)
    pd.DataFrame({"pdb_id": ["x3a", "x2a"], "ligand_key": ["x3a:A:Q00:12B", "x2a:A:Q00:701"],
                  "resname": ["Q00", "Q00"], "sites": ["S1", ""],
                  "reason": ["no_cavity_at_site", "no_site"]}).to_csv(run / "sites" / "cavfind_misses.csv",
                                                                       index=False)
    aligned = run / "structures" / "x1a_aligned.pdb"
    aligned.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n",
                       encoding="utf-8")
    pd.DataFrame({"pdb_id": ["x0a", "x1a"], "aligned_file": [str(run / "gone.pdb"), str(aligned)]}).to_csv(
        run / "metadata" / "master.csv", index=False)
    return run


def test_missed_ligands_follow_the_misses_file(tmp_path):
    run = _run(tmp_path)
    assert [Path(f).name for f in missed_ligands(run)] == ["x3a_A_Q00_12.pdb", "x2a_A_Q00_701.pdb"]


def test_no_misses_file_means_none(tmp_path):
    assert missed_ligands(tmp_path) == []


def test_frame_is_the_first_aligned_structure_on_disk(tmp_path):
    run = _run(tmp_path)
    assert frame_structure(run).name == "x1a_aligned.pdb"
    assert frame_structure(tmp_path) is None


def test_missed_scene_draws_every_miss_brick_red(tmp_path):
    run = _run(tmp_path)
    sites = pd.DataFrame({"site_id": ["S1"], "centre_x": [0.0], "centre_y": [0.0], "centre_z": [0.0],
                          "n_structures": [3], "n_holo": [1], "n_ligands": [1]})
    html = site_page.missed_scene(sites, frame_structure(run), missed_ligands(run), offline=False)
    assert html.count(site_page.MISSED_CARBON) == 2
    assert "2 missed ligands brick red" in html and "3 structures" in html
    assert site_page.MISSES not in set(sites["site_id"])
