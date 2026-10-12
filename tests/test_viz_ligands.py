"""The ligand sheet's choices and drawing, outside the notebook."""
import pandas as pd

from qpocket.viz import ligands


def _inputs():
    rows = pd.DataFrame({"pdb_id": ["A", "B", "C", "X"], "code": ["L1", "L1", "L2", "L9"],
                         "canonical": ["CCO", "CCO", "c1ccccc1", "CCN"]})
    info = pd.DataFrame({"pdb_id": ["A", "B", "C"], "cluster_id": [0, 1, 1]})
    return rows, info


def test_rows_outside_the_run_are_counted_and_dropped():
    rows, info = _inputs()
    scoped, outside, cluster_of = ligands.in_scope(rows, info, "all clusters")
    assert outside == 1 and set(scoped["pdb_id"]) == {"A", "B", "C"}
    assert set(ligands.in_scope(rows, info, "1")[0]["pdb_id"]) == {"B", "C"}


def test_the_most_crystallised_molecule_comes_first():
    rows, info = _inputs()
    scoped, _, cluster_of = ligands.in_scope(rows, info, "all clusters")
    unique = ligands.distinct_molecules(scoped, cluster_of)
    assert unique[0]["canonical"] == "CCO" and unique[0]["clusters"] == [0, 1]


def test_drawing_and_properties_cover_every_molecule():
    rows, info = _inputs()
    scoped, outside, cluster_of = ligands.in_scope(rows, info, "all clusters")
    unique = ligands.distinct_molecules(scoped, cluster_of)
    svg, mols = ligands.depictions(unique, n_cols=2, all_clusters=True)
    assert svg.startswith("<svg") and len(mols) == 2
    assert list(ligands.properties(unique, mols)["ligand"]) == ["L1", "L2"]
    assert ligands.notes(unique, unique, scoped, outside, 0)[-1].startswith("1 rows name")
