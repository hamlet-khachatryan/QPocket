"""Finalise freezes a site's clustering, representatives, residue profile and actives into final/."""
import json

import pandas as pd
import pytest

from qpocket.arms.sites import final
from qpocket.arms import sites as steps
from tests.test_site_steps import FAR, _finished_run


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    steps.shape_cluster_site(root, FAR)
    return root


def test_finalise_writes_the_four_files(run):
    folder = final.finalise_site(run, FAR)
    assert sorted(p.name for p in folder.iterdir()) == sorted(final.FINAL_FILES)
    clusters = pd.read_csv(folder / "clusters.csv")
    assert list(clusters.columns) == ["pdb_id", "cluster", "cluster_id", "is_representative",
                                      "dist_to_representative", "holo"]
    assert clusters.groupby("cluster")["is_representative"].sum().eq(1).all()
    assert clusters.set_index("pdb_id").loc["P2", "holo"]
    assert list(pd.read_csv(folder / "residue_profile.csv").columns) == ["cluster", "residue", "contacted_pct",
                                                                         "lining_pct", "n_members",
                                                                         "n_with_contacts"]
    ligands = pd.read_csv(folder / "ligands.csv", keep_default_na=False)
    assert list(ligands.columns) == ["cluster", "pdb_id", "ligand_key", "resname", "smiles"]
    assert ligands["resname"].tolist() == ["CCC"]
    settings = json.loads((folder / "settings.json").read_text())
    for key in ("site_id", "finalised_at", "shape_only", "features", "clustering", "n_structures", "n_clusters",
                "representatives", "matrices", "clustering_hash"):
        assert key in settings
    assert settings["matrices"] and list(settings["matrices"]) == ["shape_distance_matrix.csv"]
    assert settings["representatives"] == clusters.loc[clusters["is_representative"], "pdb_id"].tolist()


def test_new_clustering_makes_the_final_stale(run):
    assert final.final_status(run, FAR) == "missing"
    final.finalise_site(run, FAR)
    assert final.final_status(run, FAR) == "current"
    table = pd.read_csv(run / "sites" / FAR / "metadata" / "clustering_metadata.csv")
    table["cluster_id"] = 0
    table.to_csv(run / "sites" / FAR / "metadata" / "clustering_metadata.csv", index=False)
    assert final.final_status(run, FAR) == "stale"


def _one_cluster(run):
    """FAR's structures in one cluster, its first the representative; returns the table."""
    path = run / "sites" / FAR / "metadata" / "clustering_metadata.csv"
    table = pd.read_csv(path, dtype={"pdb_id": str})
    table["cluster_id"] = 0
    table["is_representative"] = [True] + [False] * (len(table) - 1)
    shape = pd.read_csv(path.with_name("shape_distance_matrix.csv"), index_col=0)
    shape.index, shape.columns = shape.index.astype(str), shape.columns.astype(str)
    table["dist_to_exemplar"] = [float(shape.loc[p, table["pdb_id"].iloc[0]]) for p in table["pdb_id"]]
    table.to_csv(path, index=False)
    return table


def test_a_representative_swap_makes_the_final_stale(run):
    table = _one_cluster(run)
    final.finalise_site(run, FAR)
    steps.set_representative(run, FAR, 0, table["pdb_id"].iloc[1])
    assert final.final_status(run, FAR) == "stale"


def test_a_failed_finalise_keeps_the_previous_final(run, monkeypatch):
    final.finalise_site(run, FAR)
    before = (run / "sites" / FAR / "final" / "clusters.csv").read_text()

    def broken(*args, **kwargs):
        raise AttributeError("old ligands.csv layout")

    monkeypatch.setattr(final, "_ligands", broken)
    with pytest.raises(AttributeError):
        final.finalise_site(run, FAR)
    folder = run / "sites" / FAR / "final"
    assert final.final_status(run, FAR) == "current"
    assert (folder / "clusters.csv").read_text() == before
    assert not (folder / "history").exists() and not list(folder.glob(".incoming*"))


