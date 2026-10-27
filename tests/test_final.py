"""Finalise freezes a site's clustering, representatives, residue profile and actives into final/."""
import json
from datetime import datetime

import pandas as pd
import pytest

from qpocket.arms import final
from qpocket.arms import sites as steps
from tests.test_site_steps import FAR, NEAR, _finished_run


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
                                                                         "lining_pct"]
    ligands = pd.read_csv(folder / "ligands.csv", keep_default_na=False)
    assert list(ligands.columns) == ["cluster", "pdb_id", "ligand_key", "resname", "smiles"]
    assert ligands["resname"].tolist() == ["CCC"]
    settings = json.loads((folder / "settings.json").read_text())
    for key in ("site_id", "finalised_at", "shape_only", "features", "clustering", "n_structures", "n_clusters",
                "representatives", "matrices", "clustering_hash"):
        assert key in settings
    assert settings["matrices"] and list(settings["matrices"]) == ["shape_distance_matrix.csv"]
    assert settings["representatives"] == clusters.loc[clusters["is_representative"], "pdb_id"].tolist()


def test_shape_only_final_records_no_features(run):
    settings = json.loads((final.finalise_site(run, FAR) / "settings.json").read_text())
    assert settings["shape_only"] is True and settings["features"] == []


def test_two_finalises_keep_two_histories(run):
    now = datetime(2026, 10, 27, 18, 4, 0)
    final.finalise_site(run, FAR, now=now)
    final.finalise_site(run, FAR, now=now)
    folder = final.finalise_site(run, FAR, now=now)
    history = sorted(p.name for p in (folder / "history").iterdir())
    assert history == ["2026-10-27T18-04-00", "2026-10-27T18-04-00-2"]
    assert all((folder / "history" / h / "clusters.csv").exists() for h in history)
    assert not (folder / "history" / history[0] / "history").exists()


def test_finalise_without_interactions_writes_an_empty_profile(run):
    for name in ("observed_interactions.h5", "potential_interactions.h5"):
        (run / "sites" / FAR / "metadata" / name).unlink(missing_ok=True)
    profile = pd.read_csv(final.finalise_site(run, FAR) / "residue_profile.csv")
    assert profile.empty and "contacted_pct" in profile.columns


def test_ligands_are_this_sites_members_only(run):
    table = pd.read_csv(run / "structures" / "ligands.csv", dtype=str, keep_default_na=False)
    stray = table.iloc[[2]].assign(pdb_id="X9")       # a ligand at FAR in a structure the site does not hold
    pd.concat([table, stray]).to_csv(run / "structures" / "ligands.csv", index=False)
    ligands = pd.read_csv(final.finalise_site(run, FAR) / "ligands.csv", keep_default_na=False)
    assert ligands["pdb_id"].tolist() == ["P2"]


def test_new_clustering_makes_the_final_stale(run):
    assert final.final_status(run, FAR) == "missing"
    final.finalise_site(run, FAR)
    assert final.final_status(run, FAR) == "current"
    table = pd.read_csv(run / "sites" / FAR / "metadata" / "clustering_metadata.csv")
    table["cluster_id"] = 0
    table.to_csv(run / "sites" / FAR / "metadata" / "clustering_metadata.csv", index=False)
    steps.set_representative(run, FAR, 0, "P2")
    assert final.final_status(run, FAR) == "stale"


def test_finalise_before_clustering_is_a_site_error(run):
    with pytest.raises(steps.SiteError, match="cluster this site first"):
        final.finalise_site(run, NEAR)


def test_read_final_returns_the_tables(run):
    final.finalise_site(run, FAR)
    got = final.read_final(run, FAR)
    assert set(got) == {"clusters", "settings", "residue_profile", "ligands"}
    assert isinstance(got["settings"], dict) and len(got["clusters"]) == 3
