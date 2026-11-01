"""Per-site steps on a finished run: featurise, shape clustering, two-stage, regroup."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpocket.arms import sites as steps
from qpocket.config import QPocketConfig, evolve
from qpocket.io import layout
from qpocket.pipelines import run, structure

NEAR, FAR = "S1-Y1", "S2-Y1"


def _finished_run(tmp_path, monkeypatch):
    """Three structures; the near site (S1-Y1) holds two ligands, the far site (S2-Y1) one."""
    from dataclasses import replace

    from qpocket.core.types import RawStructure, Sources
    from tests.site_helpers import _lig, _structure

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    near = _lig(inputs, "P1", "AAA", 1, [("C1", "C", 0, 2, 0), ("C2", "C", 3, 2, 0)])
    close = _lig(inputs, "P1", "BBB", 2, [("C1", "C", 6.5, 2, 0), ("C2", "C", 7, 2, 0)])
    far = _lig(inputs, "P2", "CCC", 3, [("C1", "C", 50, 2, 0), ("C2", "C", 51, 2, 0)])
    rng = np.random.default_rng(0)

    def points():  # a jittered line through both sites, different per structure
        xs = list(range(-3, 12)) + list(range(47, 55))
        return [(float(x), 1.0 + rng.uniform(0, 2), rng.uniform(-1, 1)) for x in xs]

    records = {"P1": _structure(inputs, "P1", [near, close], points=points()),
               "P2": _structure(inputs, "P2", [far], points=points()),
               "APO1": _structure(inputs, "APO1", points=points(), state="UNBOUND_STATE")}
    raws = tuple(RawStructure(pid, inputs / f"{pid}.pdb") for pid in records)

    def process(ref, p, ws, raw):
        layout.write_part(ws, raw.pdb_id, records[raw.pdb_id].to_json())
        return records[raw.pdb_id]

    monkeypatch.setattr(run, "ingest", lambda cfg, ws: Sources(raws[0], raws))
    monkeypatch.setattr(run, "build_reference", lambda cfg, ws, src: "REF")
    monkeypatch.setattr(run, "structure_params", lambda cfg: None)
    monkeypatch.setattr(run, "process", process)
    cfg = evolve(QPocketConfig(), "paths.workdir", str(tmp_path / "work"))
    cfg = evolve(cfg, "reference.pdb_id", "1ABC")
    return run.run(replace(cfg, compute=replace(cfg.compute, workers=1, backend="thread")))


@pytest.fixture
def fake_features(monkeypatch):
    """featurize writes two random feature columns beside the cloud, with no pdb2pqr or APBS."""
    calls = []

    def featurize(csv_path, pdb_path, features, charges, options=None, charge_params=None):
        calls.append(csv_path.parent.name)
        table = pd.read_csv(csv_path)
        rng = np.random.default_rng(len(table) + len(calls))
        table.assign(apbs=rng.normal(size=len(table)), hbond=rng.normal(size=len(table))).to_csv(
            csv_path.with_name("featurized.csv"), index=False)

    monkeypatch.setattr(structure, "featurize", featurize)
    return calls


def test_shape_then_featurise_then_two_stage(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    status = steps.site_status(root, NEAR)
    assert status == {"clouds": True, "geodesics": True, "shape_clustered": False, "featurised": False,
                      "two_stage": False, "crossdock": False, "n_featurised": 0, "n_clouds": 3}

    steps.shape_cluster_site(root, NEAR)
    status = steps.site_status(root, NEAR)
    assert status["shape_clustered"] and not status["featurised"] and not status["two_stage"]

    assert sorted(steps.featurize_site(root, NEAR)) == ["APO1", "P1", "P2"]
    assert steps.site_status(root, NEAR)["featurised"]
    assert not steps.site_status(root, NEAR)["two_stage"]

    steps.two_stage_site(root, NEAR)
    status = steps.site_status(root, NEAR)
    assert status["two_stage"] and status["shape_clustered"] and not status["crossdock"]
    assert list((root / "sites" / NEAR / "metadata" / "clustering_history").iterdir())


def test_one_failed_featurisation_still_allows_two_stage(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    steps.featurize_site(root, NEAR)
    (root / "sites" / NEAR / "pockets" / "P2" / "featurized.csv").unlink()
    status = steps.site_status(root, NEAR)
    assert not status["featurised"] and (status["n_featurised"], status["n_clouds"]) == (2, 3)
    steps.two_stage_site(root, NEAR)
    assert steps.site_status(root, NEAR)["two_stage"]


def test_regroup_that_changes_membership_drops_the_site(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    sites_dir = root / "sites"
    (sites_dir / NEAR / "metadata" / "marker.txt").write_text("kept")

    table = steps.regroup_sites(root, min_cavity_points=100)   # the far cavities hold 64 points

    assert list(table["site_id"]) == [NEAR]
    assert (sites_dir / NEAR / "metadata" / "marker.txt").read_text() == "kept"
    assert not (sites_dir / FAR).exists()
    misses = pd.read_csv(sites_dir / "cavfind_misses.csv")
    assert misses.set_index("resname").loc["CCC", "reason"] == "no_site"


def test_regroup_that_flips_a_ligand_refreshes_the_site_tables(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    site = root / "sites" / FAR
    assert pd.read_csv(site / "metadata" / "master.csv").set_index("pdb_id").loc["P2", "protein_state"] \
        == "BOUND_STATE"
    (site / "metadata" / "marker.txt").write_text("kept")
    ligands = pd.read_csv(root / "structures" / "ligands.csv", keep_default_na=False).set_index("resname")
    path = Path(ligands.loc["CCC", "file"])
    path.write_text(path.read_text().replace("  51.000", "  70.000"))  # one of two atoms leaves the box

    steps.regroup_sites(root, ligand_fraction=0.6)

    master = pd.read_csv(site / "metadata" / "master.csv").set_index("pdb_id")
    assert master.loc["P2", "protein_state"] == "UNBOUND_STATE"
    assert "CCC" not in (site / "screening" / "actives.smi").read_text()
    assert (site / "metadata" / "marker.txt").read_text() == "kept"   # same cavities: the folder is kept


# --- manual site edits ----------------------------------------------------------


def test_drop_removes_the_site_folder(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    table = steps.add_override(root, steps.drop_override(root, FAR))
    assert list(table["site_id"]) == [NEAR] and not (root / "sites" / FAR).exists()
    entry = steps.list_overrides(root)[0]
    assert entry["op"] == "drop" and entry["label"] == FAR and entry["status"] == "applied"


def test_split_makes_one_site_per_strict_group(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    table = steps.add_override(root, steps.split_override(root, FAR))
    entry = steps.list_overrides(root)[0]
    assert entry["status"] == "applied" and entry["parts"] == 2
    assert NEAR in set(table["site_id"])  # the other site is untouched
    assert set(entry["site_ids"]) <= set(table["site_id"])


def test_split_works_on_a_grouping_made_before_cores_were_recorded(tmp_path, monkeypatch):
    import json

    root = _finished_run(tmp_path, monkeypatch)
    path = root / "sites" / "grouping.json"
    saved = json.loads(path.read_text())
    path.write_text(json.dumps({"sites": saved["sites"], "inputs": saved["inputs"]}))
    assert steps.split_override(root, FAR)["op"] == "split"
    with pytest.raises(steps.SiteError, match="one strict cavity group"):
        steps.split_override(root, NEAR)


# --- review fixes -------------------------------------------------------------

def _members(root, site_id):
    table = pd.read_csv(root / "sites" / "cavities.csv", keep_default_na=False)
    return {(r.pdb_id, r.cavity) for r in table.itertuples() if site_id in r.site_ids.split(";")}


def test_a_merged_site_keeps_its_ligands_and_members(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    union = _members(root, NEAR) | _members(root, FAR)
    table = steps.add_override(root, steps.merge_override(root, [NEAR, FAR]))
    row = table.iloc[0]
    assert len(table) == 1 and row.n_ligands == 3 and row.n_holo == 2
    assert _members(root, row.site_id) == union


# --- representative override ----------------------------------------------------

def _clustered(root, site_id):
    steps.shape_cluster_site(root, site_id)
    return pd.read_csv(root / "sites" / site_id / "metadata" / "clustering_metadata.csv")


def _a_swap(table):
    """A (cluster_id, pdb_id) whose pdb_id is not yet its cluster's representative, or None."""
    for cluster, members in table.groupby("cluster_id"):
        others = members[~members["is_representative"].astype(bool)]
        if len(others):
            return int(cluster), str(others["pdb_id"].iloc[0])
    return None


def test_set_representative_swaps_it_everywhere(tmp_path, monkeypatch):
    import json

    root = _finished_run(tmp_path, monkeypatch)
    table = _clustered(root, NEAR)
    swap = _a_swap(table)
    if swap is None:  # every cluster a singleton: put all in one cluster so a swap exists
        table["cluster_id"] = 0
        table["is_representative"] = [True] + [False] * (len(table) - 1)
        table.to_csv(root / "sites" / NEAR / "metadata" / "clustering_metadata.csv", index=False)
        swap = _a_swap(table)
    cluster, pdb_id = swap
    steps.set_representative(root, NEAR, cluster, pdb_id)
    meta = root / "sites" / NEAR / "metadata"
    after = pd.read_csv(meta / "clustering_metadata.csv")
    reps = after[(after["cluster_id"] == cluster) & after["is_representative"].astype(bool)]
    assert reps["pdb_id"].tolist() == [pdb_id] and reps["dist_to_exemplar"].iloc[0] == 0
    selected = pd.read_csv(meta / "selected_representatives.csv")
    assert pdb_id in selected["pdb_id"].astype(str).tolist()
    assert selected["cluster_id"].tolist().count(cluster) == 1
    params = json.loads((meta / "clustering_params.json").read_text())
    assert params["representative_overrides"] == {str(cluster): pdb_id}


def _meta_dir(tmp_path, mode):
    """A clustering of A, B, C in one cluster with A its representative; shape and spectral distances differ."""
    import json

    ids = ["A", "B", "C"]
    pd.DataFrame({"pdb_id": ids, "cluster_id": 0, "is_representative": [True, False, False],
                  "dist_to_exemplar": [0.0, 1.0, 2.0]}).to_csv(tmp_path / "clustering_metadata.csv", index=False)
    shape = pd.DataFrame([[0, 1, 2], [1, 0, 3], [2, 3, 0]], index=ids, columns=ids, dtype=float)
    shape.to_csv(tmp_path / "shape_distance_matrix.csv")
    (shape * 10).to_csv(tmp_path / "spectral_distance_matrix.csv")
    (tmp_path / "clustering_params.json").write_text(json.dumps({"mode": mode}))
    return tmp_path


