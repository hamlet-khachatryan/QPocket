"""Per-site steps on a finished run: featurise, shape clustering, two-stage, regroup."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpocket.arms import sites as steps
from qpocket.config import QPocketConfig, evolve, load_run_config
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


def test_shape_clustering_one_pocket_is_a_site_error(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    for pid in ("P1", "APO1"):
        (root / "sites" / NEAR / "pockets" / pid / "binding_site.csv").unlink()
    with pytest.raises(steps.SiteError, match="fewer than 2 pockets"):
        steps.shape_cluster_site(root, NEAR)


def test_two_stage_that_writes_no_matrix_raises(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    steps.featurize_site(root, NEAR)
    steps.two_stage_site(root, NEAR)  # an earlier matrix must not pass for this run's
    monkeypatch.setattr(steps, "compute_spectral_distance_matrix", lambda cfg, ws: None)
    with pytest.raises(steps.SiteError, match="wrote no"):
        steps.two_stage_site(root, NEAR)


def test_featurise_records_the_feature_set(tmp_path, monkeypatch, fake_features):
    import json

    root = _finished_run(tmp_path, monkeypatch)
    record = root / "sites" / NEAR / "metadata" / "features.json"
    assert not record.exists()
    steps.featurize_site(root, NEAR)
    assert json.loads(record.read_text())["enabled"] == list(load_run_config(root).features.enabled)


def test_two_stage_before_featurise_raises(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    with pytest.raises(steps.SiteError, match="Featurise"):
        steps.two_stage_site(root, NEAR)


def test_featurise_skips_done(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    steps.featurize_site(root, NEAR)
    assert steps.featurize_site(root, NEAR) == []
    cloud = root / "sites" / NEAR / "pockets" / "P2" / "binding_site.csv"
    later = cloud.with_name("featurized.csv").stat().st_mtime + 10
    os.utime(cloud, (later, later))  # a touched but unchanged cloud is still featurised
    assert steps.featurize_site(root, NEAR) == []
    cloud.write_text(cloud.read_text() + "0.0,0.0,0.0\n")
    assert steps.featurize_site(root, NEAR) == ["P2"]


def test_rerun_leaves_features_current(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    steps.featurize_site(root, NEAR)
    cloud = root / "sites" / NEAR / "pockets" / "P2" / "binding_site.csv"
    before = cloud.stat().st_mtime_ns
    run.run(load_run_config(root))
    assert cloud.stat().st_mtime_ns == before
    assert steps.featurize_site(root, NEAR) == []


def test_regroup_keeps_unchanged_sites(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    sites_dir = root / "sites"
    steps.shape_cluster_site(root, FAR)
    for sid in (NEAR, FAR):
        (sites_dir / sid / "metadata" / "marker.txt").write_text("kept")
    (sites_dir / FAR / "ligands.pdb").write_text("stale")

    table = steps.regroup_sites(root, ligand_fraction=0.4)

    assert list(table["site_id"]) == [NEAR, FAR]
    for sid in (NEAR, FAR):
        assert (sites_dir / sid / "metadata" / "marker.txt").read_text() == "kept"
    assert "CCC" in (sites_dir / FAR / "ligands.pdb").read_text()
    assert load_run_config(root).sites.ligand_fraction == 0.4
    kept = pd.read_csv(sites_dir / FAR / "metadata" / "clustering_metadata.csv")
    assert all(Path(p).exists() and Path(p).parents[2].name == FAR for p in kept["pocket_file"])


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


def test_regroup_after_a_crash_extracts_the_changed_sites(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    sites_dir = root / "sites"

    def crash(*args, **kwargs):
        raise RuntimeError("killed")

    with monkeypatch.context() as m:
        m.setattr(steps, "extract_all_sites", crash)
        with pytest.raises(RuntimeError):
            steps.regroup_sites(root, cavity_overlap=0.4)
    table = steps.regroup_sites(root, cavity_overlap=0.4)
    for sid in table["site_id"]:
        assert (sites_dir / sid / "metadata" / "master.csv").exists()
        assert (sites_dir / sid / "pockets" / "APO1" / "binding_site.csv").exists()


def test_leftover_parked_folders_do_not_break_regroup(tmp_path, monkeypatch):
    import shutil

    root = _finished_run(tmp_path, monkeypatch)
    sites_dir = root / "sites"
    shutil.copytree(sites_dir / FAR, sites_dir / f".regroup_{FAR}")   # its name is taken: dropped
    shutil.move(sites_dir / NEAR, sites_dir / f".regroup_{NEAR}")     # its name is free: it is NEAR
    (sites_dir / f".regroup_{NEAR}" / "metadata" / "marker.txt").write_text("restored")

    steps.regroup_sites(root, ligand_fraction=0.5)
    assert not list(sites_dir.glob(".regroup_*"))
    assert (sites_dir / NEAR / "metadata" / "marker.txt").read_text() == "restored"
    assert (sites_dir / FAR / "metadata" / "master.csv").exists()


def test_regroup_needs_a_known_parameter(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="at least one"):
        steps.regroup_sites(root)
    with pytest.raises(ValueError, match="link_cutoff"):
        steps.regroup_sites(root, link_cutoff=3.0)


def test_unknown_site_is_refused(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    for step in (steps.site_status, steps.featurize_site, steps.shape_cluster_site, steps.two_stage_site):
        with pytest.raises(steps.SiteError, match="unknown site 'site_99'"):
            step(root, "site_99")
    assert not (root / "sites" / "site_99").exists()


def test_a_run_saved_with_the_silhouette_defaults_still_clusters(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    root = _finished_run(tmp_path, monkeypatch)
    path = Path(root) / "metadata" / "config.yaml"
    saved = OmegaConf.load(path)
    saved.distance.clustering.shape = {"method": "hierarchical", "linkage": "average", "k": None,
                                       "k_range": [2, 10], "criterion": "count", "threshold": None}
    saved.distance.clustering.chem = {"method": "kmedoids", "k": None, "k_range": [2, 6],
                                      "criterion": "count", "threshold": None}
    path.write_text(OmegaConf.to_yaml(saved), encoding="utf-8")
    steps.shape_cluster_site(root, NEAR)
    assert steps.site_status(root, NEAR)["shape_clustered"]


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
