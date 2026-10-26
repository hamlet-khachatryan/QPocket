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


def _finished_run(tmp_path, monkeypatch):
    """Three structures; site_01 holds two ligands 3.5 Å apart, site_02 one ligand far away."""
    from dataclasses import replace

    from qpocket.core.types import RawStructure, Sources
    from tests.site_helpers import _lig, _structure

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    near = _lig(inputs, "P1", "AAA", 1, [("C1", "C", 0, 0, 0), ("C2", "C", 3, 0, 0)])
    close = _lig(inputs, "P1", "BBB", 2, [("C1", "C", 6.5, 0, 0), ("C2", "C", 7, 0, 0)])
    far = _lig(inputs, "P2", "CCC", 3, [("C1", "C", 50, 0, 0), ("C2", "C", 51, 0, 0)])
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
    status = steps.site_status(root, "site_01")
    assert status == {"clouds": True, "geodesics": True, "shape_clustered": False, "featurised": False,
                      "two_stage": False, "crossdock": False}

    steps.shape_cluster_site(root, "site_01")
    status = steps.site_status(root, "site_01")
    assert status["shape_clustered"] and not status["featurised"] and not status["two_stage"]

    assert sorted(steps.featurize_site(root, "site_01")) == ["APO1", "P1", "P2"]
    assert steps.site_status(root, "site_01")["featurised"]
    assert not steps.site_status(root, "site_01")["two_stage"]

    steps.two_stage_site(root, "site_01")
    status = steps.site_status(root, "site_01")
    assert status["two_stage"] and status["shape_clustered"] and not status["crossdock"]
    assert list((root / "sites" / "site_01" / "metadata" / "clustering_history").iterdir())


def test_two_stage_before_featurise_raises(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    with pytest.raises(steps.SiteError, match="Featurise"):
        steps.two_stage_site(root, "site_01")


def test_featurise_skips_done(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    steps.featurize_site(root, "site_01")
    assert steps.featurize_site(root, "site_01") == []
    cloud = root / "sites" / "site_01" / "pockets" / "P2" / "binding_site.csv"
    later = cloud.with_name("featurized.csv").stat().st_mtime + 10
    os.utime(cloud, (later, later))  # a touched but unchanged cloud is still featurised
    assert steps.featurize_site(root, "site_01") == []
    cloud.write_text(cloud.read_text() + "0.0,0.0,0.0\n")
    assert steps.featurize_site(root, "site_01") == ["P2"]


def test_rerun_leaves_features_current(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    steps.featurize_site(root, "site_01")
    cloud = root / "sites" / "site_01" / "pockets" / "P2" / "binding_site.csv"
    before = cloud.stat().st_mtime_ns
    run.run(load_run_config(root))
    assert cloud.stat().st_mtime_ns == before
    assert steps.featurize_site(root, "site_01") == []


def test_regroup_keeps_unchanged_site(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    sites_dir = root / "sites"
    assert sorted(p.name for p in sites_dir.glob("site_*")) == ["site_01", "site_02"]
    steps.shape_cluster_site(root, "site_02")
    (sites_dir / "site_02" / "metadata" / "marker.txt").write_text("kept")
    (sites_dir / "site_02" / "ligands.pdb").write_text("stale")
    (sites_dir / "site_01" / "metadata" / "marker.txt").write_text("stale")

    table = steps.regroup_sites(root, 3.0)

    assert list(table["site_id"]) == ["site_01", "site_02", "site_03"]
    assert (sites_dir / "site_03" / "metadata" / "marker.txt").read_text() == "kept"
    assert not (sites_dir / "site_01" / "metadata" / "marker.txt").exists()
    assert not (sites_dir / "site_02" / "metadata" / "marker.txt").exists()
    for sid in ("site_01", "site_02", "site_03"):
        assert (sites_dir / sid / "metadata" / "master.csv").exists()
        assert (sites_dir / sid / "pockets" / "APO1" / "binding_site.csv").exists()
    assert load_run_config(root).sites.link_cutoff == 3.0
    assert "CCC" in (sites_dir / "site_03" / "ligands.pdb").read_text()
    kept = pd.read_csv(sites_dir / "site_03" / "metadata" / "clustering_metadata.csv")
    assert all(Path(p).exists() and Path(p).parents[2].name == "site_03" for p in kept["pocket_file"])


def test_regroup_after_a_crash_extracts_the_changed_sites(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    sites_dir = root / "sites"

    def crash(*args, **kwargs):
        raise RuntimeError("killed")

    with monkeypatch.context() as m:
        m.setattr(steps, "extract_all_sites", crash)
        with pytest.raises(RuntimeError):
            steps.regroup_sites(root, 3.0)
    steps.regroup_sites(root, 3.0)
    for sid in ("site_01", "site_02", "site_03"):
        assert (sites_dir / sid / "metadata" / "master.csv").exists()
        assert (sites_dir / sid / "pockets" / "APO1" / "binding_site.csv").exists()


def test_leftover_parked_folders_do_not_break_regroup(tmp_path, monkeypatch):
    import shutil

    root = _finished_run(tmp_path, monkeypatch)
    sites_dir = root / "sites"
    shutil.copytree(sites_dir / "site_02", sites_dir / ".regroup_site_02")  # its name is taken: dropped
    shutil.move(sites_dir / "site_01", sites_dir / ".regroup_site_01")  # its name is free: it is site_01
    (sites_dir / ".regroup_site_01" / "metadata" / "marker.txt").write_text("restored")

    steps.regroup_sites(root, 4.0)
    assert not list(sites_dir.glob(".regroup_*"))
    assert (sites_dir / "site_01" / "metadata" / "marker.txt").read_text() == "restored"
    assert (sites_dir / "site_02" / "metadata" / "master.csv").exists()


def test_unknown_site_is_refused(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    for step in (steps.site_status, steps.featurize_site, steps.shape_cluster_site, steps.two_stage_site):
        with pytest.raises(steps.SiteError, match="unknown site 'site_99'"):
            step(root, "site_99")
    assert not (root / "sites" / "site_99").exists()
