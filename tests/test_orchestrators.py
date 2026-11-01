"""One run for both modes; they differ only in the structure source."""
import pytest

from qpocket.config import FragmentScreenParams, QPocketConfig, evolve
from qpocket.io import layout
from qpocket.pipelines import run


def test_fragment_screen_mode_ingests_local_files(tmp_path, monkeypatch):
    from qpocket.plugins.sources import local

    seen = []
    monkeypatch.setattr(local, "copy_inputs", lambda params, ws: seen.append("local") or [])
    cfg = evolve(QPocketConfig(mode="fragment_screen"), "fragment_screen",
                 FragmentScreenParams(input_type="paths", structure_paths=["a.pdb"]))
    with pytest.raises(ValueError, match="ingested no structures"):
        run.ingest(cfg, layout.create(str(tmp_path), "run"))
    assert seen == ["local"]


def test_pdb_query_mode_searches_rcsb(tmp_path, monkeypatch):
    from qpocket.io import rcsb

    seen = []
    monkeypatch.setattr(rcsb, "fetch_structure_ids", lambda **kw: seen.append(kw["ref_pdb_id"]) or [])
    cfg = evolve(QPocketConfig(), "reference.pdb_id", "7KG6")
    with pytest.raises(ValueError, match="no structures"):
        run.ingest(cfg, layout.create(str(tmp_path), "run"))
    assert seen == ["7KG6"]


# ── the whole run, with the structure chain faked ────────────────────────────

def _fake_run(tmp_path, monkeypatch, holo=True, **compute):
    """A two-structure run: a holo structure and an apo one, both with cavities at the site."""
    from dataclasses import replace

    from qpocket.core.types import RawStructure, Sources
    from tests.site_helpers import _lig, _structure

    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    ligs = [_lig(inputs, "P1", "LIG", 1, [("C1", "C", 0, 0, 0), ("C2", "C", 3, 0, 0)])] if holo else []
    records = {"P1": _structure(inputs, "P1", ligs, state="BOUND_STATE" if holo else "UNBOUND_STATE"),
               "APO1": _structure(inputs, "APO1", state="UNBOUND_STATE")}
    raws = tuple(RawStructure(pid, inputs / f"{pid}.pdb") for pid in records)
    monkeypatch.setattr(run, "ingest", lambda cfg, ws: Sources(raws[0], raws))
    monkeypatch.setattr(run, "build_reference", lambda cfg, ws, src: "REF")
    monkeypatch.setattr(run, "structure_params", lambda cfg: None)
    monkeypatch.setattr(run, "process", lambda ref, p, ws, raw: records[raw.pdb_id])
    cfg = evolve(QPocketConfig(), "paths.workdir", str(tmp_path / "work"))
    cfg = evolve(cfg, "reference.pdb_id", "1ABC")
    return replace(cfg, compute=replace(cfg.compute, **{"workers": 1, "backend": "thread", **compute}))


def test_run_writes_sites_and_no_matrices(tmp_path, monkeypatch):
    import pandas as pd

    from qpocket.config import load_run_config

    from tests.site_helpers import site_ids

    root = run.run(_fake_run(tmp_path, monkeypatch))
    assert (root / "sites" / "sites.csv").exists()
    master = pd.read_csv(root / "sites" / site_ids(root)[0] / "metadata" / "master.csv")
    assert list(master["pdb_id"]) == ["P1", "APO1"]
    assert list(pd.read_csv(root / "metadata" / "master.csv")["pdb_id"]) == ["P1", "APO1"]
    assert (root / "metadata" / "failures.csv").exists()
    assert not (root / "metadata" / "observed_interactions.h5").exists()
    assert not (root / "screening" / "actives.smi").exists()
    assert not list(root.rglob("*distance_matrix.csv"))
    assert not list(root.rglob("featurized.csv"))
    assert load_run_config(root).sites.cavity_overlap == 0.5


def test_rerun_reconciles_renumbered_sites(tmp_path, monkeypatch):
    """A third structure joins the far site, which takes rank 1: only the untouched site keeps its results."""
    from dataclasses import replace

    from qpocket.core.types import RawStructure, Sources
    from tests.site_helpers import INSIDE, _lig, _structure, site_ids

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    points = INSIDE + [(float(x), 2.0, 0.0) for x in range(47, 55)]
    near = _lig(inputs, "P1", "AAA", 1, [("C1", "C", 0, 0, 0), ("C2", "C", 3, 0, 0)])
    records = {"P1": _structure(inputs, "P1", [near], points=points),
               "P2": _structure(inputs, "P2", [_lig(inputs, "P2", "CCC", 2, [("C1", "C", 50, 2, 0)])],
                                points=points)}
    current = {}
    monkeypatch.setattr(run, "ingest", lambda cfg, ws: Sources(current["raws"][0], current["raws"]))
    monkeypatch.setattr(run, "build_reference", lambda cfg, ws, src: "REF")
    monkeypatch.setattr(run, "structure_params", lambda cfg: None)
    monkeypatch.setattr(run, "process", lambda ref, p, ws, raw: records[raw.pdb_id])
    cfg = evolve(evolve(QPocketConfig(), "paths.workdir", str(tmp_path / "work")), "reference.pdb_id", "1ABC")
    cfg = replace(cfg, compute=replace(cfg.compute, workers=1, backend="thread"))

    current["raws"] = tuple(RawStructure(pid, inputs / f"{pid}.pdb") for pid in records)
    root = run.run(cfg)
    sites = root / "sites"
    assert site_ids(root) == ["S1-Y1", "S2-Y1"]           # tied on structures: the near site first
    assert "AAA" in (sites / "S1-Y1" / "ligands.pdb").read_text()
    (sites / "S1-Y1" / "metadata" / "keep.txt").write_text("near")
    (sites / "S2-Y1" / "metadata" / "keep.txt").write_text("far")

    far_only = [(float(x), 2.0, 0.0) for x in range(47, 55)]
    records["P3"] = _structure(inputs, "P3", [_lig(inputs, "P3", "DDD", 3, [("C1", "C", 51, 2, 0)])],
                               points=far_only)
    current["raws"] = tuple(RawStructure(pid, inputs / f"{pid}.pdb") for pid in records)
    run.run(cfg)

    assert site_ids(root) == ["S1-Y1", "S2-Y1"]           # the far site now has three structures
    assert "AAA" in (sites / "S2-Y1" / "ligands.pdb").read_text()
    assert (sites / "S2-Y1" / "metadata" / "keep.txt").read_text() == "near"
    assert not (sites / "S1-Y1" / "metadata" / "keep.txt").exists()
    assert (sites / "S1-Y1" / "pockets" / "P3" / "binding_site.csv").exists()
    assert not list(sites.glob(".regroup_*"))


def test_run_with_no_ligands(tmp_path, monkeypatch):
    import pandas as pd

    root = run.run(_fake_run(tmp_path, monkeypatch, holo=False))
    sites = pd.read_csv(root / "sites" / "sites.csv")
    assert len(sites) == 1 and sites["n_holo"].tolist() == [0]   # the cavities still make a site
    assert list(pd.read_csv(root / "metadata" / "master.csv")["pdb_id"]) == ["P1", "APO1"]


