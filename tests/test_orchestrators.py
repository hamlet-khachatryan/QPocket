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


def test_every_core_module_is_under_the_layer_rules():
    from tests.test_layers import CORE, CORE_MODULES_DONE
    modules = sorted(p.name for p in CORE.glob("*.py") if p.name != "__init__.py")
    assert sorted(CORE_MODULES_DONE) == modules


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

    root = run.run(_fake_run(tmp_path, monkeypatch))
    assert (root / "sites" / "sites.csv").exists()
    master = pd.read_csv(root / "sites" / "site_01" / "metadata" / "master.csv")
    assert list(master["pdb_id"]) == ["P1", "APO1"]
    assert list(pd.read_csv(root / "metadata" / "master.csv")["pdb_id"]) == ["P1", "APO1"]
    assert (root / "metadata" / "failures.csv").exists()
    assert not (root / "metadata" / "observed_interactions.h5").exists()
    assert not (root / "screening" / "actives.smi").exists()
    assert not list(root.rglob("*distance_matrix.csv"))
    assert not list(root.rglob("featurized.csv"))
    assert load_run_config(root).sites.link_cutoff == 4.0


def test_rerun_keeps_site_results(tmp_path, monkeypatch):
    cfg = _fake_run(tmp_path, monkeypatch)
    root = run.run(cfg)
    marker = root / "sites" / "site_01" / "metadata" / "keep.txt"
    marker.write_text("x")
    cloud = root / "sites" / "site_01" / "pockets" / "APO1" / "binding_site.csv"
    run.run(cfg)
    assert marker.exists() and cloud.exists()


def test_run_with_no_ligands(tmp_path, monkeypatch):
    import pandas as pd

    root = run.run(_fake_run(tmp_path, monkeypatch, holo=False))
    assert pd.read_csv(root / "sites" / "sites.csv").empty
    assert list(pd.read_csv(root / "metadata" / "master.csv")["pdb_id"]) == ["P1", "APO1"]


def test_site_extraction_runs_on_worker_processes(tmp_path, monkeypatch):
    """The per-pair worker crosses a process boundary, so it must pickle."""
    import pandas as pd

    from qpocket.pipelines.sites import discover_sites, extract_all_sites

    cfg = _fake_run(tmp_path, monkeypatch, workers=2, backend="process")
    records = [run.process(None, None, None, raw) for raw in run.ingest(cfg, None).structures]
    ws = layout.create(cfg.paths.workdir, "run")
    extract_all_sites(cfg, ws, discover_sites(cfg, ws, records), records)
    master = pd.read_csv(ws.sites / "site_01" / "metadata" / "master.csv")
    assert list(master["protein_state"]) == ["BOUND_STATE", "UNBOUND_STATE"]


def test_saved_config_loads_back_through_validation(tmp_path):
    from qpocket.config import load_qpocket_config, load_run_config, save_run_config

    cfg = load_qpocket_config(overrides=["reference.pdb_id=1ABC", "sites.link_cutoff=6.5"])
    ws = layout.create(str(tmp_path), "run")
    save_run_config(cfg, ws)
    loaded = load_run_config(ws.root)
    assert loaded.sites.link_cutoff == 6.5 and loaded.reference.pdb_id == "1ABC"
