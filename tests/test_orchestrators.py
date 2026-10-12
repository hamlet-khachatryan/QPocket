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
