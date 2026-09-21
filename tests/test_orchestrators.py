"""One orchestrator per pipeline, built from the shared stages."""
from pathlib import Path

from qpocket.pipelines import fragment_screen, pdb_query, run, stages

SHARED = (
    stages.align_structures,
    stages.clean_structures,
    stages.extract_pockets_and_interactions,
    stages.compute_spectral_distance_matrix,
)


def test_pipelines_differ_only_in_how_structures_come_in():
    assert pdb_query.STAGES == (stages.download_structures,) + SHARED
    assert fragment_screen.STAGES == (stages.ingest_local_structures,) + SHARED


def test_run_dispatches_on_mode(monkeypatch):
    seen = []
    monkeypatch.setattr(fragment_screen, "run", lambda cfg: seen.append(cfg.mode) or Path("x"))

    class Cfg:
        mode = "fragment_screen"

    assert run.run(Cfg()) == Path("x") and seen == ["fragment_screen"]


def test_every_core_module_is_under_the_layer_rules():
    from tests.test_layers import CORE, CORE_MODULES_DONE
    modules = sorted(p.name for p in CORE.glob("*.py") if p.name != "__init__.py")
    assert sorted(CORE_MODULES_DONE) == modules
