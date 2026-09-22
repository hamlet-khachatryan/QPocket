"""Per-structure work is isolated, so parallel results equal serial ones."""
import ast
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src/qpocket"


def test_per_structure_code_writes_no_shared_file():
    """A shared append (actives.smi) would race; collect rewrites it from the sidecars."""
    source = (SRC / "pipelines/pockets.py").read_text(encoding="utf-8")
    assert "actives.smi" not in source


def test_extraction_stage_uses_run_map():
    tree = ast.parse((SRC / "pipelines/stages.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "featurize_structures")
    assert "run_map" in ast.unparse(fn)


@pytest.mark.parametrize("workers", [1, 3])
def test_every_structure_is_processed_once(tmp_path, monkeypatch, workers):
    from qpocket.pipelines import stages

    seen = tmp_path / "seen"
    seen.mkdir()

    def fake_one(cfg, dirs, row, context):
        (seen / f"{row['pdb_id']}.txt").write_text("done", encoding="utf-8")

    monkeypatch.setattr(stages, "process_one_structure", fake_one)
    df = pd.DataFrame({"pdb_id": [f"P{i}" for i in range(5)]})

    class Compute:
        pass

    class Cfg:
        compute = Compute()

    Cfg.compute.workers = workers
    Cfg.compute.backend = "thread"   # a fake defined in a test module is not picklable
    stages.featurize_structures(Cfg(), {"root": tmp_path}, df, context=None)
    assert sorted(p.stem for p in seen.iterdir()) == [f"P{i}" for i in range(5)]


def test_alignment_context_is_picklable():
    """Workers reload the reference from disk; Biopython entities never cross a process."""
    import pickle

    from qpocket.pipelines.stages import AlignmentContext

    ctx = AlignmentContext(reference_path="a.pdb", reference_chain="A", reference_sequence="ACD",
                           uniprot_id=None, aux_sequences={}, aux_centroids={}, aux_uniprots={},
                           renumber=False, interface_mode=False)
    assert pickle.loads(pickle.dumps(ctx)) == ctx


def test_reference_residues_are_loaded_once_per_process(tmp_path):
    from qpocket.pipelines import stages

    stages.reference_residues.cache_clear()
    pdb = tmp_path / "ref.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n",
        encoding="utf-8")
    first = stages.reference_residues(str(pdb), "A")
    assert stages.reference_residues(str(pdb), "A") is first
    assert stages.reference_residues.cache_info().hits == 1


def test_chain_covers_the_same_units_as_the_stage_path():
    """`qpocket task <stage>` still drives the stage-wise path; the chain reuses its units."""
    src = (SRC / "pipelines/stages.py").read_text(encoding="utf-8")
    chain = next(n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.FunctionDef) and n.name == "structure_chain")
    body = ast.unparse(chain)
    for unit in ("align_one", "clean_one", "process_one_structure", "compute_geodesic"):
        assert unit in body


def test_pipelines_run_the_chained_phase():
    from qpocket.pipelines import fragment_screen, pdb_query, stages

    for module, ingest in ((pdb_query, stages.download_structures),
                           (fragment_screen, stages.ingest_local_structures)):
        assert module.STAGES == (ingest, stages.prepare_and_featurize,
                                 stages.compute_spectral_distance_matrix)
