"""Per-structure work is isolated, so parallel results equal serial ones."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src/qpocket"


def test_per_structure_code_writes_no_shared_file():
    """A shared append (actives.smi) would race; collect writes it once from the records."""
    source = (SRC / "pipelines/structure.py").read_text(encoding="utf-8")
    assert "actives.smi" not in source


def _function(path, name):
    tree = ast.parse((SRC / path).read_text(encoding="utf-8"))
    return ast.unparse(next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == name))


def test_the_run_fans_out_with_run_map():
    assert "run_map" in _function("pipelines/run.py", "run")


def test_the_chain_covers_every_step():
    body = _function("pipelines/structure.py", "process")
    for step in ("align", "clean", "locate", "first_ligand", "protonate", "read_contacts",
                 "refine_and_featurize", "geodesics", "finish"):
        assert step in body, step


@pytest.mark.parametrize("workers", [1, 3])
def test_every_structure_is_processed_once_and_collected_in_order(tmp_path, monkeypatch, workers):
    from qpocket.config import evolve
    from qpocket.core.types import RawStructure, Sources, Structure
    from qpocket.pipelines import run
    from tests.conftest import make_config

    raws = tuple(RawStructure(f"P{i}", tmp_path / f"P{i}.pdb") for i in range(5))
    seen, collected = [], {}
    monkeypatch.setattr(run, "ingest", lambda cfg, ws: Sources(raws[0], raws))
    monkeypatch.setattr(run, "build_reference", lambda cfg, ws, src: "REF")
    monkeypatch.setattr(run, "structure_params", lambda cfg: None)
    monkeypatch.setattr(run, "process",
                        lambda ref, p, ws, raw: seen.append(raw.pdb_id) or Structure(raw.pdb_id, "x"))
    monkeypatch.setattr(run, "collect_run",
                        lambda ws, records, failures, features, compression:
                        collected.update(ids=[r.pdb_id for r in records], failures=failures))
    monkeypatch.setattr(run, "compute_spectral_distance_matrix", lambda cfg, ws: None)
    cfg = evolve(make_config(), "paths.workdir", str(tmp_path))
    cfg = evolve(cfg, "compute.workers", workers)
    cfg = evolve(cfg, "compute.backend", "thread")  # the fakes above are not picklable
    run.run(cfg)
    assert sorted(seen) == [r.pdb_id for r in raws]
    assert collected == {"ids": [r.pdb_id for r in raws], "failures": []}


def test_the_reference_pickles_for_worker_processes():
    """Workers reload the reference from disk; Biopython entities never cross a process."""
    import pickle

    from qpocket.core.types import AlignmentContext, Reference, Structure

    ctx = AlignmentContext("a.pdb", "A", "ACD", None, {}, {}, {}, False, False)
    ref = Reference("R", ctx, Structure("R", "r.pdb"), None, None)
    assert pickle.loads(pickle.dumps(ref)) == ref


def test_reference_residues_are_loaded_once_per_process(tmp_path):
    from qpocket.pipelines import reference

    reference.reference_residues.cache_clear()
    pdb = tmp_path / "ref.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n",
        encoding="utf-8")
    first = reference.reference_residues(str(pdb), "A")
    assert reference.reference_residues(str(pdb), "A") is first
    assert reference.reference_residues.cache_info().hits == 1
