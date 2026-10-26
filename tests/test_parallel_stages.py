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
    for step in ("align", "clean", "find_ligands_step", "protonate", "read_contacts",
                 "detect_cavities"):
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
    monkeypatch.setattr(run, "collect_structures",
                        lambda ws, records, failures:
                        collected.update(ids=[r.pdb_id for r in records], failures=failures))
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
    ref = Reference("R", ctx, Structure("R", "r.pdb"))
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


def test_the_sites_task_writes_what_the_serial_run_writes(tmp_path, monkeypatch):
    """One implementation of the science, two ways of scheduling it: same records in, same tables out."""
    import shutil

    from qpocket.config import evolve
    from qpocket.core.result import Failure
    from qpocket.core.types import RawStructure, Sources
    from qpocket.io import layout
    from qpocket.pipelines import run, tasks
    from tests.conftest import make_config
    from tests.site_helpers import _lig, _structure

    lig = _lig(tmp_path, "P1", "LIG", 1, [(f"C{i}", "C", float(i), 2.0, 0.0) for i in range(6)])
    records = {"P1": _structure(tmp_path, "P1", ligs=[lig]),
               "P2": _structure(tmp_path, "P2", state="UNBOUND_STATE")}
    raws = tuple(RawStructure(p, tmp_path / f"{p}.pdb") for p in ("F1", "P1", "P2"))

    def fake_process(ref, params, ws, raw):
        if raw.pdb_id == "F1":
            raise RuntimeError("boom")
        return records[raw.pdb_id]

    monkeypatch.setattr(run, "ingest", lambda cfg, ws: Sources(raws[0], raws))
    monkeypatch.setattr(run, "build_reference", lambda cfg, ws, src: "REF")
    monkeypatch.setattr(run, "structure_params", lambda cfg: None)
    monkeypatch.setattr(run, "process", fake_process)
    cfg = evolve(make_config(), "paths.workdir", str(tmp_path / "work"))
    cfg = evolve(cfg, "compute.workers", 1)
    cfg = evolve(cfg, "compute.backend", "thread")
    root = run.run(cfg)

    outputs = ("metadata/master.csv", "metadata/failures.csv", "sites/sites.csv",
               "sites/site_01/metadata/master.csv", "sites/site_01/pockets/P2/binding_site.csv")
    serial = {rel: (root / rel).read_text() for rel in outputs}

    ws = layout.workspace(root)
    shutil.rmtree(ws.sites)
    for rel in outputs[:2]:
        (root / rel).unlink()
    for pid in ("F1", "P1", "P2"):
        (ws.original / f"{pid}.pdb").write_text("END\n")
    layout.write_part(ws, "F1", Failure("F1", "structure", repr(RuntimeError("boom"))).to_json())
    for pid, record in records.items():
        layout.write_part(ws, pid, record.to_json())
    tasks.task_sites(cfg, ws)

    assert {rel: (root / rel).read_text() for rel in outputs} == serial


def test_the_site_featurize_task_reuses_the_site_featuriser():
    assert "featurize_pocket" in _function("arms/sites.py", "run_site_task")
    assert "_featurize_pocket" in _function("arms/sites.py", "featurize_pocket")
