"""Per-structure work is isolated, so parallel results equal serial ones."""
import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src/qpocket"


def _function(path, name):
    tree = ast.parse((SRC / path).read_text(encoding="utf-8"))
    return ast.unparse(next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == name))


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
               "sites/S1-Y1/metadata/master.csv", "sites/S1-Y1/pockets/P2/binding_site.csv")
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


