"""The run layout: one Workspace value, a versioned run record."""
import json

from qpocket.io import layout
from qpocket.viz.runs import is_run

FIELDS = {"root", "structures", "original", "aligned", "cleaned", "extracted_ligands",
          "pockets", "metadata", "screening", "decoys", "actives"}


def test_create_makes_every_directory(tmp_path):
    ws = layout.create(str(tmp_path), "1ABC")
    assert set(ws._fields) == FIELDS
    assert all(path.is_dir() for path in ws)
    assert ws.root == tmp_path / "1ABC"


def test_create_writes_the_run_record(tmp_path):
    ws = layout.create(str(tmp_path), "1ABC")
    record = json.loads((ws.metadata / "run.json").read_text())
    assert record["layout_version"] == layout.LAYOUT_VERSION == 2
    assert "qpocket_version" in record
    assert layout.read_run_record(ws.root) == record


def test_workspace_builds_paths_without_creating(tmp_path):
    ws = layout.workspace(tmp_path / "r")
    assert ws.metadata == tmp_path / "r" / "metadata" and not ws.metadata.exists()


def test_string_keys_still_work_during_the_migration(tmp_path):
    ws = layout.workspace(tmp_path)
    assert ws["metadata"] == ws.metadata and ws[0] == ws.root


def test_is_run_requires_the_current_layout(tmp_path):
    ws = layout.create(str(tmp_path), "r")
    (ws.metadata / "master.csv").write_text("pdb_id\n")
    assert is_run(ws.root)
    (ws.metadata / "run.json").write_text(json.dumps({"layout_version": 1}))
    assert not is_run(ws.root)
    (ws.metadata / "run.json").unlink()
    assert not is_run(ws.root)
