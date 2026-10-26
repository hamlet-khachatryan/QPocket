"""The run layout: one Workspace value, a versioned run record."""
import json

from qpocket.io import layout
from qpocket.viz.runs import is_run

FIELDS = {"root", "structures", "original", "aligned", "cleaned", "extracted_ligands",
          "pockets", "metadata", "screening", "decoys", "actives", "sites", "cavities"}


def test_create_makes_every_directory(tmp_path):
    ws = layout.create(str(tmp_path), "1ABC")
    assert set(ws._fields) == FIELDS
    assert all(path.is_dir() for path in ws)
    assert ws.root == tmp_path / "1ABC"


def test_create_writes_the_run_record(tmp_path):
    ws = layout.create(str(tmp_path), "1ABC")
    record = json.loads((ws.metadata / "run.json").read_text())
    assert record["layout_version"] == layout.LAYOUT_VERSION == 3
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


def test_site_workspace_keeps_run_structures(tmp_path):
    ws = layout.create(str(tmp_path), "run")
    site = layout.site_workspace(ws, "site_01")
    assert site.root == ws.root / "sites" / "site_01"
    assert site.original == ws.original and site.cleaned == ws.cleaned
    assert site.cavities == ws.cavities == ws.structures / "cavities"
    assert site.pockets == site.root / "pockets" and site.metadata.is_dir()
    assert not (site.metadata / "run.json").exists()
    assert layout.workspace(site.root) == site
    assert layout.run_root(site) == ws.root == layout.run_root(ws)


def test_layout_version_is_3(tmp_path):
    ws = layout.create(str(tmp_path), "run")
    assert layout.read_run_record(ws.root)["layout_version"] == 3


def test_run_inside_a_directory_named_sites_is_not_a_site(tmp_path):
    ws = layout.create(str(tmp_path / "sites"), "1ABC")
    assert ws.structures == ws.root / "structures" and layout.run_root(ws) == ws.root
