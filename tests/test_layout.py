"""The run layout: one Workspace value, a versioned run record."""
import json

from qpocket.io import layout
from qpocket.viz.runs import is_run

FIELDS = {"root", "structures", "original", "aligned", "cleaned", "extracted_ligands",
          "pockets", "metadata", "screening", "decoys", "actives", "sites", "cavities"}


def test_is_run_requires_the_current_layout(tmp_path):
    ws = layout.create(str(tmp_path), "r")
    (ws.metadata / "master.csv").write_text("pdb_id\n")
    assert is_run(ws.root)
    (ws.metadata / "run.json").write_text(json.dumps({"layout_version": 1}))
    assert not is_run(ws.root)
    (ws.metadata / "run.json").unlink()
    assert not is_run(ws.root)


