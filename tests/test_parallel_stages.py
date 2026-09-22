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
