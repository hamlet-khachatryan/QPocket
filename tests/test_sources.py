"""Structure sources: what was ingested, and which one is the reference."""
import pandas as pd
import pytest

from qpocket.core.ports import SOURCES
from qpocket.io import layout
from qpocket.plugins.sources.local import copy_inputs

ATOM = "ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"


def _file(path, text=ATOM):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def ws(tmp_path):
    return layout.create(str(tmp_path), "run")


def _params(**kw):
    base = dict(input_type="paths", csv_path=None, structure_paths=(), reference_path=None,
                reference_index=0)
    base.update(kw)
    return SOURCES.get("local").params(**base)


def test_local_puts_the_reference_first_and_flags_it(ws, tmp_path):
    ref, a = _file(tmp_path / "in/PARG ref.pdb"), _file(tmp_path / "in/soak_A07.pdb")
    sources = SOURCES.get("local").fn(_params(structure_paths=(str(a),), reference_path=str(ref)), ws)
    assert sources.reference.pdb_id == "PARG_ref"
    assert [s.pdb_id for s in sources.structures] == ["PARG_ref", "soak_A07"]
    inputs = pd.read_csv(ws.metadata / "inputs.csv")
    assert inputs.set_index("pdb_id")["is_reference"].to_dict() == {"PARG_ref": True, "soak_A07": False}


def test_two_files_with_the_same_name_do_not_collide(ws, tmp_path):
    one, two = _file(tmp_path / "a/xtal.pdb"), _file(tmp_path / "b/xtal.pdb")
    ids = [r["pdb_id"] for r in copy_inputs(_params(structure_paths=(str(one), str(two))), ws)]
    assert ids == ["xtal", "xtal_2"]


