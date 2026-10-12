"""Structure sources: what was ingested, and which one is the reference."""
import pandas as pd
import pytest

from qpocket.core.ports import SOURCES
from qpocket.io import layout
from qpocket.plugins.sources.local import copy_inputs, input_paths

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


def test_structure_files_are_sorted_over_every_suffix(ws):
    for name in ("b.pdb", "a.cif", "c.ent", "notes.txt"):
        _file(ws.original / name)
    assert [p.name for p in layout.structure_files(ws)] == ["a.cif", "b.pdb", "c.ent"]


def test_local_puts_the_reference_first_and_flags_it(ws, tmp_path):
    ref, a = _file(tmp_path / "in/PARG ref.pdb"), _file(tmp_path / "in/soak_A07.pdb")
    sources = SOURCES.get("local").fn(_params(structure_paths=(str(a),), reference_path=str(ref)), ws)
    assert sources.reference.pdb_id == "PARG_ref"
    assert [s.pdb_id for s in sources.structures] == ["PARG_ref", "soak_A07"]
    inputs = pd.read_csv(ws.metadata / "inputs.csv")
    assert inputs.set_index("pdb_id")["is_reference"].to_dict() == {"PARG_ref": True, "soak_A07": False}


def test_without_a_reference_path_the_index_selects_one(ws, tmp_path):
    a, b = _file(tmp_path / "in/a.pdb"), _file(tmp_path / "in/b.pdb")
    sources = SOURCES.get("local").fn(_params(structure_paths=(str(a), str(b)), reference_index=1), ws)
    assert sources.reference.pdb_id == "b"


def test_two_files_with_the_same_name_do_not_collide(ws, tmp_path):
    one, two = _file(tmp_path / "a/xtal.pdb"), _file(tmp_path / "b/xtal.pdb")
    ids = [r["pdb_id"] for r in copy_inputs(_params(structure_paths=(str(one), str(two))), ws)]
    assert ids == ["xtal", "xtal_2"]


def test_missing_and_unknown_files_are_skipped(ws, tmp_path):
    a = _file(tmp_path / "in/a.pdb")
    notes = _file(tmp_path / "in/notes.txt", "x")
    ids = [r["pdb_id"] for r in copy_inputs(
        _params(structure_paths=(str(a), "/nope/b.pdb", str(notes))), ws)]
    assert ids == ["a"]


def test_an_empty_ingest_raises(ws):
    with pytest.raises(ValueError, match="ingested no structures"):
        SOURCES.get("local").fn(_params(structure_paths=("/nope/a.pdb",)), ws)


def test_a_csv_without_the_expected_column_yields_nothing(tmp_path):
    csv = tmp_path / "list.csv"
    pd.DataFrame({"path": ["a.pdb"]}).to_csv(csv, index=False)
    assert input_paths(_params(input_type="csv", csv_path=str(csv))) == []


def test_pdb_query_appends_and_returns_the_reference(ws, monkeypatch):
    from qpocket.io import rcsb

    monkeypatch.setattr(rcsb, "fetch_structure_ids", lambda **kw: ["4B1H"])
    monkeypatch.setattr(rcsb, "download_structures",
                        lambda ids, out: [_file(out / f"{i}.cif") for i in ids])
    params = SOURCES.get("pdb_query").params(
        pdb_id="7KG6", chain_id="A", sequence_identity=0.9, min_length=None, max_length=None,
        resolution_limit=None, experimental_method=None, enforce_same_uniprot=False)
    sources = SOURCES.get("pdb_query").fn(params, ws)
    assert sources.reference.path == ws.original / "7KG6.cif"
    assert [s.pdb_id for s in sources.structures] == ["4B1H", "7KG6"]


def test_pdb_query_with_no_hits_raises(ws, monkeypatch):
    from qpocket.io import rcsb

    monkeypatch.setattr(rcsb, "fetch_structure_ids", lambda **kw: [])
    params = SOURCES.get("pdb_query").params(
        pdb_id="7KG6", chain_id="A", sequence_identity=0.9, min_length=None, max_length=None,
        resolution_limit=None, experimental_method=None, enforce_same_uniprot=False)
    with pytest.raises(ValueError, match="no structures"):
        SOURCES.get("pdb_query").fn(params, ws)


def test_parts_and_reference_round_trip(ws):
    layout.write_part(ws, "7KG6", {"pdb_id": "7KG6"})
    assert layout.read_part(ws, "7KG6") == {"pdb_id": "7KG6"}
    assert layout.read_part(ws, "nope") is None
    with pytest.raises(FileNotFoundError, match="prepare"):
        layout.read_reference(ws)
    layout.write_reference(ws, {"pdb_id": "7KG6"})
    assert layout.read_reference(ws) == {"pdb_id": "7KG6"}
