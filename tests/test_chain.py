"""The chain: step semantics that keep today's numbers, and collection."""
import json

import numpy as np
import pandas as pd
import pytest

from qpocket.core.result import Failure
from qpocket.core.types import AlignmentContext, Reference, Structure
from qpocket.io import layout
from qpocket.pipelines import collect, structure

CTX = AlignmentContext("/r/ref.pdb", "A", "M", None, {}, {}, {}, False, False)


@pytest.fixture
def ws(tmp_path):
    return layout.create(str(tmp_path), "run")


def _params(**kw):
    from tests.conftest import make_config

    return structure.structure_params(make_config())._replace(**kw) if kw else \
        structure.structure_params(make_config())


def test_params_are_frozen_and_built_from_config():
    p = _params()
    assert p.pocket == "cavities" and p.charges == "pdb2pqr" and p.contacts == ("prolif",)


def test_the_reference_is_not_realigned(ws):
    record = Structure("REF", "ref.cif", 0.0, "a.pdb", cleaned_file="c.pdb")
    ref = Reference("REF", CTX, record, None, None)
    from qpocket.core.types import RawStructure

    assert structure.align(ref, _params(), ws, RawStructure("REF", ws.original / "REF.cif")) is record


def test_a_failed_cleaning_is_a_failure(ws, monkeypatch):
    monkeypatch.setattr(structure, "clean_file", lambda record, ws, allowed: None)
    out = structure.clean(_params(), ws, Structure("T", "t.pdb", 0.1, "t_aligned.pdb"))
    assert out == Failure("T", "clean", "cleaning failed")


def test_a_pocket_state_is_kept_and_skips_the_ligand(ws, monkeypatch, tmp_path):
    cleaned = tmp_path / "t_clean.pdb"
    cleaned.write_text("END\n")
    rec = Structure("T", "t.pdb", 0.1, "t_aligned.pdb", cleaned_file=str(cleaned))
    ref = Reference("REF", CTX, Structure("REF", "r"), None, None)
    monkeypatch.setattr(structure, "_detect", lambda p, rec, ref: (None, "CAVITY_CLOSED"))
    located = structure.locate(ref, _params(), ws, rec)
    assert located.protein_state == "CAVITY_CLOSED" and located.points is None
    assert structure.first_ligand(ref, _params(), ws, located) is located


def test_without_a_ligand_id_the_reference_branch_is_reference(ws):
    ref = Reference("REF", CTX, Structure("REF", "r"), np.ones(3), (np.zeros(3), np.ones(3)))
    rec = Structure("REF", "r", 0.0, "a.pdb", cleaned_file="c.pdb", points=np.zeros((3, 3)))
    out = structure.first_ligand(ref, _params(reference_ligand_id=None), ws, rec)
    assert out.protein_state == "REFERENCE" and out.ligand_file is None


def test_contacts_need_a_bound_ligand(ws):
    rec = Structure("T", "t", protein_state="UNBOUND_STATE", cleaned_file="/nope.pdb")
    out = structure.read_contacts(_params(), ws, rec)
    assert out.observed == ("no_ligand_found",) and out.potential == ()


def test_a_failed_re_pick_keeps_the_first_ligand(ws, monkeypatch):
    rec = Structure("T", "t", 0.1, "/nope_aligned.pdb", cleaned_file="/nope.pdb",
                    points=np.zeros((4, 3)), protein_state="BOUND_STATE", ligand_file="l.pdb",
                    ligand_id="LIG", observed=("TYR_1_A|HBDonor",))
    monkeypatch.setattr(structure, "featurize", lambda *a, **k: None)
    (ws.pockets / "T").mkdir(parents=True)  # locate made it
    out = structure.refine_and_featurize(Reference("REF", CTX, Structure("REF", "r"), None, None),
                                         _params(pocket_definition="Interactions"), ws, rec)
    assert (out.protein_state, out.ligand_id) == ("BOUND_STATE", "LIG")


def test_collect_writes_master_actives_failures_and_archives(ws):
    a = Structure("A", "a", 0.0, "a_al.pdb", cleaned_file="a_c.pdb", protein_state="BOUND_STATE",
                  ligand_file="A_L.pdb", ligand_id="L", potential=("R_1_A",), observed=("R_1_A|X",),
                  smiles=("CC", "L"))
    b = Structure("B", "b", 0.5, "b_al.pdb", cleaned_file="b_c.pdb", protein_state="UNBOUND_STATE",
                  observed=("no_ligand_found",))
    collect.collect_run(ws, [a, b], [Failure("C", "align", "x")], ["hbond"], "gzip")
    master = pd.read_csv(ws.metadata / "master.csv")
    assert list(master["pdb_id"]) == ["A", "B"]
    assert list(master.columns) == ["pdb_id", "rmsd_to_ref", "aligned_file", "cleaned_file",
                                    "ligand_file", "ligand_id", "protein_state"]
    assert (ws.screening / "actives.smi").read_text() == "SMILES\tPDB_ID\tLigand_Name\nCC\tA\tL\n"
    assert pd.read_csv(ws.metadata / "failures.csv").to_dict("records") == [
        {"pdb_id": "C", "stage": "align", "reason": "x"}]
    assert (ws.metadata / "observed_interactions.h5").exists()
    assert json.loads((ws.metadata / "features.json").read_text())["enabled"] == ["hbond"]


def test_collect_from_parts_keeps_order_and_reports_gaps(ws):
    layout.write_part(ws, "B", Structure("B", "b", 0.1, "b", cleaned_file="b",
                                         protein_state="UNBOUND_STATE").to_json())
    layout.write_part(ws, "A", Failure("A", "align", "x").to_json())
    (layout.parts_dir(ws) / "D.json").write_text("{not json")
    records, failures = collect.collect_from_parts(ws, ["A", "B", "C", "D"], ["hbond"], "gzip")
    assert [r.pdb_id for r in records] == ["B"]
    assert [(f.pdb_id, f.stage) for f in failures] == [("A", "align"), ("C", "featurize"),
                                                      ("D", "featurize")]


STEPS = ("clean", "locate", "first_ligand", "protonate", "read_contacts",
         "refine_and_featurize", "geodesics", "finish")


def _counting_chain(monkeypatch, result):
    """Replace every step: align returns `result` and counts; the rest pass it through."""
    calls = []
    monkeypatch.setattr(structure, "align", lambda ref, p, ws, raw: calls.append(raw.pdb_id) or result)
    for name in STEPS:
        monkeypatch.setattr(structure, name, lambda *args: args[-1])
    return calls


def _inputs(tmp_path):
    from qpocket.core.types import RawStructure

    raw_file = tmp_path / "T.pdb"
    raw_file.write_text("ATOM\n")
    return RawStructure("T", raw_file), Reference("REF", CTX, Structure("REF", "r"), None, None)


def test_a_finished_structure_is_not_processed_again(ws, monkeypatch, tmp_path):
    raw, ref = _inputs(tmp_path)
    calls = _counting_chain(monkeypatch, Structure("T", "t", protein_state="UNBOUND_STATE"))
    first = structure.process(ref, _params(), ws, raw)
    again = structure.process(ref, _params(), ws, raw)
    assert calls == ["T"] and again == first


def test_a_changed_input_or_setting_is_processed_again(ws, monkeypatch, tmp_path):
    raw, ref = _inputs(tmp_path)
    calls = _counting_chain(monkeypatch, Structure("T", "t", protein_state="UNBOUND_STATE"))
    structure.process(ref, _params(), ws, raw)
    structure.process(ref, _params(interaction_padding=9.0), ws, raw)
    raw.path.write_text("ATOM changed\n")
    structure.process(ref, _params(interaction_padding=9.0), ws, raw)
    assert calls == ["T", "T", "T"]


def test_a_failed_structure_is_retried(ws, monkeypatch, tmp_path):
    raw, ref = _inputs(tmp_path)
    calls = _counting_chain(monkeypatch, Failure("T", "align", "x"))
    structure.process(ref, _params(), ws, raw)
    structure.process(ref, _params(), ws, raw)
    assert calls == ["T", "T"]
