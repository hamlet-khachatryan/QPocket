"""The chain: step semantics that keep today's numbers, and collection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpocket.core.result import Failure
from qpocket.core.types import AlignmentContext, Ligand, Reference, Structure
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
    assert p.contacts == ("prolif",) and p.min_heavy_atoms == 6


def test_the_reference_is_not_realigned(ws):
    record = Structure("REF", "ref.cif", 0.0, "a.pdb", cleaned_file="c.pdb")
    ref = Reference("REF", CTX, record)
    from qpocket.core.types import RawStructure

    assert structure.align(ref, _params(), ws, RawStructure("REF", ws.original / "REF.cif")) is record


def test_a_failed_cleaning_is_a_failure(ws, monkeypatch):
    monkeypatch.setattr(structure, "clean_file", lambda record, ws, allowed: None)
    out = structure.clean(_params(), ws, Structure("T", "t.pdb", 0.1, "t_aligned.pdb"))
    assert out == Failure("T", "clean", "cleaning failed")


def _het(serial, name, resname, resseq, xyz, element="C", chain="A"):
    return (f"HETATM{serial:5d} {name:<4s} {resname:>3s} {chain}{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}\n")


def test_every_ligand_is_extracted_with_its_smiles(ws, monkeypatch, tmp_path):
    from qpocket.io import interactions

    aligned = tmp_path / "T_aligned.pdb"
    aligned.write_text("".join(_het(i + 1, f"C{i}", "LIG", 300, (i, 0, 0)) for i in range(6))
                       + "".join(_het(i + 7, f"C{i}", "LIG", 301, (i, 20, 0)) for i in range(6))
                       + _het(13, "ZN", "ZN", 400, (0, 0, 30), "ZN") + "END\n")
    monkeypatch.setattr(interactions, "ligand_smiles", lambda path: "CCCCCC")
    out = structure.find_ligands_step(_params(), ws, Structure("T", "t", 0.0, str(aligned)))
    assert [(l.resname, l.resseq) for l in out.ligands] == [("LIG", 300), ("LIG", 301)]
    assert out.protein_state == "BOUND_STATE" and all(l.smiles == "CCCCCC" for l in out.ligands)
    assert [Path(l.file).name for l in out.ligands] == ["T_A_LIG_300.pdb", "T_A_LIG_301.pdb"]
    assert all(Path(l.file).exists() for l in out.ligands)


def test_a_structure_without_ligands_is_unbound(ws, tmp_path):
    aligned = tmp_path / "T_aligned.pdb"
    aligned.write_text(_het(1, "ZN", "ZN", 400, (0, 0, 0), "ZN") + "END\n")
    out = structure.find_ligands_step(_params(), ws, Structure("T", "t", 0.0, str(aligned)))
    assert out.ligands == () and out.protein_state == "UNBOUND_STATE"


def test_contacts_are_read_per_ligand_by_its_residue_number(ws, monkeypatch):
    from qpocket.plugins.contacts import prolif

    monkeypatch.setattr(prolif, "interactions_for",
                        lambda path, lig, enabled, parameters, ligand_resseq: [f"R_{ligand_resseq}_A|X"])
    a, b = Ligand("T", "A", "LIG", 300, " ", 6), Ligand("T", "B", "LIG", 300, " ", 6)
    rec = Structure("T", "t", protonated_file="/p/T.pdb", ligands=(a, b))
    out = structure.read_contacts(_params(), ws, rec)
    assert out.contacts == ((a.key, ("R_900_A|X",)), (b.key, ("R_901_A|X",)))


def test_without_a_protonated_complex_no_contacts_are_guessed(ws, monkeypatch):
    """The aligned file cannot tell BTN A301 from BTN B301 by name and number alone."""
    from qpocket.plugins.contacts import prolif

    monkeypatch.setattr(prolif, "interactions_for", lambda *a, **k: ["R_1_A|X"])
    a, b = Ligand("T", "A", "BTN", 301, " ", 16), Ligand("T", "B", "BTN", 301, " ", 16)
    out = structure.read_contacts(_params(), ws, Structure("T", "t", aligned_file="/a.pdb", ligands=(a, b)))
    assert out.contacts == ((a.key, ("no_ligand_found",)), (b.key, ("no_ligand_found",)))


def _fake_protonation(monkeypatch, written):
    from qpocket.core import hydroxyl
    from qpocket.io import interactions, tools

    def write_complex(path, pqr, ligands):
        written.append([name for _, name in ligands])
        Path(path).write_text("END\n")
        return Path(path)

    monkeypatch.setattr(tools, "protonate_for_interactions", lambda pdb, pqr: pqr)
    monkeypatch.setattr(interactions, "write_complex", write_complex)
    monkeypatch.setattr(interactions, "load_molecule", lambda path: None)
    monkeypatch.setattr(interactions, "write_molecule", lambda path, mol: path)
    monkeypatch.setattr(hydroxyl, "optimise_hydroxyls", lambda mol: mol)
    monkeypatch.setattr(structure, "_protonated_ligand", lambda path: "mol")


def test_a_complex_is_rebuilt_when_the_ligand_list_changes(ws, monkeypatch, tmp_path):
    written = []
    _fake_protonation(monkeypatch, written)
    rec = _cleaned(tmp_path)
    one = (Ligand("T", "A", "AAA", 1, " ", 6),)
    two = one + (Ligand("T", "A", "BBB", 2, " ", 6),)
    structure.protonate(ws, rec._replace(ligands=one))
    structure.protonate(ws, rec._replace(ligands=one))       # same list: reused
    out = structure.protonate(ws, rec._replace(ligands=two))  # new list: rebuilt
    assert written == [["AAA"], ["AAA", "BBB"]] and out.protonated_file


def test_per_site_settings_do_not_change_a_structure_fingerprint(tmp_path):
    from qpocket.config import evolve
    from qpocket.core.types import RawStructure
    from tests.conftest import make_config

    raw = tmp_path / "T.pdb"
    raw.write_text("ATOM\n")
    ref = Reference("REF", CTX, Structure("REF", "r"))
    base = make_config()
    other = evolve(evolve(base, "features.enabled", ["hbond"]), "distance.geo_graph_neighbor_radius", 9.5)
    key = lambda cfg: structure.fingerprint(ref, structure.structure_params(cfg), RawStructure("T", raw))
    assert key(base) == key(other)


def test_contacts_need_a_ligand(ws):
    rec = Structure("T", "t", protein_state="UNBOUND_STATE", cleaned_file="/nope.pdb")
    assert structure.read_contacts(_params(), ws, rec).contacts == ()


def _cleaned(tmp_path):
    cleaned = tmp_path / "T_clean.pdb"
    cleaned.write_text("END\n")
    return Structure("T", "t", 0.0, "a.pdb", cleaned_file=str(cleaned), protein_state="BOUND_STATE")


def test_every_cavity_is_written_with_its_detection_index(ws, monkeypatch, tmp_path):
    found = [np.zeros((2, 3)), np.ones((3, 3))]
    monkeypatch.setattr(structure, "run_cavfind", lambda path, name: found)
    out = structure.detect_cavities(ws, _cleaned(tmp_path))
    assert out.cavities_file == str(ws.cavities / "T.csv") and out.protein_state == "BOUND_STATE"
    cav = pd.read_csv(out.cavities_file)
    assert list(cav.columns) == ["x", "y", "z", "cavity"] and list(cav["cavity"]) == [0, 0, 1, 1, 1]


def test_no_cavity_or_a_failed_detection_is_a_state(ws, monkeypatch, tmp_path):
    monkeypatch.setattr(structure, "run_cavfind", lambda path, name: [])
    out = structure.detect_cavities(ws, _cleaned(tmp_path))
    assert (out.protein_state, out.cavities_file) == ("NO_ANY_CAVITY_FOUND", None)

    def boom(path, name):
        raise RuntimeError("x")

    monkeypatch.setattr(structure, "run_cavfind", boom)
    assert structure.detect_cavities(ws, _cleaned(tmp_path)).protein_state == "FAILED"


FIXTURES = Path(__file__).resolve().parent / "fixtures/complexes"


@pytest.mark.skipif(not (FIXTURES / "1STP.pdb").exists(), reason="run scripts/make_complex_fixtures.py first")
def test_chain_records_all_ligands_and_cavities(ws, monkeypatch, tmp_path):
    from qpocket.core.types import RawStructure

    aligned = tmp_path / "1STP_aligned.pdb"
    aligned.write_text((FIXTURES / "1STP.pdb").read_text())
    ref = Reference("1STP", CTX, Structure("1STP", str(aligned), 0.0, str(aligned)))
    monkeypatch.setattr(structure, "run_cavfind", lambda path, name: [np.zeros((4, 3)) + 9.0])
    rec = structure.process(ref, _params(), ws, RawStructure("1STP", aligned))
    assert [l.resname for l in rec.ligands] == ["BTN"]
    assert Path(rec.cavities_file).exists()
    cav = pd.read_csv(rec.cavities_file)
    assert list(cav.columns) == ["x", "y", "z", "cavity"] and len(cav) > 0
    assert dict(rec.contacts)[rec.ligands[0].key]           # non-empty contacts
    assert not (ws.pockets / "1STP").exists()               # no per-structure pocket any more
    assert Structure.from_json(layout.read_part(ws, "1STP")) == rec


def test_collect_writes_master_actives_failures_and_archives(ws):
    lig = Ligand("A", "A", "L", 300, " ", 6, smiles="CC")
    a = Structure("A", "a", 0.0, "a_al.pdb", cleaned_file="a_c.pdb", protein_state="BOUND_STATE",
                  ligands=(lig, lig._replace(resseq=301, smiles="")),
                  contacts=((lig.key, ("R_1_A|X",)), ("A:A:L:301", ("R_2_A|Y", "R_1_A|X"))))
    b = Structure("B", "b", 0.5, "b_al.pdb", cleaned_file="b_c.pdb", protein_state="UNBOUND_STATE")
    collect.collect_run(ws, [a, b], [Failure("C", "align", "x")], ["hbond"], "gzip")
    master = pd.read_csv(ws.metadata / "master.csv")
    assert list(master["pdb_id"]) == ["A", "B"] and list(master["n_ligands"]) == [2, 0]
    assert list(master.columns) == ["pdb_id", "rmsd_to_ref", "aligned_file", "cleaned_file",
                                    "protonated_file", "cavities_file", "n_ligands", "protein_state"]
    assert (ws.screening / "actives.smi").read_text() == "SMILES\tPDB_ID\tLigand_Name\nCC\tA\tL\n"
    assert pd.read_csv(ws.metadata / "failures.csv").to_dict("records") == [
        {"pdb_id": "C", "stage": "align", "reason": "x"}]
    from qpocket.io import hdf5

    observed = hdf5.read_interactions(ws.metadata / "observed_interactions.h5")
    assert sorted(observed["A"]) == ["R_1_A|X", "R_2_A|Y"] and list(observed["B"]) == ["no_ligand_found"]
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


STEPS = ("clean", "find_ligands_step", "protonate", "read_contacts", "detect_cavities")


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
    return RawStructure("T", raw_file), Reference("REF", CTX, Structure("REF", "r"))


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
    structure.process(ref, _params(min_heavy_atoms=9), ws, raw)
    raw.path.write_text("ATOM changed\n")
    structure.process(ref, _params(min_heavy_atoms=9), ws, raw)
    assert calls == ["T", "T", "T"]


def test_a_failed_structure_is_retried(ws, monkeypatch, tmp_path):
    raw, ref = _inputs(tmp_path)
    calls = _counting_chain(monkeypatch, Failure("T", "align", "x"))
    structure.process(ref, _params(), ws, raw)
    structure.process(ref, _params(), ws, raw)
    assert calls == ["T", "T"]
