"""The chain: step semantics that keep today's numbers, and collection."""
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


def _het(serial, name, resname, resseq, xyz, element="C", chain="A"):
    return (f"HETATM{serial:5d} {name:<4s} {resname:>3s} {chain}{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}\n")


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


def _cleaned(tmp_path):
    cleaned = tmp_path / "T_clean.pdb"
    cleaned.write_text("END\n")
    return Structure("T", "t", 0.0, "a.pdb", cleaned_file=str(cleaned), protein_state="BOUND_STATE")


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


def test_run_level_collect_writes_master_and_failures_only(ws):
    lig = Ligand("A", "A", "L", 300, " ", 6, smiles="CC")
    a = Structure("A", "a", 0.0, "a_al.pdb", cleaned_file="a_c.pdb", protein_state="BOUND_STATE",
                  ligands=(lig, lig._replace(resseq=301, smiles="")),
                  contacts=((lig.key, ("R_1_A|X",)), ("A:A:L:301", ("R_2_A|Y", "R_1_A|X"))))
    b = Structure("B", "b", 0.5, "b_al.pdb", cleaned_file="b_c.pdb", protein_state="UNBOUND_STATE")
    collect.collect_structures(ws, [a, b], [Failure("C", "align", "x")])
    master = pd.read_csv(ws.metadata / "master.csv")
    assert list(master["pdb_id"]) == ["A", "B"] and list(master["n_ligands"]) == [2, 0]
    assert list(master.columns) == ["pdb_id", "rmsd_to_ref", "aligned_file", "cleaned_file",
                                    "protonated_file", "cavities_file", "n_ligands", "protein_state"]
    assert pd.read_csv(ws.metadata / "failures.csv").to_dict("records") == [
        {"pdb_id": "C", "stage": "align", "reason": "x"}]
    assert not (ws.screening / "actives.smi").exists()
    assert not list(ws.metadata.glob("*.h5")) and not (ws.metadata / "features.json").exists()


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


