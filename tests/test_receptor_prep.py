"""Receptors: nearby loops rebuilt, protonated by a plug-in, written for docking."""
from pathlib import Path

import numpy as np
import pytest

from qpocket.core.ports import RECEPTORS
from qpocket.core.receptor import gaps_to_build


def test_only_gaps_flanked_near_the_pocket_are_built():
    pocket = np.zeros((3, 3))
    flanks = {"near": np.array([[3.0, 0, 0], [40.0, 0, 0]]), "far": np.array([[30.0, 0, 0], [31.0, 0, 0]])}
    assert gaps_to_build(flanks, pocket, 10.0) == {"near"}
    assert gaps_to_build(flanks, pocket, 0.0) == set()
    assert gaps_to_build({}, pocket, 10.0) == set()


def test_pdb2pqr_is_the_default_receptor_plugin():
    plugin = RECEPTORS.get("pdb2pqr")
    assert plugin.tools == ("pdb2pqr",)
    assert (plugin.params().ph, plugin.params().forcefield) == (7.4, "AMBER")


def test_the_plugin_passes_ph_and_forcefield(monkeypatch, tmp_path):
    from qpocket.plugins.receptors import pdb2pqr

    seen = {}
    monkeypatch.setattr(pdb2pqr, "protonate_receptor",
                        lambda pdb, out, ph, forcefield: seen.update(ph=ph, ff=forcefield) or out)
    plugin = RECEPTORS.get("pdb2pqr")
    plugin.fn(tmp_path / "r.pdb", tmp_path / "r.pqr", plugin.params(ph=6.5, forcefield="PARSE"))
    assert seen == {"ph": 6.5, "ff": "PARSE"}


def test_prepare_receptor_chains_repair_protonation_and_pdbqt(monkeypatch, tmp_path):
    from qpocket.arms.screen import receptors
    from qpocket.screen_config import ScreenConfig

    calls = []
    monkeypatch.setattr(receptors, "repair", lambda src, dst, pocket, radius, sequences=None: calls.append(("repair", radius))
                        or Path(dst).write_text("ATOM\n") or {"loops_built": 0})
    monkeypatch.setattr(receptors, "_protonate",
                        lambda pdb, pqr, cfg: calls.append(("protonate", cfg.receptor.use)) or Path(pqr).write_text("x") or Path(pqr))
    monkeypatch.setattr(receptors, "receptor_from_pqr",
                        lambda pqr, out: calls.append(("pdbqt",)) or Path(out).write_text("x") or True)
    cleaned = tmp_path / "A_clean.pdb"
    cleaned.write_text("ATOM\n")
    out = receptors.prepare_receptor(cleaned, np.zeros((2, 3)), tmp_path / "receptors", "A", ScreenConfig())
    assert out == tmp_path / "receptors" / "A.pdbqt"
    assert calls == [("repair", 10.0), ("protonate", "pdb2pqr"), ("pdbqt",)]


def test_repair_can_be_switched_off(monkeypatch, tmp_path):
    from qpocket.arms.screen import receptors
    from qpocket.screen_config import ScreenConfig

    monkeypatch.setattr(receptors, "repair", lambda *a: pytest.fail("repair ran"))
    monkeypatch.setattr(receptors, "_protonate", lambda pdb, pqr, cfg: Path(pqr).write_text("x") or Path(pqr))
    monkeypatch.setattr(receptors, "receptor_from_pqr", lambda pqr, out: Path(out).write_text("x") or True)
    cfg = ScreenConfig()
    cfg.receptor.repair = False
    cleaned = tmp_path / "A_clean.pdb"
    cleaned.write_text("ATOM\n")
    assert receptors.prepare_receptor(cleaned, np.zeros((2, 3)), tmp_path / "r", "A", cfg) is not None
    assert (tmp_path / "r" / "A.pdb").read_text() == "ATOM\n"


def test_an_unknown_receptor_preparation_is_a_config_error():
    from qpocket.screen_config import ScreenConfig, validate_screen_config

    cfg = ScreenConfig()
    cfg.receptor.use = "nope"
    with pytest.raises(ValueError, match="receptor.use"):
        validate_screen_config(cfg)


def test_pdbfixer_adds_missing_side_chain_atoms_without_moving_others(tmp_path):
    pytest.importorskip("pdbfixer")
    from qpocket.io.repair import repair

    # Two-residue peptide; the lysine side chain is truncated after CB.
    pdb = tmp_path / "in.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1      -0.966   0.493   1.500  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       0.257   0.418   0.692  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1      -0.094   0.017  -0.716  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1      -1.056  -0.682  -0.923  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       1.204  -0.620   1.296  1.00  0.00           C\n"
        "ATOM      6  N   LYS A   2       0.661   0.484  -1.693  1.00  0.00           N\n"
        "ATOM      7  CA  LYS A   2       0.433   0.168  -3.105  1.00  0.00           C\n"
        "ATOM      8  C   LYS A   2       1.732   0.322  -3.904  1.00  0.00           C\n"
        "ATOM      9  O   LYS A   2       2.722   0.880  -3.448  1.00  0.00           O\n"
        "ATOM     10  CB  LYS A   2      -0.637   1.081  -3.703  1.00  0.00           C\n"
        "END\n")
    stats = repair(pdb, tmp_path / "out.pdb", np.zeros((1, 3)), 10.0)
    text = (tmp_path / "out.pdb").read_text()
    assert " NZ  LYS" in text and stats["atoms_added"] >= 3
    assert "  0.257   0.418   0.692" in text  # CA of residue 1 unmoved


PARG = Path(__file__).resolve().parents[1] / "experiments/PARG/7KG6/structures/original"


@pytest.mark.skipif(not PARG.exists(), reason="needs the bundled PARG structures")
def test_chain_sequences_come_from_the_original_entry():
    pytest.importorskip("gemmi")
    from qpocket.io.repair import chain_sequences

    seqs = chain_sequences(PARG / "4B1H.cif")
    assert len(seqs["A"]) == 531 and all(len(name) == 3 for name in seqs["A"])


def _single_chain_a(tmp_path):
    """4B1H chain A with one conformer per atom, as cleaning leaves it, written as PDB."""
    import gemmi

    st = gemmi.read_structure(str(PARG / "4B1H.cif"))
    st.remove_alternative_conformations()
    st.remove_ligands_and_waters()
    for name in [ch.name for ch in st[0] if ch.name != "A"]:
        st[0].remove_chain(name)
    st.setup_entities()
    path = tmp_path / "4B1H_A.pdb"
    st.write_pdb(str(path))
    return path, st


def test_a_missing_loop_near_the_pocket_is_built_and_nothing_moves(tmp_path):
    pytest.importorskip("pdbfixer")
    from biopandas.pdb import PandasPdb

    from qpocket.io.repair import chain_sequences, repair

    pdb, st = _single_chain_a(tmp_path)
    # Pocket right at the gap (residues 524-529 are unmodelled in 4B1H).
    flank = [r for r in st[0]["A"] if r.seqid.num in (523, 530)]
    pocket = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in flank for a in r])
    seqs = chain_sequences(PARG / "4B1H.cif")

    far = repair(pdb, tmp_path / "far.pdb", pocket + 1000.0, 10.0, sequences=seqs)
    near = repair(pdb, tmp_path / "near.pdb", pocket, 10.0, sequences=seqs)
    assert far["loops_built"] == 0 and far["loops_skipped"] >= 1
    assert near["loops_built"] >= 1

    key = ["chain_id", "residue_number", "residue_name", "atom_name"]
    before = PandasPdb().read_pdb(str(pdb)).df["ATOM"]
    after = PandasPdb().read_pdb(str(tmp_path / "near.pdb")).df["ATOM"]
    m = before.merge(after, on=key, suffixes=("_in", "_out"))
    shift = np.abs(m[["x_coord_in", "y_coord_in", "z_coord_in"]].to_numpy()
                   - m[["x_coord_out", "y_coord_out", "z_coord_out"]].to_numpy()).max()
    assert len(m) == len(before) and shift < 1e-3
    assert set(range(524, 530)) <= set(after["residue_number"])
