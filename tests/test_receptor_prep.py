"""Receptors: nearby loops rebuilt, protonated by a plug-in, written for docking."""
from pathlib import Path

import numpy as np



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


PARG = Path(__file__).resolve().parents[1] / "experiments/PARG/7KG6/structures/original"


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


