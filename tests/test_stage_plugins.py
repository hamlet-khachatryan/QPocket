"""Charge models and contact detectors as plug-ins."""
from pathlib import Path

from qpocket.core.ports import CHARGES, CONTACTS


def test_pdb2pqr_declares_its_tool():
    assert CHARGES.get("pdb2pqr").tools == ("pdb2pqr",)


def test_prolif_passes_its_settings_through(monkeypatch):
    from qpocket.plugins.contacts import prolif

    seen = {}
    monkeypatch.setattr(prolif, "interactions_for",
                        lambda path, lig, enabled, parameters, ligand_resseq: seen.update(
                            path=path, lig=lig, enabled=enabled, parameters=parameters,
                            resseq=ligand_resseq) or ["X|Y"])
    plugin = CONTACTS.get("prolif")
    out = plugin.fn(Path("c.pdb"), "WDG", plugin.params(enabled=("HBDonor",), parameters={"a": 1}),
                    ligand_resseq=901)
    assert out == ["X|Y"] and seen == {"path": Path("c.pdb"), "lig": "WDG",
                                       "enabled": ["HBDonor"], "parameters": {"a": 1}, "resseq": 901}
