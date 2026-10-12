"""Pocket detectors, charge models and contact detectors as plug-ins."""
from pathlib import Path

import numpy as np

from qpocket.core.ports import CHARGES, CONTACTS, POCKETS
from qpocket.core.types import AlignmentContext, Reference, Structure

CTX = AlignmentContext("/r/ref.pdb", "A", "M", None, {}, {}, {}, False, False)


def _ref(box=None, centroid=None):
    return Reference("REF", CTX, Structure("REF", "ref.pdb"), centroid, box)


def _rec(pdb_id="T"):
    return Structure(pdb_id, f"{pdb_id}.pdb", cleaned_file=f"/r/{pdb_id}_clean.pdb")


def _cavities(monkeypatch, found):
    from qpocket.plugins.pockets import cavities

    monkeypatch.setattr(cavities, "run_cavfind", lambda path, name: found)
    return POCKETS.get("cavities")


def test_no_cavity_at_all(monkeypatch):
    plugin = _cavities(monkeypatch, [])
    assert plugin.fn(_rec(), _ref(), plugin.params()) == (None, "NO_ANY_CAVITY_FOUND")


def test_detection_error_is_failed(monkeypatch):
    from qpocket.plugins.pockets import cavities

    def boom(path, name):
        raise RuntimeError("x")

    monkeypatch.setattr(cavities, "run_cavfind", boom)
    plugin = POCKETS.get("cavities")
    assert plugin.fn(_rec(), _ref(), plugin.params()) == (None, "FAILED")


def test_a_target_takes_the_cavity_overlapping_the_reference_box(monkeypatch):
    near, far = np.zeros((5, 3)) + 1.0, np.zeros((5, 3)) + 50.0
    plugin = _cavities(monkeypatch, [far, near])
    points, state = plugin.fn(_rec(), _ref(box=(np.zeros(3), np.full(3, 2.0))), plugin.params())
    assert state is None and np.array_equal(points, near)


def test_no_overlap_is_a_closed_cavity(monkeypatch):
    plugin = _cavities(monkeypatch, [np.zeros((5, 3)) + 50.0])
    assert plugin.fn(_rec(), _ref(box=(np.zeros(3), np.ones(3))), plugin.params()) == (None, "CAVITY_CLOSED")


def test_the_reference_takes_the_cavity_at_its_ligand(monkeypatch):
    near = np.zeros((5, 3)) + 1.0
    plugin = _cavities(monkeypatch, [np.zeros((5, 3)) + 50.0, near])
    points, _ = plugin.fn(_rec("REF"), _ref(box=(np.zeros(3), np.ones(3)), centroid=np.ones(3)),
                          plugin.params())
    assert np.array_equal(points, near)


def test_a_box_without_dimensions_fails():
    plugin = POCKETS.get("box")
    assert plugin.fn(_rec(), _ref(), plugin.params(center=None, dimensions=None)) == (None, "FAILED")


def test_a_box_fills_the_given_region():
    plugin = POCKETS.get("box")
    points, state = plugin.fn(_rec(), _ref(), plugin.params(center=(0.0, 0.0, 0.0),
                                                            dimensions=(4.0, 4.0, 4.0)))
    assert state is None and len(points) > 0


def test_pdb2pqr_declares_its_tool():
    assert CHARGES.get("pdb2pqr").tools == ("pdb2pqr",)


def test_prolif_passes_its_settings_through(monkeypatch):
    from qpocket.plugins.contacts import prolif

    seen = {}
    monkeypatch.setattr(prolif, "interactions_for",
                        lambda path, lig, enabled, parameters: seen.update(
                            path=path, lig=lig, enabled=enabled, parameters=parameters) or ["X|Y"])
    plugin = CONTACTS.get("prolif")
    out = plugin.fn(Path("c.pdb"), "WDG", plugin.params(enabled=("HBDonor",), parameters={"a": 1}))
    assert out == ["X|Y"] and seen == {"path": Path("c.pdb"), "lig": "WDG",
                                       "enabled": ["HBDonor"], "parameters": {"a": 1}}
