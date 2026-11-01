"""Ingesting a fragment campaign: local files, a reference, and a ligand.

The claim under test is that a structure keeps its identity. The previous scheme
renamed every input to ``fragment_0000`` and recorded the real name only in the
log, so a cluster could not be traced back to the crystal it came from, and the
reference could not be found at all — ``pdb_id == cfg.reference.pdb_id`` never
matched a synthetic ID, so the reference pocket bounding box was silently never
computed and cavity selection lost its ranking.
"""

from __future__ import annotations

import pytest

from qpocket.core.ports import SOURCES
from qpocket.io import layout
from qpocket.plugins.sources.local import copy_inputs
from qpocket.config import (FragmentScreenParams, QPocketConfig)
from qpocket.config import evolve


def structure(path, text="ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def campaign(tmp_path):
    """A reference plus two soaks, and the run directories they land in."""
    src = tmp_path / "src"
    ref = structure(src / "PARG reference.pdb")
    a = structure(src / "soak_A07.pdb")
    b = structure(src / "soak_B11.pdb")

    dirs = layout.create(str(tmp_path), "run")

    cfg = QPocketConfig(mode="fragment_screen")
    cfg = evolve(cfg, "fragment_screen", FragmentScreenParams(
        input_type="paths",
        structure_paths=[str(a), str(b)],
        reference_path=str(ref),
    ))
    return cfg, dirs, {"ref": ref, "a": a, "b": b}


def params(cfg):
    fs = cfg.fragment_screen
    return SOURCES.get("local").params(
        input_type=fs.input_type, csv_path=fs.csv_path,
        structure_paths=tuple(fs.structure_paths or ()), reference_path=fs.reference_path,
        reference_index=fs.reference_index)


def ingest(cfg, dirs):
    return [r["pdb_id"] for r in copy_inputs(params(cfg), dirs)]


# --- identifiers -----------------------------------------------------------


# --- the reference ---------------------------------------------------------


def test_a_reference_also_listed_as_a_structure_is_ingested_once(campaign):
    cfg, dirs, files = campaign
    cfg = evolve(cfg, "fragment_screen.structure_paths", [
        str(files["ref"]), str(files["a"]), str(files["b"])
    ])
    assert ingest(cfg, dirs).count("PARG_reference") == 1


# --- provenance ------------------------------------------------------------


# --- bad input -------------------------------------------------------------


# --- run naming and renumbering -------------------------------------------


# --- validation ------------------------------------------------------------


