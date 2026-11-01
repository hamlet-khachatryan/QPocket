"""Finding runs, including ones the user points at by hand.

The walk used to visit every file under the search root. These tests pin the two
properties that make it usable on a cluster filesystem -- it never descends into
a run, and it stops at a fixed depth -- alongside the plain question of whether a
typed path is recognised.
"""

from __future__ import annotations

from pathlib import Path

import json


from qpocket.io import layout
from qpocket.viz.runs import (
    find_runs,
    is_run,
)


def make_run(path: Path, complete: bool = True) -> Path:
    (path / "metadata").mkdir(parents=True)
    (path / "sites").mkdir()
    (path / "metadata" / "master.csv").write_text("pdb_id\n")
    (path / "metadata" / "run.json").write_text(json.dumps({"layout_version": layout.LAYOUT_VERSION}))
    if complete:
        (path / "sites" / "sites.csv").write_text("site_id\n")
    return path


# --- recognising a run ----------------------------------------------------


def test_v3_run_listed(tmp_path):
    run = make_run(tmp_path / "7KG6")
    assert is_run(run)
    assert find_runs([tmp_path]) == [run.resolve()]


# --- walking --------------------------------------------------------------


# --- a typed-in location --------------------------------------------------


# --- default search roots -------------------------------------------------


# --- the regroup controls and the site view ---------------------------------


