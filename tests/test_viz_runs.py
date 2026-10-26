"""Finding runs, including ones the user points at by hand.

The walk used to visit every file under the search root. These tests pin the two
properties that make it usable on a cluster filesystem -- it never descends into
a run, and it stops at a fixed depth -- alongside the plain question of whether a
typed path is recognised.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from qpocket.io import layout
from qpocket.viz.runs import (
    default_roots,
    find_runs,
    is_complete,
    is_run,
    resolve_location,
    run_label,
)


def make_run(path: Path, complete: bool = True) -> Path:
    (path / "metadata").mkdir(parents=True)
    (path / "pockets").mkdir()
    (path / "metadata" / "master.csv").write_text("pdb_id\n")
    (path / "metadata" / "run.json").write_text(json.dumps({"layout_version": layout.LAYOUT_VERSION}))
    if complete:
        (path / "metadata" / "spectral_distance_matrix.csv").write_text(",\n")
    return path


# --- recognising a run ----------------------------------------------------


def test_a_directory_with_master_and_pockets_is_a_run(tmp_path):
    assert is_run(make_run(tmp_path / "7KG6"))


def test_master_without_pockets_is_not_a_run(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "master.csv").write_text("pdb_id\n")
    assert not is_run(tmp_path)


def test_a_run_without_a_distance_matrix_is_incomplete(tmp_path):
    assert not is_complete(make_run(tmp_path / "r", complete=False))


def test_an_incomplete_run_is_flagged_in_its_label(tmp_path):
    assert "incomplete" in run_label(make_run(tmp_path / "PARG" / "7KG6", complete=False))


# --- walking --------------------------------------------------------------


def test_runs_nested_under_a_project_are_found(tmp_path):
    a = make_run(tmp_path / "PARG" / "7KG6")
    b = make_run(tmp_path / "USP7" / "5N9R")
    assert find_runs([tmp_path]) == sorted([a.resolve(), b.resolve()], key=str)


def test_the_walk_never_enters_a_run(tmp_path):
    """A run's pockets/ can hold thousands of files; none should be visited."""
    run = make_run(tmp_path / "7KG6")
    make_run(run / "pockets" / "decoy")  # would be found if the walk descended
    assert find_runs([tmp_path]) == [run.resolve()]


def test_the_walk_stops_at_the_depth_limit(tmp_path):
    deep = make_run(tmp_path / "a" / "b" / "c" / "d" / "e" / "run")
    assert find_runs([tmp_path], max_depth=3) == []


def test_hidden_directories_are_skipped(tmp_path):
    make_run(tmp_path / ".cache" / "run")
    assert find_runs([tmp_path]) == []


def test_a_missing_root_is_ignored(tmp_path):
    assert find_runs([tmp_path / "absent"]) == []


def test_the_same_run_under_two_roots_appears_once(tmp_path):
    run = make_run(tmp_path / "p" / "run")
    assert find_runs([tmp_path, tmp_path / "p"]) == [run.resolve()]


# --- a typed-in location --------------------------------------------------


def test_typing_a_run_directory_selects_that_run(tmp_path):
    run = make_run(tmp_path / "7KG6")
    assert resolve_location(str(run)).runs == [run.resolve()]


def test_typing_a_project_directory_lists_its_runs(tmp_path):
    make_run(tmp_path / "PARG" / "7KG6")
    make_run(tmp_path / "USP7" / "5N9R")
    assert len(resolve_location(str(tmp_path)).runs) == 2


def test_tilde_and_variables_are_expanded(tmp_path, monkeypatch):
    run = make_run(tmp_path / "7KG6")
    monkeypatch.setenv("MYRUNS", str(tmp_path))
    assert resolve_location("$MYRUNS/7KG6").runs == [run.resolve()]


def test_an_empty_box_is_not_an_error(tmp_path):
    assert resolve_location("   ") == ([], None)


@pytest.mark.parametrize("case", ["missing", "file", "empty_dir"])
def test_a_useless_path_says_why(tmp_path, case):
    target = {"missing": tmp_path / "nope",
              "file": tmp_path / "f.txt",
              "empty_dir": tmp_path}[case]
    if case == "file":
        target.write_text("x")
    result = resolve_location(str(target))
    assert result.runs == [] and result.message


# --- default search roots -------------------------------------------------


def test_the_environment_variable_adds_search_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("QPOCKET_RUNS", f"{tmp_path / 'a'}:{tmp_path / 'b'}")
    roots = default_roots(tmp_path)
    assert tmp_path / "a" in roots and tmp_path / "b" in roots


def test_command_line_roots_are_included(tmp_path):
    assert tmp_path / "cli" in default_roots(tmp_path, extra=[str(tmp_path / "cli")])
