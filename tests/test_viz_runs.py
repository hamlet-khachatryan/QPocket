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
    site_parameters,
    site_view_ligands,
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


def test_layout2_run_not_listed(tmp_path):
    run = make_run(tmp_path / "7KG6")
    (run / "metadata" / "run.json").write_text(json.dumps({"layout_version": 2}))
    assert not is_run(run)
    assert find_runs([tmp_path]) == []


def test_master_without_sites_is_not_a_run(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "master.csv").write_text("pdb_id\n")
    assert not is_run(tmp_path)


def test_a_run_without_a_site_table_is_incomplete(tmp_path):
    assert not is_complete(make_run(tmp_path / "r", complete=False))


def test_an_incomplete_run_is_flagged_in_its_label(tmp_path):
    assert "incomplete" in run_label(make_run(tmp_path / "PARG" / "7KG6", complete=False))


# --- walking --------------------------------------------------------------


def test_runs_nested_under_a_project_are_found(tmp_path):
    a = make_run(tmp_path / "PARG" / "7KG6")
    b = make_run(tmp_path / "USP7" / "5N9R")
    assert find_runs([tmp_path]) == sorted([a.resolve(), b.resolve()], key=str)


def test_the_walk_never_enters_a_run(tmp_path):
    """A run's sites/ can hold thousands of files; none should be visited."""
    run = make_run(tmp_path / "7KG6")
    make_run(run / "sites" / "decoy")  # would be found if the walk descended
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


# --- the regroup controls and the site view ---------------------------------


def test_site_parameters_read_from_run(tmp_path):
    from qpocket.config import QPocketConfig, evolve, save_run_config

    ws = layout.create(str(tmp_path), "r")
    save_run_config(evolve(evolve(QPocketConfig(), "reference.pdb_id", "1ABC"), "sites.cavity_overlap", 0.4), ws)
    got = site_parameters(ws.root)
    assert got["cavity_overlap"] == 0.4 and got["ligand_fraction"] == 0.5
    assert set(got) == {"min_cavity_points", "cavity_overlap", "min_structure_fraction",
                        "ligand_fraction", "box_padding"}


def test_an_unreadable_config_falls_back_to_the_defaults(tmp_path):
    from qpocket.config import QPocketConfig

    assert site_parameters(tmp_path)["cavity_overlap"] == QPocketConfig().sites.cavity_overlap


def test_site_view_ligands_one_per_holo_structure_largest_first(tmp_path):
    import pandas as pd

    ws = layout.create(str(tmp_path), "r")
    rows = [("P1", "A", "LIG", 1, " ", 6, "p1_small.pdb", "", "S1-H21"),
            ("P1", "A", "LIG", 2, " ", 9, "p1_big.pdb", "", "S1-H21;S2-A5"),
            ("P2", "A", "LIG", 1, " ", 7, "p2.pdb", "", "S1-H21"),
            ("P3", "A", "LIG", 1, " ", 7, "p3.pdb", "", "S2-A5")]
    pd.DataFrame(rows, columns=["pdb_id", "chain", "resname", "resseq", "icode", "n_heavy", "file", "smiles",
                                "site_ids"]).to_csv(ws.structures / "ligands.csv", index=False)
    meta = ws.sites / "S1-H21" / "metadata"
    meta.mkdir(parents=True)
    pd.DataFrame({"pdb_id": ["P2", "P1", "APO"], "protein_state": ["BOUND_STATE", "BOUND_STATE",
                                                                     "UNBOUND_STATE"]}).to_csv(meta / "master.csv",
                                                                                               index=False)
    assert site_view_ligands(ws.root, "S1-H21", 25) == ["p2.pdb", "p1_big.pdb"]
    assert site_view_ligands(ws.root, "S1-H21", 1) == ["p2.pdb"]
