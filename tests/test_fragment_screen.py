"""Ingesting a fragment campaign: local files, a reference, and a ligand.

The claim under test is that a structure keeps its identity. The previous scheme
renamed every input to ``fragment_0000`` and recorded the real name only in the
log, so a cluster could not be traced back to the crystal it came from, and the
reference could not be found at all — ``pdb_id == cfg.reference.pdb_id`` never
matched a synthetic ID, so the reference pocket bounding box was silently never
computed and cavity selection lost its ranking.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qpocket.pipelines import stages
from qpocket.pipelines.stages import (
    _fragment_input_paths,
    _load_fragment_screen_structures,
    reference_id,
    run_id,
    sanitise_id,
    should_renumber,
)
from qpocket.config import FragmentScreenParams, QPocketConfig, validate_config


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

    root = tmp_path / "run"
    dirs = {
        "root": root,
        "metadata": root / "metadata",
        "original": root / "structures" / "original",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cfg = QPocketConfig(mode="fragment_screen")
    cfg.fragment_screen = FragmentScreenParams(
        input_type="paths",
        structure_paths=[str(a), str(b)],
        reference_path=str(ref),
    )
    cfg.reference.ligand_id = "LIG"
    return cfg, dirs, {"ref": ref, "a": a, "b": b}


def ingest(cfg, dirs):
    return _load_fragment_screen_structures(cfg, dirs["original"])


# --- identifiers -----------------------------------------------------------


def test_structures_are_named_after_their_files(campaign):
    cfg, dirs, _ = campaign
    assert set(ingest(cfg, dirs)) == {"PARG_reference", "soak_A07", "soak_B11"}


def test_an_awkward_filename_becomes_a_safe_identifier():
    """It has to serve as a directory name and a CSV key."""
    assert sanitise_id("PARG apo/form v2.pdb") == "PARG_apo_form_v2.pdb".replace(".", "_")


def test_an_empty_stem_still_produces_a_usable_name():
    assert sanitise_id("") == "structure"


def test_two_files_with_the_same_name_do_not_collide(tmp_path):
    """Same basename in two directories must not overwrite each other."""
    one = structure(tmp_path / "a" / "xtal.pdb")
    two = structure(tmp_path / "b" / "xtal.pdb")
    root = tmp_path / "run"
    dirs = {"metadata": root / "metadata", "original": root / "structures" / "original"}
    dirs["original"].mkdir(parents=True)
    cfg = QPocketConfig(mode="fragment_screen")
    cfg.fragment_screen = FragmentScreenParams(
        input_type="paths", structure_paths=[str(one), str(two)], reference_path=str(one)
    )
    assert len(set(_load_fragment_screen_structures(cfg, dirs["original"]))) == 2


# --- the reference ---------------------------------------------------------


def test_the_reference_is_ingested_first(campaign):
    cfg, dirs, _ = campaign
    assert ingest(cfg, dirs)[0] == "PARG_reference"


def test_the_reference_is_flagged_in_the_provenance_record(campaign):
    cfg, dirs, _ = campaign
    ingest(cfg, dirs)
    table = pd.read_csv(dirs["metadata"] / "inputs.csv")
    assert list(table[table["is_reference"]]["pdb_id"]) == ["PARG_reference"]


def test_a_reference_also_listed_as_a_structure_is_ingested_once(campaign):
    cfg, dirs, files = campaign
    cfg.fragment_screen.structure_paths = [
        str(files["ref"]), str(files["a"]), str(files["b"])
    ]
    assert ingest(cfg, dirs).count("PARG_reference") == 1


def test_the_reference_need_not_be_one_of_the_screened_files(campaign):
    cfg, dirs, _ = campaign
    assert "PARG_reference" in ingest(cfg, dirs)


def test_reference_id_matches_what_was_ingested(campaign):
    """Break 7: this comparison never matched, so no row was the reference."""
    cfg, dirs, _ = campaign
    ingest(cfg, dirs)
    assert reference_id(cfg, dirs) in ingest(cfg, dirs)


def test_without_a_reference_path_the_index_selects_one(campaign):
    cfg, dirs, _ = campaign
    cfg.fragment_screen.reference_path = None
    cfg.fragment_screen.reference_index = 1
    ingest(cfg, dirs)
    table = pd.read_csv(dirs["metadata"] / "inputs.csv")
    assert table.iloc[1]["is_reference"]


def test_some_structure_is_always_the_reference(campaign):
    cfg, dirs, _ = campaign
    cfg.fragment_screen.reference_path = None
    ingest(cfg, dirs)
    assert pd.read_csv(dirs["metadata"] / "inputs.csv")["is_reference"].sum() == 1


# --- provenance ------------------------------------------------------------


def test_every_structure_maps_back_to_its_source_file(campaign):
    cfg, dirs, files = campaign
    ingest(cfg, dirs)
    table = pd.read_csv(dirs["metadata"] / "inputs.csv").set_index("pdb_id")
    assert table.loc["soak_A07", "source_path"] == str(files["a"].resolve())


def test_the_copied_file_keeps_its_format(campaign):
    cfg, dirs, _ = campaign
    ingest(cfg, dirs)
    assert (dirs["original"] / "soak_A07.pdb").exists()


# --- bad input -------------------------------------------------------------


def test_a_missing_file_is_skipped_and_the_rest_survive(campaign):
    cfg, dirs, files = campaign
    cfg.fragment_screen.structure_paths = [str(files["a"]), "/nope/absent.pdb"]
    assert ingest(cfg, dirs) == ["PARG_reference", "soak_A07"]


def test_an_unrecognised_format_is_skipped(campaign, tmp_path):
    cfg, dirs, files = campaign
    notes = structure(tmp_path / "src" / "notes.txt")
    cfg.fragment_screen.structure_paths = [str(files["a"]), str(notes)]
    assert "notes" not in ingest(cfg, dirs)


def test_an_empty_ingest_raises_rather_than_reaching_alignment(campaign):
    """It used to return [] and fail later, somewhere far less informative."""
    cfg, dirs, _ = campaign
    cfg.fragment_screen.reference_path = None
    cfg.fragment_screen.structure_paths = ["/nope/a.pdb"]
    with pytest.raises(ValueError, match="ingested no structures"):
        stages.fetch_structures(cfg, dirs["original"])


def test_a_csv_without_the_expected_column_yields_nothing(campaign, tmp_path):
    cfg, dirs, _ = campaign
    csv = tmp_path / "in.csv"
    csv.write_text("path\n/a.pdb\n")
    cfg.fragment_screen.input_type = "csv"
    cfg.fragment_screen.csv_path = str(csv)
    cfg.fragment_screen.reference_path = None
    assert _fragment_input_paths(cfg) == []


# --- run naming and renumbering -------------------------------------------


def test_the_run_is_named_after_the_reference_file(campaign):
    cfg, _, _ = campaign
    assert run_id(cfg) == "PARG_reference"


def test_an_explicit_pdb_id_overrides_the_derived_name(campaign):
    cfg, _, _ = campaign
    cfg.reference.pdb_id = "MYRUN"
    assert run_id(cfg) == "MYRUN"


def test_a_run_with_no_name_at_all_is_refused():
    cfg = QPocketConfig(mode="fragment_screen")
    with pytest.raises(ValueError, match="no name"):
        run_id(cfg)


def test_renumbering_is_off_by_default_for_local_structures(campaign):
    """They have no UniProt entry; the lookup would silently degrade."""
    cfg, _, _ = campaign
    assert should_renumber(cfg) is False


def test_renumbering_is_on_by_default_for_a_pdb_query():
    cfg = QPocketConfig()
    cfg.reference.pdb_id = "1ABC"
    assert should_renumber(cfg) is True


def test_an_explicit_setting_beats_the_mode_default(campaign):
    cfg, _, _ = campaign
    cfg.alignment.renumber_residues = True
    assert should_renumber(cfg) is True


# --- validation ------------------------------------------------------------


def test_a_fragment_campaign_needs_no_pdb_id(campaign):
    """A local file has no PDB accession; requiring one blocked the whole mode."""
    cfg, _, _ = campaign
    validate_config(cfg)


def test_a_pdb_query_still_requires_a_four_character_id():
    with pytest.raises(ValueError, match="pdb_id"):
        validate_config(QPocketConfig())


def test_a_fragment_run_name_must_be_directory_safe(campaign):
    cfg, _, _ = campaign
    cfg.reference.pdb_id = "bad/name"
    with pytest.raises(ValueError, match="pdb_id"):
        validate_config(cfg)
