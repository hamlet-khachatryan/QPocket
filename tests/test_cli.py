"""The command line, which until now had no tests at all.

That absence is why the `run` command could build an eighty-line list of Hydra
overrides and then never pass it — `--mode fragment_screen` reported "Mode:
fragment_screen" and ran a PDB query anyway, and eight other flags did nothing.
Every test here asserts that a flag actually reaches the configuration, because
"the option exists" and "the option works" turned out to be different claims.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qpocket.cli import run as cli_mod  # the module that calls the pipeline
from qpocket.cli import cli


@pytest.fixture
def captured(monkeypatch):
    """Run the CLI without running the pipeline, keeping what it passed on."""
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli_mod, "run_qpocket_pipeline", fake_run)
    return seen


def overrides_of(seen):
    return list(seen.get("extra_overrides") or [])


def invoke(*args):
    return CliRunner().invoke(cli, ["run", *args])


# --- the overrides actually travel ----------------------------------------


def test_mode_reaches_the_pipeline(captured):
    """The bug in one line: this used to be echoed to the terminal and dropped."""
    invoke("--mode", "fragment_screen", "--pdb-id", "1ABC")
    assert "mode=fragment_screen" in overrides_of(captured)


def test_structure_paths_reach_the_pipeline(captured):
    invoke("--mode", "fragment_screen", "--structure-paths", "/a.pdb",
           "--structure-paths", "/b.pdb")
    assert any("fragment_screen.structure_paths" in o for o in overrides_of(captured))


def test_reference_path_reaches_the_pipeline(captured, tmp_path):
    ref = tmp_path / "ref.pdb"
    ref.write_text("ATOM\n")
    invoke("--mode", "fragment_screen", "--reference-path", str(ref))
    assert f"fragment_screen.reference_path={ref}" in overrides_of(captured)


def test_input_type_reaches_the_pipeline(captured):
    invoke("--mode", "fragment_screen", "--input-type", "paths")
    assert "fragment_screen.input_type=paths" in overrides_of(captured)


def test_csv_path_reaches_the_pipeline(captured, tmp_path):
    csv = tmp_path / "in.csv"
    csv.write_text("pdb_path\n/a.pdb\n")
    invoke("--mode", "fragment_screen", "--csv-path", str(csv))
    assert any("fragment_screen.csv_path" in o for o in overrides_of(captured))


def test_pocket_definition_reaches_the_pipeline(captured):
    invoke("--pocket-definition", "Interactions")
    assert "reference.pocket_definition=Interactions" in overrides_of(captured)


def test_interaction_padding_reaches_the_pipeline(captured):
    invoke("--interaction-padding", "7.5")
    assert "reference.interaction_padding=7.5" in overrides_of(captured)


def test_reference_index_reaches_the_pipeline(captured):
    invoke("--mode", "fragment_screen", "--reference-index", "2")
    assert "fragment_screen.reference_index=2" in overrides_of(captured)


# --- renumbering is tri-state ---------------------------------------------


def test_no_renumber_residues_is_forwarded(captured):
    """`if renumber_residues:` swallowed False, so this flag never did anything."""
    invoke("--no-renumber-residues", "--pdb-id", "1ABC")
    assert captured.get("renumber_residues") is False


def test_renumber_residues_is_forwarded(captured):
    invoke("--renumber-residues", "--pdb-id", "1ABC")
    assert captured.get("renumber_residues") is True


def test_renumbering_is_left_to_the_mode_when_unset(captured):
    invoke("--pdb-id", "1ABC")
    assert captured.get("renumber_residues") is None


# --- the surface itself ----------------------------------------------------


@pytest.mark.parametrize("group", ["slurm", "task"])
def test_the_command_group_is_registered(group):
    assert CliRunner().invoke(cli, [group, "--help"]).exit_code == 0


@pytest.mark.parametrize(
    "stage", ["prepare", "structure", "sites", "site_featurize", "site_cluster"]
)
def test_every_array_stage_has_a_task_command(stage):
    """The generated sbatch scripts invoke these by name."""
    assert CliRunner().invoke(cli, ["task", stage, "--help"]).exit_code == 0


def test_validate_accepts_a_fragment_screen_config(tmp_path):
    ref = tmp_path / "ref.pdb"
    ref.write_text("ATOM\n")
    config = tmp_path / "c.yaml"
    config.write_text(
        "mode: fragment_screen\n"
        "fragment_screen:\n"
        "  input_type: paths\n"
        f"  reference_path: {ref}\n"
        "  structure_paths: [/a.pdb]\n"
    )
    result = CliRunner().invoke(cli, ["validate", str(config)])
    assert result.exit_code == 0, result.output


# --- precedence: a flag you did not type must not outrank your config --------


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "mode: fragment_screen\n"
        "reference:\n"
        "  chain_id: 'B'\n"
        "  pocket_definition: 'Interactions'\n"
        "  interaction_padding: 9.0\n"
        "fragment_screen:\n"
        "  input_type: paths\n"
        "  structure_paths: ['/a.pdb', '/b.pdb', '/c.pdb']\n"
        "  reference_index: 2\n"
    )
    return path


def effective(captured, config_file):
    from qpocket.config import load_qpocket_config

    return load_qpocket_config(
        config_path=str(config_file),
        overrides=list(captured.get("extra_overrides") or []) or None,
    )


def test_untyped_flags_add_no_overrides_at_all(captured, config_file):
    """Every eager default here used to be appended, and so beat --config."""
    invoke("--config", str(config_file))
    assert overrides_of(captured) == []


@pytest.mark.parametrize(
    "attr, expected",
    [
        ("mode", "fragment_screen"),
        ("reference.chain_id", "B"),
        ("reference.pocket_definition", "Interactions"),
        ("reference.interaction_padding", 9.0),
        ("fragment_screen.reference_index", 2),
    ],
)
def test_the_config_file_survives_an_untyped_flag(captured, config_file, attr, expected):
    invoke("--config", str(config_file))
    cfg = effective(captured, config_file)
    for part in attr.split("."):
        cfg = getattr(cfg, part)
    assert cfg == expected


def test_an_explicitly_typed_flag_still_beats_the_config(captured, config_file):
    invoke("--config", str(config_file), "--chain-id", "C")
    assert effective(captured, config_file).reference.chain_id == "C"


def test_renumber_residues_is_not_forced_to_the_string_none(captured, config_file):
    """It was appended unconditionally, emitting the literal `=None`."""
    invoke("--config", str(config_file))
    assert not any("renumber_residues" in o for o in overrides_of(captured))
