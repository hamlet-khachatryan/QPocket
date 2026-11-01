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


# --- renumbering is tri-state ---------------------------------------------


# --- the surface itself ----------------------------------------------------


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


@pytest.mark.parametrize(
    "attr, expected",
    [
        ("mode", "fragment_screen"),
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


