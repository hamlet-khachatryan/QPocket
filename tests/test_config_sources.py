"""Keeping the one configuration schema from becoming four.

Before this, a default was written down in four places -- the repo-root YAML,
the packaged YAML, a hardcoded template inside `qpocket init`, and
`docs/configuration.md` -- and three of them had drifted. `qpocket init` was
missing 45 of 80 keys, including every clustering and SLURM setting, so a user
who started from it could not discover half the pipeline.

These tests do not tidy that up; they make it impossible to reintroduce.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from qpocket.pipelines.api import CONF_PATH
from qpocket.cli import cli
from qpocket.config import QPocketConfig, _plain

REPO_YAML = Path("conf/config.yaml")
PACKAGED_YAML = CONF_PATH / "config.yaml"
DOCS = Path("docs/configuration.md")


def leaf_paths(node, prefix=""):
    """Every dotted leaf path in a nested mapping."""
    out = set()
    for key, value in (node or {}).items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            out |= leaf_paths(value, path)
        else:
            out.add(path)
    return out


#: `slurm.stages` maps a stage name to resources, so its children are chosen by
#: the user and have no fixed schema. Everything else is compared exactly.
FREE_FORM = "slurm.stages."


def fixed(keys):
    """Collapse the free-form subtree to its root, so both sides compare."""
    collapsed = {k for k in keys if not k.startswith(FREE_FORM)}
    if any(k.startswith(FREE_FORM) for k in keys):
        collapsed.add("slurm.stages")
    return collapsed


def yaml_keys(path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    data.pop("defaults", None)  # hydra's own composition list
    return leaf_paths(data)


@pytest.fixture(scope="module")
def schema_keys():
    return leaf_paths(_plain(QPocketConfig()))


# --- the two YAML copies ---------------------------------------------------


@pytest.mark.skipif(not REPO_YAML.exists(), reason="repo-root conf/ not present")
def test_the_two_yaml_copies_are_identical():
    """They are hand-synced, and nothing has ever checked that they agree."""
    assert REPO_YAML.read_text() == PACKAGED_YAML.read_text()


def test_the_packaged_yaml_defines_nothing_the_schema_lacks(schema_keys):
    """A YAML key with no dataclass behind it is a setting nobody can reach.

    The reverse is allowed: the file deliberately omits settings the chosen
    method ignores, and `fill_defaults` supplies them.
    """
    assert fixed(yaml_keys(PACKAGED_YAML)) - fixed(schema_keys) == set()


# --- qpocket init ----------------------------------------------------------


@pytest.fixture
def generated(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as here:
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0, result.output
        yield Path(here) / "qpocket_config.yaml"


def test_init_writes_a_parseable_file(generated):
    assert yaml.safe_load(generated.read_text())


def test_init_reproduces_the_packaged_defaults(generated):
    """It used to emit 35 of 80 keys, and none of the clustering or SLURM ones."""
    assert yaml_keys(generated) == yaml_keys(PACKAGED_YAML)


def test_init_drops_hydras_composition_list(generated):
    """`defaults:` steers Hydra's own composition and is noise in a user file."""
    assert "defaults:" not in generated.read_text().split("\n")[0]


def test_init_refuses_to_clobber_an_existing_file(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("qpocket_config.yaml").write_text("mine\n")
        result = runner.invoke(cli, ["init"], input="n\n")
        assert Path("qpocket_config.yaml").read_text() == "mine\n"


# --- the documentation -----------------------------------------------------


@pytest.mark.skipif(not DOCS.exists(), reason="docs not present")
def test_every_setting_is_documented(schema_keys):
    """The reference had no `slurm` section at all while the code read 21 of them."""
    text = DOCS.read_text()
    undocumented = sorted(k for k in schema_keys if k.rsplit(".", 1)[-1] not in text)
    assert undocumented == []


@pytest.mark.skipif(not DOCS.exists(), reason="docs not present")
def test_the_docs_point_at_the_config_the_code_loads():
    """They named the repo-root copy, which nothing reads."""
    assert "src/qpocket/conf/config.yaml" in DOCS.read_text()
