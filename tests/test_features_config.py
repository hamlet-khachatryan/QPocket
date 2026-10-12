"""features.enabled drives the columns, the tools that run, and what a run records."""
import json

import pytest
from click.testing import CliRunner

from qpocket.config import validate_config
from tests.conftest import make_config
from qpocket.config import evolve


def test_defaults_are_todays_six_columns():
    assert make_config().features.enabled == ["apbs", "cp_lp", "hbond", "aromatic"]


def test_unknown_provider_is_a_config_error():
    cfg = make_config()
    cfg = evolve(cfg, "features.enabled", ["apbs", "quantum"])
    with pytest.raises(ValueError, match="features.enabled"):
        validate_config(cfg)


def test_empty_selection_is_a_config_error():
    cfg = make_config()
    cfg = evolve(cfg, "features.enabled", [])
    with pytest.raises(ValueError, match="features.enabled"):
        validate_config(cfg)


@pytest.mark.parametrize("flags,expected", [
    (["--no-apbs"], "features.enabled=[cp_lp,hbond,aromatic]"),
    (["--features", "hbond,aromatic"], "features.enabled=[hbond,aromatic]"),
])
def test_cli_flags_narrow_the_set(monkeypatch, flags, expected):
    from qpocket import cli as cli_mod

    seen = {}
    monkeypatch.setattr(cli_mod, "run_qpocket_pipeline", lambda **kw: seen.update(kw))
    result = CliRunner().invoke(cli_mod.cli, ["run", "--pdb-id", "7KG6", *flags])
    assert result.exit_code == 0, result.output
    assert expected in seen["extra_overrides"]


def test_collect_records_the_feature_set(tmp_path):
    from qpocket.pipelines.collect import write_feature_record

    write_feature_record(tmp_path, ["hbond", "aromatic"])
    payload = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))
    assert payload == {"enabled": ["hbond", "aromatic"],
                       "columns": ["h_donor", "h_acceptor", "aromatic"]}
