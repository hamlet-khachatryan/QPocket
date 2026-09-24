"""Both log formats work; the text one is what a first run asks for."""
from qpocket.io import logging as qlog
from tests.conftest import make_config


def _configure(tmp_path, log_format):
    cfg = make_config()
    cfg.io.log_format = log_format
    qlog.configure(cfg, tmp_path / "run.log")


def test_text_format_configures_without_error(tmp_path, capsys):
    """eliot 1.15 renamed pretty_print to pretty_format; text mode used to crash."""
    from eliot import log_message

    _configure(tmp_path, "text")
    log_message(message_type="smoke", pdb_id="7KG6")
    assert "smoke" in capsys.readouterr().out


def test_json_format_writes_records_to_the_file(tmp_path):
    from eliot import log_message

    _configure(tmp_path, "json")
    log_message(message_type="smoke", pdb_id="7KG6")
    assert "smoke" in (tmp_path / "run.log").read_text(encoding="utf-8")
