"""`qpocket site` on the finished fixture run from test_site_steps."""
import pytest
from click.testing import CliRunner

from qpocket.arms import sites as steps
from qpocket.cli import cli
from tests.test_site_steps import _finished_run, fake_features  # noqa: F401


def _invoke(*args):
    return CliRunner().invoke(cli, ["site", *args])


def test_list_shows_status(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    result = _invoke("list", str(root))
    assert result.exit_code == 0, result.output
    assert "site_02" in result.output and "shape_clustered" in result.output


def test_shape_featurize_cluster_regroup(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    site = root / "sites" / "site_01"
    assert _invoke("shape", str(root), "--site", "site_01").exit_code == 0
    result = _invoke("featurize", str(root), "--site", "site_01", "--workers", "1")
    assert result.exit_code == 0 and "featurised 3" in result.output
    assert (site / "pockets" / "P1" / "featurized.csv").exists()
    result = _invoke("cluster", str(root), "--site", "site_01")
    assert result.exit_code == 0, result.output
    result = _invoke("regroup", str(root), "--cutoff", "3.0")
    assert result.exit_code == 0 and "site_03" in result.output


def test_missing_site_option(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    result = _invoke("shape", str(root))
    assert result.exit_code != 0 and "Missing option '--site'" in result.output


@pytest.mark.parametrize("command,message", [("shape", "unknown site"), ("cluster", "unknown site")])
def test_unknown_site_is_a_one_line_error(tmp_path, monkeypatch, command, message):
    root = _finished_run(tmp_path, monkeypatch)
    result = _invoke(command, str(root), "--site", "site_99")
    assert result.exit_code != 0 and message in result.output
    assert "Traceback" not in result.output and result.exception.__class__ is SystemExit


def test_two_stage_before_featurise_is_a_one_line_error(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    result = _invoke("cluster", str(root), "--site", "site_01")
    assert result.exit_code != 0 and "Featurise" in result.output


def test_a_directory_that_is_not_a_run_is_a_one_line_error(tmp_path):
    result = _invoke("list", str(tmp_path))
    assert result.exit_code != 0 and "is not a QPocket run" in result.output
    assert isinstance(result.exception, SystemExit)


def test_featurize_unknown_site(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    result = _invoke("featurize", str(root), "--site", "site_99")
    assert result.exit_code != 0 and "unknown site" in result.output


def test_unexpected_errors_keep_their_traceback(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyError("x")

    monkeypatch.setattr(steps, "featurize_site", boom)
    result = _invoke("featurize", str(tmp_path), "--site", "site_01")
    assert isinstance(result.exception, KeyError)
