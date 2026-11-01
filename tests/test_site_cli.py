"""`qpocket site` on the finished fixture run from test_site_steps."""
from click.testing import CliRunner

from qpocket.cli import cli
from tests.test_site_steps import _finished_run, fake_features  # noqa: F401


def _invoke(*args):
    return CliRunner().invoke(cli, ["site", *args])


def test_list_shows_status(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    result = _invoke("list", str(root))
    assert result.exit_code == 0, result.output
    assert "S2-Y1" in result.output and "shape_clustered" in result.output


def test_shape_featurize_cluster_regroup(tmp_path, monkeypatch, fake_features):
    root = _finished_run(tmp_path, monkeypatch)
    site = root / "sites" / "S1-Y1"
    assert _invoke("shape", str(root), "--site", "S1-Y1").exit_code == 0
    result = _invoke("featurize", str(root), "--site", "S1-Y1", "--workers", "1")
    assert result.exit_code == 0 and "featurised 3" in result.output
    assert (site / "pockets" / "P1" / "featurized.csv").exists()
    result = _invoke("cluster", str(root), "--site", "S1-Y1")
    assert result.exit_code == 0, result.output
    result = _invoke("regroup", str(root), "--ligand-fraction", "0.4")
    assert result.exit_code == 0 and "S2-Y1" in result.output


