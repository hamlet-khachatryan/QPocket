"""The cross-docking heatmap, outside the notebook."""
import pandas as pd

from qpocket.viz import crossdock


def test_heatmap_is_representative_by_active():
    pairs = pd.DataFrame({"representative": ["R1", "R1", "R2"], "active_id": ["A", "B", "A"],
                          "top1_rmsd": [1.0, 3.0, 0.5]})
    fig = crossdock.rmsd_heatmap(pairs, 2.0)
    assert list(fig.data[0].y) == ["R1", "R2"] and list(fig.data[0].x) == ["A", "B"]
    assert "2.0" in fig.layout.title.text


def test_the_cli_accepts_every_registered_engine():
    from click.testing import CliRunner

    from qpocket.cli import cli

    result = CliRunner().invoke(cli, ["screen", "prepare", "--help"])
    assert "--engine" in result.output and "quickvina" in result.output
