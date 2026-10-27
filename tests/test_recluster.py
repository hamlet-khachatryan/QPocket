"""Recluster a finished run from its stored matrices, with history."""
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from qpocket.cli import cli
from qpocket.arms import recluster

GOLD = Path(__file__).resolve().parent / "golden/expected/7KG6"


def _copy_run(tmp_path):
    run = tmp_path / "7KG6"
    shutil.copytree(GOLD / "metadata", run / "metadata")
    return run


pytestmark = pytest.mark.skipif(not (GOLD / "metadata/clustering_params.json").exists(),
                                reason="needs golden run")


def test_reclustering_with_the_same_settings_is_stable(tmp_path):
    """The run's settings (old ones take the default percentiles) give the same labels twice."""
    run = _copy_run(tmp_path)
    settings = recluster.load_settings(run)
    recluster.run(run, settings)
    before = pd.read_csv(run / "metadata/clustering_metadata.csv")
    recluster.run(run, settings)
    after = pd.read_csv(run / "metadata/clustering_metadata.csv")
    assert before["cluster_id"].tolist() == after["cluster_id"].tolist()
    assert before["is_representative"].tolist() == after["is_representative"].tolist()


def test_percentile_settings_are_applied_recorded_and_archived(tmp_path):
    run = _copy_run(tmp_path)
    settings = recluster.load_settings(run)
    counts = {}
    for p in (0.5, 50.0):
        shape = replace(settings.shape, method="hierarchical", percentile=p)
        recluster.run(run, replace(settings, shape=shape))
        params = json.loads((run / "metadata/clustering_params.json").read_text())
        assert params["shape"]["percentile"] == p
        counts[p] = pd.read_csv(run / "metadata/clustering_metadata.csv")["shape_cluster_id"].nunique()
    assert counts[0.5] > counts[50.0]                 # a lower percentile cuts finer
    history = list((run / "metadata/clustering_history").iterdir())
    assert len(history) == 2 and (history[0] / "clustering_metadata.csv").exists()


def test_cli_recluster(tmp_path):
    run = _copy_run(tmp_path)
    res = CliRunner().invoke(cli, ["recluster", str(run), "--shape-percentile", "30", "--chem-k", "2"])
    assert res.exit_code == 0, res.output
    params = json.loads((run / "metadata/clustering_params.json").read_text())
    assert params["shape"]["percentile"] == 30.0 and params["chem"]["k"] == 2
