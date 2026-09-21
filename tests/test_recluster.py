"""Recluster a finished run from its stored matrices, with history."""
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from qpocket.cli import cli
from qpocket.pipelines import recluster

GOLD = Path(__file__).resolve().parent / "golden/expected/7KG6"


def _copy_run(tmp_path):
    run = tmp_path / "7KG6"
    shutil.copytree(GOLD / "metadata", run / "metadata")
    return run


pytestmark = pytest.mark.skipif(not (GOLD / "metadata/clustering_params.json").exists(),
                                reason="needs golden run")


def test_the_run_own_settings_reproduce_its_labels(tmp_path):
    run = _copy_run(tmp_path)
    before = pd.read_csv(run / "metadata/clustering_metadata.csv")
    recluster.run(run, recluster.load_settings(run))
    after = pd.read_csv(run / "metadata/clustering_metadata.csv")
    assert before["cluster_id"].tolist() == after["cluster_id"].tolist()
    assert before["is_representative"].tolist() == after["is_representative"].tolist()


def test_threshold_settings_are_applied_recorded_and_archived(tmp_path):
    run = _copy_run(tmp_path)
    settings = recluster.load_settings(run)
    shape = replace(settings.shape, method="hierarchical", criterion="threshold", threshold=1e-9)
    recluster.run(run, replace(settings, shape=shape))
    params = json.loads((run / "metadata/clustering_params.json").read_text())
    assert params["shape"]["criterion"] == "threshold" and params["shape"]["threshold"] == 1e-9
    labels = pd.read_csv(run / "metadata/clustering_metadata.csv")["shape_cluster_id"]
    assert labels.nunique() == len(labels)          # a near-zero cut separates everything
    history = list((run / "metadata/clustering_history").iterdir())
    assert len(history) == 1 and (history[0] / "clustering_metadata.csv").exists()


def test_cli_recluster(tmp_path):
    run = _copy_run(tmp_path)
    res = CliRunner().invoke(cli, ["recluster", str(run), "--shape-threshold", "0.01", "--chem-k", "2"])
    assert res.exit_code == 0, res.output
    params = json.loads((run / "metadata/clustering_params.json").read_text())
    assert params["shape"]["threshold"] == 0.01 and params["chem"]["k"] == 2
