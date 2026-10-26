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


def _stale_stages(settings):
    """Count stages saved with k null: they chose k by a score that no longer exists."""
    return [n for n in ("single", "shape", "chem")
            if getattr(settings, n).criterion == "count" and getattr(settings, n).k is None
            and getattr(settings, n).method != "affinity"]


def _runnable(settings):
    """The run's settings, with each stale count stage cut at half height instead."""
    return replace(settings, **{n: replace(getattr(settings, n), method="hierarchical",
                                           criterion="relative", threshold=0.5)
                                for n in _stale_stages(settings)})


def test_the_run_own_settings_reproduce_its_labels(tmp_path):
    """Runnable saved settings reproduce the saved labels; stale ones name the fix, and once fixed are stable."""
    run = _copy_run(tmp_path)
    settings = recluster.load_settings(run)
    if _stale_stages(settings):
        with pytest.raises(ValueError, match="set k or use criterion"):
            recluster.run(run, settings)
        settings = _runnable(settings)
        recluster.run(run, settings)
    before = pd.read_csv(run / "metadata/clustering_metadata.csv")
    recluster.run(run, settings)
    after = pd.read_csv(run / "metadata/clustering_metadata.csv")
    assert before["cluster_id"].tolist() == after["cluster_id"].tolist()
    assert before["is_representative"].tolist() == after["is_representative"].tolist()


def test_threshold_settings_are_applied_recorded_and_archived(tmp_path):
    run = _copy_run(tmp_path)
    settings = recluster.load_settings(run)
    shape = replace(settings.shape, method="hierarchical", criterion="threshold", threshold=1e-9)
    chem = replace(settings.chem, method="hierarchical", criterion="relative", threshold=0.5)
    recluster.run(run, replace(settings, shape=shape, chem=chem))
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
