"""Applying a clustering in the notebook writes the run's metadata.

The explorer previews with the same function the pipeline clusters with, and
Apply persists it, so the sections below read the file rather than an
in-memory copy that could drift from it.
"""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qpocket.arms import recluster
from qpocket.viz import two_stage as ts


@pytest.fixture
def run(tmp_path):
    """A run with the three matrices and a clustering already stored."""
    ids = [f"P{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.2, (4, 2)), rng.normal(8, 0.2, (4, 2))])
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(d, 0.0)
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    frame = pd.DataFrame(d, index=ids, columns=ids)
    for name in ("spectral_distance_matrix", "shape_distance_matrix",
                 "feature_distance_matrix"):
        frame.to_csv(meta / f"{name}.csv")
    pd.DataFrame({"pdb_id": ids, "cluster_id": [0] * 8}).to_csv(
        meta / "clustering_metadata.csv", index=False)
    (meta / "clustering_params.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_apply_writes_the_clustering_the_preview_showed(run):
    matrices = ts.load_matrices(run)
    settings = ts.settings_for(recluster.load_settings(run), "two_stage",
                               shape_cut=4.0, chem_cut=0.6)
    previewed = ts.preview(matrices, settings)

    recluster.run(run, settings)

    saved = pd.read_csv(run / "metadata/clustering_metadata.csv")
    assert saved["cluster_id"].nunique() == previewed.n_clusters
    assert (saved.set_index("pdb_id")["cluster_id"].loc[previewed.frame["pdb_id"]].to_numpy()
            == previewed.frame["cluster_id"].to_numpy()).all()


def test_apply_archives_the_previous_clustering_once(run):
    settings = ts.settings_for(recluster.load_settings(run), "affinity")
    recluster.run(run, settings)
    history = list((run / "metadata" / "clustering_history").iterdir())
    assert len(history) == 1
    assert (history[0] / "clustering_metadata.csv").exists()


def test_the_settings_that_were_applied_are_recorded(run):
    settings = ts.settings_for(recluster.load_settings(run), "two_stage",
                               shape_cut=4.0, chem_cut=0.6)
    recluster.run(run, settings)
    recorded = json.loads((run / "metadata/clustering_params.json").read_text(encoding="utf-8"))
    assert recorded, "nothing was recorded"


def test_a_reopened_run_starts_from_what_was_applied(run):
    """load_settings feeds the sliders, so reopening resumes the last decision."""
    applied = ts.settings_for(recluster.load_settings(run), "two_stage",
                              shape_cut=4.0, chem_cut=0.6)
    recluster.run(run, applied)
    reloaded = recluster.load_settings(run)
    assert reloaded.mode == "two_stage"
    assert reloaded.shape.threshold == pytest.approx(4.0)


def test_load_matrices_shape_only(run):
    (run / "metadata/feature_distance_matrix.csv").unlink()
    m = ts.load_matrices(run)
    assert m.chem is None
    np.testing.assert_array_equal(m.combined, m.shape)


def test_recluster_shape_only_without_chem(run):
    (run / "metadata/feature_distance_matrix.csv").unlink()
    settings = replace(recluster.load_settings(run), mode="shape_only")
    recluster.run(run, settings)
    assert json.loads((run / "metadata/clustering_params.json").read_text())["mode"] == "shape_only"
    saved = pd.read_csv(run / "metadata/clustering_metadata.csv")
    assert saved["is_representative"].sum() == saved["cluster_id"].nunique()


def test_recluster_two_stage_without_chem_raises(run):
    (run / "metadata/feature_distance_matrix.csv").unlink()
    with pytest.raises(ValueError, match="needs featurised pockets"):
        recluster.run(run, recluster.load_settings(run))


def test_shape_only_matrices_preview_with_shape_only_settings(run):
    (run / "metadata/feature_distance_matrix.csv").unlink()
    m = ts.load_matrices(run)
    assert ts.chem_height_range(m, np.zeros(len(m.ids), dtype=int)) == (0.0, 1.0)
    settings = ts.shape_only_settings(recluster.load_settings(run), 4.0)
    assert settings.mode == "shape_only"
    assert ts.preview(m, settings).n_clusters >= 1
