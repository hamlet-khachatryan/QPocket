"""Applying a clustering in the notebook writes the run's metadata.

The explorer previews with the same function the pipeline clusters with, and
Apply persists it, so the sections below read the file rather than an
in-memory copy that could drift from it.
"""

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
                               shape_cut=40.0, chem_cut=20.0)
    previewed = ts.preview(matrices, settings)

    recluster.run(run, settings)

    saved = pd.read_csv(run / "metadata/clustering_metadata.csv")
    assert saved["cluster_id"].nunique() == previewed.n_clusters
    assert (saved.set_index("pdb_id")["cluster_id"].loc[previewed.frame["pdb_id"]].to_numpy()
            == previewed.frame["cluster_id"].to_numpy()).all()


