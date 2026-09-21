"""Recluster a finished run from its stored distance matrices, keeping the old clustering."""
from __future__ import annotations

import json
import shutil
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qpocket.config import ClusteringParams, StageClustering
from qpocket.pipelines.distance import cluster_pockets, write_clustering

CLUSTERING_FILES = ("clustering_metadata.csv", "selected_representatives.csv", "clustering_params.json")


def _stage(payload: dict, default: StageClustering) -> StageClustering:
    known = {f.name for f in fields(StageClustering)}
    values = {k: v for k, v in payload.items() if k in known}
    return StageClustering(**{**default.__dict__, **values})


def load_settings(run_dir: Path) -> ClusteringParams:
    """The clustering settings a run was made with, from its clustering_params.json.

    Args:
        run_dir: Run root (holds metadata/).

    Returns:
        The settings; defaults for anything the file does not record.
    """
    defaults = ClusteringParams()
    path = Path(run_dir) / "metadata" / "clustering_params.json"
    if not path.exists():
        return defaults
    params = json.loads(path.read_text(encoding="utf-8"))
    return ClusteringParams(
        mode=params.get("mode", defaults.mode),
        min_cluster_size=int(params.get("min_cluster_size", defaults.min_cluster_size)),
        single=_stage(params.get("single", {}), defaults.single),
        shape=_stage(params.get("shape", {}), defaults.shape),
        chem=_stage(params.get("chem", {}), defaults.chem),
    )


def run(run_dir: Path, settings: ClusteringParams) -> Path:
    """Recluster a run and make the result its clustering; the old one moves to history.

    Args:
        run_dir: Run root (holds metadata/ with the three distance matrices).
        settings: Clustering settings to apply.

    Returns:
        Path of the new clustering_metadata.csv.
    """
    meta = Path(run_dir) / "metadata"
    combined = pd.read_csv(meta / "spectral_distance_matrix.csv", index_col=0)
    ids = [str(i) for i in combined.index]
    shape = pd.read_csv(meta / "shape_distance_matrix.csv", index_col=0).loc[combined.index, combined.columns]
    chem = pd.read_csv(meta / "feature_distance_matrix.csv", index_col=0).loc[combined.index, combined.columns]

    old = meta / "clustering_metadata.csv"
    file_by_id = {}
    if old.exists():
        previous = pd.read_csv(old)
        if "pocket_file" in previous.columns:
            file_by_id = dict(zip(previous["pdb_id"].astype(str), previous["pocket_file"]))

    assignment = cluster_pockets(settings, shape.to_numpy(float), chem.to_numpy(float), combined.to_numpy(float))
    # min_cluster_size is not part of the stage records; keep it with the rest.
    assignment.params.setdefault("min_cluster_size", int(settings.min_cluster_size))

    present = [meta / f for f in CLUSTERING_FILES if (meta / f).exists()]
    if present:
        archive = meta / "clustering_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        for path in present:
            shutil.copy2(path, archive / path.name)

    return write_clustering(meta, ids, assignment, combined.to_numpy(float), file_by_id)
