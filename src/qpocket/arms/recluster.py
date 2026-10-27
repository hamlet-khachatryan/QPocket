"""Recluster a finished run from its stored distance matrices, keeping the old clustering."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pandas as pd

from qpocket.config import ClusteringParams, StageClustering
from qpocket.pipelines.clustering import archive_clustering, cluster_pockets, write_clustering


def _stage(payload: dict, default: StageClustering) -> StageClustering:
    known = {f.name for f in fields(StageClustering)}
    values = {k: v for k, v in payload.items() if k in known}
    if payload and "percentile" not in payload and values.get("method", default.method) == "kmedoids" \
            and values.get("k") is None:
        # Saved before percentile cuts, when a kmedoids stage without k chose k itself.
        values["method"] = "hierarchical"
    if payload and "percentile" not in payload and values.get("method", default.method) == "hierarchical":
        values["k"] = None  # an old hierarchical count cut: percentile cuts replace it
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
        run_dir: Run root (holds metadata/ with the distance matrices).
        settings: Clustering settings to apply.

    Returns:
        Path of the new clustering_metadata.csv.
    """
    meta = Path(run_dir) / "metadata"
    shape_df = pd.read_csv(meta / "shape_distance_matrix.csv", index_col=0)
    ids = [str(i) for i in shape_df.index]
    chem = combined = None
    if (meta / "feature_distance_matrix.csv").exists():
        combined = pd.read_csv(meta / "spectral_distance_matrix.csv", index_col=0)
        shape_df = shape_df.loc[combined.index, combined.columns]
        ids = [str(i) for i in combined.index]
        chem = pd.read_csv(meta / "feature_distance_matrix.csv", index_col=0).loc[combined.index, combined.columns]
    shape = shape_df

    old = meta / "clustering_metadata.csv"
    file_by_id = {}
    if old.exists():
        previous = pd.read_csv(old)
        if "pocket_file" in previous.columns:
            file_by_id = dict(zip(previous["pdb_id"].astype(str), previous["pocket_file"]))

    # Shape-only measures distance to the exemplar on shape, whatever else exists.
    chem_m = None if chem is None else chem.to_numpy(float)
    combined_m = shape.to_numpy(float) if combined is None or settings.mode == "shape_only" else combined.to_numpy(float)
    assignment = cluster_pockets(settings, shape.to_numpy(float), chem_m, combined_m)
    # min_cluster_size is not part of the stage records; keep it with the rest.
    assignment.params.setdefault("min_cluster_size", int(settings.min_cluster_size))

    archive_clustering(meta)
    return write_clustering(meta, ids, assignment, combined_m, file_by_id)
