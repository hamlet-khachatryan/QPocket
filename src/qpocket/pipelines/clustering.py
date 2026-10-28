"""Grouping pockets from their distance matrices; methods come from CLUSTERING."""
from __future__ import annotations

from types import SimpleNamespace
import json
import shutil
from datetime import datetime, timezone
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
from pathlib import Path
import pandas as pd
from eliot import log_message
from qpocket.config import QPocketConfig
from qpocket.core import clustering as cl
from qpocket.core.ports import CLUSTERING



def stage_labels(distances, method, linkage, k, criterion="count", threshold=None,
                 damping=cl.PIPELINE_DAMPING, max_iter=cl.PIPELINE_MAX_ITER) -> cl.Clustering:
    """One clustering stage through the CLUSTERING registry.

    Args:
        distances: Square distance matrix.
        method: Plug-in name.
        linkage: Linkage for hierarchical.
        k: Cluster count for a count stage.
        criterion: "count", "threshold" or "relative".
        threshold: Cut height for a threshold stage; the fraction of the tallest merge for a relative one.
        damping: Affinity damping.
        max_iter: Affinity iteration cap.

    Returns:
        The clustering.
    """
    stage = SimpleNamespace(linkage=linkage, k=k, criterion=criterion,
                            threshold=threshold, damping=damping, max_iter=max_iter)
    return CLUSTERING.get(method).fn(distances, stage)


class ClusterAssignment(NamedTuple):
    """The grouping written to ``clustering_metadata.csv``.

    Attributes:
        labels: Final cluster index per structure.
        shape_labels: Stage-one index, or all zeros in single-stage mode.
        chem_labels: Stage-two index within the shape cluster, or all zeros.
        exemplar_positions: Row position of each cluster's representative.
        params: What was actually run, for ``clustering_params.json``.
    """

    labels: np.ndarray
    shape_labels: np.ndarray
    chem_labels: np.ndarray
    exemplar_positions: Tuple[int, ...]
    params: Dict[str, object]


def stage_cut(stage, site_distances: np.ndarray) -> Tuple[str, Optional[float]]:
    """How a stage is cut: a hierarchical stage at its percentile of the whole site's distances.

    Args:
        stage: Stage settings (StageClustering).
        site_distances: The site's full matrix for this stage, so a chemistry percentile means the same in every shape cluster.

    Returns:
        (criterion, threshold) for :func:`stage_labels`: a height for a hierarchical stage, a count otherwise.
    """
    if stage.method == "hierarchical":
        height = cl.percentile_height(site_distances, float(stage.percentile))
        return "threshold", height * (1 + 1e-9) + 1e-12   # merges at the cut carry float noise
    return "count", None


def _stage_params(stage, cut_height: Optional[float] = None) -> Dict[str, object]:
    """A stage's settings as plain types, for the params JSON."""
    return {
        "method": stage.method,
        "linkage": stage.linkage,
        "percentile": float(stage.percentile),
        "cut_height": None if cut_height is None else float(cut_height),
        "k": None if stage.k is None else int(stage.k),
        "damping": float(stage.damping),
        "max_iter": int(stage.max_iter),
    }


def assign_clusters(cfg: QPocketConfig, d_shape: np.ndarray, d_chem: np.ndarray,
                    combined: np.ndarray) -> ClusterAssignment:
    """Group the pockets with the run's clustering settings.

    Args:
        cfg: QPocket configuration.
        d_shape: Shape distance matrix.
        d_chem: Physicochemical distance matrix.
        combined: The blend.

    Returns:
        The assignment.
    """
    return cluster_pockets(cfg.distance.clustering, d_shape, d_chem, combined)


def cluster_shape_only(stage, d_shape: np.ndarray) -> ClusterAssignment:
    """Group the pockets on shape alone, one representative per shape cluster.

    Args:
        stage: Stage-one settings (StageClustering).
        d_shape: Shape distance matrix.

    Returns:
        The assignment; representatives are the medoids on d_shape.
    """
    criterion, height = stage_cut(stage, d_shape)
    result = stage_labels(d_shape, stage.method, stage.linkage, stage.k,
                          criterion, height, stage.damping, stage.max_iter)
    zeros = np.zeros(len(result.labels), dtype=int)
    exemplar_map = cl.exemplars(d_shape, result.labels)
    log_message(message_type="clustering_result", mode="shape_only", algorithm=result.method,
                n_clusters=result.n_clusters)
    return ClusterAssignment(
        labels=result.labels,
        shape_labels=result.labels,
        chem_labels=zeros,
        exemplar_positions=tuple(exemplar_map[c] for c in sorted(exemplar_map)),
        params={
            "mode": "shape_only",
            "shape": {**_stage_params(stage, height), "n_clusters": result.n_clusters},
            "n_clusters": result.n_clusters,
        },
    )


def cluster_pockets(
    settings,
    d_shape: np.ndarray,
    d_chem: Optional[np.ndarray],
    combined: Optional[np.ndarray],
) -> ClusterAssignment:
    """Group the pockets, in one pass or two depending on the config.

    Args:
        settings: distance.clustering settings (ClusteringParams).
        d_shape: Shape distance matrix.
        d_chem: Physicochemical distance matrix; None in shape-only mode.
        combined: The blend, used for representatives and for single-stage mode; None in shape-only mode.

    Returns:
        The assignment, with a record of what was run.
    """
    if settings.mode == "shape_only":
        return cluster_shape_only(settings.shape, d_shape)

    if d_chem is None or combined is None:
        raise ValueError("two-stage clustering needs featurised pockets; run Featurise first")

    if settings.mode == "single":
        stage = settings.single
        criterion, height = stage_cut(stage, combined)
        result = stage_labels(combined, stage.method, stage.linkage, stage.k,
                              criterion, height, stage.damping, stage.max_iter)

        zeros = np.zeros(len(result.labels), dtype=int)
        exemplar_map = cl.exemplars(combined, result.labels)
        log_message(
            message_type="clustering_result",
            mode="single",
            algorithm=result.method,
            n_clusters=result.n_clusters,
        )
        return ClusterAssignment(
            labels=result.labels,
            shape_labels=zeros,
            chem_labels=zeros,
            exemplar_positions=tuple(exemplar_map[c] for c in sorted(exemplar_map)),
            params={
                "mode": "single",
                "single": _stage_params(stage, height),
                "n_clusters": result.n_clusters,
            },
        )

    shape_cfg, chem_cfg = settings.shape, settings.chem
    shape_criterion, shape_height = stage_cut(shape_cfg, d_shape)
    chem_criterion, chem_height = stage_cut(chem_cfg, d_chem)   # the whole site's scale
    result = cl.two_stage_labels(
        d_shape,
        d_chem,
        shape_method=shape_cfg.method,
        shape_linkage=shape_cfg.linkage,
        shape_k=None if shape_cfg.k is None else int(shape_cfg.k),
        chem_method=chem_cfg.method,
        chem_linkage=chem_cfg.linkage,
        chem_k=None if chem_cfg.k is None else int(chem_cfg.k),
        min_cluster_size=int(settings.min_cluster_size),
        combined_distances=combined,
        shape_criterion=shape_criterion,
        shape_threshold=shape_height,
        shape_damping=float(shape_cfg.damping),
        shape_max_iter=int(shape_cfg.max_iter),
        chem_criterion=chem_criterion,
        chem_threshold=chem_height,
        chem_damping=float(chem_cfg.damping),
        chem_max_iter=int(chem_cfg.max_iter),
        stage_labels=stage_labels,
    )

    n_final = int(len(np.unique(result.labels)))
    log_message(
        message_type="clustering_result",
        mode="two_stage",
        n_shape_clusters=result.shape_clustering.n_clusters,
        n_clusters=n_final,
    )
    return ClusterAssignment(
        labels=result.labels,
        shape_labels=result.shape_labels,
        chem_labels=result.chem_labels,
        exemplar_positions=result.exemplar_positions,
        params={
            "mode": "two_stage",
            "shape": {
                **_stage_params(shape_cfg, shape_height),
                "n_clusters": result.shape_clustering.n_clusters,
            },
            "chem": {
                **_stage_params(chem_cfg, chem_height),
                "n_subclusters": {
                    str(k): (None if v is None else v.n_clusters)
                    for k, v in result.chem_clusterings.items()
                },
            },
            "n_clusters": n_final,
        },
    )


CLUSTERING_FILES = ("clustering_metadata.csv", "selected_representatives.csv", "clustering_params.json")


def archive_clustering(meta: Path) -> None:
    """Copy the current clustering, if any, into ``clustering_history/<UTC stamp>/``.

    Args:
        meta: A run's or a site's metadata directory.
    """
    present = [meta / f for f in CLUSTERING_FILES if (meta / f).exists()]
    if present:
        archive = meta / "clustering_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        for path in present:
            shutil.copy2(path, archive / path.name)


def write_clustering(meta_dir: Path, ids: List[str], assignment: ClusterAssignment,
                     combined: np.ndarray, file_by_id: Dict[str, Path]) -> Path:
    """Write clustering_metadata.csv, selected_representatives.csv and clustering_params.json.

    Args:
        meta_dir: The run's metadata directory.
        ids: Pocket IDs in matrix order.
        assignment: Result of cluster_pockets.
        combined: Blended distance matrix, for dist_to_exemplar.
        file_by_id: Featurised pocket file per ID.

    Returns:
        Path of clustering_metadata.csv.
    """
    exemplar_by_cluster = {
        int(assignment.labels[pos]): pos
        for pos in assignment.exemplar_positions
    }

    results_data = []
    for i, pid in enumerate(ids):
        cluster_id = int(assignment.labels[i])
        exemplar_idx = exemplar_by_cluster.get(cluster_id, i)
        results_data.append({
            "pdb_id": pid,
            "pocket_file": str(file_by_id.get(pid, "")),
            "cluster_id": cluster_id,
            "shape_cluster_id": int(assignment.shape_labels[i]),
            "chem_cluster_id": int(assignment.chem_labels[i]),
            "is_representative": bool(i == exemplar_idx),
            "dist_to_exemplar": float(combined[i, exemplar_idx]),
        })

    df_res = pd.DataFrame(results_data)
    out_meta = meta_dir / "clustering_metadata.csv"
    df_res.to_csv(out_meta, index=False)

    df_rep = df_res[df_res["is_representative"]]
    out_rep = meta_dir / "selected_representatives.csv"
    df_rep.to_csv(out_rep, index=False)

    out_params = meta_dir / "clustering_params.json"
    out_params.write_text(json.dumps(assignment.params, indent=2) + "\n", encoding="utf-8")

    log_message(
        message_type="clustering_saved",
        n_clusters=int(len(set(assignment.labels.tolist()))),
        n_representatives=len(df_rep),
        metadata_path=str(out_meta),
        representatives_path=str(out_rep),
        params_path=str(out_params),
    )
    return out_meta


def set_representative(meta_dir: Path, cluster_id: int, pdb_id: str) -> Path:
    """Make ``pdb_id`` its cluster's representative in the applied clustering, and record the choice.

    The swap is kept in ``clustering_params.json``: ``representative_overrides`` (cluster to member, gone again
    once the cluster is back on its medoid) and ``representative_history`` (every swap, in order).

    Args:
        meta_dir: The site's metadata directory, holding the applied clustering.
        cluster_id: The cluster.
        pdb_id: A member of that cluster.

    Returns:
        Path of clustering_metadata.csv.

    Raises:
        FileNotFoundError: If the site holds no clustering or no distance matrix.
        ValueError: If ``pdb_id`` is not in the cluster.
    """
    meta_dir = Path(meta_dir)
    meta_csv, params_json = meta_dir / "clustering_metadata.csv", meta_dir / "clustering_params.json"
    table = pd.read_csv(meta_csv)
    table["pdb_id"] = table["pdb_id"].astype(str)
    members = table["cluster_id"] == int(cluster_id)
    if not (members & (table["pdb_id"] == str(pdb_id))).any():
        raise ValueError(f"{pdb_id} is not in cluster {cluster_id}")
    params = json.loads(params_json.read_text(encoding="utf-8")) if params_json.exists() else {}
    # dist_to_exemplar is measured on the matrix the clustering used.
    spectral = meta_dir / "spectral_distance_matrix.csv"
    use = spectral if spectral.exists() and params.get("mode") != "shape_only" else meta_dir / "shape_distance_matrix.csv"
    if not use.exists():
        raise FileNotFoundError(f"no distance matrix to measure the swap on ({use.name} is missing)")
    matrix = pd.read_csv(use, index_col=0)
    matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
    current = table.loc[members & table["is_representative"].astype(bool), "pdb_id"]
    previous = str(current.iloc[0]) if len(current) else None
    table.loc[members, "is_representative"] = table.loc[members, "pdb_id"] == str(pdb_id)
    table.loc[members, "dist_to_exemplar"] = [float(matrix.loc[p, str(pdb_id)]) if p in matrix.index else float("nan")
                                              for p in table.loc[members, "pdb_id"]]
    key = str(int(cluster_id))
    medoids = params.setdefault("representative_medoids", {})
    medoids.setdefault(key, previous)
    overrides = params.setdefault("representative_overrides", {})
    if str(pdb_id) == medoids[key]:  # back on the medoid: the clustering is as it was made
        overrides.pop(key, None)
        medoids.pop(key)
    else:
        overrides[key] = str(pdb_id)
    for name in ("representative_overrides", "representative_medoids"):
        if not params[name]:
            del params[name]
    params.setdefault("representative_history", []).append(
        {"cluster_id": int(cluster_id), "from": previous, "to": str(pdb_id),
         "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()})
    # Every file is written aside first and swapped in last, so a failure leaves the clustering as it was.
    outputs = {meta_csv: table.to_csv(index=False),
               meta_dir / "selected_representatives.csv": table[table["is_representative"].astype(bool)].to_csv(
                   index=False),
               params_json: json.dumps(params, indent=2) + "\n"}
    for path, text in outputs.items():
        path.with_name(path.name + ".incoming").write_text(text, encoding="utf-8")
    for path in outputs:
        path.with_name(path.name + ".incoming").replace(path)
    log_message(message_type="representative_set", cluster_id=int(cluster_id), pdb_id=str(pdb_id))
    return meta_csv
