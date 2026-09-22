"""Distance stage: geodesics, fingerprints, distance matrices and clustering for a run."""
from __future__ import annotations

import json
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from pathlib import Path
import pandas as pd
from eliot import log_message, start_action
from joblib import Parallel, delayed

from qpocket.config import QPocketConfig
from qpocket.core import clustering as cl
from qpocket.core import spectral as sp
from qpocket.core.spectral import pad_or_truncate  # noqa: F401  (re-exported)
from qpocket.io import cache, tables
from qpocket.types import Fingerprint


def _workers(cfg: QPocketConfig) -> int:
    """Worker count for the distance stage.

    Args:
        cfg: QPocket configuration.

    Returns:
        joblib n_jobs: the legacy distance.parallel_jobs when it was set, else compute.workers.
    """
    from qpocket.pipelines.parallel import resolve_workers

    legacy = cfg.distance.parallel_jobs
    return legacy if legacy != -1 else resolve_workers(cfg.compute.workers)


def compute_geodesic(csv_path: Path, neighbor_radius: float = 2.0) -> Union[Path, None]:
    """Builds a geodesic distance matrix from a featurised pocket CSV.

    Args:
        csv_path: Path to a featurised pocket CSV with columns x, y, z, …features….
        neighbor_radius: Maximum Euclidean distance (Å) to connect two points in the neighbourhood graph.

    Returns:
        Path to the cached ``geospectral_cache.npz`` file, or ``None`` if the point cloud has fewer than 5 points or an error occurs.
    """
    cache_path = csv_path.with_name("geospectral_cache.npz")
    # Keyed on content: the Interactions pass rewrites featurized.csv after a first geodesic.
    source_hash = tables.file_hash(csv_path) if csv_path.exists() else ""
    if cache_path.exists() and cache.is_current(cache_path, source_hash, neighbor_radius):
        return cache_path

    with start_action(
        action_type="compute_geodesic",
        pocket=csv_path.parent.name,
        neighbor_radius=neighbor_radius,
    ):
        try:
            df = pd.read_csv(csv_path)

            # Use explicit column names — never rely on positional order.
            coord_cols = ["x", "y", "z"]
            feature_cols = [c for c in df.columns if c not in coord_cols]

            if not all(c in df.columns for c in coord_cols):
                log_message(
                    message_type="error",
                    stage="compute_geodesic",
                    pocket=csv_path.parent.name,
                    error="featurized.csv missing x/y/z columns",
                )
                return None

            coords = df[coord_cols].to_numpy(dtype=float)
            features = df[feature_cols].to_numpy(dtype=float) if feature_cols else np.empty((len(df), 0))
            n_points = len(coords)

            if n_points < 5:
                log_message(
                    message_type="skip",
                    reason="too_few_points",
                    n_points=n_points,
                    pocket=csv_path.parent.name,
                )
                return None

            D_geo = sp.geodesic_matrix(coords, neighbor_radius)
            if D_geo is None:
                log_message(
                    message_type="warning",
                    stage="compute_geodesic",
                    pocket=csv_path.parent.name,
                    warning="no edges in geodesic graph — increase geo_graph_neighbor_radius",
                )
                return None

            cache.save_geodesic(cache_path, D_geo, features, source_hash, neighbor_radius)

            log_message(
                message_type="geodesic_complete",
                pocket=csv_path.parent.name,
                n_points=n_points,
            )
            return cache_path

        except Exception as e:
            log_message(
                message_type="error",
                stage="compute_geodesic",
                pocket=csv_path.parent.name,
                error=str(e),
            )
            return None


def compute_spectral_fingerprint(
    D_geo: np.ndarray,
    F: np.ndarray,
    global_k: int,
    sigma: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Converts a geodesic matrix and feature matrix into a spectral fingerprint.

    Args:
        D_geo: Square geodesic distance matrix (N × N).
        F: Feature matrix (N × D).
        global_k: Target dimensionality of the fingerprint.
        sigma: Standard deviation of the Gaussian heat kernel.

    Returns:
        Tuple of ``(eigenvalues, spectral_feature_vector)``.
    """
    with start_action(action_type="compute_spectral_fingerprint", n_points=D_geo.shape[0]):
        return sp.spectral_fingerprint(D_geo, F, global_k, sigma)


def data_processing_spectral(
    cfg: QPocketConfig,
    geo_dist_paths: List[Path],
    scaler_out: Optional[Path] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads geodesic NPZ caches, normalises features, and computes spectral fingerprints.

    Args:
        cfg: QPocket configuration.
        geo_dist_paths: List of paths to ``geospectral_cache.npz`` files.
        scaler_out: Where to save the fitted scaler, if given.

    Returns:
        Mapping from pocket ID to ``(eigenvalues, spectral_feature_vector)``.
    """
    with start_action(
        action_type="data_processing_spectral",
        n_inputs=len(geo_dist_paths),
    ):
        raw_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        valid_ids: List[str] = []

        for p in geo_dist_paths:
            try:
                pid = p.parent.name if p.name == "geospectral_cache.npz" else p.stem
                D, F = cache.load_geodesic(p)
                if D.shape[0] <= cfg.distance.max_points:
                    raw_data[pid] = (D, F)
                    valid_ids.append(pid)
            except Exception as e:
                log_message(
                    message_type="load_error",
                    stage="data_processing_spectral",
                    file=str(p),
                    error=str(e),
                )

        log_message(
            message_type="loaded_pockets",
            n_valid=len(valid_ids),
            n_total=len(geo_dist_paths),
        )

        if not raw_data:
            log_message(
                message_type="error",
                stage="data_processing_spectral",
                error="no data loaded",
            )
            return {}

        # Global Z-score feature normalisation across all pockets.
        with start_action(action_type="normalise_features"):
            all_feats = [f for _, f in raw_data.values()]
            scaler = sp.fit_scaler(all_feats)
            if scaler_out is not None:
                cache.save_scaler(scaler_out, scaler)
            log_message(
                message_type="scaler_fit",
                feature_shape=[sum(len(f) for f in all_feats), all_feats[0].shape[1]],
            )

        def _process_item(pid: str, D: np.ndarray, F: np.ndarray):
            fp = sp.fingerprint(D, F, scaler, cfg.distance.global_k, cfg.distance.sigma)
            return (pid, None, None) if fp is None else (pid, fp.eigenvalues, fp.spectrum)

        results = Parallel(
            n_jobs=_workers(cfg),
            verbose=0,
        )(
            delayed(_process_item)(pid, D, F) for pid, (D, F) in raw_data.items()
        )

        processed_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for pid, lam, spec_vec in results:
            if lam is not None:
                processed_cache[pid] = (lam, spec_vec)

        log_message(
            message_type="fingerprints_computed",
            n_successful=len(processed_cache),
        )
        return processed_cache


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


def pocket_manifest(cfg: QPocketConfig, dirs: Dict[str, Path]) -> Tuple[List[Path], List[str]]:
    """Featurised pocket files named by ``master.csv``, and their PDB IDs.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary from ``setup_directory_hierarchy()``.

    Returns:
        Parallel lists of ``featurized.csv`` paths and PDB IDs, in master order.

    Raises:
        FileNotFoundError: If ``master.csv`` is absent.
    """
    master_csv_path = dirs["metadata"] / "master.csv"
    if not master_csv_path.exists():
        raise FileNotFoundError(f"master.csv not found at {master_csv_path}")

    df_master = pd.read_csv(master_csv_path)
    pocket_files: List[Path] = []
    pdb_ids: List[str] = []

    for _, row in df_master.iterrows():
        pdb_id = row["pdb_id"]
        candidate = dirs["pockets"] / pdb_id / "featurized.csv"
        if candidate.exists():
            pocket_files.append(candidate)
            pdb_ids.append(pdb_id)

    return pocket_files, pdb_ids


def build_distance_matrices(
    cfg: QPocketConfig,
    fingerprints: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn spectral fingerprints into the three distance matrices.

    Args:
        cfg: QPocket configuration.
        fingerprints: Pocket ID to ``(eigenvalues, spectral_feature_vector)``.
        ids: Pocket IDs, in the row order all three matrices will use.

    Returns:
        ``(shape, chemistry, combined)`` square matrices.
    """
    shape_weight = cfg.distance.shape_weight
    chem_scale = cfg.distance.chem_scale

    with start_action(
        action_type="pairwise_distance_computation",
        matrix_size=len(ids),
        shape_weight=shape_weight,
        chem_scale=chem_scale,
    ):
        return sp.distance_matrices(
            [Fingerprint(*fingerprints[pid]) for pid in ids], shape_weight, chem_scale
        )


def _stage_params(stage) -> Dict[str, object]:
    """A stage's settings as plain types, for the params JSON."""
    return {
        "method": stage.method,
        "linkage": stage.linkage,
        "k": None if stage.k is None else int(stage.k),
        "k_range": [int(v) for v in stage.k_range],
        "damping": float(stage.damping),
        "max_iter": int(stage.max_iter),
        "criterion": getattr(stage, "criterion", "count"),
        "threshold": None if getattr(stage, "threshold", None) is None else float(stage.threshold),
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


def cluster_pockets(
    settings,
    d_shape: np.ndarray,
    d_chem: np.ndarray,
    combined: np.ndarray,
) -> ClusterAssignment:
    """Group the pockets, in one pass or two depending on the config.

    Args:
        settings: distance.clustering settings (ClusteringParams).
        d_shape: Shape distance matrix.
        d_chem: Physicochemical distance matrix.
        combined: The blend, used for representatives and for single-stage mode.

    Returns:
        The assignment, with a record of what was run.
    """
    if settings.mode == "single":
        stage = settings.single
        if stage.method == "affinity":
            result = cl.affinity_labels(
                combined,
                damping=stage.damping,
                preference_percentile=cl.PIPELINE_PREFERENCE_PCT,
                max_iter=stage.max_iter,
            )
        elif stage.k is None:
            method = cl.KMEDOIDS if stage.method == "kmedoids" else stage.linkage
            result = cl.best_k_by_silhouette(
                combined, method=method, k_range=stage.k_range
            )
        else:
            method = cl.KMEDOIDS if stage.method == "kmedoids" else stage.linkage
            result = cl.cluster_labels(
                combined, method=method, n_clusters=int(stage.k)
            )

        zeros = np.zeros(len(result.labels), dtype=int)
        exemplar_map = cl.exemplars(combined, result.labels)
        log_message(
            message_type="clustering_result",
            mode="single",
            algorithm=result.method,
            n_clusters=result.n_clusters,
            silhouette=result.silhouette,
        )
        return ClusterAssignment(
            labels=result.labels,
            shape_labels=zeros,
            chem_labels=zeros,
            exemplar_positions=tuple(exemplar_map[c] for c in sorted(exemplar_map)),
            params={
                "mode": "single",
                "single": _stage_params(stage),
                "n_clusters": result.n_clusters,
                "silhouette": result.silhouette,
            },
        )

    shape_cfg, chem_cfg = settings.shape, settings.chem
    result = cl.two_stage_labels(
        d_shape,
        d_chem,
        shape_method=shape_cfg.method,
        shape_linkage=shape_cfg.linkage,
        shape_k=None if shape_cfg.k is None else int(shape_cfg.k),
        shape_k_range=shape_cfg.k_range,
        chem_method=chem_cfg.method,
        chem_linkage=chem_cfg.linkage,
        chem_k=None if chem_cfg.k is None else int(chem_cfg.k),
        chem_k_range=chem_cfg.k_range,
        min_cluster_size=int(settings.min_cluster_size),
        combined_distances=combined,
        shape_criterion=getattr(shape_cfg, "criterion", "count"),
        shape_threshold=getattr(shape_cfg, "threshold", None),
        shape_damping=float(shape_cfg.damping),
        shape_max_iter=int(shape_cfg.max_iter),
        chem_criterion=getattr(chem_cfg, "criterion", "count"),
        chem_threshold=getattr(chem_cfg, "threshold", None),
        chem_damping=float(chem_cfg.damping),
        chem_max_iter=int(chem_cfg.max_iter),
    )

    n_final = int(len(np.unique(result.labels)))
    log_message(
        message_type="clustering_result",
        mode="two_stage",
        n_shape_clusters=result.shape_clustering.n_clusters,
        n_clusters=n_final,
        shape_silhouette=result.shape_clustering.silhouette,
    )
    return ClusterAssignment(
        labels=result.labels,
        shape_labels=result.shape_labels,
        chem_labels=result.chem_labels,
        exemplar_positions=result.exemplar_positions,
        params={
            "mode": "two_stage",
            "shape": {
                **_stage_params(shape_cfg),
                "n_clusters": result.shape_clustering.n_clusters,
                "silhouette": result.shape_clustering.silhouette,
            },
            "chem": {
                **_stage_params(chem_cfg),
                "n_subclusters": {
                    str(k): (None if v is None else v.n_clusters)
                    for k, v in result.chem_clusterings.items()
                },
            },
            "n_clusters": n_final,
        },
    )


def compute_spectral_distance_matrix(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
) -> None:
    """Orchestrates spectral fingerprinting, pairwise distances, and clustering.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary from ``setup_directory_hierarchy()``.

    Raises:
        ValueError: If fewer than two featurised pockets are available.
    """
    with start_action(action_type="compute_spectral_distance_matrix"):

        try:
            pocket_files, pdb_ids = pocket_manifest(cfg, dirs)
        except FileNotFoundError as exc:
            log_message(
                message_type="error",
                stage="compute_spectral_distance_matrix",
                error=str(exc),
            )
            return

        if len(pocket_files) < 2:
            log_message(
                message_type="error",
                stage="compute_spectral_distance_matrix",
                error="fewer than 2 featurised pockets found",
                n_found=len(pocket_files),
            )
            raise ValueError("Need at least 2 featurised pockets to compute the distance matrix.")

        log_message(message_type="pocket_discovery", n_pockets=len(pocket_files))

        # ── Geodesic computation (parallel) ──────────────────────────────────
        with start_action(action_type="compute_geodesics_batch", n_pockets=len(pocket_files)):
            geo_results = Parallel(
                n_jobs=_workers(cfg),
                verbose=0,
            )(
                delayed(compute_geodesic)(
                    p,
                    neighbor_radius=cfg.distance.geo_graph_neighbor_radius,
                )
                for p in pocket_files
            )

        path_cache: List[Path] = []
        for i, res_path in enumerate(geo_results):
            if res_path is not None:
                path_cache.append(res_path)
            else:
                log_message(
                    message_type="skip",
                    stage="geodesic_batch",
                    pdb_id=pdb_ids[i],
                    reason="geodesic_failed",
                )

        # Keyed by PDB ID rather than by position: the two lists are filtered
        # differently further down, and pairing them positionally is what used to
        # attach the wrong pocket_file to a row.
        file_by_id = {pid: path for pid, path in zip(pdb_ids, pocket_files)}

        log_message(
            message_type="spectral_fingerprint_start",
            n_valid=len(path_cache),
            global_k=cfg.distance.global_k,
            sigma=cfg.distance.sigma,
        )

        # ── Spectral fingerprints ─────────────────────────────────────────────
        fingerprints = data_processing_spectral(
            cfg, path_cache, scaler_out=dirs["metadata"] / "scaler.npz"
        )
        ids = sorted(fingerprints.keys())
        n_samples = len(ids)

        if n_samples == 0:
            log_message(
                message_type="error",
                stage="compute_spectral_distance_matrix",
                error="no spectral fingerprints produced",
            )
            return

        # ── Distance matrices ─────────────────────────────────────────────────
        d_shape, d_chem, combined = build_distance_matrices(cfg, fingerprints, ids)

        for name, matrix in (
            ("spectral_distance_matrix.csv", combined),
            ("shape_distance_matrix.csv", d_shape),
            ("feature_distance_matrix.csv", d_chem),
        ):
            out_path = dirs["metadata"] / name
            pd.DataFrame(matrix, index=ids, columns=ids).to_csv(out_path)
            log_message(
                message_type="distance_matrix_saved",
                path=str(out_path),
                shape=[n_samples, n_samples],
            )

        # ── Clustering ────────────────────────────────────────────────────────
        with start_action(
            action_type="clustering",
            n_samples=n_samples,
            mode=cfg.distance.clustering.mode,
        ):
            assignment = assign_clusters(cfg, d_shape, d_chem, combined)

            write_clustering(dirs["metadata"], ids, assignment, combined, file_by_id)


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
