"""Distance stage: geodesics, fingerprints, distance matrices and clustering for a run."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pathlib import Path
import pandas as pd
from eliot import log_message, start_action
from joblib import Parallel, delayed

from qpocket.config import QPocketConfig
from qpocket.core import spectral as sp
from qpocket.core.spectral import pad_or_truncate  # noqa: F401  (re-exported)
from qpocket.io import cache, tables
from qpocket.types import Fingerprint
from qpocket.pipelines.clustering import assign_clusters, write_clustering


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


def pocket_manifest(cfg: QPocketConfig, dirs: Dict[str, Path]) -> Tuple[List[Path], List[str]]:
    """Featurised pocket files named by ``master.csv``, and their PDB IDs.

    Args:
        cfg: QPocket configuration.
        dirs: Directory map from ``layout.create``.

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


def compute_spectral_distance_matrix(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
) -> None:
    """Orchestrates spectral fingerprinting, pairwise distances, and clustering.

    Args:
        cfg: QPocket configuration.
        dirs: Directory map from ``layout.create``.

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


