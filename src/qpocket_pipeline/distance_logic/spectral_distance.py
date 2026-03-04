from typing import Union, Dict, List, Tuple, Optional

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
from eliot import log_message, start_action
from joblib import Parallel, delayed

from qpocket_pipeline.config import QPocketConfig


def compute_geodesic(csv_path: Path, neighbor_radius: float = 2.0) -> Union[Path, None]:
    """Builds a geodesic distance matrix from a featurised pocket CSV.

    Loads point coordinates and pharmacophoric features from *csv_path*,
    constructs a sparse neighbourhood graph (edges between points whose
    Euclidean distance is ≤ *neighbor_radius*), and computes all-pairs
    shortest-path distances via Dijkstra's algorithm.  The result is cached
    as a compressed NPZ file alongside the CSV.

    Args:
        csv_path: Path to a featurised pocket CSV (columns: x, y, z, …features…).
        neighbor_radius: Maximum Euclidean distance (Å) to connect two points
            in the neighbourhood graph.

    Returns:
        Path to the cached ``geospectral_cache.npz`` file, or ``None`` if the
        point cloud has fewer than 5 points or an error occurs.
    """
    cache_path = csv_path.with_name("geospectral_cache.npz")

    if cache_path.exists():
        try:
            with np.load(cache_path) as _:
                pass
            return cache_path
        except Exception:
            pass

    with start_action(
        action_type="compute_geodesic",
        pocket=csv_path.parent.name,
        neighbor_radius=neighbor_radius,
    ):
        try:
            df = pd.read_csv(csv_path)
            coords = df.iloc[:, 0:3].values
            features = df.iloc[:, 3:].values
            n_points = len(coords)

            if n_points < 5:
                log_message(
                    message_type="skip",
                    reason="too_few_points",
                    n_points=n_points,
                    pocket=csv_path.parent.name,
                )
                return None

            dist_euclidean = squareform(pdist(coords))
            adjacency = dist_euclidean.copy()
            adjacency[adjacency > neighbor_radius] = 0

            graph = csr_matrix(adjacency)
            D_geo = shortest_path(csgraph=graph, directed=False, method="D")

            finite_vals = D_geo[np.isfinite(D_geo)]
            max_finite = np.max(finite_vals) if len(finite_vals) > 0 else 100.0
            D_geo[~np.isfinite(D_geo)] = max_finite * 1.5

            if D_geo.max() > 0:
                D_geo /= D_geo.max()

            weights = np.ones(n_points) / n_points

            np.savez_compressed(
                cache_path, D_geo=D_geo, features=features, weights=weights
            )

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


def pad_or_truncate(vector: np.ndarray, target_k: int) -> np.ndarray:
    """Ensures *vector* is exactly *target_k* elements long.

    Pads with zeros on the right (high-frequency components carry no energy)
    or truncates to remove high-frequency noise.
    """
    curr_k = len(vector)
    if curr_k < target_k:
        return np.pad(vector, (0, target_k - curr_k), "constant")
    return vector[:target_k]


def compute_spectral_fingerprint(
    D_geo: np.ndarray,
    F: np.ndarray,
    global_k: int,
    sigma: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Converts a geodesic matrix and feature matrix into a spectral fingerprint.

    Constructs a Gaussian heat kernel on the geodesic graph, computes the
    normalised graph Laplacian, performs eigen-decomposition, and applies the
    Graph Fourier Transform to produce a rotationally invariant fingerprint.

    Args:
        D_geo: Square geodesic distance matrix (N × N).
        F: Feature matrix (N × D).
        global_k: Target dimensionality of the fingerprint.
        sigma: Standard deviation of the Gaussian heat kernel.

    Returns:
        Tuple of ``(eigenvalues, spectral_feature_vector)``.  Both arrays are
        of length *global_k* (padded / truncated as necessary).  Returns
        ``(None, None)`` when the matrix is too small.
    """
    with start_action(action_type="compute_spectral_fingerprint", n_points=D_geo.shape[0]):
        N = D_geo.shape[0]

        # Gaussian heat kernel: W_ij = exp(-d² / σ²)
        W = np.exp(-(D_geo ** 2) / (sigma ** 2))
        np.fill_diagonal(W, 0)

        # Normalised Laplacian: L = I - D^-0.5 W D^-0.5
        degrees = np.sum(W, axis=1)
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 1e-12)
        d_inv_sqrt[degrees <= 1e-12] = 0.0
        D_mat = np.diag(d_inv_sqrt)
        L = np.eye(N) - (D_mat @ W @ D_mat)

        # Eigen-decomposition — skip trivial mode 0
        k_req = min(global_k + 1, N - 1)
        if k_req < 1:
            return None, None

        eigvals, eigvecs = linalg.eigh(L, subset_by_index=[0, k_req])

        lam = eigvals[1:]
        U = eigvecs[:, 1:]

        # Graph Fourier Transform power spectrum (sign-invariant)
        spec_feat = np.abs(U.T @ F) ** 2
        lam_fixed = pad_or_truncate(lam, global_k)

        curr_k = spec_feat.shape[0]
        if curr_k < global_k:
            spec_feat = np.pad(spec_feat, ((0, global_k - curr_k), (0, 0)), "constant")
        else:
            spec_feat = spec_feat[:global_k, :]

        return lam_fixed, spec_feat.flatten()


def _cluster_with_affinity_propagation(
    dist_matrix: np.ndarray,
    damping: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clusters a precomputed distance matrix with Affinity Propagation.

    Converts the distance matrix to a similarity matrix (negation), then
    runs Affinity Propagation which automatically determines the number of
    clusters.

    Args:
        dist_matrix: Square symmetric distance matrix (N × N).
        damping: AP damping factor in ``[0.5, 1.0)``.
        max_iter: Maximum AP iterations.

    Returns:
        Tuple of ``(labels, exemplar_indices)`` where *labels* assigns a
        cluster index to each sample and *exemplar_indices* are the row
        indices of the selected exemplars (cluster representatives).
    """
    similarity = -dist_matrix.astype(np.float64)

    ap = AffinityPropagation(
        affinity="precomputed",
        damping=damping,
        max_iter=max_iter,
        random_state=42,
    )
    labels = ap.fit_predict(similarity)
    exemplar_indices = ap.cluster_centers_indices_

    n_clusters = len(set(labels))
    log_message(
        message_type="clustering_result",
        algorithm="affinity_propagation",
        n_clusters=n_clusters,
        n_samples=dist_matrix.shape[0],
        converged=ap.n_iter_ < max_iter,
    )
    return labels, exemplar_indices


def data_processing_spectral(
    cfg: QPocketConfig,
    geo_dist_paths: List[Path],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads geodesic NPZ caches, normalises features, and computes spectral fingerprints.

    Args:
        cfg: QPocket configuration.
        geo_dist_paths: List of paths to ``geospectral_cache.npz`` files.

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
                with np.load(p) as d:
                    D, F = d["D_geo"], d["features"]
                    if D.shape[0] <= cfg.spectral_distance_matrix.max_points:
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
            log_message(message_type="error", stage="data_processing_spectral", error="no data loaded")
            return {}

        # Global Z-score feature normalisation
        with start_action(action_type="normalise_features"):
            all_feats = [f for _, f in raw_data.values()]
            stacked = np.vstack(all_feats)
            scaler = StandardScaler()
            scaler.fit(stacked)
            log_message(
                message_type="scaler_fit",
                feature_shape=list(stacked.shape),
            )

        def _process_item(pid: str, D: np.ndarray, F: np.ndarray):
            F_norm = np.clip(scaler.transform(F), -5, 5)
            lam, spec_vec = compute_spectral_fingerprint(
                D,
                F_norm,
                cfg.spectral_distance_matrix.global_k,
                cfg.spectral_distance_matrix.sigma,
            )
            return pid, lam, spec_vec

        results = Parallel(
            n_jobs=cfg.spectral_distance_matrix.paralell_jobs,
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


def compute_spectral_distance_matrix(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
) -> None:
    """Orchestrates spectral fingerprinting, pairwise distance computation, and clustering.

    Pipeline:
        1. Reads pocket file paths from ``master.csv``.
        2. Computes / loads geodesic distance caches in parallel.
        3. Normalises features and computes spectral fingerprints.
        4. Builds a pairwise distance matrix combining eigenvalue and feature
           distances (weighted by ``alpha`` and ``beta``).
        5. Clusters with Affinity Propagation (automatic *k* selection).
        6. Writes ``spectral_distance_matrix.csv``, ``clustering_metadata.csv``,
           and ``selected_representatives.csv`` to the metadata directory.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary from ``setup_directory_hierarchy()``.
    """
    with start_action(action_type="compute_spectral_distance_matrix"):

        master_csv_path = dirs["metadata"] / "master.csv"
        if not master_csv_path.exists():
            log_message(
                message_type="error",
                stage="compute_spectral_distance_matrix",
                error="master.csv not found",
                path=str(master_csv_path),
            )
            return

        df_master = pd.read_csv(master_csv_path)
        pocket_files: List[Path] = []
        pdb_ids: List[str] = []

        for _, row in df_master.iterrows():
            pdb_id = row["pdb_id"]
            cav_dir = dirs["pockets"] / pdb_id
            if cav_dir.exists():
                bs_files = list(cav_dir.glob("featurized.csv"))
                if bs_files:
                    pocket_files.append(bs_files[0])
                    pdb_ids.append(pdb_id)

        if len(pocket_files) < 2:
            log_message(
                message_type="error",
                stage="compute_spectral_distance_matrix",
                error="fewer than 2 featurised pockets found",
                n_found=len(pocket_files),
            )
            raise ValueError("Need at least 2 featurised pockets to compute the distance matrix.")

        log_message(
            message_type="pocket_discovery",
            n_pockets=len(pocket_files),
        )

        # ── Geodesic computation (parallel) ──────────────────────────────────
        with start_action(action_type="compute_geodesics_batch", n_pockets=len(pocket_files)):
            geo_results = Parallel(
                n_jobs=cfg.spectral_distance_matrix.paralell_jobs,
                verbose=0,
            )(
                delayed(compute_geodesic)(
                    p,
                    neighbor_radius=cfg.spectral_distance_matrix.geo_graph_neighbor_radius,
                )
                for p in pocket_files
            )

        path_cache: List[Path] = []
        valid_indices: List[int] = []
        for i, res_path in enumerate(geo_results):
            if res_path is not None:
                path_cache.append(res_path)
                valid_indices.append(i)
            else:
                log_message(
                    message_type="skip",
                    stage="geodesic_batch",
                    pdb_id=pdb_ids[i],
                    reason="geodesic_failed",
                )

        valid_pdbs = [pdb_ids[i] for i in valid_indices]
        valid_files = [pocket_files[i] for i in valid_indices]
        n_samples = len(path_cache)

        log_message(
            message_type="spectral_fingerprint_start",
            n_valid=n_samples,
            global_k=cfg.spectral_distance_matrix.global_k,
            sigma=cfg.spectral_distance_matrix.sigma,
        )

        # ── Spectral fingerprints ─────────────────────────────────────────────
        cache = data_processing_spectral(cfg, path_cache)
        ids = sorted(cache.keys())
        N = len(ids)

        if N == 0:
            log_message(
                message_type="error",
                stage="compute_spectral_distance_matrix",
                error="no spectral fingerprints produced",
            )
            return

        # ── Pairwise distance matrix ──────────────────────────────────────────
        dist_matrix = np.zeros((N, N), dtype=np.float32)
        alpha = cfg.spectral_distance_matrix.alpha
        beta = cfg.spectral_distance_matrix.beta
        lams = [cache[pid][0] for pid in ids]
        feats = [cache[pid][1] for pid in ids]

        with start_action(
            action_type="pairwise_distance_computation",
            matrix_size=N,
            alpha=alpha,
            beta=beta,
        ):
            def _compute_row(i: int):
                row_vals = []
                for j in range(i + 1, N):
                    d_geo = float(np.linalg.norm(lams[i] - lams[j]))
                    d_feat = float(np.linalg.norm(feats[i] - feats[j]))
                    d_total = alpha * d_geo + (1.0 - alpha) * beta * d_feat
                    row_vals.append((j, d_total))
                return i, row_vals

            row_results = Parallel(
                n_jobs=cfg.spectral_distance_matrix.paralell_jobs,
            )(delayed(_compute_row)(i) for i in range(N))

            for i, row_vals in row_results:
                for j, val in row_vals:
                    dist_matrix[i, j] = val
                    dist_matrix[j, i] = val

        output_csv = dirs["metadata"] / "spectral_distance_matrix.csv"
        pd.DataFrame(dist_matrix, index=ids, columns=ids).to_csv(output_csv)
        log_message(
            message_type="distance_matrix_saved",
            path=str(output_csv),
            shape=[N, N],
        )

        # ── Affinity Propagation clustering ──────────────────────────────────
        with start_action(
            action_type="affinity_propagation_clustering",
            n_samples=N,
            damping=cfg.spectral_distance_matrix.affinity_damping,
            max_iter=cfg.spectral_distance_matrix.affinity_max_iter,
        ):
            try:
                labels, exemplar_indices = _cluster_with_affinity_propagation(
                    dist_matrix,
                    damping=cfg.spectral_distance_matrix.affinity_damping,
                    max_iter=cfg.spectral_distance_matrix.affinity_max_iter,
                )

                # Map ids → valid_pdbs order (ids are sorted; valid_pdbs may differ)
                id_to_vp_idx = {pid: i for i, pid in enumerate(valid_pdbs)}

                results_data = []
                for i, pid in enumerate(ids):
                    cluster_id = int(labels[i])
                    exemplar_idx = int(exemplar_indices[cluster_id])
                    dist_to_exemplar = float(dist_matrix[i, exemplar_idx])
                    vp_idx = id_to_vp_idx.get(pid, i)

                    results_data.append({
                        "pdb_id": pid,
                        "pocket_file": str(valid_files[vp_idx]) if vp_idx < len(valid_files) else "",
                        "cluster_id": cluster_id,
                        "is_representative": bool(i in exemplar_indices),
                        "dist_to_exemplar": dist_to_exemplar,
                    })

                df_res = pd.DataFrame(results_data)
                out_meta = dirs["metadata"] / "clustering_metadata.csv"
                df_res.to_csv(out_meta, index=False)

                df_rep = df_res[df_res["is_representative"]]
                out_rep = dirs["metadata"] / "selected_representatives.csv"
                df_rep.to_csv(out_rep, index=False)

                log_message(
                    message_type="clustering_saved",
                    n_clusters=int(labels.max()) + 1,
                    n_representatives=len(df_rep),
                    metadata_path=str(out_meta),
                    representatives_path=str(out_rep),
                )

            except Exception as e:
                log_message(
                    message_type="error",
                    stage="affinity_propagation_clustering",
                    error=str(e),
                )