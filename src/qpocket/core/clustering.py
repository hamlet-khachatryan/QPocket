"""Grouping pockets from a precomputed distance matrix."""

# The pipeline blends two distances into one before it clusters: a shape distance
# between Laplacian eigenvalue vectors, and a physicochemical distance between
# Graph-Fourier power spectra. Clustering the blend answers "which pockets are
# alike", but it cannot answer "alike in shape, different in chemistry", because by
# the time the clustering runs the two have been added together.
#
# :func:`two_stage_labels` keeps them apart: it groups on shape alone, then
# subdivides each shape group on chemistry alone. The result is a two-level label
# that says which pockets share a fold and, within that, which share a
# pharmacophore.
#
# Everything here takes a square distance matrix and nothing else, so the same
# functions serve the pipeline and the explorer notebook. Only linkages defined for
# an arbitrary distance matrix are offered: ward, centroid and median all assume
# Euclidean coordinates and return nonsense when handed condensed distances.

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

#: Linkages valid for a precomputed, non-Euclidean distance matrix.
LINKAGES: Tuple[str, ...] = ("average", "complete", "single", "weighted")

#: The pipeline's original algorithm.
AFFINITY = "affinity propagation"

#: Partitioning around medoids, for a distance matrix with no coordinates.
KMEDOIDS = "k-medoids"

#: Methods the explorer notebook offers, the pipeline's original first. Kept as
#: the notebook's picker contract; use :data:`ALL_METHODS` to include k-medoids.
METHODS: Tuple[str, ...] = (AFFINITY,) + LINKAGES

#: Every method :func:`cluster_labels` understands.
ALL_METHODS: Tuple[str, ...] = (AFFINITY, KMEDOIDS) + LINKAGES

#: How a hierarchical tree is cut.
CRITERIA: Tuple[str, ...] = ("cluster count", "distance threshold")

COUNT, THRESHOLD = CRITERIA

#: Affinity Propagation damping used by the pipeline.
PIPELINE_DAMPING = 0.9

#: Affinity Propagation iteration cap used by the pipeline.
PIPELINE_MAX_ITER = 300

#: Preference percentile that matches scikit-learn's default of the median
#: similarity, and therefore the pipeline.
PIPELINE_PREFERENCE_PCT = 50.0

#: Seed for every method that draws at random, so a run is reproducible.
RANDOM_STATE = 42


def symmetrise(distances: np.ndarray) -> np.ndarray:
    """Force a saved distance matrix to be a clean symmetric matrix.

    Args:
        distances: Square distance matrix.

    Returns:
        A symmetric copy with a zeroed diagonal.

    Raises:
        ValueError: If the matrix is not square.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"expected a square matrix, got {d.shape}")
    out = 0.5 * (d + d.T)
    np.fill_diagonal(out, 0.0)
    return out


class Clustering(NamedTuple):
    """A grouping of structures, and how good it is.

    Attributes:
        labels: Cluster index per structure, 0-based, largest cluster first.
        linkage_matrix: The scipy linkage used, reusable for a dendrogram, or ``None`` for a method that builds no tree.
        method: One of :data:`METHODS`.
        n_clusters: Number of clusters produced.
        threshold: Cut height for a hierarchical method, otherwise ``nan``.
        silhouette: Mean silhouette over all structures, or ``None`` when it is undefined (fewer than two clusters, or one cluster per structure).
        silhouette_per_structure: Per-structure silhouette, or ``None``.
        exemplar_positions: Positions the method itself nominated as cluster representatives, when it nominates any.
    """

    labels: np.ndarray
    linkage_matrix: Optional[np.ndarray]
    method: str
    n_clusters: int
    threshold: float
    silhouette: Optional[float]
    silhouette_per_structure: Optional[np.ndarray]
    exemplar_positions: Optional[Tuple[int, ...]] = None


class ClusterComparison(NamedTuple):
    """How far one grouping sits from another.

    Attributes:
        ari: Adjusted Rand index. 1.0 is identical, 0.0 is chance agreement.
        n_clusters_a: Cluster count in the first grouping.
        n_clusters_b: Cluster count in the second grouping.
        n_moved: Structures that change group once clusters are matched up as well as they can be.
        moved: Which structures those are, by position.
        pairs_split: Pairs together in the first grouping but apart in the second.
        pairs_joined: Pairs apart in the first grouping but together in the second.
    """

    ari: float
    n_clusters_a: int
    n_clusters_b: int
    n_moved: int
    moved: Tuple[int, ...]
    pairs_split: int
    pairs_joined: int


class DendrogramData(NamedTuple):
    """Line segments for drawing a dendrogram without a plotting backend.

    Attributes:
        segments: ``(xs, ys)`` pairs, one per link, in leaf-position units.
        leaf_positions: X position of each leaf, in the same units.
        leaf_order: Original indices in left-to-right display order.
        max_height: Tallest merge, for scaling the height axis.
    """

    segments: Tuple[Tuple[Tuple[float, ...], Tuple[float, ...]], ...]
    leaf_positions: np.ndarray
    leaf_order: Tuple[int, ...]
    max_height: float


class TwoStageClustering(NamedTuple):
    """A grouping made in two passes: shape first, then chemistry within it.

    Attributes:
        labels: Final composite cluster index per structure, 0-based, largest cluster first.
        shape_labels: Stage-one cluster index per structure.
        chem_labels: Stage-two cluster index per structure, numbered **within** its shape cluster, so ``0`` in one shape cluster is unrelated to ``0`` in another.
        shape_clustering: The stage-one result, with its silhouette.
        chem_clusterings: Stage-two result per shape cluster, or ``None`` where the cluster was too small to subdivide.
        exemplar_positions: Medoid position of each final cluster.
    """

    labels: np.ndarray
    shape_labels: np.ndarray
    chem_labels: np.ndarray
    shape_clustering: Clustering
    chem_clusterings: Dict[int, Optional[Clustering]]
    exemplar_positions: Tuple[int, ...]


def linkage_matrix(distances: np.ndarray, method: str = "average") -> np.ndarray:
    """Build a linkage from a square distance matrix.

    Args:
        distances: Square, symmetric distance matrix.
        method: One of :data:`LINKAGES`.

    Returns:
        The scipy linkage matrix.

    Raises:
        ValueError: If ``method`` is not a linkage valid for precomputed distances.
    """
    if method not in LINKAGES:
        raise ValueError(
            f"linkage {method!r} is not defined for a precomputed distance matrix;"
            f" use one of {LINKAGES}"
        )
    return linkage(squareform(symmetrise(distances), checks=False), method=method)


def threshold_for_clusters(link: np.ndarray, n_clusters: int) -> float:
    """Cut height that splits a tree into ``n_clusters``.

    Args:
        link: A scipy linkage matrix.
        n_clusters: Wanted cluster count.

    Returns:
        A height between the two merges that bracket that count.
    """
    heights = np.asarray(link)[:, 2]
    n = len(heights) + 1
    k = int(np.clip(n_clusters, 1, n))
    if k <= 1:
        return float(heights[-1] * 1.05) if len(heights) else 0.0
    if k >= n:
        return float(heights[0] * 0.95) if len(heights) else 0.0
    upper = heights[-(k - 1)]
    lower = heights[-k] if k <= len(heights) else 0.0
    return float(0.5 * (lower + upper))


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber labels 0, 1, 2… with the largest cluster first."""
    values, counts = np.unique(labels, return_counts=True)
    # Size descending, original label ascending as the tie-break.
    order = values[np.lexsort((values, -counts))]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[v] for v in labels], dtype=int)


def silhouette(
    distances: np.ndarray, labels: Sequence[int]
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Silhouette of a grouping, measured on the distances themselves.

    Args:
        distances: Square, symmetric distance matrix.
        labels: Cluster index per structure.

    Returns:
        Mean silhouette and per-structure values, or ``(None, None)`` when the grouping has too few or too many clusters for it to be defined.
    """
    from sklearn.metrics import silhouette_samples

    lab = np.asarray(labels)
    n_labels = len(np.unique(lab))
    if not 2 <= n_labels <= len(lab) - 1:
        return None, None
    values = silhouette_samples(symmetrise(distances), lab, metric="precomputed")
    return float(values.mean()), values


def affinity_labels(
    distances: np.ndarray,
    damping: float = PIPELINE_DAMPING,
    preference_percentile: float = PIPELINE_PREFERENCE_PCT,
    max_iter: int = PIPELINE_MAX_ITER,
) -> Clustering:
    """Cluster by Affinity Propagation, the way the pipeline does.

    Args:
        distances: Square, symmetric distance matrix.
        damping: Damping factor, in ``[0.5, 1.0)``.
        preference_percentile: Percentile of the similarity matrix used as the preference.
        max_iter: Iteration cap.

    Returns:
        The grouping, with the exemplars Affinity Propagation nominated.
    """
    from sklearn.cluster import AffinityPropagation

    d = symmetrise(distances)
    similarity = -d.astype(np.float64)
    preference = float(np.percentile(similarity, preference_percentile))

    model = AffinityPropagation(
        affinity="precomputed",
        damping=damping,
        max_iter=max_iter,
        preference=preference,
        random_state=RANDOM_STATE,
    )
    raw = model.fit_predict(similarity)
    centres = model.cluster_centers_indices_

    if centres is None or not len(centres):
        # scikit-learn signals non-convergence this way; the pipeline treats each
        # structure as its own cluster, so match that rather than inventing one.
        labels = np.arange(len(d))
        exemplar_positions: Optional[Tuple[int, ...]] = tuple(range(len(d)))
    else:
        labels = _relabel_by_size(raw)
        # Re-attach each exemplar to its cluster under the new numbering.
        exemplar_positions = tuple(sorted(int(c) for c in centres))

    mean_sil, per_sil = silhouette(d, labels)
    return Clustering(
        labels=labels,
        linkage_matrix=None,
        method=AFFINITY,
        n_clusters=int(len(np.unique(labels))),
        threshold=float("nan"),
        silhouette=mean_sil,
        silhouette_per_structure=per_sil,
        exemplar_positions=exemplar_positions,
    )


def _medoid_of(d: np.ndarray, members: np.ndarray) -> int:
    """Position of the member with the smallest total distance to the rest."""
    totals = d[np.ix_(members, members)].sum(axis=1)
    return int(members[int(np.argmin(totals))])


def _kmedoids_seed(d: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """Choose ``k`` starting medoids, spread out, deterministically seeded."""
    n = len(d)
    first = int(np.argmin(d.sum(axis=1)))
    chosen = [first]
    nearest = d[first].copy()

    while len(chosen) < k:
        weights = nearest**2
        total = float(weights.sum())
        if total <= 0.0:
            # Every remaining point coincides with a medoid; fill in order so the
            # result stays deterministic rather than raising.
            chosen.extend(i for i in range(n) if i not in chosen)
            return chosen[:k]
        nxt = int(rng.choice(n, p=weights / total))
        if nxt in chosen:
            nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, d[nxt])

    return chosen


def kmedoids_labels(
    distances: np.ndarray,
    n_clusters: int,
    max_iter: int = PIPELINE_MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> Clustering:
    """Partition around medoids, for a matrix with no underlying coordinates.

    Args:
        distances: Square, symmetric distance matrix.
        n_clusters: Wanted cluster count, clipped to ``[1, n]``.
        max_iter: Cap on assign/update rounds.
        random_state: Seed for the medoid seeding draw.

    Returns:
        The grouping, with the final medoids as ``exemplar_positions``.
    """
    d = symmetrise(distances)
    n = len(d)
    k = int(np.clip(n_clusters, 1, n))
    rng = np.random.default_rng(random_state)

    medoids = np.array(_kmedoids_seed(d, k, rng), dtype=int)
    raw = np.argmin(d[medoids], axis=0)

    for _ in range(max_iter):
        updated = medoids.copy()
        for c in range(k):
            members = np.flatnonzero(raw == c)
            if len(members):
                updated[c] = _medoid_of(d, members)
        new_raw = np.argmin(d[updated], axis=0)
        if np.array_equal(new_raw, raw) and np.array_equal(updated, medoids):
            medoids = updated
            break
        medoids, raw = updated, new_raw

    labels = _relabel_by_size(raw)
    # Report medoids under the final numbering, recomputed from the labels so an
    # empty cluster cannot leave a stale position behind.
    exemplar_positions = tuple(
        _medoid_of(d, np.flatnonzero(labels == c)) for c in sorted(np.unique(labels))
    )
    mean_sil, per_sil = silhouette(d, labels)
    return Clustering(
        labels=labels,
        linkage_matrix=None,
        method=KMEDOIDS,
        n_clusters=int(len(np.unique(labels))),
        threshold=float("nan"),
        silhouette=mean_sil,
        silhouette_per_structure=per_sil,
        exemplar_positions=exemplar_positions,
    )


def cluster_labels(
    distances: np.ndarray,
    method: str = AFFINITY,
    criterion: str = COUNT,
    n_clusters: int = 2,
    threshold: Optional[float] = None,
    damping: float = PIPELINE_DAMPING,
    preference_percentile: float = PIPELINE_PREFERENCE_PCT,
) -> Clustering:
    """Cluster a distance matrix by any of the offered methods.

    Args:
        distances: Square, symmetric distance matrix.
        method: One of :data:`ALL_METHODS`.
        criterion: ``"cluster count"`` to ask for a number of clusters, or ``"distance threshold"`` to cut at a height.
        n_clusters: Wanted cluster count, used by the count criterion.
        threshold: Cut height, used by the threshold criterion.
        damping: Affinity Propagation damping.
        preference_percentile: Affinity Propagation preference percentile.

    Returns:
        The grouping, with its silhouette.

    Raises:
        ValueError: If ``method`` or ``criterion`` is unknown.
    """
    d = symmetrise(distances)

    if method == AFFINITY:
        return affinity_labels(
            d, damping=damping, preference_percentile=preference_percentile
        )

    if method == KMEDOIDS:
        return kmedoids_labels(d, n_clusters=n_clusters)

    link = linkage_matrix(d, method)

    if criterion == COUNT:
        k = int(np.clip(n_clusters, 1, len(d)))
        raw = fcluster(link, t=k, criterion="maxclust")
        cut = threshold_for_clusters(link, k)
    elif criterion == THRESHOLD:
        cut = float(link[:, 2].max() if threshold is None else threshold)
        raw = fcluster(link, t=cut, criterion="distance")
    else:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {CRITERIA}")

    labels = _relabel_by_size(raw)
    mean_sil, per_sil = silhouette(d, labels)
    return Clustering(
        labels=labels,
        linkage_matrix=link,
        method=method,
        n_clusters=int(len(np.unique(labels))),
        threshold=float(cut),
        silhouette=mean_sil,
        silhouette_per_structure=per_sil,
    )


def best_k_by_silhouette(
    distances: np.ndarray,
    method: str,
    k_range: Sequence[int] = (2, 10),
    **kwargs: object,
) -> Clustering:
    """Cluster at the ``k`` with the best silhouette over a range.

    Args:
        distances: Square, symmetric distance matrix.
        method: One of :data:`ALL_METHODS`, excluding Affinity Propagation, which chooses its own cluster count.
        k_range: Inclusive ``(lo, hi)`` bounds on the cluster count.
        **kwargs: Passed through to :func:`cluster_labels`.

    Returns:
        The best-scoring grouping.

    Raises:
        ValueError: If ``method`` is Affinity Propagation.
    """
    if method == AFFINITY:
        raise ValueError(
            "affinity propagation chooses its own cluster count; "
            "it cannot be scanned over k_range"
        )

    d = symmetrise(distances)
    n = len(d)
    lo = max(2, int(k_range[0]))
    hi = min(int(k_range[1]), n - 1)

    best: Optional[Clustering] = None
    for k in range(lo, hi + 1):
        candidate = cluster_labels(d, method=method, n_clusters=k, **kwargs)
        if candidate.silhouette is None:
            continue
        if best is None or candidate.silhouette > best.silhouette:
            best = candidate

    if best is None:
        return cluster_labels(d, method=method, n_clusters=max(lo, 1), **kwargs)
    return best


def _stage_labels(
    distances: np.ndarray,
    method: str,
    linkage_name: str,
    k: Optional[int],
    k_range: Sequence[int],
    criterion: str = "count",
    threshold: Optional[float] = None,
    damping: float = PIPELINE_DAMPING,
    max_iter: int = PIPELINE_MAX_ITER,
) -> Clustering:
    """Run one stage: affinity, k-medoids, or a hierarchical cut by count or height."""
    resolved = AFFINITY if method == "affinity" else (
        KMEDOIDS if method == "kmedoids" else linkage_name
    )
    if criterion in ("threshold", THRESHOLD):
        if resolved in (AFFINITY, KMEDOIDS):
            raise ValueError(f"a threshold cut needs a hierarchical stage, not {method!r}")
        return cluster_labels(distances, method=resolved, criterion=THRESHOLD, threshold=threshold)
    if resolved == AFFINITY:
        return affinity_labels(distances, damping=damping, max_iter=max_iter)
    if k is None:
        return best_k_by_silhouette(distances, method=resolved, k_range=k_range)
    return cluster_labels(distances, method=resolved, n_clusters=int(k))


def two_stage_labels(
    shape_distances: np.ndarray,
    chem_distances: np.ndarray,
    shape_method: str = "hierarchical",
    shape_linkage: str = "average",
    shape_k: Optional[int] = None,
    shape_k_range: Sequence[int] = (2, 10),
    chem_method: str = "kmedoids",
    chem_linkage: str = "average",
    chem_k: Optional[int] = None,
    chem_k_range: Sequence[int] = (2, 6),
    min_cluster_size: int = 4,
    combined_distances: Optional[np.ndarray] = None,
    shape_criterion: str = "count",
    shape_threshold: Optional[float] = None,
    shape_damping: float = PIPELINE_DAMPING,
    shape_max_iter: int = PIPELINE_MAX_ITER,
    chem_criterion: str = "count",
    chem_threshold: Optional[float] = None,
    chem_damping: float = PIPELINE_DAMPING,
    chem_max_iter: int = PIPELINE_MAX_ITER,
) -> TwoStageClustering:
    """Group on shape, then subdivide each group on chemistry.

    Args:
        shape_distances: Square shape distance matrix.
        chem_distances: Square physicochemical distance matrix, same order.
        shape_method: ``"hierarchical"``, ``"kmedoids"`` or ``"affinity"``.
        shape_linkage: Linkage used when ``shape_method`` is hierarchical.
        shape_k: Pinned stage-one cluster count, or ``None`` to choose by silhouette.
        shape_k_range: Inclusive bounds scanned when ``shape_k`` is ``None``.
        chem_method: As ``shape_method``, for stage two.
        chem_linkage: As ``shape_linkage``, for stage two.
        chem_k: As ``shape_k``, for stage two.
        chem_k_range: As ``shape_k_range``, for stage two.
        min_cluster_size: Shape clusters smaller than this are not subdivided.
        combined_distances: Matrix used to pick each final cluster's medoid.
        shape_criterion: "count" (k or silhouette) or "threshold" (cut the shape tree at a height).
        shape_threshold: Cut height for a threshold shape stage.
        shape_damping: Affinity Propagation damping for an affinity shape stage.
        shape_max_iter: Affinity Propagation iteration cap for the shape stage.
        chem_criterion: As shape_criterion, for stage two.
        chem_threshold: Cut height for a threshold chemistry stage, applied within each shape cluster.
        chem_damping: As shape_damping, for stage two.
        chem_max_iter: As shape_max_iter, for stage two.

    Returns:
        The two-level grouping.

    Raises:
        ValueError: If the two matrices describe different numbers of structures.
    """
    shape_d = symmetrise(shape_distances)
    chem_d = symmetrise(chem_distances)
    if shape_d.shape != chem_d.shape:
        raise ValueError(
            f"shape and chemistry matrices cover {shape_d.shape} and "
            f"{chem_d.shape} structures"
        )

    n = len(shape_d)
    stage_one = _stage_labels(
        shape_d, shape_method, shape_linkage, shape_k, shape_k_range,
        shape_criterion, shape_threshold, shape_damping, shape_max_iter,
    )
    shape_labels = stage_one.labels

    chem_labels = np.zeros(n, dtype=int)
    chem_clusterings: Dict[int, Optional[Clustering]] = {}

    for value in sorted(np.unique(shape_labels)):
        members = np.flatnonzero(shape_labels == value)
        # Three members cannot support a silhouette-backed split, and a cluster
        # below the configured floor is deliberately left alone.
        if len(members) < max(min_cluster_size, 3):
            chem_clusterings[int(value)] = None
            continue
        sub = chem_d[np.ix_(members, members)]
        stage_two = _stage_labels(
            sub, chem_method, chem_linkage, chem_k, chem_k_range,
            chem_criterion, chem_threshold, chem_damping, chem_max_iter,
        )
        chem_clusterings[int(value)] = stage_two
        chem_labels[members] = stage_two.labels

    # Flatten (shape, chem) into one index, then renumber largest-first so the
    # final cluster_id follows the same convention as every other method here.
    pairs = list(zip(shape_labels.tolist(), chem_labels.tolist()))
    unique_pairs = sorted(set(pairs))
    pair_to_flat = {p: i for i, p in enumerate(unique_pairs)}
    labels = _relabel_by_size(np.array([pair_to_flat[p] for p in pairs], dtype=int))

    medoid_source = symmetrise(
        shape_d if combined_distances is None else combined_distances
    )
    exemplar_positions = tuple(
        _medoid_of(medoid_source, np.flatnonzero(labels == c))
        for c in sorted(np.unique(labels))
    )

    return TwoStageClustering(
        labels=labels,
        shape_labels=shape_labels,
        chem_labels=chem_labels,
        shape_clustering=stage_one,
        chem_clusterings=chem_clusterings,
        exemplar_positions=exemplar_positions,
    )


def compare_labelings(a: Sequence[int], b: Sequence[int]) -> ClusterComparison:
    """Measure how far two groupings of the same structures are apart.

    Args:
        a: Cluster index per structure in the first grouping.
        b: Cluster index per structure in the second grouping.

    Returns:
        Agreement measures and which structures changed group.

    Raises:
        ValueError: If the two groupings cover different numbers of structures.
    """
    from sklearn.metrics import adjusted_rand_score

    first, second = np.asarray(a), np.asarray(b)
    if first.shape != second.shape:
        raise ValueError(f"labelings cover {first.shape} and {second.shape} structures")

    values_a, values_b = np.unique(first), np.unique(second)
    contingency = np.zeros((len(values_a), len(values_b)), dtype=int)
    for i, va in enumerate(values_a):
        for j, vb in enumerate(values_b):
            contingency[i, j] = int(np.sum((first == va) & (second == vb)))

    rows, cols = linear_sum_assignment(-contingency)
    mapping = {values_a[r]: values_b[c] for r, c in zip(rows, cols)}
    moved = tuple(
        int(i) for i in range(len(first)) if mapping.get(first[i], None) != second[i]
    )

    iu = np.triu_indices(len(first), 1)
    same_a = first[iu[0]] == first[iu[1]]
    same_b = second[iu[0]] == second[iu[1]]
    return ClusterComparison(
        ari=float(adjusted_rand_score(first, second)),
        n_clusters_a=len(values_a),
        n_clusters_b=len(values_b),
        n_moved=len(moved),
        moved=moved,
        pairs_split=int(np.sum(same_a & ~same_b)),
        pairs_joined=int(np.sum(~same_a & same_b)),
    )


def block_order(
    distances: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    within_cluster: bool = True,
) -> List[int]:
    """Order structures so similar ones sit together on a heatmap diagonal.

    Args:
        distances: Square, symmetric distance matrix.
        labels: Cluster index per structure.
        within_cluster: Order inside each cluster by average-linkage leaf order.

    Returns:
        Positions in display order.
    """
    d = symmetrise(distances)

    def leaves(idx: List[int]) -> List[int]:
        if not within_cluster or len(idx) < 3:
            return list(idx)
        sub = d[np.ix_(idx, idx)]
        order = leaves_list(linkage(squareform(sub, checks=False), method="average"))
        return [idx[i] for i in order]

    if labels is None:
        return leaves(list(range(len(d))))

    lab = np.asarray(labels)
    order: List[int] = []
    for value in sorted(np.unique(lab)):
        order.extend(leaves([int(i) for i in np.flatnonzero(lab == value)]))
    return order


def exemplars(distances: np.ndarray, labels: Sequence[int]) -> Dict[int, int]:
    """The most central structure of each cluster.

    Args:
        distances: Square, symmetric distance matrix.
        labels: Cluster index per structure.

    Returns:
        Cluster index to the position of its medoid, the member with the smallest total distance to the rest of its cluster.
    """
    d = symmetrise(distances)
    lab = np.asarray(labels)
    out: Dict[int, int] = {}
    for value in np.unique(lab):
        out[int(value)] = _medoid_of(d, np.flatnonzero(lab == value))
    return out


def dendrogram_segments(
    link: np.ndarray, threshold: Optional[float] = None
) -> DendrogramData:
    """Line segments for a dendrogram, ready to hand to any plotting library.

    Args:
        link: A scipy linkage matrix.
        threshold: Colour threshold, only used to decide leaf ordering the same way a coloured dendrogram would.

    Returns:
        Segments in leaf-position units, where leaf ``i`` sits at ``10 * i + 5``.
    """
    data = dendrogram(
        np.asarray(link),
        no_plot=True,
        color_threshold=threshold if threshold is not None else 0.0,
    )
    segments = tuple(
        (tuple(float(v) for v in xs), tuple(float(v) for v in ys))
        for xs, ys in zip(data["icoord"], data["dcoord"])
    )
    leaves = tuple(int(i) for i in data["leaves"])
    positions = np.array([10.0 * i + 5.0 for i in range(len(leaves))])
    heights = np.asarray(link)[:, 2]
    return DendrogramData(
        segments=segments,
        leaf_positions=positions,
        leaf_order=leaves,
        max_height=float(heights.max()) if len(heights) else 0.0,
    )
