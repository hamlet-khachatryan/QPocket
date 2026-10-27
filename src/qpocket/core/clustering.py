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

from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

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

#: How a hierarchical tree is cut: into a number of clusters, at a height, at
#: a fraction of the tree's tallest merge, or at a percentile of the pairwise distances.
CRITERIA: Tuple[str, ...] = ("cluster count", "distance threshold", "relative height", "site percentile")

COUNT, THRESHOLD, RELATIVE, PERCENTILE = CRITERIA

#: The config's short criterion names, onto :data:`CRITERIA`.
_CRITERION_NAMES = {"count": COUNT, "threshold": THRESHOLD, "relative": RELATIVE, "percentile": PERCENTILE}

#: What a count stage without k is told; old runs saved k null under "count".
NEEDS_K = "a 'count' cut has no k: set k or use criterion 'threshold' or 'relative'"

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
    """A grouping of structures.

    Attributes:
        labels: Cluster index per structure, 0-based, largest cluster first.
        linkage_matrix: The scipy linkage used, reusable for a dendrogram, or ``None`` for a method that builds no tree.
        method: One of :data:`METHODS`.
        n_clusters: Number of clusters produced.
        threshold: Cut height for a hierarchical method, otherwise ``nan``.
        exemplar_positions: Positions the method itself nominated as cluster representatives, when it nominates any.
    """

    labels: np.ndarray
    linkage_matrix: Optional[np.ndarray]
    method: str
    n_clusters: int
    threshold: float
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
        shape_clustering: The stage-one result.
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

    return Clustering(
        labels=labels,
        linkage_matrix=None,
        method=AFFINITY,
        n_clusters=int(len(np.unique(labels))),
        threshold=float("nan"),
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
    return Clustering(
        labels=labels,
        linkage_matrix=None,
        method=KMEDOIDS,
        n_clusters=int(len(np.unique(labels))),
        threshold=float("nan"),
        exemplar_positions=exemplar_positions,
    )


def pair_distances(distances: np.ndarray) -> np.ndarray:
    """Every pairwise distance once: the strict upper triangle of a square matrix."""
    d = np.asarray(distances, dtype=float)
    return d[np.triu_indices(len(d), k=1)]


def percentile_height(distances: np.ndarray, percentile: float) -> float:
    """The cut height a percentile names: that quantile of the matrix's pairwise distances.

    Args:
        distances: Square distance matrix of the whole site.
        percentile: In (0, 100).

    Returns:
        The distance, or 0.0 when there are fewer than two structures.
    """
    pairs = pair_distances(distances)
    return float(np.percentile(pairs, percentile)) if len(pairs) else 0.0


def to_percentile(distances: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Where values fall among the matrix's pairwise distances, as percentiles in [0, 100].

    Args:
        distances: Square distance matrix of the whole site.
        values: Distances to place, e.g. merge heights.

    Returns:
        The share of pairs at or below each value, times 100.
    """
    pairs = np.sort(pair_distances(distances))
    values = np.asarray(values, dtype=float)
    if not len(pairs):
        return np.zeros_like(values)
    return np.clip(np.searchsorted(pairs, values, side="right") / len(pairs) * 100.0, 0.0, 100.0)


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
        criterion: One of :data:`CRITERIA`, or its short name ``"count"``, ``"threshold"``, ``"relative"`` or ``"percentile"``.
        n_clusters: Wanted cluster count, used by the count criterion.
        threshold: Cut height for the threshold criterion; for the relative one, the fraction of the tallest merge to cut at; for the percentile one, the percentile of the pairwise distances. When every distance is equal and non-zero, every merge sits at the same height, so a fraction below 1 leaves each structure on its own.
        damping: Affinity Propagation damping.
        preference_percentile: Affinity Propagation preference percentile.

    Returns:
        The grouping.

    Raises:
        ValueError: If ``method`` or ``criterion`` is unknown, or a threshold or relative cut has no threshold.
    """
    d = symmetrise(distances)
    criterion = _CRITERION_NAMES.get(criterion, criterion)

    if method == AFFINITY:
        return affinity_labels(
            d, damping=damping, preference_percentile=preference_percentile
        )

    if method == KMEDOIDS:
        return kmedoids_labels(d, n_clusters=n_clusters)

    if criterion in (THRESHOLD, RELATIVE, PERCENTILE) and threshold is None:
        short = {v: k for k, v in _CRITERION_NAMES.items()}[criterion]
        raise ValueError(f"a '{short}' cut needs a threshold")

    if criterion == RELATIVE and len(d) < 3:
        # Too few pockets for a tree worth cutting: one group, never an error.
        return Clustering(labels=np.zeros(len(d), dtype=int),
                          linkage_matrix=linkage_matrix(d, method) if len(d) == 2 else None,
                          method=method, n_clusters=1, threshold=float("nan"))

    link = linkage_matrix(d, method)

    if criterion == COUNT:
        k = int(np.clip(n_clusters, 1, len(d)))
        raw = fcluster(link, t=k, criterion="maxclust")
        cut = threshold_for_clusters(link, k)
    elif criterion == THRESHOLD:
        cut = float(threshold)
        raw = fcluster(link, t=cut, criterion="distance")
    elif criterion == RELATIVE:
        cut = float(threshold) * float(link[:, 2].max())
        raw = fcluster(link, t=cut, criterion="distance")
    elif criterion == PERCENTILE:
        cut = percentile_height(d, float(threshold))
        # the cut is one of the distances; merges at it carry float noise
        raw = fcluster(link, t=cut * (1 + 1e-9) + 1e-12, criterion="distance")
    else:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {CRITERIA}")

    labels = _relabel_by_size(raw)
    return Clustering(
        labels=labels,
        linkage_matrix=link,
        method=method,
        n_clusters=int(len(np.unique(labels))),
        threshold=float(cut),
    )


def _stage_labels(
    distances: np.ndarray,
    method: str,
    linkage_name: str,
    k: Optional[int],
    criterion: str = "count",
    threshold: Optional[float] = None,
    damping: float = PIPELINE_DAMPING,
    max_iter: int = PIPELINE_MAX_ITER,
) -> Clustering:
    """Run one stage: affinity, k-medoids, or a hierarchical cut by count, height or relative height."""
    resolved = AFFINITY if method == "affinity" else (
        KMEDOIDS if method == "kmedoids" else linkage_name
    )
    if resolved == AFFINITY:  # chooses its own cluster count, whatever the criterion
        return affinity_labels(distances, damping=damping, max_iter=max_iter)
    criterion = _CRITERION_NAMES.get(criterion, criterion)
    if criterion in (THRESHOLD, RELATIVE):
        if resolved == KMEDOIDS:
            raise ValueError(f"a {criterion} cut needs a hierarchical stage, not {method!r}")
        return cluster_labels(distances, method=resolved, criterion=criterion, threshold=threshold)
    if k is None:
        raise ValueError(NEEDS_K)
    return cluster_labels(distances, method=resolved, n_clusters=int(k))


def two_stage_labels(
    shape_distances: np.ndarray,
    chem_distances: np.ndarray,
    shape_method: str = "hierarchical",
    shape_linkage: str = "average",
    shape_k: Optional[int] = None,
    chem_method: str = "hierarchical",
    chem_linkage: str = "average",
    chem_k: Optional[int] = None,
    min_cluster_size: int = 4,
    combined_distances: Optional[np.ndarray] = None,
    shape_criterion: str = "relative",
    shape_threshold: Optional[float] = 0.5,
    shape_damping: float = PIPELINE_DAMPING,
    shape_max_iter: int = PIPELINE_MAX_ITER,
    chem_criterion: str = "relative",
    chem_threshold: Optional[float] = 0.5,
    chem_damping: float = PIPELINE_DAMPING,
    chem_max_iter: int = PIPELINE_MAX_ITER,
    stage_labels: Optional[Callable[..., Clustering]] = None,
) -> TwoStageClustering:
    """Group on shape, then subdivide each group on chemistry.

    Args:
        shape_distances: Square shape distance matrix.
        chem_distances: Square physicochemical distance matrix, same order.
        shape_method: ``"hierarchical"``, ``"kmedoids"`` or ``"affinity"``.
        shape_linkage: Linkage used when ``shape_method`` is hierarchical.
        shape_k: Stage-one cluster count, used by the count criterion.
        chem_method: As ``shape_method``, for stage two.
        chem_linkage: As ``shape_linkage``, for stage two.
        chem_k: As ``shape_k``, for stage two.
        min_cluster_size: Shape clusters smaller than this are not subdivided.
        combined_distances: Matrix used to pick each final cluster's medoid.
        shape_criterion: "count" (k clusters), "threshold" (cut the shape tree at a height) or "relative" (cut at a fraction of its tallest merge).
        shape_threshold: Cut height for a threshold shape stage; the fraction for a relative one.
        shape_damping: Affinity Propagation damping for an affinity shape stage.
        shape_max_iter: Affinity Propagation iteration cap for the shape stage.
        chem_criterion: As shape_criterion, for stage two.
        chem_threshold: As shape_threshold, for stage two, applied within each shape cluster (a relative cut takes each cluster's own tallest merge).
        chem_damping: As shape_damping, for stage two.
        chem_max_iter: As shape_max_iter, for stage two.
        stage_labels: Per-stage clustering function; the built-in dispatch when None.

    Returns:
        The two-level grouping.

    Raises:
        ValueError: If the two matrices describe different numbers of structures.
    """
    run_stage = stage_labels or _stage_labels
    shape_d = symmetrise(shape_distances)
    chem_d = symmetrise(chem_distances)
    if shape_d.shape != chem_d.shape:
        raise ValueError(
            f"shape and chemistry matrices cover {shape_d.shape} and "
            f"{chem_d.shape} structures"
        )

    n = len(shape_d)
    stage_one = run_stage(
        shape_d, shape_method, shape_linkage, shape_k,
        shape_criterion, shape_threshold, shape_damping, shape_max_iter,
    )
    shape_labels = stage_one.labels

    chem_labels = np.zeros(n, dtype=int)
    chem_clusterings: Dict[int, Optional[Clustering]] = {}

    for value in sorted(np.unique(shape_labels)):
        members = np.flatnonzero(shape_labels == value)
        # Fewer than three members leave no tree worth cutting, and a cluster
        # below the configured floor is deliberately left alone.
        if len(members) < max(min_cluster_size, 3):
            chem_clusterings[int(value)] = None
            continue
        sub = chem_d[np.ix_(members, members)]
        stage_two = run_stage(
            sub, chem_method, chem_linkage, chem_k,
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
