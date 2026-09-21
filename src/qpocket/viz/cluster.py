"""Re-clustering a precomputed pocket distance matrix.

The pipeline stores one clustering per run. This module recomputes it under
whatever method and cut the user asks for, which is how you find out whether the
stored answer is a property of the pockets or of one parameter choice, and
reports how far a new grouping has moved from the stored one.

Affinity Propagation is offered first because it is what the pipeline ran for
single-stage runs: at its default damping and preference it reproduces the stored
labels, so a comparison against them starts from zero disagreement. It chooses its
own cluster count, which is why its knob is a preference percentile rather than a
``k``.

The hierarchical alternatives take a ``k`` or a cut height and will disagree with
the stored grouping substantially, since they optimise something else entirely.

The algorithms themselves live in :mod:`qpocket.core.clustering`,
because the pipeline needs them too and must not import a presentation package.
This module is the notebook's view of them: :data:`METHODS` is the picker's list,
which deliberately omits k-medoids.
"""

from __future__ import annotations

from qpocket.core.clustering import (
    AFFINITY,
    ALL_METHODS,
    COUNT,
    CRITERIA,
    KMEDOIDS,
    LINKAGES,
    METHODS,
    PIPELINE_DAMPING,
    PIPELINE_MAX_ITER,
    PIPELINE_PREFERENCE_PCT,
    THRESHOLD,
    ClusterComparison,
    Clustering,
    DendrogramData,
    TwoStageClustering,
    _relabel_by_size,
    affinity_labels,
    best_k_by_silhouette,
    block_order,
    cluster_labels,
    compare_labelings,
    dendrogram_segments,
    exemplars,
    kmedoids_labels,
    linkage_matrix,
    silhouette,
    symmetrise,
    threshold_for_clusters,
    two_stage_labels,
)

__all__ = [
    "AFFINITY",
    "ALL_METHODS",
    "COUNT",
    "CRITERIA",
    "KMEDOIDS",
    "LINKAGES",
    "METHODS",
    "PIPELINE_DAMPING",
    "PIPELINE_MAX_ITER",
    "PIPELINE_PREFERENCE_PCT",
    "THRESHOLD",
    "ClusterComparison",
    "Clustering",
    "DendrogramData",
    "TwoStageClustering",
    "_relabel_by_size",
    "affinity_labels",
    "best_k_by_silhouette",
    "block_order",
    "cluster_labels",
    "compare_labelings",
    "dendrogram_segments",
    "exemplars",
    "kmedoids_labels",
    "linkage_matrix",
    "silhouette",
    "symmetrise",
    "threshold_for_clusters",
    "two_stage_labels",
]
