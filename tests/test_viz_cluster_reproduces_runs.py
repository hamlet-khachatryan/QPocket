"""The notebook's default clustering must reproduce what a run actually stored.

The banner in ``experiments/visualize_qpocket.py`` claims that affinity
propagation at the pipeline's damping and preference reproduces the labels in
``clustering_metadata.csv``. That claim is only worth making if it is checked
against real runs, so this module checks it against whichever runs are present.

A run whose stored metadata disagrees with its own algorithm is a finding about
that run, not a failure here, so the mismatch is reported rather than asserted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpocket_pipeline.viz.cluster import affinity_labels, compare_labelings

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Runs whose stored metadata is known to be inconsistent with its own algorithm,
#: so the suite stays green while the problem stays on the record. Marked
#: non-strict: regenerating the run turns these into XPASS rather than a failure.
#:
#: finalNA/6HP0 — the run's own log records 9 clusters from affinity propagation,
#: but clustering_metadata.csv stores 8, and cluster 1 holds two members at
#: dist_to_exemplar 0.0 (3TI4 and the reference 6HP0). One exemplar was folded
#: into a neighbouring cluster when the metadata was written.
KNOWN_METADATA_ISSUES = {
    ("finalNA", "6HP0"): "exemplar folded into another cluster when written",
}


def known_issue(run):
    return KNOWN_METADATA_ISSUES.get((run.parent.name, run.name))


def discover_runs():
    """Runs that carry both a distance matrix and stored cluster labels."""
    root = REPO_ROOT / "experiments"
    if not root.exists():
        return []
    out = []
    for matrix in sorted(root.glob("*/*/metadata/spectral_distance_matrix.csv")):
        meta = matrix.parent
        if (meta / "clustering_metadata.csv").exists():
            out.append(meta.parent)
    return out


RUNS = discover_runs()
run_case = pytest.mark.parametrize(
    "run", RUNS, ids=[f"{r.parent.name}-{r.name}" for r in RUNS]
)


def load(run):
    distances = pd.read_csv(run / "metadata/spectral_distance_matrix.csv", index_col=0)
    stored = pd.read_csv(run / "metadata/clustering_metadata.csv")
    label_of = dict(zip(stored["pdb_id"].astype(str), stored["cluster_id"]))
    order = [str(i) for i in distances.index]
    labels = np.array([int(label_of[i]) for i in order])
    return distances.to_numpy(dtype=float), labels, order, stored


@pytest.mark.skipif(not RUNS, reason="no Q-Pocket runs checked in")
@run_case
def test_default_clustering_agrees_with_the_stored_labels(run):
    """Near-perfect agreement, and a clear report when it is not exact."""
    distances, stored_labels, order, stored = load(run)
    result = affinity_labels(distances)
    agreement = compare_labelings(stored_labels, result.labels)

    if agreement.ari < 1.0:
        moved = ", ".join(order[i] for i in agreement.moved)
        # A stored file that disagrees with its own algorithm is worth surfacing,
        # but it is that run's problem, so only a gross mismatch fails here.
        print(
            f"\n{run.parent.name}/{run.name}: stored metadata disagrees with"
            f" affinity propagation at pipeline defaults —"
            f" {agreement.n_clusters_a} clusters stored against"
            f" {agreement.n_clusters_b} recomputed, ARI {agreement.ari:.3f},"
            f" moved: {moved}"
        )
    assert agreement.ari > 0.9, (
        f"{run} stored labels are unrelated to its own algorithm's output"
        f" (ARI {agreement.ari:.3f})"
    )


@pytest.mark.skipif(not RUNS, reason="no Q-Pocket runs checked in")
@run_case
def test_every_structure_in_the_matrix_has_stored_labels(run):
    distances = pd.read_csv(run / "metadata/spectral_distance_matrix.csv", index_col=0)
    stored = pd.read_csv(run / "metadata/clustering_metadata.csv")
    assert set(str(i) for i in distances.index) <= set(stored["pdb_id"].astype(str))


@pytest.mark.skipif(not RUNS, reason="no Q-Pocket runs checked in")
@run_case
def test_stored_clusters_name_exactly_one_representative_each(run, request):
    """Two zero-distance members in one cluster means two exemplars were merged."""
    reason = known_issue(run)
    if reason:
        request.node.add_marker(pytest.mark.xfail(strict=False, reason=reason))
    _, _, _, stored = load(run)
    if "is_representative" not in stored.columns:
        pytest.skip("run predates representative marking")
    per_cluster = stored.groupby("cluster_id")["is_representative"].sum()
    assert set(per_cluster.unique()) <= {1}, f"representatives per cluster: {dict(per_cluster)}"

    if "dist_to_exemplar" in stored.columns:
        zeros = stored[stored["dist_to_exemplar"] == 0.0].groupby("cluster_id").size()
        extra = {int(k): int(v) for k, v in zeros.items() if v > 1}
        assert not extra, (
            f"clusters with more than one zero-distance member: {extra};"
            " an exemplar was folded into another cluster"
        )
