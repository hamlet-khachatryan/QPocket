"""Interactive two-stage clustering: slider ranges, live regrouping and dendrograms."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from qpocket.core.clustering import (dendrogram_segments, linkage_matrix, percentile_height, sorted_pairs,
                                     symmetrise, threshold_for_clusters, to_percentile)
from qpocket.pipelines.clustering import cluster_pockets
from qpocket.viz.style import GREY, apply


class RunMatrices(NamedTuple):
    """A run's three distance matrices in one row order.

    Attributes:
        ids: Pocket IDs.
        shape: Shape distances.
        chem: Physicochemical distances, None when the pockets were never featurised.
        combined: Blended distances; the shape distances when chem is None.
    """

    ids: List[str]
    shape: np.ndarray
    chem: Optional[np.ndarray]
    combined: Optional[np.ndarray]


def load_matrices(run_dir: Path) -> Optional[RunMatrices]:
    """Read shape, chemistry and combined matrices from metadata/ (the shape matrix alone for an unfeaturised site).

    Args:
        run_dir: Run root.

    Returns:
        The matrices, or None if the run has none.
    """
    meta = Path(run_dir) / "metadata"
    names = ("shape_distance_matrix", "feature_distance_matrix", "spectral_distance_matrix")
    if (meta / f"{names[0]}.csv").exists() and not (meta / f"{names[1]}.csv").exists():
        shape = pd.read_csv(meta / f"{names[0]}.csv", index_col=0)
        return RunMatrices([str(i) for i in shape.index], shape.to_numpy(float), None, shape.to_numpy(float))
    if not all((meta / f"{n}.csv").exists() for n in names):
        return None
    combined = pd.read_csv(meta / "spectral_distance_matrix.csv", index_col=0)
    order = combined.index
    shape, chem = (pd.read_csv(meta / f"{n}.csv", index_col=0).loc[order, order] for n in names[:2])
    return RunMatrices([str(i) for i in order], shape.to_numpy(float), chem.to_numpy(float),
                       combined.to_numpy(float))


def height_range(distances: np.ndarray, linkage: str = "average") -> Tuple[float, float]:
    """Lowest and highest merge height of a hierarchical tree, for a cut slider.

    Args:
        distances: Square distance matrix.
        linkage: Linkage method.

    Returns:
        (min height, max height).
    """
    heights = linkage_matrix(symmetrise(distances), linkage)[:, 2]
    return float(heights.min()), float(heights.max())


def height_for_count(distances: np.ndarray, n_clusters: int, linkage: str = "average") -> float:
    """Cut height that yields a given number of clusters.

    Args:
        distances: Square distance matrix.
        n_clusters: Wanted count.
        linkage: Linkage method.

    Returns:
        The height.
    """
    return float(threshold_for_clusters(linkage_matrix(symmetrise(distances), linkage), n_clusters))


def chem_height_range(m: RunMatrices, shape_labels: np.ndarray, linkage: str = "average") -> Tuple[float, float]:
    """Merge-height range pooled over the chemistry trees of every shape cluster.

    Args:
        m: The run's matrices.
        shape_labels: Stage-one label per pocket.
        linkage: Linkage method.

    Returns:
        (min height, max height); (0, 1) when no cluster can be split.
    """
    if m.chem is None:
        return 0.0, 1.0
    lows, highs = [], []
    for value in np.unique(shape_labels):
        members = np.flatnonzero(shape_labels == value)
        if len(members) >= 3:
            lo, hi = height_range(m.chem[np.ix_(members, members)], linkage)
            lows.append(lo)
            highs.append(hi)
    return (min(lows), max(highs)) if lows else (0.0, 1.0)


def percentile_settings(settings, shape_cut: float, chem_cut: Optional[float],
                        min_cluster_size: Optional[int] = None):
    """Clustering settings with hierarchical percentile cuts on both stages.

    Args:
        settings: Starting ClusteringParams (usually the run's own).
        shape_cut: Shape cut, a percentile of the site's shape distances.
        chem_cut: Chemistry cut, a percentile of the site's chemistry distances, or None to keep the run's chemistry stage.
        min_cluster_size: Override for the smallest shape cluster that is split.

    Returns:
        New ClusteringParams in two-stage mode.
    """
    shape = replace(settings.shape, method="hierarchical", percentile=float(shape_cut))
    chem = settings.chem if chem_cut is None else replace(settings.chem, method="hierarchical",
                                                          percentile=float(chem_cut))
    return replace(settings, mode="two_stage", shape=shape, chem=chem,
                   min_cluster_size=min_cluster_size or settings.min_cluster_size)


def shape_only_settings(settings, shape_cut: float):
    """Clustering settings that cut the shape tree at a percentile and stop there.

    Args:
        settings: Starting ClusteringParams (usually the run's own).
        shape_cut: Shape cut, a percentile of the site's shape distances.

    Returns:
        New ClusteringParams in shape-only mode.
    """
    return replace(settings, mode="shape_only",
                   shape=replace(settings.shape, method="hierarchical", percentile=float(shape_cut)))


def regroup(m: RunMatrices, settings) -> pd.DataFrame:
    """Cluster a run's matrices and tabulate it like clustering_metadata.csv.

    Args:
        m: The run's matrices.
        settings: ClusteringParams.

    Returns:
        pdb_id, cluster_id, shape_cluster_id, chem_cluster_id, is_representative, dist_to_exemplar.
    """
    a = cluster_pockets(settings, m.shape, m.chem, m.combined)
    exemplar_of: Dict[int, int] = {int(a.labels[p]): p for p in a.exemplar_positions}
    rows = []
    for i, pid in enumerate(m.ids):
        ex = exemplar_of.get(int(a.labels[i]), i)
        rows.append({"pdb_id": pid, "cluster_id": int(a.labels[i]),
                     "shape_cluster_id": int(a.shape_labels[i]), "chem_cluster_id": int(a.chem_labels[i]),
                     "is_representative": i == ex, "dist_to_exemplar": float(m.combined[i, ex])})
    return pd.DataFrame(rows)


def dendrogram_figure(distances: np.ndarray, ids: List[str], threshold: Optional[float],
                      title: str, linkage: str = "average", height: int = 320):
    """Plotly dendrogram with a dashed line at the cut height.

    Args:
        distances: Square distance matrix.
        ids: Leaf labels, same order.
        threshold: Cut height to draw, or None.
        title: Figure title.
        linkage: Linkage method.
        height: Figure height (px).

    Returns:
        A plotly Figure.
    """
    import plotly.graph_objects as go

    tree = dendrogram_segments(linkage_matrix(symmetrise(distances), linkage), threshold)
    fig = go.Figure()
    for xs, ys in tree.segments:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#555", width=1),
                                 hoverinfo="skip", showlegend=False))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color="#d62728", dash="dash"))
    fig.update_layout(title=title, height=height, margin=dict(l=40, r=10, t=40, b=80),
                      xaxis=dict(tickmode="array", tickvals=list(tree.leaf_positions),
                                 ticktext=[ids[i] for i in tree.leaf_order], tickangle=-60),
                      yaxis=dict(title="merge height"), plot_bgcolor="white")
    return fig


def shape_groups(m: RunMatrices, percentile: float, linkage: str = "average") -> np.ndarray:
    """Stage-one labels for a shape cut.

    Args:
        m: The run's matrices.
        percentile: Cut, a percentile of the site's shape distances.
        linkage: Linkage method.

    Returns:
        Label per pocket.
    """
    from qpocket.core.clustering import PERCENTILE, cluster_labels

    return cluster_labels(m.shape, method=linkage, criterion=PERCENTILE, threshold=percentile).labels


#: Method key to the label shown in the picker.
METHODS: Dict[str, str] = {
    "affinity": "Affinity propagation on the combined score",
    "hierarchical": "One hierarchical cut on the combined score",
    "two_stage": "Two stages: shape, then chemistry",
}


def settings_for(base, method: str, combined_cut: Optional[float] = None,
                 shape_cut: Optional[float] = None, chem_cut: Optional[float] = None,
                 damping: float = 0.9, min_cluster_size: Optional[int] = None):
    """Clustering settings for one of the offered methods; every cut is a site percentile.

    Args:
        base: The run's own settings, used for linkage and anything not set here.
        method: One of METHODS.
        combined_cut: Percentile cut for the single hierarchical method.
        shape_cut: Stage-one percentile cut.
        chem_cut: Stage-two percentile cut, measured on the whole site's chemistry distances.
        damping: Affinity propagation damping.
        min_cluster_size: Smallest shape group that is still subdivided.

    Returns:
        ClusteringParams the pipeline can run and recluster can save.

    Raises:
        ValueError: If method is not one of METHODS.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")

    if method == "affinity":
        return replace(base, mode="single",
                       single=replace(base.single, method="affinity", damping=float(damping)))
    if method == "hierarchical":
        return replace(base, mode="single",
                       single=replace(base.single, method="hierarchical", percentile=float(combined_cut)))
    return replace(
        base,
        mode="two_stage",
        min_cluster_size=int(min_cluster_size if min_cluster_size is not None
                             else base.min_cluster_size),
        shape=replace(base.shape, method="hierarchical", percentile=float(shape_cut)),
        chem=replace(base.chem, method="hierarchical", percentile=float(chem_cut)),
    )


class Preview(NamedTuple):
    """What a set of settings would produce, without writing anything.

    Attributes:
        frame: pdb_id, cluster_id, shape_cluster_id, chem_cluster_id.
        n_clusters: Number of clusters.
        shape_labels: Stage-one label per pocket, in matrix order.
        labels: Final label per pocket, in matrix order.
    """

    frame: pd.DataFrame
    n_clusters: int
    shape_labels: np.ndarray
    labels: np.ndarray


def preview(m: RunMatrices, settings) -> Preview:
    """Run the pipeline's own clustering, without saving it.

    Args:
        m: The run's distance matrices.
        settings: ClusteringParams.

    Returns:
        The grouping these settings produce.
    """
    assignment = cluster_pockets(settings, m.shape, m.chem, m.combined)
    labels = np.asarray(assignment.labels, dtype=int)
    frame = pd.DataFrame({
        "pdb_id": list(m.ids),
        "cluster_id": labels,
        "shape_cluster_id": np.asarray(assignment.shape_labels, dtype=int),
        "chem_cluster_id": np.asarray(assignment.chem_labels, dtype=int),
    })
    return Preview(frame=frame, n_clusters=int(len(set(labels.tolist()))),
                   shape_labels=np.asarray(assignment.shape_labels, dtype=int),
                   labels=labels)


def segment_clusters(data, threshold: float, labels) -> List[Optional[int]]:
    """Which cluster each dendrogram branch belongs to, or None above the cut.

    Args:
        data: Segments from dendrogram_segments.
        threshold: The cut height.
        labels: Cluster per pocket, in the matrix's own order.

    Returns:
        One entry per segment: a cluster id, or None for a branch above the cut.
    """
    labels = np.asarray(labels, dtype=int)
    positions = np.asarray(data.leaf_positions, dtype=float)
    order = list(data.leaf_order)

    out: List[Optional[int]] = []
    for xs, ys in data.segments:
        if max(ys) >= threshold:
            out.append(None)
            continue
        # Below the cut every leaf under a branch shares one cluster, so the
        # first leaf inside the branch's span answers for all of them.
        lo, hi = min(xs), max(xs)
        inside = np.flatnonzero((positions >= lo - 1e-9) & (positions <= hi + 1e-9))
        out.append(int(labels[order[int(inside[0])]]) if len(inside) else None)
    return out


def chem_subtrees(m: RunMatrices, shape_labels, linkage: str = "average"):
    """One chemistry dendrogram per shape group, in group order.

    Args:
        m: The run's matrices.
        shape_labels: Stage-one label per pocket, in matrix order.
        linkage: Linkage method.

    Returns:
        ``(group_id, members, data_or_None)`` per shape group, members in matrix order.
    """
    shape_labels = np.asarray(shape_labels, dtype=int)
    out = []
    for group in sorted(set(shape_labels.tolist())):
        members = np.flatnonzero(shape_labels == group)
        if len(members) < 2:
            out.append((int(group), members, None))
            continue
        sub = symmetrise(m.chem[np.ix_(members, members)])
        out.append((int(group), members, dendrogram_segments(linkage_matrix(sub, linkage))))
    return out


def mirrored_figure(m: RunMatrices, preview_result, shape_cut: float,
                    chem_cut: Optional[float], linkage: str = "average",
                    height: int = 520):
    """Shape tree and the chemistry trees of its groups, facing each other.

    Args:
        m: The run's matrices.
        preview_result: What preview() returned for the current settings.
        shape_cut: Stage-one cut height.
        chem_cut: Stage-two cut height, or None to draw no chemistry line.
        linkage: Linkage method.
        height: Figure height (px).

    Returns:
        A plotly Figure with the two panels and the shared leaf labels.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from qpocket.viz.ramps import cluster_colour

    grey = "#9a9a97"
    labels = np.asarray(preview_result.labels, dtype=int)
    shape_data = dendrogram_segments(linkage_matrix(symmetrise(m.shape), linkage), shape_cut)
    shape_colours = segment_clusters(shape_data, shape_cut, labels)

    figure = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02,
                           subplot_titles=("shape", "chemistry, within each group"))

    # Left panel: the shape tree, laid on its side so leaves run down the middle.
    for (xs, ys), cluster in zip(shape_data.segments, shape_colours):
        figure.add_trace(
            go.Scatter(x=list(ys), y=list(xs), mode="lines", hoverinfo="skip",
                       showlegend=False,
                       line=dict(color=grey if cluster is None else cluster_colour(cluster),
                                 width=1.2 if cluster is None else 2.0)),
            row=1, col=1)
    figure.add_vline(x=shape_cut, line=dict(color="#c0392b", dash="dash", width=1),
                     row=1, col=1)

    # Right panel: each group's chemistry tree, stacked against the same axis.
    offset = 0.0
    for _group, members, data in chem_subtrees(m, preview_result.shape_labels, linkage):
        span = max(len(members), 1) * 10.0
        if data is not None:
            colours = segment_clusters(data, chem_cut if chem_cut else 0.0,
                                       labels[members])
            for (xs, ys), cluster in zip(data.segments, colours):
                figure.add_trace(
                    go.Scatter(x=list(ys), y=[v + offset for v in xs], mode="lines",
                               hoverinfo="skip", showlegend=False,
                               line=dict(color=grey if cluster is None else cluster_colour(cluster),
                                         width=1.2 if cluster is None else 2.0)),
                    row=1, col=2)
        offset += span
    if chem_cut:
        figure.add_vline(x=chem_cut, line=dict(color="#c0392b", dash="dash", width=1),
                         row=1, col=2)

    ticks = [str(m.ids[i]) for i in shape_data.leaf_order]
    figure.update_yaxes(tickmode="array", tickvals=list(shape_data.leaf_positions),
                        ticktext=ticks, row=1, col=1)
    figure.update_xaxes(title_text="merge height", row=1, col=1)
    figure.update_xaxes(title_text="merge height", autorange="reversed", row=1, col=2)
    figure.update_layout(height=height, plot_bgcolor="white",
                         margin=dict(l=90, r=20, t=40, b=40))
    return figure


# ── collapsed views for large sites ─────────────────────────────────────────

SHAPE_CUT_COLOUR = "#d1495b"
CHEM_CUT_COLOUR = "#1f5f8b"


def _members(link: np.ndarray, node: int, n: int) -> List[int]:
    """Leaves under a node of a linkage tree."""
    out, stack = [], [node]
    while stack:
        current = stack.pop()
        if current < n:
            out.append(current)
        else:
            stack.extend(int(c) for c in link[current - n, :2])
    return out


def _shape_colour(colours: Dict, frame: pd.DataFrame, shape: int) -> str:
    """A shape cluster's own hue: that of its largest chemistry subcluster."""
    rows = frame[frame["shape_cluster_id"] == shape]
    top = int(rows["chem_cluster_id"].value_counts().idxmax()) if len(rows) else 0
    return colours.get((shape, top), GREY)


def collapsed_figure(m: RunMatrices, preview_result, shape_cut: float, chem_cut: Optional[float],
                     linkage: str, colours: Dict, height: int = 440):
    """The shape tree drawn only down to its cut, one leaf per cluster, with each cluster's chemistry tree below.

    Args:
        m: The run's matrices.
        preview_result: What preview() returned for the current settings.
        shape_cut: Shape cut, a site percentile.
        chem_cut: Chemistry cut, a site percentile, or None for shape only.
        linkage: Linkage method.
        colours: ``(shape, chem)`` colours from :func:`qpocket.viz.style.cluster_colours`.
        height: Figure height (px).

    Returns:
        A styled plotly Figure; axes read in percentiles, the leaf markers (trace "clusters") carry the cluster ids.
    """
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import dendrogram

    frame = preview_result.frame
    shape_labels = np.asarray(preview_result.shape_labels, dtype=int)
    chem_labels = frame["chem_cluster_id"].to_numpy(dtype=int)
    n = len(m.ids)
    with_chem = chem_cut is not None and m.chem is not None
    fig = go.Figure()
    link = linkage_matrix(symmetrise(m.shape), linkage) if n >= 2 else None
    groups = sorted(set(shape_labels.tolist()))
    shape_pairs = sorted_pairs(m.shape)
    lines: Dict[Tuple[str, float], Tuple[list, list]] = {}   # (colour, width) -> xs, ys with None breaks

    def _line(xs, ys, colour, width):
        bx, by = lines.setdefault((colour, width), ([], []))
        bx.extend(list(xs) + [None])
        by.extend(list(ys) + [None])

    if link is not None and len(groups) >= 2:
        tree = dendrogram(link, truncate_mode="lastp", p=len(groups), no_plot=True)
        for xs, ys in zip(tree["icoord"], tree["dcoord"]):
            _line(xs, to_percentile(None, ys, pairs=shape_pairs), GREY, 1.1)
        tops = [(np.mean(xs[1:3]), ys[1]) for xs, ys in zip(tree["icoord"], tree["dcoord"])]
        top_pct = to_percentile(None, [h for _, h in tops], pairs=shape_pairs)
        fig.add_trace(go.Scatter(
            x=[x for x, _ in tops], y=top_pct, mode="markers",
            marker=dict(size=6, color="rgba(0,0,0,0)"), showlegend=False, name="merges",
            hovertext=[f"shape distance {h:.3g} · p{p:.0f}" for (_, h), p in zip(tops, top_pct)],
            hoverinfo="text"))
        leaf_ids = []
        for node in tree["leaves"]:
            members = _members(link, int(node), n)
            leaf_ids.append(int(pd.Series(shape_labels[members]).mode().iloc[0]))
        leaf_x = [5.0 + 10.0 * i for i in range(len(leaf_ids))]
    else:
        leaf_ids, leaf_x = groups, [5.0]

    sizes = pd.Series(shape_labels).value_counts()
    leaf_colours = [_shape_colour(colours, frame, c) for c in leaf_ids]
    for x, colour in zip(leaf_x, leaf_colours):
        _line([x, x], [0, shape_cut], colour, 2.2)
    fig.add_trace(go.Scatter(
        x=leaf_x, y=[0.0] * len(leaf_x), mode="markers", name="clusters", customdata=leaf_ids,
        marker=dict(symbol="square", size=11, color=leaf_colours, line=dict(width=0)),
        hovertext=[f"S{c + 1} · {int(sizes[c])} pockets" for c in leaf_ids], hoverinfo="text", showlegend=False))
    for x, c, colour in zip(leaf_x, leaf_ids, leaf_colours):
        if colour != GREY:
            fig.add_annotation(x=x, y=0, yshift=-14 if not with_chem else 14, showarrow=False,
                               text=f"<b>S{c + 1}</b> {int(sizes[c])}", font=dict(size=9, color=colour))

    if with_chem:
        chem_height = percentile_height(m.chem, chem_cut)
        chem_pairs = sorted_pairs(m.chem)
        for x, c in zip(leaf_x, leaf_ids):
            members = np.flatnonzero(shape_labels == c)
            if len(members) < 3 or _shape_colour(colours, frame, c) == GREY:
                continue
            data = dendrogram_segments(linkage_matrix(symmetrise(m.chem[np.ix_(members, members)]), linkage))
            span = max(10.0 * (len(members) - 1), 1.0)
            for (xs, ys), sub in zip(data.segments, segment_clusters(data, chem_height, chem_labels[members])):
                colour = GREY if sub is None else colours.get((c, sub), GREY)
                _line([x - 4 + (v - 5) / span * 8 for v in xs],
                      [-v for v in to_percentile(None, ys, pairs=chem_pairs)], colour, 1.1)

    # every branch drawn as one trace per colour, so a large site stays a light figure
    for (colour, width), (xs, ys) in lines.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=colour, width=width), connectgaps=False))

    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=shape_cut, y1=shape_cut,
                  line=dict(color=SHAPE_CUT_COLOUR, width=1, dash="dash"))
    if with_chem:
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=-chem_cut, y1=-chem_cut,
                      line=dict(color=CHEM_CUT_COLOUR, width=1, dash="dash"))
    ticks = [-100, -50, 0, 50, 100] if with_chem else [0, 25, 50, 75, 100]
    title = ("Shape (up) · physicochemistry (down), distance percentile" if with_chem
             else "Shape distance (percentile)")
    fig.update_layout(height=height, showlegend=False,
                      xaxis=dict(showticklabels=False, showline=False, ticks="", range=[0, 10.0 * len(leaf_x)]),
                      yaxis=dict(title=title, tickvals=ticks, ticktext=[str(abs(t)) for t in ticks],
                                 range=[-102 if with_chem else -4, 102]))
    return apply(fig)


def subtree_figure(m: RunMatrices, member_ids, kind: str, linkage: str, height: int = 300):
    """One cluster's member-level tree, on the site's percentile scale; pocket IDs on hover only.

    Args:
        m: The run's matrices.
        member_ids: The cluster's pocket IDs.
        kind: "shape" or "chem".
        linkage: Linkage method.
        height: Figure height (px).

    Returns:
        A styled plotly Figure.
    """
    import plotly.graph_objects as go

    full = m.shape if kind == "shape" or m.chem is None else m.chem
    index = [m.ids.index(i) for i in member_ids]
    fig = go.Figure()
    if len(index) >= 2:
        data = dendrogram_segments(linkage_matrix(symmetrise(full[np.ix_(index, index)]), linkage))
        for xs, ys in data.segments:
            fig.add_trace(go.Scatter(x=xs, y=to_percentile(full, ys), mode="lines", hoverinfo="skip",
                                     showlegend=False, line=dict(color="#555", width=1)))
        fig.add_trace(go.Scatter(x=list(data.leaf_positions), y=[0.0] * len(index), mode="markers",
                                 marker=dict(size=4, color="#555"), showlegend=False,
                                 hovertext=[member_ids[i] for i in data.leaf_order], hoverinfo="text"))
    fig.update_layout(height=height, xaxis=dict(showticklabels=False, ticks="", showline=False),
                      yaxis=dict(title=f"{'Shape' if full is m.shape else 'Physicochemical'} distance (percentile)",
                                 range=[-3, 102]))
    return apply(fig)
