"""One figure style for the explorer, and cluster colours that agree across every view."""
from __future__ import annotations

import colorsys
from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import plotly.graph_objects as go

#: Branches above a cut, small clusters, and anything that is not a cluster of its own.
GREY = "#9aa3ad"

#: Shape-cluster hues, largest cluster first; colour-blind-safe and distinct in print.
HUES: Tuple[str, ...] = ("#1f5f8b", "#2a9d8f", "#e9a03b", "#c4566a", "#6a4c93",
                         "#8ab17d", "#3d5a80", "#b07d62", "#4f772d", "#9c6644")

_AXIS = dict(showgrid=False, zeroline=False, showline=True, linecolor="#222", linewidth=0.8,
             ticks="outside", tickcolor="#222", ticklen=3, tickfont=dict(size=9, color="#555"),
             title=dict(font=dict(size=10, color="#333")))

TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family="Helvetica, Arial, sans-serif", size=10, color="#222"),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=_AXIS, yaxis=_AXIS,
    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", bordercolor="#c9ced4", font=dict(size=10, color="#222")),
    colorway=list(HUES),
))


def apply(fig: go.Figure) -> go.Figure:
    """Give a figure the explorer's style: the template and tight margins."""
    fig.update_layout(template=TEMPLATE, margin=dict(l=48, r=16, t=28, b=36))
    return fig


def _tint(hex_colour: str, step: int) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, min(0.88, l + 0.15 * step), s)
    return "#%02x%02x%02x" % tuple(round(c * 255) for c in (r, g, b))


def cluster_colours(shape_labels: Sequence[int], chem_labels: Optional[Sequence[int]],
                    min_size: int) -> Dict[Tuple[int, int], str]:
    """One colour per (shape, chemistry) cluster, the same wherever the cluster is drawn.

    Args:
        shape_labels: Shape cluster per structure.
        chem_labels: Chemistry subcluster per structure, or None for shape only.
        min_size: Clusters smaller than this are grey.

    Returns:
        Colour per ``(shape, chem)``; ``chem`` is 0 when there are no subclusters.
    """
    chem = list(chem_labels) if chem_labels is not None else [0] * len(shape_labels)
    shape_sizes = Counter(int(s) for s in shape_labels)
    sub_sizes = Counter((int(s), int(c)) for s, c in zip(shape_labels, chem))
    order = sorted(shape_sizes, key=lambda s: (-shape_sizes[s], s))
    colours: Dict[Tuple[int, int], str] = {}
    for rank, shape in enumerate(order):
        hue = HUES[rank % len(HUES)]
        subs = sorted((key for key in sub_sizes if key[0] == shape), key=lambda k: (-sub_sizes[k], k[1]))
        for step, key in enumerate(subs):
            # the largest subcluster carries its shape cluster's hue even when it is small itself
            small = shape_sizes[shape] < min_size or (step > 0 and sub_sizes[key] < min_size)
            colours[key] = GREY if small else _tint(hue, step)
    return colours
