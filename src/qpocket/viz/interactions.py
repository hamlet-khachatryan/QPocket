"""Per-cluster interaction fingerprints and cluster-pair exclusivity."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qpocket.core.interactions import (
    ENGAGED_FRAC,
    MIN_TESTABLE,
    Profile,
    exclusivity,
    interaction_type_shift,
    observed_frequencies,
    potential_frequencies,
    profile,
)
from qpocket.viz.pockets import overlap_matrix

#: Entries the detector writes in place of a residue when it could not run.
SENTINELS = frozenset({"no_ligand_found", "file_not_found", "analysis_error"})

#: Fixed hue per interaction type, so a type keeps its colour across runs and is
#: never recoloured by how many types happen to be present in one run.
ITYPE_COLORS: Dict[str, str] = {
    "H-bond Donor": "#2a78d6",
    "H-bond Acceptor": "#eb6834",
    "Hydrophobic": "#1baf7a",
    "Pi-stacking (P)": "#eda100",
    "Pi-stacking (T)": "#e87ba4",
    "Saltbridge (+)": "#008300",
    "Saltbridge (-)": "#4a3aa7",
    "Halogen Bond": "#e34948",
}

#: Label and hue for the types with no reserved colour, folded together rather
#: than given invented hues that would clash with the reserved ones.
OTHER = "Other"
OTHER_COLOR = "#6b6b66"


def strip_sentinels(entries: Mapping[str, Sequence[str]]) -> Dict[str, Tuple[str, ...]]:
    """Drop the detector's error markers, keeping every structure as a key.

    Args:
        entries: Structure ID to raw archive entries.

    Returns:
        Structure ID to real entries, empty for a structure that had only markers.
    """
    return {
        pid: tuple(e for e in rows if e.split("|", 1)[0] not in SENTINELS)
        for pid, rows in entries.items()
    }


def residue_axis(
    observed: Mapping[str, Sequence[str]], potential: Mapping[str, Sequence[str]]
) -> Tuple[str, ...]:
    """The one residue order every chart in the notebook follows.

    Args:
        observed: Structure ID to observed entries, sentinels already removed.
        potential: Structure ID to pocket-lining residues.

    Returns:
        Residue keys, most widely contacted first, then most widely lining the pocket, then by name.
    """
    contacted: Counter = Counter()
    lining: Counter = Counter()
    for rows in observed.values():
        contacted.update({e.split("|", 1)[0] for e in rows})
    for rows in potential.values():
        lining.update(set(rows))
    names = set(contacted) | set(lining)
    # The name is the final tie-break: the counts come from iterating sets, whose
    # order varies with string hash randomisation.
    return tuple(sorted(names, key=lambda r: (-contacted[r], -lining[r], r)))


class RunInteractions(NamedTuple):
    """Both interaction archives of one run, ready to slice by cluster.

    Attributes:
        observed: Structure ID to observed entries, sentinels removed.
        potential: Structure ID to the residues lining its pocket.
        holo: Structures left with at least one observed entry.
        residues: The shared residue axis, most contacted first.
        engaged: Residues contacted in at least one structure.
        types: Interaction types in legend order, with Other last when it is needed.
        folded: Types folded into Other, for the note under the chart.
    """

    observed: Dict[str, Tuple[str, ...]]
    potential: Dict[str, Tuple[str, ...]]
    holo: Tuple[str, ...]
    residues: Tuple[str, ...]
    engaged: frozenset
    types: Tuple[str, ...]
    folded: Tuple[str, ...]


def load_run_interactions(run_dir: Path) -> Optional[RunInteractions]:
    """Read a run's two interaction archives onto one residue axis.

    Args:
        run_dir: The run directory.

    Returns:
        The archives, or None when neither file is present.
    """
    # Imported here rather than at module load, so importing qpocket.viz does not
    # pull in h5py for a notebook section that may never be opened.
    from qpocket.io.hdf5 import read_interactions

    meta = Path(run_dir) / "metadata"
    obs_path = meta / "observed_interactions.h5"
    pot_path = meta / "potential_interactions.h5"
    if not obs_path.exists() and not pot_path.exists():
        return None

    observed = strip_sentinels(read_interactions(obs_path)) if obs_path.exists() else {}
    potential = strip_sentinels(read_interactions(pot_path)) if pot_path.exists() else {}

    present = sorted({e.split("|", 1)[1] for rows in observed.values() for e in rows if "|" in e})
    folded = tuple(t for t in present if t not in ITYPE_COLORS)
    reserved = tuple(t for t in ITYPE_COLORS if t in present)

    return RunInteractions(
        observed=observed,
        potential=potential,
        holo=tuple(pid for pid, rows in observed.items() if rows),
        residues=residue_axis(observed, potential),
        engaged=frozenset(e.split("|", 1)[0] for rows in observed.values() for e in rows),
        types=reserved + ((OTHER,) if folded else ()),
        folded=folded,
    )


class ClusterFrequencies(NamedTuple):
    """One selection's interaction fingerprint, on the run's shared residue axis.

    Attributes:
        label: What the selection is, for the chart title.
        residues: Residue axis, most contacted first.
        observed: Residue by type percentages, over the ligand-bound members.
        potential: Percentage of all members whose pocket the residue lines.
        contacted: Percentage of ligand-bound members contacting the residue.
        n_members: Structures in the selection.
        n_holo: Those of them carrying a ligand.
    """

    label: str
    residues: Tuple[str, ...]
    observed: pd.DataFrame
    potential: pd.Series
    contacted: pd.Series
    n_members: int
    n_holo: int


def cluster_frequencies(
    run: RunInteractions, members: Sequence[str], label: str = "all clusters"
) -> ClusterFrequencies:
    """Interaction frequencies for one selection of structures.

    Args:
        run: The run's archives.
        members: PDB IDs in the selection; unknown IDs are ignored.
        label: What the selection is, for the chart title.

    Returns:
        The fingerprint, on the run's full residue axis with absent residues at zero.
    """
    axis = list(run.residues)
    known = [m for m in members if m in run.observed or m in run.potential]
    holo = [m for m in known if run.observed.get(m)]

    observed = pd.DataFrame(0.0, index=axis, columns=list(run.types))
    if holo:
        table = observed_frequencies({m: list(run.observed[m]) for m in holo})
        if table is not None and not table.empty:
            table = table.set_index("amino")
            for column in table.columns:
                # A type with no reserved hue joins Other rather than inventing one.
                target = column if column in observed.columns else OTHER
                observed[target] = observed[target] + table[column].reindex(axis).fillna(0.0)

    potential = pd.Series(0.0, index=axis)
    lining = {m: list(run.potential.get(m, ())) for m in known}
    # potential_frequencies sorts a frame it builds from the residues it saw, so an
    # all-empty input would leave it sorting columns that do not exist.
    if any(lining.values()):
        table = potential_frequencies(lining)
        if table is not None and not table.empty:
            potential = table.set_index("Residue")["Frequency (%)"].reindex(axis).fillna(0.0)

    return ClusterFrequencies(
        label=label,
        residues=tuple(axis),
        observed=observed,
        potential=potential,
        contacted=observed.sum(axis=1),
        n_members=len(known),
        n_holo=len(holo),
    )


def filter_residues(
    freqs: ClusterFrequencies, min_percent: float = 0.0, only_unused: bool = False
) -> ClusterFrequencies:
    """Trim the residue axis without disturbing its order.

    Args:
        freqs: A fingerprint from :func:`cluster_frequencies`.
        min_percent: Drop residues below this in both contacted and lining percentage.
        only_unused: Keep only residues this selection never contacts.

    Returns:
        The same fingerprint over fewer residues, counts unchanged.
    """
    keep = [
        r
        for r in freqs.residues
        if (not only_unused or freqs.contacted[r] <= 0.0)
        and max(float(freqs.contacted[r]), float(freqs.potential[r])) >= min_percent
    ]
    return freqs._replace(
        residues=tuple(keep),
        observed=freqs.observed.loc[keep],
        potential=freqs.potential.loc[keep],
        contacted=freqs.contacted.loc[keep],
    )


#: Tones for the potential panel: the pocket lining behind, the contacted share in front.
LINING_COLOR = "#c9d3de"
CONTACTED_COLOR = "#2a78d6"

#: Chart height per residue row, and the room a title and legend need.
ROW_PX = 24
CHROME_PX = 120


def _figure_height(n_rows: int, height_per_row: int) -> int:
    """Chart height that fits the rows without collapsing on a short axis."""
    return max(420, height_per_row * n_rows + CHROME_PX)


def observed_figure(freqs: ClusterFrequencies, height_per_row: int = ROW_PX):
    """Stacked bars of how often each residue makes each kind of contact.

    Args:
        freqs: A fingerprint from :func:`cluster_frequencies`.
        height_per_row: Pixels of chart height per residue.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    order = list(reversed(freqs.residues))
    figure = go.Figure()
    for itype in freqs.observed.columns:
        column = freqs.observed[itype].reindex(order).fillna(0.0)
        if column.sum() <= 0:
            continue
        figure.add_bar(
            y=order,
            x=column,
            name=itype,
            orientation="h",
            marker=dict(color=ITYPE_COLORS.get(itype, OTHER_COLOR),
                        line=dict(color="white", width=1)),
            customdata=[[r] for r in order],
            hovertemplate=f"%{{y}}<br>{itype}: %{{x:.1f}}%<extra></extra>",
        )

    figure.update_layout(
        barmode="stack",
        bargap=0.25,
        template="plotly_white",
        height=_figure_height(len(order), height_per_row),
        title=f"Observed interactions · {freqs.label} · {freqs.n_holo} ligand-bound structures",
        xaxis=dict(title="ligand-bound structures forming the interaction (%)",
                   gridcolor="#e6e6e3", range=[0, 100]),
        yaxis=dict(title="", categoryorder="array", categoryarray=order, showgrid=False),
        # traceorder="normal": Plotly reverses the legend for horizontal stacks,
        # which would leave legend order disagreeing with the stacking order.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title="",
                    traceorder="normal"),
        margin=dict(l=10, r=20, t=90, b=10),
    )
    return figure


def potential_figure(
    freqs: ClusterFrequencies,
    height_per_row: int = ROW_PX,
    show_ticklabels: bool = False,
):
    """Grouped bars of what lines the pocket against what is actually contacted.

    Args:
        freqs: A fingerprint from :func:`cluster_frequencies`.
        height_per_row: Pixels of chart height per residue.
        show_ticklabels: Draw the residue names; off when this panel sits beside the observed one.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    order = list(reversed(freqs.residues))
    figure = go.Figure()
    # The two bars divide by different things on purpose: an apo structure has no
    # interactions but its pocket still has a lining, and dropping it would blank
    # the panel for an all-apo cluster. Both counts are named in the hover.
    for name, series, colour, denominator in (
        ("lines the pocket", freqs.potential, LINING_COLOR, f"{freqs.n_members} structures"),
        ("contacted", freqs.contacted, CONTACTED_COLOR, f"{freqs.n_holo} ligand-bound"),
    ):
        figure.add_bar(
            y=order,
            x=series.reindex(order).fillna(0.0),
            name=name,
            orientation="h",
            marker=dict(color=colour),
            customdata=[[r] for r in order],
            hovertemplate=f"%{{y}}<br>{name}: %{{x:.1f}}% of {denominator}<extra></extra>",
        )

    figure.update_layout(
        barmode="group",
        bargap=0.25,
        template="plotly_white",
        height=_figure_height(len(order), height_per_row),
        title=(f"Pocket lining · {freqs.label} · {freqs.n_members} structures, "
               f"{freqs.n_holo} ligand-bound"),
        xaxis=dict(title="structures (%)", gridcolor="#e6e6e3", range=[0, 100]),
        yaxis=dict(title="", categoryorder="array", categoryarray=order, showgrid=False,
                   showticklabels=show_ticklabels),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title="",
                    traceorder="normal"),
        margin=dict(l=10, r=20, t=90, b=10),
    )
    return figure


#: Jmol colours for the elements a side chain can show, carbon excluded. 3Dmol
#: applies this map by element, so a class colour never repaints a heteroatom.
ELEMENT_COLORS: Dict[str, str] = {
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "SE": "#FFA100",
    "P": "#FF8000",
    "F": "#90E050",
    "CL": "#1FF01F",
    "BR": "#A62929",
    "I": "#940094",
    "H": "#FFFFFF",
}

#: Carbon colour for a highlighted residue that is contacted in this run.
OBSERVED_COLOR = "#eb6834"

#: Carbon colour for a highlighted residue that only ever lines the pocket.
POTENTIAL_COLOR = "#7b5cd6"


def carbon_scheme(carbon: str) -> Dict[str, object]:
    """A 3Dmol colour scheme that repaints carbon and leaves CPK heteroatoms alone.

    Args:
        carbon: Hex colour for carbon.

    Returns:
        A scheme 3Dmol applies by element.
    """
    return {"prop": "elem", "map": {**ELEMENT_COLORS, "C": carbon}}


def classify_residues(
    run: RunInteractions, picked: Sequence[str]
) -> Dict[str, Tuple[str, ...]]:
    """Split picked residues into those contacted in this run and those merely lining it.

    Args:
        run: The run's archives.
        picked: Residue keys the user highlighted.

    Returns:
        Up to two groups, observed first, each in the run's axis order; empty groups are left out.
    """
    wanted = set(picked)
    observed = tuple(r for r in run.residues if r in wanted and r in run.engaged)
    potential = tuple(r for r in run.residues if r in wanted and r not in run.engaged)
    return {name: group for name, group in
            (("observed", observed), ("potential", potential)) if group}


#: At or above this on both axes, and with no type conflict, one ligand can aim
#: at both states.
CO_TARGETABLE = 0.5

#: Below this on either axis, the two states have too little in common to share a
#: ligand.
EXCLUSIVE = 0.2


class PairVerdict(NamedTuple):
    """Whether one ligand could be aimed at two pocket states at once.

    Attributes:
        a: Lower cluster id.
        b: Higher cluster id.
        residue_jaccard: Overlap of the engaged residue sets.
        shape_jaccard: Mean voxel overlap between the two clusters' clouds, nan when unknown.
        conflicts: Shared residues whose interaction type shifts significantly.
        shared: Residues both clusters engage.
        a_only: Residues only cluster A engages.
        b_only: Residues only cluster B engages.
        n_a: Ligand-bound structures in cluster A.
        n_b: Ligand-bound structures in cluster B.
        underpowered: Either cluster has fewer than MIN_TESTABLE members.
        verdict: One of co-targetable, partly exclusive, exclusive, no interactions, too few structures.
    """

    a: int
    b: int
    residue_jaccard: float
    shape_jaccard: float
    conflicts: Tuple[str, ...]
    shared: Tuple[str, ...]
    a_only: Tuple[str, ...]
    b_only: Tuple[str, ...]
    n_a: int
    n_b: int
    underpowered: bool
    verdict: str


def observed_profile(run: RunInteractions) -> Profile:
    """Engagement profile over the run's ligand-bound structures.

    Args:
        run: The run's archives.

    Returns:
        A profile carrying interaction-type counts.
    """
    return profile({pid: list(run.observed[pid]) for pid in run.holo})


def shape_overlaps(
    clouds: Mapping[str, np.ndarray],
    clusters: Mapping[int, Sequence[str]],
    spacing: float,
) -> Dict[Tuple[int, int], float]:
    """Mean voxel overlap between the point clouds of every pair of clusters.

    Args:
        clouds: PDB ID to its pocket point cloud.
        clusters: Cluster id to the PDB IDs it contains.
        spacing: Lattice spacing shared by the clouds.

    Returns:
        Cluster pair to the mean overlap, nan when either side has no cloud.
    """
    ids = list(clouds)
    matrix = overlap_matrix([clouds[i] for i in ids], spacing)
    position = {pid: i for i, pid in enumerate(ids)}

    out: Dict[Tuple[int, int], float] = {}
    for a, b in combinations(sorted(clusters), 2):
        rows = [position[p] for p in clusters[a] if p in position]
        cols = [position[p] for p in clusters[b] if p in position]
        # The off-diagonal block only: a member is never compared with itself, and
        # a representative pair would hang the answer on one structure.
        out[(a, b)] = (float(matrix[np.ix_(rows, cols)].mean())
                       if rows and cols else float("nan"))
    return out


def _verdict(
    residue_jaccard: float, shape_jaccard: float, conflicts: Sequence[str], underpowered: bool
) -> str:
    """The label a pair earns, with too little evidence checked before anything else."""
    if underpowered:
        return "too few structures"
    if math.isnan(residue_jaccard):
        return "no interactions"
    known_shape = not math.isnan(shape_jaccard)
    if residue_jaccard < EXCLUSIVE or (known_shape and shape_jaccard < EXCLUSIVE):
        return "exclusive"
    if (residue_jaccard >= CO_TARGETABLE
            and (not known_shape or shape_jaccard >= CO_TARGETABLE)
            and not conflicts):
        return "co-targetable"
    return "partly exclusive"


def pair_verdicts(
    prof: Profile,
    clusters: Mapping[int, Sequence[str]],
    shape: Optional[Mapping[Tuple[int, int], float]] = None,
    engaged_frac: float = ENGAGED_FRAC,
    alpha: float = 0.05,
) -> List[PairVerdict]:
    """Whether each pair of clusters could be targeted by one ligand.

    Args:
        prof: An engagement profile from :func:`observed_profile`.
        clusters: Cluster id to the PDB IDs it contains.
        shape: Cluster pair to voxel overlap from :func:`shape_overlaps`, omitted when the run has no lattice.
        engaged_frac: Fraction of a cluster that must show a residue for it to count as engaged.
        alpha: FDR threshold for calling a type shift a conflict.

    Returns:
        One verdict per unordered pair, in ascending cluster order.
    """
    overlaps = dict(shape or {})
    out: List[PairVerdict] = []
    for a, b in combinations(sorted(clusters), 2):
        overlap = exclusivity(prof, clusters[a], clusters[b], engaged_frac)
        conflicts = tuple(
            test.residue
            for test in interaction_type_shift(prof, clusters[a], clusters[b], alpha)
            if test.significant
        )
        shape_jaccard = float(overlaps.get((a, b), float("nan")))
        out.append(
            PairVerdict(
                a=a,
                b=b,
                residue_jaccard=overlap.jaccard,
                shape_jaccard=shape_jaccard,
                conflicts=conflicts,
                shared=overlap.shared,
                a_only=overlap.a_only,
                b_only=overlap.b_only,
                n_a=overlap.n_structures_a,
                n_b=overlap.n_structures_b,
                underpowered=overlap.underpowered,
                verdict=_verdict(overlap.jaccard, shape_jaccard, conflicts,
                                 overlap.underpowered),
            )
        )
    return out


#: How each verdict's sentence ends, once the counts have been stated.
_ENDINGS = {
    "co-targetable": "a ligand anchored on the shared core should bind both",
    "partly exclusive": "a shared core exists, but each state keeps residues of its own",
    "exclusive": "targeting one means giving up the other",
}


def describe(verdict: PairVerdict) -> str:
    """One sentence saying whether a ligand could be aimed at both clusters.

    Args:
        verdict: A verdict from :func:`pair_verdicts`.

    Returns:
        A sentence naming both clusters, the counts behind the call, and the call itself.
    """
    pair = f"Clusters {verdict.a} and {verdict.b}"
    if verdict.verdict == "too few structures":
        return (f"{pair} cannot be compared: {verdict.n_a} and {verdict.n_b} ligand-bound"
                f" structures, fewer than the {MIN_TESTABLE} a test needs.")
    if verdict.verdict == "no interactions":
        return (f"{pair} cannot be compared: neither engages a residue in at least half"
                " its members.")

    union = len(verdict.shared) + len(verdict.a_only) + len(verdict.b_only)
    clauses = [f"share {len(verdict.shared)} of {union} engaged residues"]
    if not math.isnan(verdict.shape_jaccard):
        clauses.append(f"{verdict.shape_jaccard:.0%} of their pocket volume")
    sentence = f"{pair} {' and '.join(clauses)}"
    if verdict.conflicts:
        named = ", ".join(verdict.conflicts[:3])
        more = len(verdict.conflicts) - 3
        sentence += f", but contradict each other at {named}" + (f" +{more}" if more > 0 else "")
    return f"{sentence} — {_ENDINGS[verdict.verdict]}."


def exclusivity_figure(verdicts: Sequence[PairVerdict], cluster_ids: Sequence[int]):
    """Heatmap of how far every pair of clusters could share one ligand.

    Args:
        verdicts: Verdicts from :func:`pair_verdicts`.
        cluster_ids: Clusters to show, in the order wanted.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    labels = [str(c) for c in cluster_ids]
    position = {c: i for i, c in enumerate(cluster_ids)}
    size = len(cluster_ids)
    # None rather than 0.0 for an unmeasurable cell: a number there would read as
    # evidence of disjointness, which is the opposite of what it means.
    grid: List[List[Optional[float]]] = [[None] * size for _ in range(size)]
    text = [["" for _ in range(size)] for _ in range(size)]
    hover = [["" for _ in range(size)] for _ in range(size)]

    for i in range(size):
        text[i][i] = "—"

    for verdict in verdicts:
        i, j = position[verdict.a], position[verdict.b]
        value = None if verdict.underpowered else verdict.residue_jaccard
        shape = ("unknown" if math.isnan(verdict.shape_jaccard)
                 else f"{verdict.shape_jaccard:.2f}")
        detail = (f"clusters {verdict.a} vs {verdict.b}<br>"
                  f"residues {verdict.residue_jaccard:.2f} · shape {shape}<br>"
                  f"{len(verdict.conflicts)} type conflicts<br>"
                  f"{verdict.n_a} and {verdict.n_b} ligand-bound structures")
        for row, col in ((i, j), (j, i)):
            grid[row][col] = value
            text[row][col] = verdict.verdict
            hover[row][col] = detail

    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=labels,
            y=labels,
            text=text,
            customdata=hover,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(title="shared residues"),
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=max(360, 90 * len(labels) + 120),
        title="Can two pocket states be targeted at once?",
        xaxis=dict(title="cluster", side="top"),
        yaxis=dict(title="cluster", autorange="reversed"),
        margin=dict(l=10, r=20, t=90, b=10),
    )
    return figure


def pair_figure(tests: Sequence, verdict: PairVerdict, top: int = 25):
    """Per-residue engagement in one cluster against the other, most telling first.

    Args:
        tests: ResidueTest results from qpocket.core.interactions.residue_enrichment.
        verdict: The pair's verdict, for the cluster names and the title.
        top: Most residues to draw.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    shown = list(tests)[:top]
    # residue_enrichment returns most significant first; reversing puts that row at
    # the top of a horizontal chart.
    order = [t.residue for t in reversed(shown)]
    conflicts = set(verdict.conflicts)
    labels = [f"{r} ⚠" if r in conflicts else r for r in order]
    by_residue = {t.residue: t for t in shown}

    figure = go.Figure()
    for name, sign, colour in ((f"cluster {verdict.a}", -1.0, "#2a78d6"),
                               (f"cluster {verdict.b}", 1.0, "#eb6834")):
        attribute = "freq_a" if sign < 0 else "freq_b"
        values = [sign * getattr(by_residue[r], attribute) * 100.0 for r in order]
        figure.add_bar(
            y=labels,
            x=values,
            name=name,
            orientation="h",
            marker=dict(color=colour),
            customdata=[[r, abs(v)] for r, v in zip(order, values)],
            hovertemplate="%{customdata[0]}<br>" + name + ": %{customdata[1]:.0f}%<extra></extra>",
        )

    significant = [t.residue for t in shown if t.significant]
    figure.update_layout(
        barmode="relative",
        template="plotly_white",
        height=max(360, ROW_PX * len(order) + CHROME_PX),
        title=(f"Cluster {verdict.a} against cluster {verdict.b} · {verdict.verdict}"
               f" · {len(significant)} residues differ significantly"),
        xaxis=dict(title="structures engaging the residue (%)", gridcolor="#e6e6e3",
                   range=[-100, 100],
                   tickvals=[-100, -50, 0, 50, 100], ticktext=["100", "50", "0", "50", "100"]),
        yaxis=dict(title="", categoryorder="array", categoryarray=labels, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title="",
                    traceorder="normal"),
        margin=dict(l=10, r=20, t=90, b=10),
    )
    return figure
