"""Per-cluster interaction fingerprints and cluster-pair exclusivity."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from qpocket.core.interactions import observed_frequencies, potential_frequencies

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
