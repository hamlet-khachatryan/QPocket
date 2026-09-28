"""Per-cluster interaction fingerprints and cluster-pair exclusivity."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

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
