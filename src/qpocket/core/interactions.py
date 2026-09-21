"""Testing whether two pocket clusters engage different residues."""

# A cluster comparison invites an eyeball judgement — the bars look different, so
# the clusters must be different. They frequently are not: with eight structures on
# one side and five on the other, a residue contacted by six and two looks decisive
# and is not.
#
# Three questions are worth separating, and this module answers each with its own
# test:
#
# *Which residues differ?*
#     :func:`residue_enrichment` puts each residue in a 2x2 table of engaged versus
#     not, against cluster A versus B, and runs a two-sided Fisher exact test.
#     Fisher rather than chi-square because the counts are small by construction.
#
# *Does a shared residue differ in kind?*
#     :func:`interaction_type_shift` catches the residue both clusters touch but in
#     different ways — an H-bond donor in one, a hydrophobic contact in the other.
#     Presence/absence is blind to this, which is why both tests exist.
#
# *Are the repertoires disjoint?*
#     :func:`exclusivity` is the direct answer to "mutually exclusive": the Jaccard
#     overlap of the residue sets the two clusters engage. Near zero means disjoint.
#
# Every p-value is corrected across residues with Benjamini-Hochberg, and anything
# resting on too few structures is returned with ``underpowered`` set rather than a
# q-value near one — those two look identical on a plot and mean opposite things.
#
# Only numpy, scipy and pandas are used; files are read by :mod:`qpocket.io.hdf5`.

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

import numpy as np
from scipy.stats import chi2_contingency, false_discovery_control, fisher_exact

#: Below this many structures a cluster cannot support a test, whatever the data.
MIN_TESTABLE = 3

#: Expected cell count below which the chi-square approximation is not trusted.
MIN_EXPECTED = 5.0

#: Fraction of a cluster's structures that must show a residue for it to count as
#: engaged by that cluster, when reducing a profile to a set.
ENGAGED_FRAC = 0.5


class Profile(NamedTuple):
    """Which residues each structure engages, and how.

    Attributes:
        presence: Boolean ``(n_structures, n_residues)`` engagement matrix.
        structures: PDB IDs, in row order.
        residues: Residue keys (``RESNAME_RESNUM_CHAIN``), in column order.
        type_counts: ``(n_structures, n_residues, n_types)`` interaction counts, or ``None`` for a potential-interaction file, which records no types.
        types: Interaction type names in the last axis' order, empty when ``type_counts`` is ``None``.
    """

    presence: np.ndarray
    structures: Tuple[str, ...]
    residues: Tuple[str, ...]
    type_counts: Optional[np.ndarray]
    types: Tuple[str, ...]


class ResidueTest(NamedTuple):
    """One residue's differential engagement between two clusters.

    Attributes:
        residue: Residue key.
        n_a: Structures in cluster A engaging it.
        n_b: Structures in cluster B engaging it.
        freq_a: Fraction of cluster A engaging it, in ``[0, 1]``.
        freq_b: Fraction of cluster B engaging it.
        difference: ``freq_a - freq_b``.
        odds_ratio: Fisher odds ratio, ``inf`` when B never engages it.
        p_value: Two-sided Fisher exact p-value.
        q_value: Benjamini-Hochberg adjusted p-value across all residues tested.
        significant: ``q_value <= alpha`` and not underpowered.
        underpowered: Either cluster has fewer than :data:`MIN_TESTABLE` members.
    """

    residue: str
    n_a: int
    n_b: int
    freq_a: float
    freq_b: float
    difference: float
    odds_ratio: float
    p_value: float
    q_value: float
    significant: bool
    underpowered: bool


class TypeShiftTest(NamedTuple):
    """One residue's change of interaction type between two clusters.

    Attributes:
        residue: Residue key.
        types: Interaction types present for this residue in either cluster.
        counts_a: Per-type counts in cluster A, aligned with ``types``.
        counts_b: Per-type counts in cluster B.
        statistic: Chi-square statistic, ``nan`` when the test could not run.
        p_value: Chi-square p-value, ``nan`` when the test could not run.
        q_value: Benjamini-Hochberg adjusted p-value across residues tested.
        significant: ``q_value <= alpha`` and not underpowered.
        underpowered: Too few structures, fewer than two types, or an expected cell count below :data:`MIN_EXPECTED`.
    """

    residue: str
    types: Tuple[str, ...]
    counts_a: Tuple[int, ...]
    counts_b: Tuple[int, ...]
    statistic: float
    p_value: float
    q_value: float
    significant: bool
    underpowered: bool


class Exclusivity(NamedTuple):
    """How far two clusters' engaged-residue sets overlap.

    Attributes:
        jaccard: ``|A and B| / |A or B|``.
        shared: Residues engaged by both.
        a_only: Residues engaged only by cluster A.
        b_only: Residues engaged only by cluster B.
        n_structures_a: Members of cluster A.
        n_structures_b: Members of cluster B.
        underpowered: Either cluster has fewer than :data:`MIN_TESTABLE` members.
    """

    jaccard: float
    shared: Tuple[str, ...]
    a_only: Tuple[str, ...]
    b_only: Tuple[str, ...]
    n_structures_a: int
    n_structures_b: int
    underpowered: bool


def profile(entries: Dict[str, List[str]]) -> Profile:
    """Engagement matrix from decoded interaction entries.

    Args:
        entries: Structure ID to entries, in the row order wanted.

    Returns:
        The profile, with sorted residue and type axes.
    """
    structures = tuple(entries.keys())

    per_structure: Dict[str, Dict[str, Dict[str, int]]] = {}
    residues: set = set()
    types: set = set()

    for pid, rows in entries.items():
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for entry in rows:
            # maxsplit=1: a subtype such as "Pi-stacking (T)" may contain a "|".
            residue, _, itype = entry.partition("|")
            residues.add(residue)
            if itype:
                types.add(itype)
                counts[residue][itype] += 1
            else:
                counts[residue]  # touch, so presence registers with no type
        per_structure[pid] = counts

    residue_axis = tuple(sorted(residues))
    type_axis = tuple(sorted(types))
    r_index = {r: i for i, r in enumerate(residue_axis)}
    t_index = {t: i for i, t in enumerate(type_axis)}

    presence = np.zeros((len(structures), len(residue_axis)), dtype=bool)
    counts_arr = (
        np.zeros((len(structures), len(residue_axis), len(type_axis)), dtype=int)
        if type_axis
        else None
    )

    for row, pid in enumerate(structures):
        for residue, by_type in per_structure[pid].items():
            presence[row, r_index[residue]] = True
            if counts_arr is not None:
                for itype, count in by_type.items():
                    counts_arr[row, r_index[residue], t_index[itype]] = count

    return Profile(
        presence=presence,
        structures=structures,
        residues=residue_axis,
        type_counts=counts_arr,
        types=type_axis,
    )


def _split_rows(
    profile: Profile, members_a: Sequence[str], members_b: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Row indices for two clusters, keeping only structures the profile holds."""
    position = {pid: i for i, pid in enumerate(profile.structures)}
    rows_a = np.array([position[p] for p in members_a if p in position], dtype=int)
    rows_b = np.array([position[p] for p in members_b if p in position], dtype=int)
    return rows_a, rows_b


def _adjust(p_values: List[float]) -> np.ndarray:
    """Benjamini-Hochberg over the p-values that exist, tolerating an empty list."""
    if not p_values:
        return np.empty(0, dtype=float)
    return np.asarray(false_discovery_control(np.asarray(p_values), method="bh"))


def residue_enrichment(
    profile: Profile,
    members_a: Sequence[str],
    members_b: Sequence[str],
    alpha: float = 0.05,
) -> List[ResidueTest]:
    """Which residues two clusters engage at different rates.

    Args:
        profile: An engagement profile from :func:`build_profile`.
        members_a: PDB IDs in cluster A.
        members_b: PDB IDs in cluster B.
        alpha: FDR threshold applied to the q-values.

    Returns:
        One result per tested residue, most significant first, ties broken by the larger absolute frequency difference.
    """
    rows_a, rows_b = _split_rows(profile, members_a, members_b)
    n_a, n_b = len(rows_a), len(rows_b)
    underpowered = n_a < MIN_TESTABLE or n_b < MIN_TESTABLE

    if n_a == 0 or n_b == 0:
        return []

    hits_a = profile.presence[rows_a].sum(axis=0)
    hits_b = profile.presence[rows_b].sum(axis=0)
    tested = np.flatnonzero(hits_a + hits_b)

    records = []
    p_values: List[float] = []
    for col in tested:
        a, b = int(hits_a[col]), int(hits_b[col])
        table = [[a, n_a - a], [b, n_b - b]]
        odds, p = fisher_exact(table, alternative="two-sided")
        records.append((profile.residues[col], a, b, float(odds), float(p)))
        p_values.append(float(p))

    q_values = _adjust(p_values)

    results = [
        ResidueTest(
            residue=residue,
            n_a=a,
            n_b=b,
            freq_a=a / n_a,
            freq_b=b / n_b,
            difference=a / n_a - b / n_b,
            odds_ratio=odds,
            p_value=p,
            q_value=float(q),
            significant=bool(q <= alpha) and not underpowered,
            underpowered=underpowered,
        )
        for (residue, a, b, odds, p), q in zip(records, q_values)
    ]
    results.sort(key=lambda r: (r.q_value, -abs(r.difference)))
    return results


def interaction_type_shift(
    profile: Profile,
    members_a: Sequence[str],
    members_b: Sequence[str],
    alpha: float = 0.05,
) -> List[TypeShiftTest]:
    """Which residues change interaction type between two clusters.

    Args:
        profile: A profile carrying ``type_counts`` — that is, one built from an observed-interaction file.
        members_a: PDB IDs in cluster A.
        members_b: PDB IDs in cluster B.
        alpha: FDR threshold applied to the q-values.

    Returns:
        One result per tested residue, most significant first.
    """
    if profile.type_counts is None:
        return []

    rows_a, rows_b = _split_rows(profile, members_a, members_b)
    n_a, n_b = len(rows_a), len(rows_b)
    if n_a == 0 or n_b == 0:
        return []
    too_small = n_a < MIN_TESTABLE or n_b < MIN_TESTABLE

    totals_a = profile.type_counts[rows_a].sum(axis=0)
    totals_b = profile.type_counts[rows_b].sum(axis=0)
    both = np.flatnonzero(
        profile.presence[rows_a].any(axis=0) & profile.presence[rows_b].any(axis=0)
    )

    records = []
    p_values: List[float] = []
    for col in both:
        a_counts, b_counts = totals_a[col], totals_b[col]
        keep = np.flatnonzero(a_counts + b_counts)
        types = tuple(profile.types[t] for t in keep)
        a_keep = a_counts[keep].astype(int)
        b_keep = b_counts[keep].astype(int)

        weak = too_small or len(keep) < 2
        statistic = p = float("nan")
        if not weak:
            table = np.vstack([a_keep, b_keep])
            if table.sum(axis=1).min() > 0:
                statistic, p, _, expected = chi2_contingency(table)
                if expected.min() < MIN_EXPECTED:
                    weak = True
            else:
                weak = True

        records.append(
            (profile.residues[col], types, tuple(int(v) for v in a_keep),
             tuple(int(v) for v in b_keep), float(statistic), float(p), weak)
        )
        if not weak:
            p_values.append(float(p))

    q_values = iter(_adjust(p_values))
    results = []
    for residue, types, a_keep, b_keep, statistic, p, weak in records:
        q = float("nan") if weak else float(next(q_values))
        results.append(
            TypeShiftTest(
                residue=residue,
                types=types,
                counts_a=a_keep,
                counts_b=b_keep,
                statistic=statistic,
                p_value=p,
                q_value=q,
                significant=(not weak) and bool(q <= alpha),
                underpowered=weak,
            )
        )
    results.sort(key=lambda r: (np.isnan(r.q_value), r.q_value))
    return results


def engaged_set(
    profile: Profile, members: Sequence[str], engaged_frac: float = ENGAGED_FRAC
) -> Tuple[str, ...]:
    """Residues a cluster engages in at least ``engaged_frac`` of its members.

    Args:
        profile: An engagement profile.
        members: PDB IDs in the cluster.
        engaged_frac: Fraction of members that must show the residue.

    Returns:
        Residue keys, sorted.
    """
    rows, _ = _split_rows(profile, members, [])
    if len(rows) == 0:
        return ()
    frequency = profile.presence[rows].mean(axis=0)
    return tuple(
        profile.residues[c] for c in np.flatnonzero(frequency >= engaged_frac)
    )


def exclusivity(
    profile: Profile,
    members_a: Sequence[str],
    members_b: Sequence[str],
    engaged_frac: float = ENGAGED_FRAC,
) -> Exclusivity:
    """How far two clusters' engaged-residue sets overlap.

    Args:
        profile: An engagement profile.
        members_a: PDB IDs in cluster A.
        members_b: PDB IDs in cluster B.
        engaged_frac: Fraction of a cluster that must show a residue for it to count as engaged by that cluster.

    Returns:
        The overlap, with the residues on each side named.
    """
    set_a = set(engaged_set(profile, members_a, engaged_frac))
    set_b = set(engaged_set(profile, members_b, engaged_frac))
    rows_a, rows_b = _split_rows(profile, members_a, members_b)

    union = set_a | set_b
    shared = set_a & set_b
    return Exclusivity(
        jaccard=float("nan") if not union else len(shared) / len(union),
        shared=tuple(sorted(shared)),
        a_only=tuple(sorted(set_a - set_b)),
        b_only=tuple(sorted(set_b - set_a)),
        n_structures_a=len(rows_a),
        n_structures_b=len(rows_b),
        underpowered=len(rows_a) < MIN_TESTABLE or len(rows_b) < MIN_TESTABLE,
    )


def pairwise_exclusivity(
    profile: Profile,
    clusters: Dict[int, Sequence[str]],
    engaged_frac: float = ENGAGED_FRAC,
) -> Dict[Tuple[int, int], Exclusivity]:
    """Exclusivity for every unordered pair of clusters.

    Args:
        profile: An engagement profile.
        clusters: Cluster index to the PDB IDs it contains.
        engaged_frac: Passed through to :func:`exclusivity`.

    Returns:
        ``(a, b)`` with ``a < b`` to the overlap between them.
    """
    return {
        (a, b): exclusivity(profile, clusters[a], clusters[b], engaged_frac)
        for a, b in combinations(sorted(clusters), 2)
    }


def observed_frequencies(entries: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
    """Interaction-type shares per residue, weighted by how often it is engaged.

    Args:
        entries: Structure ID to observed entries (RESIDUE|Type).

    Returns:
        Residue x type table, or None when there are no structures.
    """
    total = len(entries)
    if total == 0:
        return None
    amino_counts: Dict[str, int] = defaultdict(int)
    raw_data: List[List[str]] = []
    for rows in entries.values():
        unique_res_in_struct: set = set()
        for entry in rows:
            if "|" in entry:
                # Use maxsplit=1 to handle subtypes that contain a second "|".
                res, inter = entry.split("|", 1)
                raw_data.append([res, inter])
                unique_res_in_struct.add(res)
        for res in unique_res_in_struct:
            amino_counts[res] += 1

    df = pd.DataFrame(raw_data, columns=["amino", "inter"])
    if df.empty:
        return pd.DataFrame()
    global_freqs = pd.Series({a: (c / total) * 100.0 for a, c in amino_counts.items()})
    ratios = pd.crosstab(df["amino"], df["inter"], normalize="index")
    weighted = ratios.mul(global_freqs, axis=0)
    return weighted.reset_index()


def potential_frequencies(entries: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
    """Share of structures whose pocket contains each residue.

    Args:
        entries: Structure ID to potential entries (RESIDUE).

    Returns:
        Residue, Frequency (%), Count table, or None when there are no structures.
    """
    total_structures = len(entries)
    if total_structures == 0:
        return None
    amino_counts: Dict[str, int] = defaultdict(int)
    for rows in entries.values():
        if len(rows) == 0:
            continue
        for res in set(rows):
            amino_counts[res] += 1

    data = [
        {"Residue": amino, "Frequency (%)": (count / total_structures) * 100.0, "Count": count}
        for amino, count in amino_counts.items()
    ]
    # Residue as the tie-break: the counts are accumulated by iterating a set,
    # whose order varies with string hash randomisation.
    return pd.DataFrame(data).sort_values(["Frequency (%)", "Residue"], ascending=[False, True])


def utilization(observed: Dict[str, List[str]], potential: Dict[str, List[str]]) -> pd.DataFrame:
    """Percentage of each pocket's residues the ligand engages.

    Args:
        observed: Structure ID to observed entries.
        potential: Structure ID to potential entries.

    Returns:
        pdb_id, utilization table, highest first; empty if nothing overlaps.
    """
    utilization_data = []
    for pdb_id in set(observed) & set(potential):
        pot_set = set(potential[pdb_id])
        if not pot_set:
            continue
        # Observed entries are "RESNAME_NUM|InteractionType" — strip the subtype.
        obs_set = {x.split("|", 1)[0] for x in observed[pdb_id]}
        utilized = obs_set.intersection(pot_set)
        utilization_data.append({"pdb_id": pdb_id, "utilization": (len(utilized) / len(pot_set)) * 100.0})
    if not utilization_data:
        return pd.DataFrame()
    return pd.DataFrame(utilization_data).sort_values("utilization", ascending=False)
