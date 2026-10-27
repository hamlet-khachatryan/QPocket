"""The cluster table: size, holo share, tightness and shared cavity volume per cluster."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qpocket.core.clustering import pair_distances, to_percentile
from qpocket.core.sites import grid_spacing

TABLE_COLUMNS = ["cluster", "level", "parent", "n", "holo", "shape_within", "chem_within",
                 "shared_volume", "representative"]


def shared_volume(a: np.ndarray, b: np.ndarray, spacing: float) -> float:
    """Overlap of two pockets' cavity points in the common frame: |A ∩ B| / |A ∪ B| on the grid.

    Args:
        a: ``(n, 3)`` cavity points of one pocket.
        b: ``(m, 3)`` cavity points of the other.
        spacing: Grid spacing (Å) the points are rounded to.

    Returns:
        A value in [0, 1]; 0 when both are empty.
    """
    cells_a = {tuple(p) for p in np.round(np.asarray(a, float) / spacing).astype(int)}
    cells_b = {tuple(p) for p in np.round(np.asarray(b, float) / spacing).astype(int)}
    union = cells_a | cells_b
    return len(cells_a & cells_b) / len(union) if union else 0.0


def _within(full: Optional[np.ndarray], index: List[int]) -> float:
    """Median within-cluster distance, as a percentile of the site's pairs."""
    if full is None or len(index) < 2:
        return float("nan")
    return float(to_percentile(full, [np.median(pair_distances(full[np.ix_(index, index)]))])[0])


def _overlap(ids: List[str], clouds: Dict[str, np.ndarray], spacing: float, max_pairs: int,
             rng: np.random.Generator) -> float:
    """Median shared volume over member pairs, sampled to at most ``max_pairs``."""
    present = [i for i in ids if i in clouds]
    pairs = list(combinations(present, 2))
    if not pairs:
        return float("nan")
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]
    return float(np.median([shared_volume(clouds[a], clouds[b], spacing) for a, b in pairs]))


def _representative(rows: pd.DataFrame, shape: np.ndarray, index: List[int]) -> str:
    """The cluster's marked representative, else its shape medoid."""
    if "is_representative" in rows and rows["is_representative"].astype(bool).any():
        return str(rows.loc[rows["is_representative"].astype(bool), "pdb_id"].iloc[0])
    return str(rows["pdb_id"].iloc[int(np.argmin(shape[np.ix_(index, index)].sum(axis=1)))])


def _row(name: str, level: str, parent: str, rows: pd.DataFrame, index: List[int], m,
         clouds: Dict[str, np.ndarray], spacing: float, max_pairs: int, rng: np.random.Generator) -> dict:
    return {"cluster": name, "level": level, "parent": parent, "n": len(rows),
            "holo": float((rows["protein_state"] == "BOUND_STATE").mean()),
            "shape_within": _within(m.shape, index), "chem_within": _within(m.chem, index),
            "shared_volume": _overlap(list(rows["pdb_id"]), clouds, spacing, max_pairs, rng),
            "representative": _representative(rows, m.shape, index)}


def cluster_table(info: pd.DataFrame, m, clouds: Dict[str, np.ndarray], min_size: int,
                  max_pairs: int = 200, seed: int = 0) -> pd.DataFrame:
    """One row per shape cluster (and per chemistry subcluster), small clusters pooled into one row.

    Args:
        info: pdb_id, shape_cluster_id, chem_cluster_id (optional), protein_state, is_representative (optional).
        m: The run's matrices (RunMatrices).
        clouds: Cavity points per pdb_id; pockets without one are left out of the shared volume.
        min_size: Shape clusters smaller than this are pooled into the "small" row.
        max_pairs: Pairs sampled per cluster for the shared volume.
        seed: Seed for that sample.

    Returns:
        Rows with :data:`TABLE_COLUMNS`; "within" values are site percentiles, holo and shared volume fractions.
    """
    rng = np.random.default_rng(seed)
    position = {pid: i for i, pid in enumerate(m.ids)}
    info = info[info["pdb_id"].isin(position)].copy()
    if "chem_cluster_id" not in info:
        info["chem_cluster_id"] = 0
    spacing = grid_spacing(list(clouds.values())) or 0.7
    rows, small = [], []
    sizes = info["shape_cluster_id"].value_counts()
    for shape in sorted(sizes.index, key=lambda s: (-sizes[s], s)):
        members = info[info["shape_cluster_id"] == shape]
        if len(members) < min_size:
            small.append(members)
            continue
        name = f"S{int(shape) + 1}"
        rows.append(_row(name, "shape", "", members, [position[p] for p in members["pdb_id"]], m, clouds,
                         spacing, max_pairs, rng))
        subs = members["chem_cluster_id"].value_counts()
        if m.chem is None or len(subs) < 2:
            continue
        for chem in sorted(subs.index, key=lambda c: (-subs[c], c)):
            part = members[members["chem_cluster_id"] == chem]
            rows.append(_row(f"{name}·c{int(chem) + 1}", "chem", name, part, [position[p] for p in part["pdb_id"]],
                             m, clouds, spacing, max_pairs, rng))
    if small:
        pooled = pd.concat(small)
        rows.append({"cluster": "small", "level": "shape", "parent": "", "n": len(pooled),
                     "holo": float((pooled["protein_state"] == "BOUND_STATE").mean()),
                     "shape_within": float("nan"), "chem_within": float("nan"), "shared_volume": float("nan"),
                     "representative": f"{len(small)} clusters < {min_size}"})
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)
