"""Binding-site discovery: ligands a structure holds, and sites built from cavity boxes (pure)."""
from __future__ import annotations

from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

from qpocket.constants import BUFFER_RESNAMES
from qpocket.core.types import Ligand


def element_symbol(name: str, element: str) -> str:
    """Upper-case element symbol, read from the atom name when the element is blank or unidentified.

    Args:
        name: Atom name.
        element: Element column, possibly blank.

    Returns:
        The symbol, or "" when neither gives one.
    """
    symbol = (element or "").strip().upper()
    if symbol in ("", "X"):
        symbol = next((c for c in name.upper() if c.isalpha()), "")
    return symbol


def _element(atom) -> str:
    """Element symbol of a Biopython atom."""
    return element_symbol(atom.name, atom.element)


def find_ligands(structure, pdb_id: str, allowed_cofactors: Sequence[str],
                 min_heavy_atoms: int, keep: Sequence[str] = ()) -> List[Tuple[Ligand, object]]:
    """Every non-water, carbon-containing HET residue of the first model that the run does not clean away.

    Args:
        structure: Parsed Biopython structure (first model only, so NMR ensembles do not repeat ligands).
        pdb_id: Identifier stamped on each ligand.
        allowed_cofactors: Residue names never reported.
        min_heavy_atoms: Residues with fewer non-hydrogen atoms are skipped.
        keep: Buffer/cryoprotectant names (BUFFER_RESNAMES, otherwise skipped) to report all the same.

    Returns:
        ``(Ligand, residue)`` pairs in chain, then residue, order.
    """
    cofactors = {name.upper() for name in allowed_cofactors}
    skipped = cofactors | (BUFFER_RESNAMES - {name.upper() for name in keep})
    found: List[Tuple[Ligand, object]] = []
    for chain in next(iter(structure)):
        for res in chain:
            if res.id[0] in (" ", "W") or res.get_resname().strip().upper() in skipped:
                continue
            elements = [_element(a) for a in res]
            if "C" not in elements:
                continue  # ions and inorganic species such as SO4 are never ligands
            n_heavy = sum(1 for e in elements if e not in ("H", "D"))
            if n_heavy < min_heavy_atoms:
                continue
            _, resseq, icode = res.id
            found.append((Ligand(pdb_id, chain.id, res.get_resname().strip(), resseq, icode, n_heavy), res))
    return found


def grid_spacing(points: Sequence[np.ndarray]) -> float:
    """Median nearest-neighbour distance over the given point clouds.

    Args:
        points: ``(n_i, 3)`` arrays.

    Returns:
        The spacing (Å), or 0.0 when there are fewer than two points.
    """
    # CavFind repeats points; spacing is measured between distinct ones.
    xyz = np.unique(np.round(np.vstack([p for p in points if len(p)] or [np.zeros((0, 3))]), 4), axis=0)
    if len(xyz) < 2:
        return 0.0
    return float(np.median(cKDTree(xyz).query(xyz, k=2)[0][:, 1]))


def voxel_box(points: np.ndarray, half: float) -> np.ndarray:
    """Axis-aligned box of a cavity, each point counted as a voxel.

    Args:
        points: ``(n, 3)`` cavity points.
        half: Half the grid spacing (Å), added to every face.

    Returns:
        ``(6,)`` array: lower corner, then upper corner.
    """
    return np.concatenate([points.min(axis=0) - half, points.max(axis=0) + half])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in ``a`` with every box in ``b``.

    Args:
        a: ``(n, 6)`` boxes.
        b: ``(m, 6)`` boxes.

    Returns:
        ``(n, m)`` IoU, 0 where boxes are disjoint.
    """
    a, b = np.atleast_2d(a)[:, None, :], np.atleast_2d(b)[None, :, :]
    inter = np.prod(np.clip(np.minimum(a[..., 3:], b[..., 3:]) - np.maximum(a[..., :3], b[..., :3]), 0, None), axis=-1)
    union = np.prod(a[..., 3:] - a[..., :3], axis=-1) + np.prod(b[..., 3:] - b[..., :3], axis=-1) - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def cavity_cores(boxes: np.ndarray, threshold: float, chunk: int = 512) -> List[List[int]]:
    """Groups of boxes in which every pair overlaps by at least ``threshold`` (complete linkage).

    Args:
        boxes: ``(n, 6)`` cavity boxes.
        threshold: IoU every pair inside a group reaches.
        chunk: Rows of the distance matrix computed at once.

    Returns:
        Index groups covering every box.
    """
    n = len(boxes)
    if n < 2:
        return [[0]] if n else []
    condensed = np.empty(n * (n - 1) // 2)
    pos = 0
    for start in range(0, n, chunk):
        block = 1.0 - box_iou(boxes[start:start + chunk], boxes)
        for row, i in enumerate(range(start, min(start + chunk, n))):
            condensed[pos:pos + n - i - 1] = block[row, i + 1:]
            pos += n - i - 1
    labels = fcluster(linkage(condensed, "complete"), 1.0 - threshold, criterion="distance")
    groups: dict = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return list(groups.values())


def median_box(boxes: np.ndarray) -> np.ndarray:
    """Element-wise median of boxes, so one outlying member cannot stretch a site."""
    return np.median(np.atleast_2d(boxes), axis=0)


def merge_sites(groups: List[List[int]], boxes: np.ndarray, threshold: float) -> List[List[int]]:
    """Merge groups whose median boxes overlap by at least ``threshold``, best pair first.

    Args:
        groups: Index groups, e.g. from :func:`cavity_cores`.
        boxes: ``(n, 6)`` boxes the indices refer to.
        threshold: IoU two site boxes need to merge.

    Returns:
        The merged groups, each sorted.
    """
    sites = [sorted(g) for g in groups]
    while len(sites) > 1:
        iou = box_iou(np.array([median_box(boxes[g]) for g in sites]), np.array([median_box(boxes[g]) for g in sites]))
        np.fill_diagonal(iou, 0.0)
        i, j = np.unravel_index(int(np.argmax(iou)), iou.shape)
        if iou[i, j] < threshold:
            break
        merged = sorted(sites[i] + sites[j])
        sites = [g for k, g in enumerate(sites) if k not in (i, j)] + [merged]
    return sites


def site_members(site_boxes: np.ndarray, boxes: np.ndarray, threshold: float,
                 groups: Optional[Sequence[Sequence[int]]] = None) -> List[List[int]]:
    """Per site, its own group plus every box overlapping the site box by at least ``threshold``.

    Args:
        site_boxes: ``(k, 6)`` site boxes.
        boxes: ``(n, 6)`` cavity boxes.
        threshold: IoU a cavity of another group needs with a site box.
        groups: The cavities each site was built from; they always belong to it.

    Returns:
        One sorted index list per site; a box may be in several.
    """
    own = [set(g) for g in groups] if groups is not None else [set() for _ in range(len(site_boxes))]
    if not len(site_boxes) or not len(boxes):
        return [sorted(g) for g in own]
    iou = box_iou(site_boxes, boxes)
    return [sorted(own[k] | set(np.flatnonzero(row >= threshold).tolist())) for k, row in enumerate(iou)]


def ligand_sites(coords: Sequence[np.ndarray], site_boxes: np.ndarray, fraction: float,
                 padding: float) -> List[List[int]]:
    """Per ligand, the sites whose padded box holds at least ``fraction`` of its heavy atoms.

    Args:
        coords: ``(n_i, 3)`` heavy-atom coordinates per ligand.
        site_boxes: ``(k, 6)`` site boxes.
        fraction: Share of atoms that must lie inside (bounds inclusive).
        padding: Å added to every face of each site box.

    Returns:
        One site-index list per ligand.
    """
    out = []
    for xyz in coords:
        if not len(xyz) or not len(site_boxes):
            out.append([])
            continue
        lo, hi = site_boxes[:, None, :3] - padding, site_boxes[:, None, 3:] + padding
        inside = np.all((xyz[None] >= lo) & (xyz[None] <= hi), axis=2).mean(axis=1)
        out.append(np.flatnonzero(inside >= fraction).tolist())
    return out


def site_name(rank: int, residue: str) -> str:
    """Short site name: ``S<rank>-<residue>``, e.g. S1-H21, or ``S<rank>`` without a residue."""
    return f"S{rank}-{residue}" if residue else f"S{rank}"


def match_site(recorded: Set[Tuple[str, int]], member_sets: Sequence[Set[Tuple[str, int]]],
               min_jaccard: float = 0.5) -> Optional[int]:
    """The site whose member cavities best overlap a recorded set, if they overlap enough.

    Args:
        recorded: ``(pdb_id, cavity)`` pairs an override was made on.
        member_sets: Each current site's ``(pdb_id, cavity)`` pairs.
        min_jaccard: Smallest |A ∩ B| / |A ∪ B| that still counts as the same site.

    Returns:
        Index into ``member_sets``, or None when no site overlaps enough.
    """
    best, best_score = None, 0.0
    for k, members in enumerate(member_sets):
        union = len(recorded | members)
        score = len(recorded & members) / union if union else 0.0
        if score > best_score:
            best, best_score = k, score
    return best if best_score >= min_jaccard else None


class Overridden(NamedTuple):
    """Site groups after manual overrides (see :func:`apply_overrides`).

    Attributes:
        groups: Candidate indices each site was built from.
        members: Per site, every candidate belonging to it; never recomputed from a box after an override.
        parts: Per site, the index groups whose median boxes, joined, make its box.
        dropped: Groups removed by a drop.
        report: Per override ``{op, status, parts, groups}``; status is applied or unmatched.
    """

    groups: List[List[int]]
    members: List[List[int]]
    parts: List[List[List[int]]]
    dropped: List[List[int]]
    report: List[Dict]


def joined_box(boxes: np.ndarray, parts: Sequence[Sequence[int]]) -> np.ndarray:
    """The box around the median boxes of several groups: a merged site covers each part where it lies."""
    medians = np.array([median_box(boxes[list(p)]) for p in parts]).reshape(-1, 6)
    return np.concatenate([medians[:, :3].min(axis=0), medians[:, 3:].max(axis=0)])


def apply_overrides(groups: List[List[int]], cores: List[List[int]], members: List[List[int]],
                    keys: Sequence[Tuple[str, int]], overrides: Sequence[Mapping]) -> Overridden:
    """Re-apply manual merges, drops and splits, in order, to the automatic site groups.

    Args:
        groups: Site groups (candidate indices) after :func:`merge_sites`.
        cores: Strict groups from :func:`cavity_cores`; a split returns the ones inside its group.
        members: Per group, every candidate belonging to it (:func:`site_members`); overrides match on these.
        keys: ``(pdb_id, cavity)`` of each candidate.
        overrides: ``{"op": "merge", "sites": [cavities, ...]}`` or ``{"op": "drop"|"split", "cavities": [[pdb, cav], ...]}``.

    Returns:
        The groups, their members and box parts, the dropped groups, and a report per override.
    """
    own = [sorted(g) for g in groups]
    mem = [set(m) for m in members]
    parts = [[list(g)] for g in own]
    dropped: List[List[int]] = []
    report: List[Dict] = []

    def keyed(k: int) -> Set[Tuple[str, int]]:
        return {tuple(keys[i]) for i in mem[k]}

    def recorded(cavities) -> Set[Tuple[str, int]]:
        return {(str(p), int(c)) for p, c in cavities}

    def replace(drop: List[int], new_own, new_mem, new_parts) -> None:
        nonlocal own, mem, parts
        keep = [k for k in range(len(own)) if k not in drop]
        own = [own[k] for k in keep] + new_own
        mem = [mem[k] for k in keep] + new_mem
        parts = [parts[k] for k in keep] + new_parts

    for override in overrides:
        op = override["op"]
        sets = [keyed(k) for k in range(len(own))]
        entry: Dict = {"op": op, "status": "unmatched"}
        if op == "merge":
            hits = sorted({k for part in override["sites"]
                           for k in [match_site(recorded(part), sets)] if k is not None})
            entry["parts"] = len(hits)
            if len(hits) >= 2:
                merged = sorted(i for k in hits for i in own[k])
                replace(hits, [merged], [set().union(*(mem[k] for k in hits))], [[p for k in hits for p in parts[k]]])
                entry.update(status="applied", groups=[merged])
        elif op in ("drop", "split"):
            k = match_site(recorded(override["cavities"]), sets)
            if k is not None and op == "drop":
                dropped.append(own[k])
                replace([k], [], [], [])
                entry["status"] = "applied"
            elif k is not None:
                inside = set(own[k])
                pieces = [sorted(c) for c in cores if c and set(c) <= inside]
                entry["parts"] = len(pieces)
                if len(pieces) >= 2:
                    # A part is its own cavities only; taking overlap members back would rebuild the site it left.
                    replace([k], pieces, [set(p) for p in pieces], [[p] for p in pieces])
                    entry.update(status="applied", groups=pieces)
        else:
            raise ValueError(f"unknown site override {op!r}; known: merge, drop, split")
        report.append(entry)
    return Overridden(own, [sorted(m) for m in mem], parts, dropped, report)
