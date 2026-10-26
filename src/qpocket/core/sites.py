"""Binding-site discovery: ligands a structure holds, and sites built from cavity boxes (pure)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

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
    xyz = np.vstack([p for p in points if len(p)] or [np.zeros((0, 3))])
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


def site_members(site_boxes: np.ndarray, boxes: np.ndarray, threshold: float) -> List[List[int]]:
    """Per site, every box overlapping the site box by at least ``threshold``; a box may join several.

    Args:
        site_boxes: ``(k, 6)`` site boxes.
        boxes: ``(n, 6)`` cavity boxes.
        threshold: IoU a cavity needs with a site box.

    Returns:
        One index list per site.
    """
    if not len(site_boxes) or not len(boxes):
        return [[] for _ in range(len(site_boxes))]
    iou = box_iou(site_boxes, boxes)
    return [np.flatnonzero(row >= threshold).tolist() for row in iou]


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
