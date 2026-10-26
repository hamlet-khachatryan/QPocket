"""Binding-site discovery: which ligands a structure holds (pure)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from qpocket.core.geometry import compute_pocket_bounding_box
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
                 min_heavy_atoms: int) -> List[Tuple[Ligand, object]]:
    """Every non-water, carbon-containing HET residue of the first model that the run does not clean away.

    Args:
        structure: Parsed Biopython structure (first model only, so NMR ensembles do not repeat ligands).
        pdb_id: Identifier stamped on each ligand.
        allowed_cofactors: Residue names never reported.
        min_heavy_atoms: Residues with fewer non-hydrogen atoms are skipped.

    Returns:
        ``(Ligand, residue)`` pairs in chain, then residue, order.
    """
    cofactors = {name.upper() for name in allowed_cofactors}
    found: List[Tuple[Ligand, object]] = []
    for chain in next(iter(structure)):
        for res in chain:
            if res.id[0] in (" ", "W") or res.get_resname().strip().upper() in cofactors:
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


def group_ligands(coords: Sequence[np.ndarray], cutoff: float) -> np.ndarray:
    """Single-linkage grouping: ligands join when any heavy-atom pair lies within ``cutoff``.

    Args:
        coords: One ``(n_i, 3)`` heavy-atom coordinate array per ligand.
        cutoff: Linking distance (Å).

    Returns:
        Integer component label per ligand (labels identify groups; they carry no order).
    """
    n = len(coords)
    if n == 0:
        return np.zeros(0, dtype=int)
    owner = np.repeat(np.arange(n), [len(c) for c in coords])
    pairs = cKDTree(np.vstack(coords)).query_pairs(cutoff, output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def site_ids(labels: np.ndarray, pdb_ids: Sequence[str]) -> List[str]:
    """Name each ligand's site ``site_NN``, the site held by the most distinct structures first.

    Args:
        labels: Group label per ligand, from :func:`group_ligands`.
        pdb_ids: Structure identifier per ligand.

    Returns:
        ``site_NN`` per ligand; ties in structure count go to the label that appears first.
    """
    holo = {}  # insertion order is first appearance
    for label, pdb_id in zip(labels.tolist(), pdb_ids):
        holo.setdefault(label, set()).add(pdb_id)
    order = sorted(holo, key=lambda label: -len(holo[label]))  # stable sort keeps first appearance on ties
    width = 3 if len(order) > 99 else 2
    name = {label: f"site_{i:0{width}d}" for i, label in enumerate(order, 1)}
    return [name[label] for label in labels.tolist()]


def site_box(coords: Sequence[np.ndarray], padding: float) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around a site's ligands.

    Args:
        coords: Heavy-atom coordinate arrays of the site's ligands.
        padding: Expansion (Å) on every face.

    Returns:
        ``(min_corner, max_corner)``.
    """
    return compute_pocket_bounding_box(np.vstack(coords), padding)


def points_in_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Points inside the box, bounds included.

    Args:
        points: ``(N, 3)`` coordinates.
        lo: Minimum corner.
        hi: Maximum corner.

    Returns:
        ``(M, 3)`` subset of ``points``.
    """
    return points[np.all((points >= lo) & (points <= hi), axis=1)]
