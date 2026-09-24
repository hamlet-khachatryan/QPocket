"""Orient rotatable hydroxyl hydrogens against the assembled complex."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

#: Hydroxyl hydrogens are placed by pdb2pqr before the ligand exists, so their
#: rotamer is arbitrary with respect to it. These are scanned and re-chosen.
STEP_DEGREES = 10.0

#: An acceptor this far from the hydroxyl oxygen can still take its hydrogen.
ACCEPTOR_MAX = 3.6

#: Below this donor-hydrogen-acceptor angle a contact is not a hydrogen bond.
ANGLE_MIN = 120.0


def _rotate(point: np.ndarray, origin: np.ndarray, axis: np.ndarray,
            radians: float) -> np.ndarray:
    """Rotate a point about an axis through an origin, by Rodrigues' formula."""
    unit = axis / np.linalg.norm(axis)
    offset = point - origin
    return (origin + offset * np.cos(radians)
            + np.cross(unit, offset) * np.sin(radians)
            + unit * np.dot(unit, offset) * (1.0 - np.cos(radians)))


def rotatable_hydroxyls(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    """Hydroxyl groups whose hydrogen can rotate freely.

    Args:
        mol: A perceived molecule with explicit hydrogens.

    Returns:
        One ``(carbon, oxygen, hydrogen)`` index triple per hydroxyl.
    """
    groups: List[Tuple[int, int, int]] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O" or atom.GetDegree() != 2:
            continue
        hydrogens = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
        heavies = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() != "H"]
        if len(hydrogens) == 1 and len(heavies) == 1:
            groups.append((heavies[0], atom.GetIdx(), hydrogens[0]))
    return groups


def _score(hydrogen: np.ndarray, oxygen: np.ndarray,
           acceptors: np.ndarray) -> float:
    """Total hydrogen-bond quality of one rotamer against nearby acceptors."""
    if not len(acceptors):
        return 0.0
    to_acceptor = acceptors - hydrogen
    distances = np.linalg.norm(to_acceptor, axis=1)
    to_donor = oxygen - hydrogen
    cosines = (to_acceptor @ to_donor) / (distances * np.linalg.norm(to_donor))
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    usable = (angles >= ANGLE_MIN) & (distances < 2.8)
    if not usable.any():
        return 0.0
    # Shorter and straighter is better; summed so a hydroxyl may serve two acceptors.
    return float((((2.8 - distances[usable]) / 2.8) * (angles[usable] / 180.0)).sum())


def optimise_hydroxyls(mol: Chem.Mol, step_degrees: float = STEP_DEGREES) -> Chem.Mol:
    """Copy of a molecule with every rotatable hydroxyl hydrogen at its best rotamer.

    Args:
        mol: A perceived complex with explicit hydrogens.
        step_degrees: Angular resolution of the scan.

    Returns:
        A copy whose hydroxyl hydrogens point at their best acceptors.
    """
    groups = rotatable_hydroxyls(mol)
    if not groups:
        return mol

    out = Chem.Mol(mol)
    conformer = out.GetConformer()
    positions = conformer.GetPositions()
    candidates = np.array([atom.GetIdx() for atom in out.GetAtoms()
                           if atom.GetSymbol() in ("O", "N")])
    steps = np.arange(0.0, 360.0, step_degrees)

    for carbon, oxygen, hydrogen in groups:
        origin, axis = positions[oxygen], positions[oxygen] - positions[carbon]
        nearby = candidates[(np.linalg.norm(positions[candidates] - origin, axis=1)
                             < ACCEPTOR_MAX)]
        # An acceptor in the same hydroxyl, or its own carbon, is not a partner.
        nearby = np.array([i for i in nearby if i not in (oxygen, carbon)])
        if not len(nearby):
            continue
        acceptors = positions[nearby]
        best_score, best_point = -1.0, positions[hydrogen]
        for degrees in steps:
            moved = _rotate(positions[hydrogen], origin, axis, np.radians(degrees))
            score = _score(moved, origin, acceptors)
            # Strictly greater keeps the lowest angle on a tie, so the scan is stable.
            if score > best_score:
                best_score, best_point = score, moved
        conformer.SetAtomPosition(hydrogen, Point3D(*(float(v) for v in best_point)))
        positions[hydrogen] = best_point
    return out
