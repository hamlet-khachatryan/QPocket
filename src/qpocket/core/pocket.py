"""Pocket definition: cavity choice, cropping residues and ligand choice (pure)."""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qpocket.constants import HET_TO_REMOVE
from qpocket.core.geometry import check_overlap, compute_volumetric_overlap, crop_pocket_by_interactions


def is_residue_ligand(residue, allowed_cofactors: List[str]) -> bool:
    """Determines whether a residue is a candidate ligand.

    Args:
        residue: BioPython residue object.
        allowed_cofactors: List of cofactor residue names to exclude.

    Returns:
        ``True`` if the residue is a potential drug-like ligand.
    """
    if residue.id[0] == " ":
        return False  # standard amino acid
    if residue.id[0] == "W":
        return False  # water
    resname = residue.get_resname().strip().upper()
    if resname in allowed_cofactors:
        return False
    if resname in HET_TO_REMOVE:
        return False
    return True


def extract_pdb_block(residue, chain_id: str) -> str:
    """Constructs a PDB-compliant HETATM block for a single residue.

    Args:
        residue: BioPython residue object.
        chain_id: Chain identifier character.

    Returns:
        PDB-formatted text block terminated with a newline.
    """
    lines: List[str] = []
    for i, atom in enumerate(residue):
        x, y, z = atom.get_coord()
        element = (atom.element or "").strip()
        name = atom.name.strip()

        # PDB atom name occupies columns 13-16 — always exactly four characters.
        # A 4-character name (common for ligand hydrogens, e.g. "H011") fills the
        # field and must NOT be given a leading space, or every later field on the
        # line shifts by one and strict parsers reject the record. Shorter names
        # get a leading space when the element symbol is 1 character (" C1 ") and
        # start at column 13 when it is 2 ("FE  ").
        if len(name) >= 4:
            atom_str = name[:4]
        elif len(element) == 1:
            atom_str = f" {name:<3}"
        else:
            atom_str = f"{name:<4}"

        res_seq = residue.id[1]

        # Legacy PDB has no room for longer values; truncating keeps the columns
        # aligned, which matters more to downstream parsers than the lost tail.
        resname = residue.resname[:3]

        # Standard HETATM record — 80 columns exactly.
        line = (
            f"HETATM{i + 1:5d} {atom_str} "   # cols 1-17 (record + serial + name + altloc)
            f"{resname:>3} "                    # cols 18-21 (resname + space)
            f"{chain_id}"                       # col  22    (chain)
            f"{res_seq:4d} "                    # cols 23-27 (resseq + icode)
            f"   "                              # cols 28-30 (padding)
            f"{x:8.3f}{y:8.3f}{z:8.3f}"       # cols 31-54 (coordinates)
            f"  1.00  0.00"                     # cols 55-66 (occupancy + B-factor)
            f"          "                       # cols 67-76 (segment ID padding)
            f"{element[:2]:>2}  "               # cols 77-80 (element + charge)
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def select_cavity_by_centroid(
    cavities: Sequence[np.ndarray], centroid: np.ndarray
) -> Optional[np.ndarray]:
    """First cavity whose padded bounding box contains a point.

    Args:
        cavities: Cavity point clouds, in detection order.
        centroid: (3,) point, usually the reference ligand centroid.

    Returns:
        That cavity's points, or None.
    """
    for coords in cavities:
        if check_overlap(centroid, coords, use_majority=True):
            return coords
    return None


def select_cavity_by_bbox(
    cavities: Sequence[np.ndarray], bbox_min: np.ndarray, bbox_max: np.ndarray
) -> Tuple[Optional[np.ndarray], int]:
    """Cavity with the most points inside a reference box.

    Args:
        cavities: Cavity point clouds, in detection order.
        bbox_min: Lower box corner.
        bbox_max: Upper box corner.

    Returns:
        (best cavity or None, its overlap count); None when nothing overlaps.
    """
    best_coords: Optional[np.ndarray] = None
    best_overlap: int = 0

    for coords in cavities:
        if len(coords) == 0:
            continue
        overlap = compute_volumetric_overlap(coords, bbox_min, bbox_max)
        if overlap > best_overlap:
            best_overlap = overlap
            best_coords = coords

    if best_coords is None or best_overlap == 0:
        return None, 0
    return best_coords, best_overlap


class LigandChoice(NamedTuple):
    """The ligand chosen for a pocket.

    Attributes:
        residue: Biopython residue.
        chain_id: Its chain.
        n_atoms: Its atom count.
        n_candidates: Ligands that overlapped the pocket.
    """

    residue: object
    chain_id: str
    n_atoms: int
    n_candidates: int


def pick_ligand(structure, pocket_coords: np.ndarray, allowed_cofactors: List[str]) -> Optional[LigandChoice]:
    """Largest non-cofactor heteroatom residue overlapping a pocket.

    Args:
        structure: Parsed Biopython structure.
        pocket_coords: (N, 3) pocket points.
        allowed_cofactors: Residue names never picked.

    Returns:
        The choice, or None when no ligand overlaps.
    """
    candidates: List[Tuple] = []
    for model in structure:
        for chain in model:
            for res in chain:
                if not is_residue_ligand(res, allowed_cofactors):
                    continue
                atoms = np.array([a.get_coord() for a in res])
                if len(atoms) == 0:
                    continue
                if check_overlap(atoms, pocket_coords, use_majority=True):
                    candidates.append((res, chain.id, len(atoms)))

    if not candidates:
        return None
    best_res, best_chain_id, n_atoms = max(candidates, key=lambda x: x[2])
    return LigandChoice(best_res, best_chain_id, n_atoms, len(candidates))


def ligand_centroid(
    structure, ligand_resname: str, search_chains: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """Centroid of every atom of a ligand residue name.

    Args:
        structure: Parsed Biopython structure.
        ligand_resname: Ligand residue name.
        search_chains: Chains to search, or None for all.

    Returns:
        (3,) centroid, or None if the ligand is absent.
    """
    atoms: List[np.ndarray] = []
    for model in structure:
        for chain in model:
            if search_chains is not None and chain.id not in search_chains:
                continue
            for residue in chain:
                if residue.get_resname().strip().upper() == ligand_resname.upper():
                    for atom in residue:
                        atoms.append(atom.get_coord())
    if not atoms:
        return None
    return np.mean(atoms, axis=0)


def interacting_residue_coords(atoms: pd.DataFrame, interactions: List[str]) -> Optional[np.ndarray]:
    """Atom coordinates of the residues named in interaction strings.

    Args:
        atoms: biopandas ATOM table.
        interactions: Strings like "TYR_795_A|H-bond Donor" (chain optional).

    Returns:
        (M, 3) coordinates, or None if no residue matches.
    """
    residue_ids: set = set()
    for interaction in interactions:
        if "|" in interaction:
            res_id = interaction.split("|", 1)[0]
            residue_ids.add(res_id)

    if not residue_ids:
        return None

    coords_list: List[np.ndarray] = []
    for res_id in residue_ids:
        parts = res_id.split("_")
        # Supported formats:
        #   "RESNAME_RESNUM"          (single-chain / legacy)
        #   "RESNAME_RESNUM_CHAINID"  (multi-chain / chain-tagged)
        if len(parts) < 2:
            continue
        res_name = parts[0]
        try:
            res_num = int(parts[1])
        except ValueError:
            continue
        mask = (atoms["residue_name"] == res_name) & (
            atoms["residue_number"] == res_num
        )
        if len(parts) >= 3:
            # Chain-tagged format: further restrict to the specific chain.
            chain_filter = parts[2]
            mask = mask & (atoms["chain_id"] == chain_filter)
        res_coords = atoms[mask][["x_coord", "y_coord", "z_coord"]].to_numpy()
        if len(res_coords) > 0:
            coords_list.append(res_coords)

    if not coords_list:
        return None
    return np.vstack(coords_list)


def residue_coords(structure, residues: Sequence[str], chain: Optional[str] = None) -> Optional[np.ndarray]:
    """Atom coordinates of residues named like TYR795 (name check optional: 795).

    Args:
        structure: Parsed Biopython structure.
        residues: Residue labels, e.g. ("TYR795", "GLU756").
        chain: Restrict to this chain, or None for all.

    Returns:
        (M, 3) coordinates, or None if none match.
    """
    wanted = []
    for label in residues:
        name = "".join(c for c in label if c.isalpha())[:3].upper()
        number = int("".join(c for c in label if c.isdigit()))
        wanted.append((name, number))
    coords = [
        atom.get_coord()
        for model in structure
        for ch in model
        if chain is None or ch.id == chain
        for res in ch
        if any(res.id[1] == num and (not name or res.get_resname().strip() == name) for name, num in wanted)
        for atom in res
    ]
    return np.array(coords) if coords else None


def select_cavity_by_residues(
    cavities: Sequence[np.ndarray], residue_xyz: np.ndarray, padding: float
) -> Optional[np.ndarray]:
    """Points of the best-lined cavity that lie within padding of given residues.

    Args:
        cavities: Cavity point clouds.
        residue_xyz: (M, 3) atoms of the lining residues.
        padding: Distance cutoff (A).

    Returns:
        Cropped points of the cavity with the most such points, or None.
    """
    best: Optional[np.ndarray] = None
    for coords in cavities:
        if len(coords) == 0:
            continue
        cropped = crop_pocket_by_interactions(coords, residue_xyz, padding=padding)
        if len(cropped) and (best is None or len(cropped) > len(best)):
            best = cropped
    return best
