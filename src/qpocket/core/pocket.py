"""Pocket definition: cavity choice and cropping residues (pure)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from qpocket.core.geometry import check_overlap, crop_pocket_by_interactions


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
