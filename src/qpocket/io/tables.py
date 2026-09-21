"""Tabular files: PQR atoms and pocket point clouds."""
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from qpocket.constants import FP_DTYPE
from qpocket.types import PQRAtoms


def read_point_table(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Read a pocket point cloud CSV.

    Args:
        csv_path: CSV with x, y, z columns.

    Returns:
        (the table, (N, 3) points).
    """
    df = pd.read_csv(csv_path)
    points = df[["x", "y", "z"]].to_numpy(dtype=FP_DTYPE)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    return df, points


def write_featurized(out_path: Path, table: pd.DataFrame, features: Dict[str, np.ndarray]) -> None:
    """Write a point table with feature columns appended.

    Args:
        out_path: Target CSV.
        table: Point table from read_point_table.
        features: Column name to (N,) values, in output order.
    """
    out = table.copy()
    for name, values in features.items():
        out[name] = values
    out.to_csv(out_path, index=False)


def write_points(coords: np.ndarray, path: Path) -> None:
    """Write a point cloud as an x,y,z CSV.

    Args:
        coords: (N, 3) points.
        path: Target CSV.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("x,y,z\n")
        for pt in coords:
            f.write(f"{pt[0]},{pt[1]},{pt[2]}\n")


def read_pqr(pqr_path: Path) -> PQRAtoms:
    """Read every atom of a whitespace-separated PQR file.

    Args:
        pqr_path: PQR written by pdb2pqr (plus repaired ions).

    Returns:
        The atoms.
    """
    coords, charges, radii, names, residues, chains, seqs = [], [], [], [], [], [], []
    with pqr_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                xyz = [float(fields[-5]), float(fields[-4]), float(fields[-3])]
                q, r = float(fields[-2]), float(fields[-1])
                seq = int(fields[-6])
            except (ValueError, IndexError):
                continue
            coords.append(xyz)
            charges.append(q)
            radii.append(r)
            names.append(fields[2])
            residues.append(fields[3])
            # Records with a chain column have 11 fields; repaired ions have none.
            chains.append(fields[4] if len(fields) >= 11 else "")
            seqs.append(seq)
    coords_arr = np.array(coords, dtype=FP_DTYPE).reshape(-1, 3)
    return PQRAtoms(coords_arr, np.array(charges, dtype=FP_DTYPE), np.array(radii, dtype=FP_DTYPE),
                    names, residues, chains, seqs)


def file_hash(path: Path) -> str:
    """sha256 of a file's bytes.

    Args:
        path: Any file.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
