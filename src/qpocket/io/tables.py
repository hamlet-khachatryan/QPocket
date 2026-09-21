"""Tabular files: PQR atoms and pocket point clouds."""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from qpocket.constants import FP_DTYPE


def parse_pqr_data(
    pqr_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Robust PQR parser handling variable whitespace"""
    coords = []
    charges = []
    atom_names = []
    res_names = []

    with pqr_path.open("r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            fields = line.strip().split()
            if len(fields) < 10:
                continue

            try:
                x, y, z = float(fields[-5]), float(fields[-4]), float(fields[-3])
                q = float(fields[-2])
                atom_name = fields[2]
                res_name = fields[3]

                coords.append([x, y, z])
                charges.append(q)
                atom_names.append(atom_name)
                res_names.append(res_name)
            except (ValueError, IndexError):
                continue

    coords_arr = np.array(coords, dtype=FP_DTYPE)
    if coords_arr.ndim == 1:
        coords_arr = coords_arr.reshape(-1, 3)

    return (coords_arr, np.array(charges, dtype=FP_DTYPE), atom_names, res_names)


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
    with open(path, "w") as f:
        f.write("x,y,z\n")
        for pt in coords:
            f.write(f"{pt[0]},{pt[1]},{pt[2]}\n")
