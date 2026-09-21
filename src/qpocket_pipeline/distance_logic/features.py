from __future__ import annotations

import math
import tempfile
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from eliot import log_message, start_action

from qpocket_pipeline.distance_logic.distance_utils import (
    get_structure_data,
    repair_pqr_with_missing_ions,
)
from qpocket_pipeline.global_vars import FP_DTYPE, SIGMA_AROM, SIGMA_HBOND, SIGMA_HYDRO

try:
    from apbs_binary import popen_apbs
except ImportError:
    raise ImportError("pip install apbs-binary")

# Maximum distance (Å) beyond which atom contributions to CP/LP are negligible.
_CP_LP_CUTOFF = 12.0


def read_dx_grid(
    dx_path: Path,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parses an APBS OpenDX (.dx) file into a numpy grid and coordinate axes.

    Args:
        dx_path: Path to the ``.dx`` output file from APBS.

    Returns:
        Tuple of ``(grid, (x_axis, y_axis, z_axis))`` where *grid* has shape
        ``(nx, ny, nz)`` and each axis array is 1-D.

    Raises:
        RuntimeError: On malformed header or data section.
    """
    with dx_path.open("r") as f:
        counts = None
        origin = None
        deltas = []
        data_start_line = 0
        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("object 1 class gridpositions counts"):
                parts = line.split()
                counts = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
            elif line.startswith("origin"):
                parts = line.split()
                origin = np.array([float(parts[-3]), float(parts[-2]), float(parts[-1])])
            elif line.startswith("delta"):
                parts = line.split()
                vals = [float(x) for x in parts[1:]]
                deltas.append(max(vals))
            elif line.startswith("object 3 class array type double"):
                data_start_line = i + 1
                break

        if counts is None or origin is None or len(deltas) != 3:
            raise RuntimeError("Malformed DX Header")

        try:
            floats = []
            for token in " ".join(lines[data_start_line:]).split():
                try:
                    floats.append(float(token))
                except ValueError:
                    break
            data_values = np.array(floats, dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to parse DX data: {e}") from e

        nx, ny, nz = counts
        grid = data_values.reshape((nx, ny, nz))

        x_axis = np.linspace(origin[0], origin[0] + (nx - 1) * deltas[0], nx)
        y_axis = np.linspace(origin[1], origin[1] + (ny - 1) * deltas[1], ny)
        z_axis = np.linspace(origin[2], origin[2] + (nz - 1) * deltas[2], nz)

        return grid, (x_axis, y_axis, z_axis)


def calculate_apbs_potential(pdb_path: Path, point_cloud: np.ndarray) -> np.ndarray:
    """Runs APBS to compute the electrostatic potential at each pocket point.

    Constructs an APBS input file with a union bounding box covering both the
    protein and the pocket point cloud, runs pdb2pqr for charge assignment,
    then interpolates the resulting DX grid onto *point_cloud*.

    Args:
        pdb_path: Path to the cleaned protein PDB file.
        point_cloud: ``(N, 3)`` array of pocket grid points.

    Returns:
        ``(N,)`` array of electrostatic potential values (kT/e units from APBS).
        Returns an array of zeros if APBS or pdb2pqr fails.
    """
    with start_action(action_type="calculate_apbs_potential"):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            base = temp_dir / "mol"
            pqr = base.with_suffix(".pqr")

            cmd = [
                "pdb2pqr",
                "--ff=PARSE",
                "--whitespace",
                "--noopt",
                str(pdb_path),
                str(pqr),
            ]
            # cwd=temp_dir: both pdb2pqr and APBS drop working files beside the
            # process, not beside their output. APBS in particular always writes
            # an io.mc, which several concurrent workers would otherwise append
            # to in whatever directory the job was launched from.
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=str(temp_dir))
            out, err = p.communicate()

            repair_pqr_with_missing_ions(pdb_path, pqr)

            if not pqr.exists() or pqr.stat().st_size == 0:
                log_message(message_type="pdb2pqr_fail", error=err.decode())
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)

            prot_coords = []
            with pqr.open() as f:
                for line in f:
                    if line.startswith("ATOM"):
                        try:
                            parts = line.split()
                            prot_coords.append(
                                [float(parts[-5]), float(parts[-4]), float(parts[-3])]
                            )
                        except (ValueError, IndexError):
                            pass

            if not prot_coords:
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)
            prot_coords_arr = np.array(prot_coords)

            cloud_min = np.min(point_cloud, axis=0)
            cloud_max = np.max(point_cloud, axis=0)
            prot_min = np.min(prot_coords_arr, axis=0)
            prot_max = np.max(prot_coords_arr, axis=0)

            union_min = np.minimum(prot_min, cloud_min)
            union_max = np.maximum(prot_max, cloud_max)
            grid_center = (union_min + union_max) / 2.0
            grid_dims = (union_max - union_min) + 30.0
            dim_x, dim_y, dim_z = grid_dims[0], grid_dims[1], grid_dims[2]

            if np.any(np.isnan(grid_center)) or np.any(np.isnan(grid_dims)):
                log_message(
                    message_type="error",
                    stage="calculate_apbs_potential",
                    error="NaN in grid center/dims",
                )
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)

            apbs_in = base.with_suffix(".in")
            cx, cy, cz = grid_center[0], grid_center[1], grid_center[2]
            pts = 129 if max(dim_x, dim_y, dim_z) > 80 else 97

            # Ion balancing: Na⁺ 0.150 M + Mg²⁺ 0.002 M → Cl⁻ 0.154 M
            conc_na = 0.150
            conc_mg = 0.002
            conc_cl = conc_na + (2 * conc_mg)

            with apbs_in.open("w") as f:
                f.write(f"""read
    mol pqr "{pqr}"
end
elec
    mg-auto
    mol 1
    dime {pts} {pts} {pts}
    cglen {dim_x:.2f} {dim_y:.2f} {dim_z:.2f}
    fglen {dim_x:.2f} {dim_y:.2f} {dim_z:.2f}
    cgcent {cx:.3f} {cy:.3f} {cz:.3f}
    fgcent {cx:.3f} {cy:.3f} {cz:.3f}
    lpbe
    bcfl sdh
    ion charge 1 conc {conc_na:.3f} radius 2.0
    ion charge -1 conc {conc_cl:.3f} radius 2.0
    ion charge 2 conc {conc_mg:.3f} radius 2.0
    pdie 2.0
    sdie 78.54
    srfm smol
    chgm spl2
    sdens 10.0
    srad 1.4
    temp 298.15
    calcenergy total
    calcforce no
    write pot dx "{base}"
end
quit
""")

            p = popen_apbs([str(apbs_in)], stdout=PIPE, stderr=PIPE, cwd=str(temp_dir))
            out, err = p.communicate()

            dx_file = base.with_suffix(".dx")
            if not dx_file.exists():
                dx_file = base.with_suffix(".pot.dx")

            if not dx_file.exists():
                log_message(message_type="apbs_binary_fail", stderr=err.decode())
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)

            try:
                grid, (ax_x, ax_y, ax_z) = read_dx_grid(dx_file)
                interpolator = RegularGridInterpolator(
                    (ax_x, ax_y, ax_z), grid, bounds_error=False, fill_value=0.0
                )
                values = interpolator(point_cloud)
                return values.astype(FP_DTYPE)
            except Exception as e:
                log_message(message_type="interpolation_error", error=str(e))
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)


def calculate_cp_lp(
    points: np.ndarray,
    atom_coords: np.ndarray,
    chg: np.ndarray,
    hp: np.ndarray,
    c1: float = 1.0,
    c2: float = SIGMA_HYDRO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the charge potential (CP) and lipophilic potential (LP) at pocket points.

    A KD-tree spatial pre-filter restricts computation to atoms within
    ``_CP_LP_CUTOFF`` (12 Å) of any pocket point, reducing the number of
    atom-pair evaluations by ~10× for typical systems without loss of accuracy
    (contributions from atoms beyond 12 Å are negligible for both potentials).

    Args:
        points: Pocket grid points ``(N, 3)``.
        atom_coords: Protein atom coordinates ``(M, 3)``.
        chg: Partial charges for each atom ``(M,)``.
        hp: Hydrophobicity scores for each atom ``(M,)``.
        c1: Fermi function steepness parameter.
        c2: Fermi function midpoint (default: SIGMA_HYDRO).

    Returns:
        Tuple ``(cp, lp)`` each of shape ``(N,)`` and dtype ``FP_DTYPE``.
    """
    if len(atom_coords) == 0:
        return (
            np.zeros(len(points), dtype=FP_DTYPE),
            np.zeros(len(points), dtype=FP_DTYPE),
        )

    # Spatial pre-filter: keep only atoms within _CP_LP_CUTOFF of any pocket point.
    tree = cKDTree(atom_coords)
    nearby_sets = tree.query_ball_point(points, r=_CP_LP_CUTOFF)
    nearby_idx = np.array(sorted({i for sub in nearby_sets for i in sub}), dtype=int)

    if len(nearby_idx) == 0:
        return (
            np.zeros(len(points), dtype=FP_DTYPE),
            np.zeros(len(points), dtype=FP_DTYPE),
        )

    atom_coords_f = atom_coords[nearby_idx]
    chg_f = chg[nearby_idx]
    hp_f = hp[nearby_idx]

    dist_sq = cdist(points, atom_coords_f, metric="sqeuclidean").astype(FP_DTYPE)

    # Charge potential: sum(q / r²) × 557 (kcal/mol·e units approximation)
    with np.errstate(divide="ignore", invalid="ignore"):
        chg_matrix = np.divide(chg_f[np.newaxis, :], dist_sq)
        chg_matrix[np.isinf(chg_matrix)] = 0.0
    cp = np.sum(chg_matrix, axis=1) * 557.0

    # Lipophilic potential: Fermi-weighted hydrophobicity
    dist = np.sqrt(dist_sq)
    fermi_wt = np.zeros_like(dist)
    num = math.exp(-c1 * c2) + 1.0
    exponent = c1 * (dist - c2)
    mask = exponent < 50.0
    fermi_wt[mask] = num / (np.exp(exponent[mask]) + 1.0)
    sum_wt = np.maximum(np.sum(fermi_wt, axis=1), 1.0e-6)
    lp = np.sum(fermi_wt * hp_f[np.newaxis, :], axis=1) / sum_wt

    return cp, lp


def calculate_geometric_potentials(
    points: np.ndarray,
    targets: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Computes a Gaussian kernel potential at pocket points from target atom positions.

    Args:
        points: Pocket grid points ``(N, 3)``.
        targets: Target atom positions ``(T, 3)``.
        sigma: Gaussian kernel width (Å).

    Returns:
        ``(N,)`` potential array.
    """
    if len(targets) == 0:
        return np.zeros(len(points), dtype=FP_DTYPE)
    d_sq = cdist(points, targets, metric="sqeuclidean").astype(FP_DTYPE)
    factor = -1.0 / (2.0 * sigma ** 2)
    return np.sum(np.exp(d_sq * factor), axis=1)


def featurize(csv_path: Path, pdb_path: Path) -> None:
    """Computes six pharmacophoric features for each point in a pocket CSV.

    Features written to ``featurized.csv`` (same directory as *csv_path*):

    * **apbs** — APBS electrostatic potential (kT/e).
    * **cp** — charge potential (Coulomb-like, inverse-square distance sum).
    * **lp** — lipophilic potential (Fermi-weighted hydrophobicity).
    * **h_donor** — geometric H-bond donor potential.
    * **h_acceptor** — geometric H-bond acceptor potential.
    * **aromatic** — geometric aromatic potential.

    Args:
        csv_path: Path to the ``binding_site.csv`` pocket point cloud
            (columns: x, y, z).
        pdb_path: Path to the cleaned protein PDB file used for featurisation.
    """
    with start_action(action_type="process_point_cloud"):
        if not csv_path.exists():
            log_message(message_type="error", error="CSV missing", path=str(csv_path))
            return

        try:
            df = pd.read_csv(csv_path)
            points = df[["x", "y", "z"]].to_numpy(dtype=FP_DTYPE)
            if points.ndim == 1:
                points = points.reshape(-1, 3)
        except Exception as e:
            log_message(message_type="error", error=f"CSV read error: {e}")
            return

        try:
            coords, chgs, hps, donors, acceptors, aromatics = get_structure_data(pdb_path)
        except Exception as e:
            log_message(message_type="fatal_error", error=str(e))
            return

        out_path = csv_path.with_name("featurized.csv")

        log_message(message_type="status", msg="Running APBS")
        df["apbs"] = calculate_apbs_potential(pdb_path, points)

        log_message(message_type="status", msg="Running CP/LP")
        cp, lp = calculate_cp_lp(points, coords, chgs, hps)
        df["cp"] = cp
        df["lp"] = lp

        log_message(message_type="status", msg="Running geometry")
        df["h_donor"] = calculate_geometric_potentials(points, acceptors, sigma=SIGMA_HBOND)
        df["h_acceptor"] = calculate_geometric_potentials(points, donors, sigma=SIGMA_HBOND)
        df["aromatic"] = calculate_geometric_potentials(points, aromatics, sigma=SIGMA_AROM)

        df.to_csv(out_path, index=False)
        log_message(message_type="success", output=str(out_path))