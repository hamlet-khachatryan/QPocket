import math
import tempfile
import numpy as np
from typing import Tuple
from subprocess import PIPE, Popen
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from pathlib import Path
from eliot import start_action, log_message

from qpocket_pipeline.distance_logic.distance_utils import repair_pqr_with_missing_ions
from qpocket_pipeline.global_vars import FP_DTYPE, SIGMA_HBOND, SIGMA_AROM, SIGMA_HYDRO
from qpocket_pipeline.distance_logic.distance_utils import get_structure_data

try:
    from apbs_binary import popen_apbs
except ImportError:
    raise ImportError("pip install apbs-binary")


def read_dx_grid(
    dx_path: Path,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parses an APBS OpenDX (.dx) file into a numpy grid and coordinate axes.
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
                origin = np.array(
                    [float(parts[-3]), float(parts[-2]), float(parts[-1])]
                )
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
            full_text = " ".join(lines[data_start_line:])
            data_values = np.fromstring(full_text, sep=" ")
        except Exception as e:
            raise RuntimeError(f"Failed to parse DX data: {e}")

        nx, ny, nz = counts
        grid = data_values.reshape((nx, ny, nz))

        x_axis = np.linspace(origin[0], origin[0] + (nx - 1) * deltas[0], nx)
        y_axis = np.linspace(origin[1], origin[1] + (ny - 1) * deltas[1], ny)
        z_axis = np.linspace(origin[2], origin[2] + (nz - 1) * deltas[2], nz)

        return grid, (x_axis, y_axis, z_axis)


def calculate_apbs_potential(pdb_path: Path, point_cloud: np.ndarray) -> np.ndarray:
    """
    Runs APBS with strict input ordering, union-box sizing, and CHARGE BALANCING.
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
            p = Popen(cmd, stdout=PIPE, stderr=PIPE)
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
                        except:
                            pass

            if not prot_coords:
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)
            prot_coords = np.array(prot_coords)

            cloud_min = np.min(point_cloud, axis=0)
            cloud_max = np.max(point_cloud, axis=0)
            prot_min = np.min(prot_coords, axis=0)
            prot_max = np.max(prot_coords, axis=0)

            # union bounding box
            union_min = np.minimum(prot_min, cloud_min)
            union_max = np.maximum(prot_max, cloud_max)
            grid_center = (union_min + union_max) / 2.0

            grid_dims = (union_max - union_min) + 30.0
            dim_x, dim_y, dim_z = grid_dims[0], grid_dims[1], grid_dims[2]

            if np.any(np.isnan(grid_center)) or np.any(np.isnan(grid_dims)):
                log_message(
                    message_type="error", msg="Calculated NaN for grid center/dims"
                )
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)

            # APBS config
            apbs_in = base.with_suffix(".in")
            cx, cy, cz = grid_center[0], grid_center[1], grid_center[2]
            pts = 129 if max(dim_x, dim_y, dim_z) > 80 else 97

            # ion balancing: 0.150(Na) + 2*0.002(Mg) = 0.154 -> Cl must be 0.154
            conc_na = 0.150
            conc_mg = 0.002
            conc_cl = conc_na + (2 * conc_mg)

            with apbs_in.open("w") as f:
                f.write(f"""
                    read
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

            # run APBS
            p = popen_apbs([str(apbs_in)], stdout=PIPE, stderr=PIPE)
            out, err = p.communicate()

            dx_file = base.with_suffix(".dx")
            if not dx_file.exists():
                dx_file = base.with_suffix(".pot.dx")

            if not dx_file.exists():
                log_message(message_type="apbs_binary_fail", stderr=err.decode())
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)

            # interpolate
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
    if len(atom_coords) == 0:
        return np.zeros(len(points), dtype=FP_DTYPE), np.zeros(
            len(points), dtype=FP_DTYPE
        )

    dist_sq = cdist(points, atom_coords, metric="sqeuclidean").astype(FP_DTYPE)

    with np.errstate(divide="ignore", invalid="ignore"):
        chg_matrix = np.divide(chg[np.newaxis, :], dist_sq)
        chg_matrix[np.isinf(chg_matrix)] = 0.0
    cp = np.sum(chg_matrix, axis=1) * 557.0

    dist = np.sqrt(dist_sq)
    fermi_wt = np.zeros_like(dist)
    num = math.exp(-c1 * c2) + 1.0
    exponent = c1 * (dist - c2)
    mask = exponent < 50.0
    fermi_wt[mask] = num / (np.exp(exponent[mask]) + 1.0)
    sum_wt = np.maximum(np.sum(fermi_wt, axis=1), 1.0e-6)
    lp = np.sum(fermi_wt * hp[np.newaxis, :], axis=1) / sum_wt

    return cp, lp


def calculate_geometric_potentials(
    points: np.ndarray, targets: np.ndarray, sigma: float
) -> np.ndarray:
    if len(targets) == 0:
        return np.zeros(len(points), dtype=FP_DTYPE)
    d_sq = cdist(points, targets, metric="sqeuclidean").astype(FP_DTYPE)
    factor = -1.0 / (2.0 * sigma**2)
    return np.sum(np.exp(d_sq * factor), axis=1)



def featurize(csv_path: Path, pdb_path: Path) -> None:

    with start_action(action_type="process_point_cloud"):
        if not csv_path.exists():
            log_message(message_type="error", error="CSV missing")
            return

        try:
            df = pd.read_csv(csv_path)
            points = df[['x', 'y', 'z']].values.astype(FP_DTYPE)
            if points.ndim == 1: points = points.reshape(-1, 3)
        except Exception as e:
            log_message(message_type="error", error=f"CSV Error: {e}")
            return

        try:
            coords, chgs, hps, donors, acceptors, aromatics = get_structure_data(pdb_path)
        except Exception as e:
            log_message(message_type="fatal_error", error=str(e))
            return

        out_path = csv_path.with_name("featurized.csv")

        log_message(message_type="status", msg="Running APBS")
        df['apbs'] = calculate_apbs_potential(pdb_path, points)

        log_message(message_type="status", msg="Running CP/LP")
        cp, lp = calculate_cp_lp(points, coords, chgs, hps)
        df['cp'] = cp
        df['lp'] = lp

        log_message(message_type="status", msg="Running Geometry")
        df['h_donor'] = calculate_geometric_potentials(points, acceptors, sigma=SIGMA_HBOND)
        df['h_acceptor'] = calculate_geometric_potentials(points, donors, sigma=SIGMA_HBOND)
        df['aromatic'] = calculate_geometric_potentials(points, aromatics, sigma=SIGMA_AROM)


        df.to_csv(out_path, index=False)
        log_message(message_type="success", output=str(out_path))
