"""Pocket featurisation: potentials and pharmacophore fields on point clouds (pure)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import is_aa
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from qpocket.constants import (
    ACCEPTOR_RESIDUES,
    AROMATIC_RESIDUES,
    DONOR_RESIDUES,
    FP_DTYPE,
    SIGMA_HYDRO,
)
from qpocket.types import StructureData

# Atom contributions to cp/lp fade out smoothly between these distances (A). A hard
# cutoff let a 0.001 A coordinate change add or drop an atom's whole contribution.
_CP_LP_R_ON = 10.0
_CP_LP_R_OFF = 12.0
_CP_LP_CUTOFF = _CP_LP_R_OFF  # the KD-tree pre-filter radius; exact, since weights are 0 beyond


def switching(r: np.ndarray, r_on: float = _CP_LP_R_ON, r_off: float = _CP_LP_R_OFF) -> np.ndarray:
    """Weight 1 below r_on, 0 from r_off, C2-smooth (smootherstep) in between.

    Args:
        r: Distances (A).
        r_on: Where switching starts.
        r_off: Where the weight reaches zero.

    Returns:
        Weights, same shape as r.
    """
    x = np.clip((r - r_on) / (r_off - r_on), 0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)

#: Column order of featurized.csv after x, y, z.
FEATURE_COLUMNS = ("apbs", "cp", "lp", "h_donor", "h_acceptor", "aromatic")


def parse_dx(text: str) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parse OpenDX text into a grid and its axes.

    Args:
        text: Contents of an APBS .dx file.

    Returns:
        (grid of shape (nx, ny, nz), (x_axis, y_axis, z_axis)).

    Raises:
        RuntimeError: On a malformed header or data section.
    """
    counts = None
    origin = None
    deltas = []
    data_start_line = 0
    lines = text.splitlines(keepends=True)

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


def apbs_grid_box(
    protein_xyz: np.ndarray, points: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """APBS grid covering the protein and the pocket with 15 A margins.

    Args:
        protein_xyz: (M, 3) protein atom coordinates.
        points: (N, 3) pocket points.

    Returns:
        (centre, lengths, points per axis), or None if the box is NaN.
    """
    cloud_min = np.min(points, axis=0)
    cloud_max = np.max(points, axis=0)
    prot_min = np.min(protein_xyz, axis=0)
    prot_max = np.max(protein_xyz, axis=0)

    union_min = np.minimum(prot_min, cloud_min)
    union_max = np.maximum(prot_max, cloud_max)
    grid_center = (union_min + union_max) / 2.0
    grid_dims = (union_max - union_min) + 30.0

    if np.any(np.isnan(grid_center)) or np.any(np.isnan(grid_dims)):
        return None
    pts = 129 if max(grid_dims[0], grid_dims[1], grid_dims[2]) > 80 else 97
    return grid_center, grid_dims, pts


def apbs_input(pqr: Path, out_base: Path, center: np.ndarray, dims: np.ndarray, pts: int) -> str:
    """Text of the APBS input file.

    Args:
        pqr: PQR file APBS reads.
        out_base: Output path stem for the potential .dx.
        center: Grid centre.
        dims: Grid lengths.
        pts: Grid points per axis.

    Returns:
        APBS input text.
    """
    dim_x, dim_y, dim_z = dims[0], dims[1], dims[2]
    cx, cy, cz = center[0], center[1], center[2]

    # Ion balancing: Na⁺ 0.150 M + Mg²⁺ 0.002 M → Cl⁻ 0.154 M
    conc_na = 0.150
    conc_mg = 0.002
    conc_cl = conc_na + (2 * conc_mg)

    return f"""read
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
    write pot dx "{out_base}"
end
quit
"""


def interpolate_grid(
    grid: np.ndarray, axes: Tuple[np.ndarray, np.ndarray, np.ndarray], points: np.ndarray
) -> np.ndarray:
    """Trilinear interpolation of a grid at points; zero outside it.

    Args:
        grid: (nx, ny, nz) values.
        axes: Grid axes.
        points: (N, 3) query points.

    Returns:
        (N,) values.
    """
    interpolator = RegularGridInterpolator(axes, grid, bounds_error=False, fill_value=0.0)
    return interpolator(points).astype(FP_DTYPE)


def calculate_cp_lp(
    points: np.ndarray,
    atom_coords: np.ndarray,
    chg: np.ndarray,
    hp: np.ndarray,
    c1: float = 1.0,
    c2: float = SIGMA_HYDRO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the charge potential (CP) and lipophilic potential (LP) at pocket points.

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

    dist = np.sqrt(dist_sq)
    weight = switching(dist).astype(dist.dtype)

    # Charge potential: sum(s(r) q / r²) × 557 (kcal/mol·e units approximation)
    with np.errstate(divide="ignore", invalid="ignore"):
        chg_matrix = np.divide(chg_f[np.newaxis, :], dist_sq)
        chg_matrix[np.isinf(chg_matrix)] = 0.0
    cp = np.sum(chg_matrix * weight, axis=1) * 557.0

    # Lipophilic potential: Fermi-weighted hydrophobicity, switched off with distance
    fermi_wt = np.zeros_like(dist)
    num = math.exp(-c1 * c2) + 1.0
    exponent = c1 * (dist - c2)
    mask = exponent < 50.0
    fermi_wt[mask] = num / (np.exp(exponent[mask]) + 1.0)
    fermi_wt *= weight
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


def structure_sites(structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H-bond donor, acceptor and aromatic-ring positions of a protein.

    Args:
        structure: Parsed Biopython structure.

    Returns:
        (donors, acceptors, aromatic ring centres), each (K, 3).
    """
    geo_donors, geo_acceptors, geo_arom = [], [], []

    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res):
                    continue
                resname = res.get_resname()

                if resname in AROMATIC_RESIDUES:
                    ring_atoms = []
                    if resname in ["PHE", "TYR"]:
                        ring_atoms = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
                    elif resname == "TRP":
                        ring_atoms = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3"]
                    elif resname in ["HIS", "HID", "HIE", "HIP"]:
                        ring_atoms = ["CG", "ND1", "CD2", "CE1", "NE2"]

                    pts = [res[at].get_coord() for at in ring_atoms if at in res]
                    if pts:
                        geo_arom.append(np.mean(pts, axis=0))

                for atom in res:
                    elem = atom.element
                    if resname in DONOR_RESIDUES and (
                        elem == "N" or (elem == "O" and resname in ["SER", "THR"])
                    ):
                        geo_donors.append(atom.get_coord())
                    if resname in ACCEPTOR_RESIDUES and (
                        elem == "O"
                        or (elem == "N" and resname in ["HIS", "HID", "HIE", "HIP"])
                    ):
                        geo_acceptors.append(atom.get_coord())

    def safe_arr(lst):
        arr = np.array(lst, dtype=FP_DTYPE)
        return arr.reshape(-1, 3) if arr.ndim == 1 else arr

    return safe_arr(geo_donors), safe_arr(geo_acceptors), safe_arr(geo_arom)


#: Providers a run uses unless features.enabled says otherwise.
DEFAULT_PROVIDERS = ("apbs", "cp_lp", "hbond", "aromatic")


def pocket_features(points: np.ndarray, data: StructureData, apbs: np.ndarray,
                    provider_names: Sequence[str] = DEFAULT_PROVIDERS) -> Dict[str, np.ndarray]:
    """The per-point features of a pocket, from the enabled providers.

    Args:
        points: (N, 3) pocket points.
        data: Protein atoms, charges, hydrophobicity and pharmacophore sites.
        apbs: (N,) electrostatic potential, used by the apbs provider.
        provider_names: Providers to run.

    Returns:
        One column per provider column, in FEATURE_COLUMNS order.
    """
    from qpocket.core.providers.base import FeatureContext, columns_for, resolve

    chosen = resolve(provider_names)
    ctx = FeatureContext(points=points, pdb_path=None, pqr_path=None,
                         structure_data=data, options={"apbs": apbs})
    produced: Dict[str, np.ndarray] = {}
    for provider in chosen:
        produced.update(provider.compute(points, ctx))
    return {column: produced[column] for column in columns_for(chosen)}
