"""Pocket comparison: any structures, pockets named by a ligand or residues, no alignment."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from qpocket.core import pocket as pk
from qpocket.core import spectral as sp
from qpocket.core import structure as st
from qpocket.core.features import DEFAULT_PROVIDERS, pocket_features
from qpocket.core.ports import FEATURES
from qpocket.io import rcsb, tables, tools
from qpocket.io import structures as sio
from qpocket.types import ComparisonResult, PocketSpec, Scaler, StructureData


def _resolve(spec: PocketSpec, workdir: Path) -> Path:
    path = Path(spec.structure)
    if path.exists():
        return path
    rcsb.download_structures([spec.structure.upper()], str(workdir))
    downloaded = workdir / f"{spec.structure.upper()}.cif"
    if not downloaded.exists():
        raise FileNotFoundError(f"{spec.structure}: not a file and not downloadable from RCSB")
    return downloaded


def featurize(spec: PocketSpec, workdir: Optional[Path] = None,
              features: Sequence[str] = DEFAULT_PROVIDERS) -> Tuple[np.ndarray, np.ndarray]:
    """Find one pocket and compute its per-point features.

    Args:
        spec: Structure and pocket definition.
        workdir: Scratch directory for downloads and tool files (a temporary one if None).
        features: Feature providers to run.

    Returns:
        (points (N, 3), features (N, len(columns)) in provider order).

    Raises:
        ValueError: If no cavity matches the pocket definition.
    """
    spec.validate()
    workdir = Path(workdir or tempfile.mkdtemp(prefix="qpocket_"))
    workdir.mkdir(parents=True, exist_ok=True)

    structure, _ = st.without_water(sio.load(_resolve(spec, workdir)))
    protein = st.keep_chains(structure, [spec.chain]) if spec.chain else structure
    clean, _ = st.cleaned(protein, [])
    name = Path(spec.structure).stem
    clean_path = workdir / f"{name}_clean.pdb"
    sio.save(clean, clean_path)

    cavities = tools.run_cavfind(clean_path, name)
    if spec.ligand:
        centroid = pk.ligand_centroid(structure, spec.ligand)
        if centroid is None:
            raise ValueError(f"{spec.structure}: ligand {spec.ligand} not found")
        points = pk.select_cavity_by_centroid(cavities, centroid)
    else:
        lining = pk.residue_coords(protein, spec.residues, spec.chain)
        if lining is None:
            raise ValueError(f"{spec.structure}: none of {', '.join(spec.residues)} found")
        points = pk.select_cavity_by_residues(cavities, lining, spec.padding)
    if points is None:
        raise ValueError(f"{spec.structure}: no cavity matches the pocket definition")

    chosen = FEATURES.resolve(features)
    data = StructureData.empty()
    apbs = np.zeros(len(points))
    if any("pqr" in provider.requires for provider in chosen):
        pqr_path = tools.protonate(clean_path, workdir / f"{name}.pqr")
        data = tools.structure_data(clean_path, tables.read_pqr(pqr_path))
        if any(provider.name == "apbs" for provider in chosen):
            apbs = tools.apbs_potential(pqr_path, points)
    feats = pocket_features(points, data, apbs, features)
    return points, np.column_stack(list(feats.values()))


def compare_features(
    pockets: Sequence[Tuple[np.ndarray, np.ndarray]],
    ids: Sequence[str],
    scaler: Optional[Scaler] = None,
    global_k: int = 100,
    sigma: float = 10.0,
    radius: float = 2.0,
    shape_weight: float = 0.85,
    chem_scale: float = 12.5,
) -> ComparisonResult:
    """Pairwise distances between already-featurised pockets.

    Args:
        pockets: (points, features) per pocket.
        ids: Labels, same order.
        scaler: Reuse a run's scaler; fitted on these pockets when None.
        global_k: Fingerprint length.
        sigma: Heat-kernel width.
        radius: Geodesic neighbour radius (A).
        shape_weight: Weight of the shape term.
        chem_scale: Scale of the chemistry term.

    Returns:
        Shape, chemistry and combined distance matrices.

    Raises:
        ValueError: If a pocket is too small or disconnected to fingerprint.
    """
    scaler = scaler or sp.fit_scaler([f for _, f in pockets])
    fingerprints = []
    for label, (points, features) in zip(ids, pockets):
        d_geo = sp.geodesic_matrix(np.asarray(points, dtype=float), radius)
        fp = None if d_geo is None else sp.fingerprint(d_geo, features, scaler, global_k, sigma)
        if fp is None:
            raise ValueError(f"{label}: pocket too small or disconnected to fingerprint")
        fingerprints.append(fp)
    shape, chem, combined = sp.distance_matrices(fingerprints, shape_weight, chem_scale)
    return ComparisonResult(tuple(ids), shape, chem, combined, scaler)


def run(specs: Sequence[PocketSpec], workdir: Optional[Path] = None, scaler: Optional[Scaler] = None,
        **options) -> ComparisonResult:
    """Featurise pockets from structures and compare them all pairwise.

    Args:
        specs: Pockets to compare (two or more).
        workdir: Scratch directory (temporary if None).
        scaler: Reuse a run's scaler; fitted on these pockets when None.
        **options: features, global_k, sigma, radius, shape_weight, chem_scale.

    Returns:
        Pairwise distances.
    """
    labels = [f"{Path(s.structure).stem}:{s.ligand or ','.join(s.residues)}" for s in specs]
    features = options.pop("features", DEFAULT_PROVIDERS)
    pockets = [featurize(s, workdir, features) for s in specs]
    return compare_features(pockets, labels, scaler, **options)
