"""The docking box of a finalised site: around its contacted residues or its representatives' clouds."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from qpocket.core.screening import calculate_box, extremum_points
from qpocket.io import final as io_final

INTERACTING, CLOUDS = "interacting_residues", "representative_pockets"


class BoxError(ValueError):
    """A box cannot be made for this site: not finalised, or nothing to build it from."""


class BoxPreview(NamedTuple):
    """One box mode for one site, with what the viewers draw.

    Attributes:
        mode: interacting_residues or representative_pockets.
        padding: Å added on every face of the tight box.
        tight: Box around the points (calculate_box keys).
        padded: The tight box grown by ``padding``.
        points: The coordinates the box is built from.
        extremes: The six points that set its faces.
        extreme_labels: -x, +x, -y, +y, -z, +z.
        residues: The contacted residues (interacting mode).
        clouds: Representative id to its cloud (cloud mode).
        cluster_of: Representative id to its cluster name.
        frame_pdb: Aligned structure the residues and cartoon come from.
        final_hash: The final's clustering hash, recorded with a saved box.
    """

    mode: str
    padding: float
    tight: Dict[str, float]
    padded: Dict[str, float]
    points: np.ndarray
    extremes: np.ndarray
    extreme_labels: List[str]
    residues: List[str]
    clouds: Dict[str, np.ndarray]
    cluster_of: Dict[str, str]
    frame_pdb: Path
    final_hash: str


def _site(run_dir: Path, site_id: str) -> Tuple[Path, dict]:
    site = Path(run_dir) / "sites" / site_id
    if io_final.final_status(site) != "current":
        raise BoxError(f"finalise this site first: {site_id} has no current final metadata")
    return site, io_final.read_final(site)


def _frame(run_dir: Path, site: Path, reps: List[str]) -> Path:
    """The aligned file of the run's reference when it is a site member, else of the first representative."""
    from qpocket.config import load_run_config
    from qpocket.io import layout
    from qpocket.pipelines.sites import _reference_id

    master = pd.read_csv(site / "metadata" / "master.csv", dtype=str, keep_default_na=False)
    aligned = dict(zip(master["pdb_id"], master["aligned_file"]))
    try:
        ref = _reference_id(load_run_config(run_dir), layout.workspace(run_dir),
                            [SimpleNamespace(pdb_id=p) for p in master["pdb_id"]])
    except Exception:  # an unreadable config only costs the choice of frame
        ref = None
    for pid in [ref, *reps, *master["pdb_id"]]:
        if pid and aligned.get(pid) and Path(aligned[pid]).exists():
            return Path(aligned[pid])
    raise BoxError(f"no aligned structure found for {site.name}")


def _preview(mode, padding, points, residues, clouds, cluster_of, frame, final) -> BoxPreview:
    extremes, labels = extremum_points(points)
    return BoxPreview(mode, float(padding), calculate_box(points, padding=0.0), calculate_box(points, padding=padding),
                      points, extremes, labels, residues, clouds, cluster_of, frame,
                      str(final["settings"].get("clustering_hash", "")))


def _reps(final) -> Tuple[List[str], Dict[str, str]]:
    clusters = final["clusters"]
    reps = clusters[clusters["is_representative"].astype(str).str.lower() == "true"]
    return list(reps["pdb_id"]), dict(zip(reps["pdb_id"], reps["cluster"]))


def interacting_box(run_dir: Path, site_id: str, padding: float) -> BoxPreview:
    """The box around every residue contacted at least once at the site, in the frame structure.

    Args:
        run_dir: The run directory.
        site_id: A finalised site.
        padding: Å on every face.

    Returns:
        The preview.

    Raises:
        BoxError: If the site is not finalised, nothing was contacted, or no residue is found in the frame.
    """
    from biopandas.pdb import PandasPdb

    from qpocket.core.pocket import interacting_residue_coords

    site, final = _site(run_dir, site_id)
    profile = final["residue_profile"]
    contacted = profile[pd.to_numeric(profile["contacted_pct"], errors="coerce") > 0] if len(profile) else profile
    residues = list(dict.fromkeys(contacted["residue"])) if len(contacted) else []
    if not residues:
        raise BoxError(f"no residue was contacted at {site_id}; use the representatives' clouds")
    reps, cluster_of = _reps(final)
    frame = _frame(run_dir, site, reps)
    points = interacting_residue_coords(PandasPdb().read_pdb(str(frame)).df["ATOM"], [f"{r}|contact" for r in residues])
    if points is None or not len(points):
        raise BoxError(f"none of {site_id}'s contacted residues is in {frame.name}")
    return _preview(INTERACTING, padding, points, residues, {}, cluster_of, frame, final)


def cloud_box(run_dir: Path, site_id: str, padding: float) -> BoxPreview:
    """The box around the final representatives' pocket clouds.

    Args:
        run_dir: The run directory.
        site_id: A finalised site.
        padding: Å on every face.

    Returns:
        The preview.

    Raises:
        BoxError: If the site is not finalised or no representative has a cloud.
    """
    site, final = _site(run_dir, site_id)
    reps, cluster_of = _reps(final)
    clouds = {}
    for rep in reps:
        path = site / "pockets" / rep / "binding_site.csv"
        if path.exists():
            clouds[rep] = pd.read_csv(path)[["x", "y", "z"]].to_numpy(float)
    if not clouds:
        raise BoxError(f"no representative of {site_id} has a pocket cloud")
    points = np.vstack(list(clouds.values()))
    return _preview(CLOUDS, padding, points, [], clouds, {r: cluster_of[r] for r in clouds},
                    _frame(run_dir, site, reps), final)


def save_box(run_dir: Path, site_id: str, preview: BoxPreview, now: Optional[datetime] = None) -> Path:
    """Write the box preparation will use, ``sites/<id>/screening/box.json``.

    Args:
        run_dir: The run directory.
        site_id: The site.
        preview: The box to save (its padded box is the search box).
        now: The time to record; now when None.

    Returns:
        Path of box.json.
    """
    path = Path(run_dir) / "sites" / site_id / "screening" / "box.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"site_id": site_id, "mode": preview.mode, "padding": preview.padding,
               "saved_at": (now or datetime.now()).replace(microsecond=0).isoformat(),
               "final_hash": preview.final_hash,
               **{k: float(v) for k, v in preview.padded.items()},
               **({"residues": preview.residues} if preview.mode == INTERACTING
                  else {"representatives": sorted(preview.clouds)})}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def saved_box(run_dir: Path, site_id: str) -> Tuple[Optional[dict], str]:
    """The site's saved box and whether it belongs to its current final.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        (box or None, "none" | "stale" | "current"); stale when the site was finalised again, or its final is stale.
    """
    site = Path(run_dir) / "sites" / site_id
    path = site / "screening" / "box.json"
    if not path.exists():
        return None, "none"
    saved = json.loads(path.read_text(encoding="utf-8"))
    if io_final.final_status(site) != "current":
        return saved, "stale"
    current = io_final.read_final(site)["settings"].get("clustering_hash")
    return saved, "current" if saved.get("final_hash") == current else "stale"
