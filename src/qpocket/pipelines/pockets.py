"""Pocket extraction, featurisation, and interaction analysis."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from eliot import log_message, start_action
from qpocket.core.geometry import (
    compute_pocket_bounding_box,
    crop_pocket_by_interactions,
    create_bounding_box_pocket,
)
from qpocket.core.pocket import (
    extract_pdb_block,
    pick_ligand,
    select_cavity_by_bbox,
    select_cavity_by_centroid,
)
from qpocket.core import pocket
from qpocket.core.features import DEFAULT_PROVIDERS, pocket_features
from qpocket.core.providers import base
from qpocket.io import tables, tools
from qpocket.types import StructureData
from qpocket.io.tables import write_points as _write_coords_csv
from qpocket.io.tools import plip_interactions as get_plip_interactions  # noqa: F401 (re-exported)
from qpocket.io.tools import run_cavfind

warnings.simplefilter("ignore", PDBConstructionWarning)


# ---------------------------------------------------------------------------
# Internal helper: safe attribute access for CavFind cavity objects
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reference pocket bounding box
# ---------------------------------------------------------------------------

def featurize(csv_path: Path, pdb_path: Path,
              providers: Sequence[str] = DEFAULT_PROVIDERS) -> None:
    """Write featurized.csv beside a pocket's point cloud.

    Args:
        csv_path: Pocket point cloud CSV (x, y, z).
        pdb_path: Cleaned protein PDB.
        providers: Feature providers to run.
    """
    with start_action(action_type="process_point_cloud"):
        if not csv_path.exists():
            log_message(message_type="error", error="CSV missing", path=str(csv_path))
            return

        try:
            table, points = tables.read_point_table(csv_path)
        except Exception as e:
            log_message(message_type="error", error=f"CSV read error: {e}")
            return

        chosen = base.resolve(providers)
        needs_pqr = any("pqr" in provider.requires for provider in chosen)
        wants_apbs = any(provider.name == "apbs" for provider in chosen)

        pqr_path = csv_path.with_name("protein.pqr")
        data = StructureData.empty()
        apbs = np.zeros(len(points))
        try:
            if needs_pqr:
                # One pdb2pqr run per structure; reused by the Interactions second pass.
                if not pqr_path.exists() or pqr_path.stat().st_mtime < Path(pdb_path).stat().st_mtime:
                    tools.protonate(pdb_path, pqr_path)
                data = tools.structure_data(pdb_path, tables.read_pqr(pqr_path))
        except Exception as e:
            log_message(message_type="fatal_error", error=str(e))
            return

        out_path = csv_path.with_name("featurized.csv")
        if wants_apbs:
            log_message(message_type="status", msg="Running APBS")
            apbs = tools.apbs_potential(pqr_path, points)
        log_message(message_type="status", msg="Running the feature providers",
                    providers=list(providers))
        tables.write_featurized(out_path, table,
                                pocket_features(points, data, apbs, providers))
        log_message(message_type="success", output=str(out_path))


def get_ref_ligand_centroid(
    ref_pdb_path: Path,
    ligand_resname: str,
    search_chains: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """Centroid of the reference ligand in a PDB file.

    Args:
        ref_pdb_path: Aligned reference PDB.
        ligand_resname: Ligand residue name.
        search_chains: Chains to search, or None for all.

    Returns:
        (3,) centroid, or None if absent or unreadable.
    """
    try:
        structure = PDBParser(QUIET=True).get_structure("reference", ref_pdb_path)
        centroid = pocket.ligand_centroid(structure, ligand_resname, search_chains)
        if centroid is None:
            searched = search_chains if search_chains is not None else "all chains"
            log_message(
                message_type="warning",
                stage="get_ref_ligand_centroid",
                warning=(
                    f"ligand {ligand_resname!r} not found in {searched} "
                    f"of {ref_pdb_path}"
                ),
            )
        return centroid
    except Exception as e:
        log_message(
            message_type="error",
            stage="get_ref_ligand_centroid",
            error=str(e),
        )
        return None


def get_interacting_residues_coords(
    pdb_path: Path,
    interactions_list: List[str],
    padding: float = 5.0,
) -> Optional[np.ndarray]:
    """Atom coordinates of the residues named in interaction strings.

    Args:
        pdb_path: Aligned PDB.
        interactions_list: Strings like "TYR_795_A|H-bond Donor".
        padding: Unused; kept for callers.

    Returns:
        (M, 3) coordinates, or None.
    """
    try:
        atoms = PandasPdb().read_pdb(str(pdb_path)).df["ATOM"]
        return pocket.interacting_residue_coords(atoms, interactions_list)
    except Exception as e:
        log_message(
            message_type="error",
            stage="get_interacting_residues_coords",
            error=str(e),
        )
        return None


def extract_reference_pocket_bbox(
    pdb_path: Path,
    pdb_id: str,
    ref_centroid: np.ndarray,
    pocket_save_dir: Path,
    save_intermediate: bool = False,
    bbox_padding: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Finds the reference binding pocket and derives its bounding box.

    Args:
        pdb_path: Path to the cleaned reference PDB file.
        pdb_id: PDB identifier (used for naming and logging).
        ref_centroid: 3-element array — geometric centroid of the reference ligand used to identify the correct cavity.
        pocket_save_dir: Directory in which to write the pocket CSV.
        save_intermediate: When *True*, also write the raw (pre-crop) CSV.
        bbox_padding: Expansion of the bounding box in Å (default 1 Å).

    Returns:
        ``(bbox_min, bbox_max)`` numpy arrays of shape ``(3,)`` defining the reference bounding box, or ``None`` if cavity detection fails.
    """
    with start_action(
        action_type="extract_reference_pocket_bbox",
        pdb_id=pdb_id,
        bbox_padding=bbox_padding,
    ):
        pocket_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            found_cavities = run_cavfind(pdb_path, pdb_id)
        except Exception as e:
            log_message(
                message_type="error",
                stage="extract_reference_pocket_bbox",
                pdb_id=pdb_id,
                error=str(e),
            )
            return None

        if not found_cavities:
            log_message(
                message_type="warning",
                stage="extract_reference_pocket_bbox",
                pdb_id=pdb_id,
                warning="no cavities found in reference structure",
            )
            return None

        coords = select_cavity_by_centroid(found_cavities, ref_centroid)
        if coords is None:
            log_message(
                message_type="warning",
                stage="extract_reference_pocket_bbox",
                pdb_id=pdb_id,
                warning="reference centroid does not overlap any cavity",
            )
            return None

        n_points = len(coords)

        pocket_csv = pocket_save_dir / "binding_site.csv"
        _write_coords_csv(coords, pocket_csv)

        if save_intermediate:
            _write_coords_csv(coords, pocket_save_dir / "binding_site_initial.csv")

        bbox_min, bbox_max = compute_pocket_bounding_box(coords, padding=bbox_padding)

        log_message(
            message_type="reference_bbox_computed",
            pdb_id=pdb_id,
            n_points=n_points,
            bbox_min=bbox_min.tolist(),
            bbox_max=bbox_max.tolist(),
        )
        return bbox_min, bbox_max


# ---------------------------------------------------------------------------
# Main pocket extraction
# ---------------------------------------------------------------------------

def extract_pocket_from_structure(
    row: pd.Series,
    dirs: Dict[str, Path],
    ref_centroid: Optional[np.ndarray],
    allowed_cofactors: str,
    pocket_mode: str = "cavity_detection",
    bounding_box_center: Optional[List[float]] = None,
    bounding_box_dims: Optional[List[float]] = None,
    pocket_definition: str = "Solvent",
    interaction_padding: float = 5.0,
    observed_interactions: Optional[List[str]] = None,
    ref_pocket_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    save_intermediate: bool = False,
    force_featurize: bool = False,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
) -> Tuple[Optional[str], str]:
    """Extracts the binding pocket and largest co-crystallised ligand.

    Args:
        row: DataFrame row with ``pdb_id``, ``cleaned_file``, ``aligned_file``.
        dirs: Directory path dictionary from ``setup_directory_hierarchy()``.
        ref_centroid: Reference ligand centroid (used for reference structure processing when ``ref_pocket_bbox`` is ``None``).
        allowed_cofactors: ``"|"``-separated list of cofactor residue names to exclude from ligand selection.
        pocket_mode: ``"cavity_detection"`` or ``"bounding_box"``.
        bounding_box_center: Box centre for bounding_box mode.
        bounding_box_dims: Box dimensions for bounding_box mode.
        pocket_definition: ``"Solvent"`` (full cavity) or ``"Interactions"`` (cropped to interacting residues).
        interaction_padding: Crop radius around interacting residues (Å).
        observed_interactions: Interaction list for Interactions mode.
        ref_pocket_bbox: Pre-computed reference bounding box ``(bbox_min, bbox_max)`` used for non-reference structures.
        save_intermediate: Persist the pre-crop pocket point cloud.
        force_featurize: When ``True``, delete any existing ``featurized.csv`` and re-run featurisation.
        providers: Feature providers to run.

    Returns:
        ``(ligand_file_path_or_None, status_string)`` where status is one of ``BOUND_STATE``, ``UNBOUND_STATE``, ``CAVITY_CLOSED``, ``NO_ANY_CAVITY_FOUND``, or ``FAILED``.
    """
    pdb_id = row["pdb_id"]
    cleaned_file = row["cleaned_file"]
    aligned_file = row["aligned_file"]

    allowed_cofactors_list = allowed_cofactors.split("|") if allowed_cofactors else []

    if pd.isna(cleaned_file) or pd.isna(aligned_file):
        return None, "FAILED"

    cleaned_path = Path(cleaned_file)
    aligned_path = Path(aligned_file)

    if not cleaned_path.exists():
        return None, "FAILED"

    with start_action(action_type="extract_pocket", pdb_id=pdb_id, pocket_mode=pocket_mode):

        pocket_save_dir = dirs["pockets"] / pdb_id
        pocket_save_dir.mkdir(parents=True, exist_ok=True)

        # ── Determine pocket coordinates ──────────────────────────────────────
        if pocket_mode == "bounding_box":
            if bounding_box_center is None or bounding_box_dims is None:
                log_message(
                    message_type="error",
                    stage="extract_pocket",
                    pdb_id=pdb_id,
                    error="bounding_box parameters not provided",
                )
                return None, "FAILED"

            target_coords = create_bounding_box_pocket(
                bounding_box_center,
                bounding_box_dims,
                tolerance=2.0,
            )

        else:  # cavity_detection
            try:
                found_cavities = run_cavfind(cleaned_path, pdb_id)
            except Exception as e:
                log_message(
                    message_type="warning",
                    stage="extract_pocket",
                    pdb_id=pdb_id,
                    warning=f"cavity detection failed: {e}",
                )
                return None, "FAILED"

            if not found_cavities:
                return None, "NO_ANY_CAVITY_FOUND"

            if ref_pocket_bbox is not None:
                bbox_min, bbox_max = ref_pocket_bbox
                target_coords = _select_best_cavity_by_bbox(
                    found_cavities, bbox_min, bbox_max, pdb_id
                )
                if target_coords is None:
                    return None, "CAVITY_CLOSED"
            else:
                target_coords = select_cavity_by_centroid(found_cavities, ref_centroid)
                if target_coords is None:
                    return None, "CAVITY_CLOSED"

        # ── Save initial pocket (optional) ────────────────────────────────────
        if save_intermediate:
            _write_coords_csv(target_coords, pocket_save_dir / "binding_site_initial.csv")

        # ── Apply pocket definition (crop if Interactions mode) ───────────────
        final_pocket_coords = target_coords
        if pocket_definition == "Interactions" and observed_interactions:
            interacting_coords = get_interacting_residues_coords(
                aligned_path,
                observed_interactions,
                padding=interaction_padding,
            )
            if interacting_coords is not None:
                cropped = crop_pocket_by_interactions(
                    target_coords, interacting_coords, padding=interaction_padding
                )
                if len(cropped) > 0:
                    final_pocket_coords = cropped
                    log_message(
                        message_type="pocket_cropped",
                        pdb_id=pdb_id,
                        n_before=len(target_coords),
                        n_after=len(final_pocket_coords),
                    )

        # ── Save final pocket CSV ─────────────────────────────────────────────
        final_pocket_csv = pocket_save_dir / "binding_site.csv"
        _write_coords_csv(final_pocket_coords, final_pocket_csv)

        # ── Featurise pocket ──────────────────────────────────────────────────
        with start_action(action_type="featurise_pocket", pdb_id=pdb_id):
            features_path = pocket_save_dir / "featurized.csv"
            if force_featurize and features_path.exists():
                features_path.unlink()

            if features_path.exists():
                log_message(
                    message_type="skip",
                    stage="featurise",
                    pdb_id=pdb_id,
                    reason="already featurised",
                )
            else:
                featurize(final_pocket_csv, cleaned_path, providers)

        # ── Ligand extraction (select largest molecule) ───────────────────────
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdb_id, str(aligned_path))
        except Exception:
            return None, "FAILED"

        choice = pick_ligand(structure, final_pocket_coords, allowed_cofactors_list)
        if choice is None:
            return None, "UNBOUND_STATE"

        best_res, best_chain_id, n_atoms = choice.residue, choice.chain_id, choice.n_atoms
        res_name = best_res.get_resname().strip()
        fname = f"{pdb_id}_{res_name}.pdb"

        block = extract_pdb_block(best_res, best_chain_id)
        lig_out = dirs["extracted_ligands"] / fname
        with open(lig_out, "w", encoding="utf-8") as f:
            f.write(block)

        log_message(
            message_type="ligand_extracted",
            pdb_id=pdb_id,
            ligand=res_name,
            n_atoms=n_atoms,
            n_candidates=choice.n_candidates,
        )
        return str(lig_out), "BOUND_STATE"


# ---------------------------------------------------------------------------
# Interaction analysis
# ---------------------------------------------------------------------------

def run_analysis_step(
    row: pd.Series,
    dirs: Dict[str, Path],
) -> Dict[str, List[str]]:
    """Derives potential and observed interactions for a structure.

    Args:
        row: DataFrame row with structure metadata.
        dirs: Directory path dictionary.

    Returns:
        ``{"potential": [...], "observed": [...]}``.
    """
    pdb_id = row["pdb_id"]
    res: Dict[str, List[str]] = {"potential": [], "observed": []}

    with start_action(action_type="analyse_interactions", pdb_id=pdb_id):

        pocket_dir = dirs["pockets"] / pdb_id
        cav_csv = (
            next(pocket_dir.glob("binding_site.csv"), None)
            if pocket_dir.exists()
            else None
        )

        if cav_csv and pd.notna(row.get("cleaned_file")):
            clean_path = Path(row["cleaned_file"])
            if clean_path.exists():
                try:
                    from scipy.spatial.distance import cdist as _cdist

                    ppdb = PandasPdb().read_pdb(str(clean_path))
                    atoms = ppdb.df["ATOM"]
                    cloud = pd.read_csv(cav_csv)[["x", "y", "z"]].to_numpy()
                    prot_coords = atoms[["x_coord", "y_coord", "z_coord"]].to_numpy()

                    dists = _cdist(prot_coords, cloud)
                    mask = dists.min(axis=1) <= 4.0
                    nearby = atoms[mask]
                    # Chain ID is included so that potential and observed
                    # interaction keys are in the same format and can be
                    # intersected correctly in utilization analysis.
                    unique_res = (
                        nearby["residue_name"]
                        + "_"
                        + nearby["residue_number"].astype(str)
                        + "_"
                        + nearby["chain_id"]
                    ).unique()
                    res["potential"] = sorted(unique_res.tolist())

                    log_message(
                        message_type="potential_interactions",
                        pdb_id=pdb_id,
                        n_residues=len(res["potential"]),
                    )
                except Exception as e:
                    log_message(
                        message_type="error",
                        stage="potential_interactions",
                        pdb_id=pdb_id,
                        error=str(e),
                    )

        if row.get("protein_state") == "BOUND_STATE" and pd.notna(row.get("ligand_file")):
            aligned_path = Path(row["aligned_file"])
            lig_name = Path(row["ligand_file"]).stem.split("_")[-1].upper()

            if aligned_path.exists():
                res["observed"] = get_plip_interactions(aligned_path, lig_name)
            else:
                res["observed"] = ["file_not_found"]
        else:
            res["observed"] = ["no_ligand_found"]

    return res


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _select_best_cavity_by_bbox(
    cavities,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    pdb_id: str,
) -> Optional[np.ndarray]:
    """Returns coords of the cavity with the highest volumetric overlap with the bbox.

    Args:
        cavities: Cavity point clouds from run_cavfind.
        bbox_min: Lower corner of the reference bounding box.
        bbox_max: Upper corner of the reference bounding box.
        pdb_id: Used for logging only.

    Returns:
        Point cloud array ``(N, 3)`` of the best cavity, or ``None`` if no cavity overlaps at all.
    """
    best_coords, best_overlap = select_cavity_by_bbox(cavities, bbox_min, bbox_max)

    if best_coords is None or best_overlap == 0:
        log_message(
            message_type="warning",
            stage="cavity_selection",
            pdb_id=pdb_id,
            warning="no cavity overlaps reference bounding box",
        )
        return None

    log_message(
        message_type="cavity_selected",
        pdb_id=pdb_id,
        volumetric_overlap=best_overlap,
        n_points=len(best_coords),
    )
    return best_coords
