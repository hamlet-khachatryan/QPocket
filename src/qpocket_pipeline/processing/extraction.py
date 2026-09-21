"""Pocket extraction, featurisation, and interaction analysis.

This module provides three public functions:

* :func:`extract_reference_pocket_bbox` — processes the reference structure,
  finds the binding pocket via centroid overlap, and returns its axis-aligned
  bounding box with 1 Å padding.

* :func:`extract_pocket_from_structure` — extracts the binding pocket for an
  arbitrary structure.  When a reference bounding box is supplied (all
  non-reference structures) it replaces the centroid-check with a volumetric
  overlap check, selecting the cavity with the largest overlap.  Ligand
  selection picks the **largest** (most heavy atoms) heteroatom molecule
  overlapping with the pocket.

* :func:`run_analysis_step` — derives potential (geometric) and observed
  (PLIP) interactions for a structure.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from eliot import log_message, start_action
from plip.structure.preparation import PDBComplex

from cavitomix.cavfind import CavFind
from qpocket_pipeline.utils import (
    check_overlap,
    compute_pocket_bounding_box,
    compute_volumetric_overlap,
    crop_pocket_by_interactions,
    create_bounding_box_pocket,
    extract_pdb_block,
    get_interacting_residues_coords,
    get_smiles_from_pdb_block,
    is_residue_ligand,
)
from qpocket_pipeline.distance_logic.features import featurize

warnings.simplefilter("ignore", PDBConstructionWarning)


# ---------------------------------------------------------------------------
# Internal helper: safe attribute access for CavFind cavity objects
# ---------------------------------------------------------------------------

def _cavity_coords(cavity) -> np.ndarray:
    """Returns the coordinate array of a CavFind cavity object.

    Centralises access to the internal ``coords`` attribute so that a single
    change here covers all call sites if cavitomix's API evolves.

    Args:
        cavity: A cavity object returned by :class:`cavitomix.cavfind.CavFind`.

    Returns:
        ``(N, 3)`` numpy array of cavity grid point coordinates.
    """
    return cavity.__dict__["coords"]


# ---------------------------------------------------------------------------
# Reference pocket bounding box
# ---------------------------------------------------------------------------

def extract_reference_pocket_bbox(
    pdb_path: Path,
    pdb_id: str,
    ref_centroid: np.ndarray,
    pocket_save_dir: Path,
    save_intermediate: bool = False,
    bbox_padding: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Finds the reference binding pocket and derives its bounding box.

    Runs cavity detection on the reference structure, finds the cavity whose
    point cloud overlaps with *ref_centroid*, writes the pocket CSV, and
    returns the axis-aligned bounding box of that point cloud expanded by
    *bbox_padding* Å on every face.

    Args:
        pdb_path: Path to the cleaned reference PDB file.
        pdb_id: PDB identifier (used for naming and logging).
        ref_centroid: 3-element array — geometric centroid of the reference
            ligand used to identify the correct cavity.
        pocket_save_dir: Directory in which to write the pocket CSV.
        save_intermediate: When *True*, also write the raw (pre-crop) CSV.
        bbox_padding: Expansion of the bounding box in Å (default 1 Å).

    Returns:
        ``(bbox_min, bbox_max)`` numpy arrays of shape ``(3,)`` defining the
        reference bounding box, or ``None`` if cavity detection fails.
    """
    with start_action(
        action_type="extract_reference_pocket_bbox",
        pdb_id=pdb_id,
        bbox_padding=bbox_padding,
    ):
        pocket_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(pdb_path, "r") as f:
                cf = CavFind(name=pdb_id, obj_name=pdb_id)
                cf.struct_from_pdb(f.read())
                found_cavities = cf.run().cavities
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

        target_cavity = None
        for cavity in found_cavities:
            coords = _cavity_coords(cavity)
            if check_overlap(ref_centroid, coords, use_majority=True):
                target_cavity = cavity
                break

        if target_cavity is None:
            log_message(
                message_type="warning",
                stage="extract_reference_pocket_bbox",
                pdb_id=pdb_id,
                warning="reference centroid does not overlap any cavity",
            )
            return None

        coords = _cavity_coords(target_cavity)
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
) -> Tuple[Optional[str], str]:
    """Extracts the binding pocket and largest co-crystallised ligand.

    Pocket identification strategy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **bounding_box** mode: the pocket is defined by an explicit box supplied
      via *bounding_box_center* / *bounding_box_dims*.
    * **cavity_detection** mode (reference structure, ``ref_pocket_bbox`` is
      ``None``): cavities are compared against *ref_centroid* using majority
      overlap.
    * **cavity_detection** mode (all other structures, ``ref_pocket_bbox``
      provided): each cavity is scored by the number of its points inside the
      reference bounding box; the cavity with the highest volumetric overlap
      is selected.

    Ligand selection
    ~~~~~~~~~~~~~~~~
    All heteroatom residues that overlap with the final pocket are collected;
    the one with the most heavy atoms is returned.

    Args:
        row: DataFrame row with ``pdb_id``, ``cleaned_file``, ``aligned_file``.
        dirs: Directory path dictionary from ``setup_directory_hierarchy()``.
        ref_centroid: Reference ligand centroid (used for reference structure
            processing when ``ref_pocket_bbox`` is ``None``).
        allowed_cofactors: ``"|"``-separated list of cofactor residue names to
            exclude from ligand selection.
        pocket_mode: ``"cavity_detection"`` or ``"bounding_box"``.
        bounding_box_center: Box centre for bounding_box mode.
        bounding_box_dims: Box dimensions for bounding_box mode.
        pocket_definition: ``"Solvent"`` (full cavity) or ``"Interactions"``
            (cropped to interacting residues).
        interaction_padding: Crop radius around interacting residues (Å).
        observed_interactions: Interaction list for Interactions mode.
        ref_pocket_bbox: Pre-computed reference bounding box
            ``(bbox_min, bbox_max)`` used for non-reference structures.
        save_intermediate: Persist the pre-crop pocket point cloud.
        force_featurize: When ``True``, delete any existing ``featurized.csv``
            and re-run featurisation.  Required for the Interactions second
            pass to keep ``featurized.csv`` consistent with the cropped
            ``binding_site.csv``.

    Returns:
        ``(ligand_file_path_or_None, status_string)`` where status is one of
        ``BOUND_STATE``, ``UNBOUND_STATE``, ``CAVITY_CLOSED``,
        ``NO_ANY_CAVITY_FOUND``, or ``FAILED``.
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
                with open(cleaned_path, "r") as f:
                    cf = CavFind(name=pdb_id, obj_name=pdb_id)
                    cf.struct_from_pdb(f.read())
                    found_cavities = cf.run().cavities
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
                target_cavity = None
                for cavity in found_cavities:
                    coords = _cavity_coords(cavity)
                    if check_overlap(ref_centroid, coords, use_majority=True):
                        target_cavity = cavity
                        break

                if target_cavity is None:
                    return None, "CAVITY_CLOSED"
                target_coords = _cavity_coords(target_cavity)

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
                featurize(final_pocket_csv, cleaned_path)

        # ── Ligand extraction (select largest molecule) ───────────────────────
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdb_id, str(aligned_path))
        except Exception:
            return None, "FAILED"

        candidates: List[Tuple] = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if not is_residue_ligand(res, allowed_cofactors_list):
                        continue
                    atoms = np.array([a.get_coord() for a in res])
                    if len(atoms) == 0:
                        continue
                    if check_overlap(atoms, final_pocket_coords, use_majority=True):
                        candidates.append((res, chain.id, len(atoms)))

        if not candidates:
            return None, "UNBOUND_STATE"

        best_res, best_chain_id, n_atoms = max(candidates, key=lambda x: x[2])
        res_name = best_res.get_resname().strip()
        fname = f"{pdb_id}_{res_name}.pdb"

        block = extract_pdb_block(best_res, best_chain_id)
        lig_out = dirs["extracted_ligands"] / fname
        with open(lig_out, "w") as f:
            f.write(block)

        smi = get_smiles_from_pdb_block(block)
        if smi:
            with open(dirs["screening"] / "actives.smi", "a") as f:
                f.write(f"{smi}\t{pdb_id}\t{res_name}\n")

        log_message(
            message_type="ligand_extracted",
            pdb_id=pdb_id,
            ligand=res_name,
            n_atoms=n_atoms,
            n_candidates=len(candidates),
        )
        return str(lig_out), "BOUND_STATE"


# ---------------------------------------------------------------------------
# Interaction analysis
# ---------------------------------------------------------------------------

def get_plip_interactions(pdb_path: Path, ligand_name: str) -> List[str]:
    """Analyses protein–ligand interactions with PLIP.

    Collects interactions from **all** binding sites whose residue name
    matches *ligand_name* (handles structures with multiple copies of the
    same ligand in different chains / symmetry mates).

    Args:
        pdb_path: Path to a PDB structure file.
        ligand_name: Residue name of the ligand to analyse (case-insensitive).

    Returns:
        List of interaction strings formatted as
        ``"RESNAME_RESNUM|InteractionType"``.  Returns ``["no_ligand_found"]``
        when no matching interactions are detected and ``["analysis_error"]``
        on PLIP failure.
    """
    interactions: List[str] = []

    try:
        protlig = PDBComplex()
        protlig.load_pdb(str(pdb_path))
        protlig.analyze()

        for k, binding_site in protlig.interaction_sets.items():
            # PLIP key format: "RESNAME:CHAIN:RESNUM:" — match on residue name only
            # to capture all chains that carry the same ligand.
            if ligand_name.upper() != k.upper().split(":")[0]:
                continue

            bind_site_interactions = [
                i for i in binding_site.all_itypes if "waterbridge" not in str(i)
            ]

            for inter in bind_site_interactions:
                inter_type = str(inter).split("(")[0]

                # Interaction strings include the chain ID so that residues on
                # different chains are unambiguously distinguished (critical for
                # interface pockets with two chains carrying the same residue numbers).
                res_key = (
                    f"{inter.restype.upper()}_{inter.resnr}_{inter.reschain}"
                )

                if inter_type == "pistack":
                    interactions.append(
                        f"{res_key}|Pi-stacking ({inter.type})"
                    )
                elif inter_type == "hbond":
                    itype = "H-bond Donor" if inter.protisdon else "H-bond Acceptor"
                    interactions.append(f"{res_key}|{itype}")
                elif inter_type == "pication":
                    itype = "Pi-Cation (Cat)" if inter.protcharged else "Pi-Cation (Pi)"
                    interactions.append(f"{res_key}|{itype}")
                elif inter_type == "saltbridge":
                    itype = "Saltbridge (+)" if inter.protispos else "Saltbridge (-)"
                    interactions.append(f"{res_key}|{itype}")
                elif inter_type == "metal_complex":
                    interactions.append(f"{res_key}|Metal Complex")
                elif inter_type == "halogenbond":
                    interactions.append(f"{res_key}|Halogen Bond")
                elif inter_type == "hydroph_interaction":
                    interactions.append(f"{res_key}|Hydrophobic")

    except Exception as e:
        log_message(
            message_type="error",
            stage="plip_analysis",
            pdb_path=str(pdb_path),
            error=str(e),
        )
        return ["analysis_error"]

    return interactions if interactions else ["no_ligand_found"]


def run_analysis_step(
    row: pd.Series,
    dirs: Dict[str, Path],
) -> Dict[str, List[str]]:
    """Derives potential and observed interactions for a structure.

    Potential interactions are all protein residues within 4 Å of the pocket
    point cloud.  Observed interactions are from PLIP analysis of the bound
    ligand (if any).

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

def _write_coords_csv(coords: np.ndarray, path: Path) -> None:
    """Writes a point cloud to a CSV file with x, y, z columns."""
    with open(path, "w") as f:
        f.write("x,y,z\n")
        for pt in coords:
            f.write(f"{pt[0]},{pt[1]},{pt[2]}\n")


def _select_best_cavity_by_bbox(
    cavities,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    pdb_id: str,
) -> Optional[np.ndarray]:
    """Returns coords of the cavity with the highest volumetric overlap with the bbox.

    Args:
        cavities: Iterable of cavity objects from CavFind.
        bbox_min: Lower corner of the reference bounding box.
        bbox_max: Upper corner of the reference bounding box.
        pdb_id: Used for logging only.

    Returns:
        Point cloud array ``(N, 3)`` of the best cavity, or ``None`` if no
        cavity overlaps at all.
    """
    best_coords: Optional[np.ndarray] = None
    best_overlap: int = 0

    for cavity in cavities:
        coords = _cavity_coords(cavity)
        if len(coords) == 0:
            continue
        overlap = compute_volumetric_overlap(coords, bbox_min, bbox_max)
        if overlap > best_overlap:
            best_overlap = overlap
            best_coords = coords

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
