import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from typing import Dict, List, Optional, Tuple, Any, Union

from qpocket_pipeline.global_vars import HET_TO_REMOVE

from eliot import start_action, log_message
from rdkit import Chem

warnings.simplefilter('ignore', PDBConstructionWarning)

def setup_directory_hierarchy(
    workdir: Optional[str],
    ref_pdb_id: str,
) -> Dict[str, Path]:
    """
    Creates the directory hierarchy for QPocket pipeline execution.

    Args:
        workdir: Working directory path, defaults to current directory if None
        ref_pdb_id: Reference PDB identifier for creating project subdirectory

    Returns:
        Dictionary mapping directory names to their Path objects
    """
    base_root = Path(workdir) if workdir is not None else Path.cwd()

    project_path = base_root / ref_pdb_id

    dirs = {
        "root": project_path,
        "structures": project_path / "structures",
        "original": project_path / "structures" / "original",
        "aligned": project_path / "structures" / "aligned",
        "cleaned": project_path / "structures" / "cleaned",
        "extracted_ligands": project_path / "structures" / "extracted_ligands",
        "pockets": project_path / "pockets",
        "metadata": project_path / "metadata",
        "screening": project_path / "screening",
        "decoys": project_path / "screening" / "decoys",
        "actives": project_path / "screening" / "actives",
    }

    for key, path in dirs.items():
        path.mkdir(exist_ok=True)

    return dirs

def calculate_sequence_identity(
        seq1: str,
        seq2: str
) -> float:
    """
    Calculates the sequence identity between two sequences after global alignment.

    This function performs a global alignment on two sequences and calculates
    the percentage of identical characters in the aligned region.
    """

    alignments = pairwise2.align.globalxx(seq1, seq2)

    if not alignments:
        return 0.0

    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2, score, begin, end = best_alignment

    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
    alignment_length = len(aligned_seq1)

    if alignment_length == 0:
        return 0.0
    identity = (matches / alignment_length) * 100

    return identity


def get_ref_ligand_centroid(
        ref_pdb_path: Path,
        ligand_resname: str,
        ref_chain: str
) -> Optional[np.ndarray]:
    """
    Calculate the geometric centroid of the reference ligand.
    """
    parser = PDBParser(QUIET=True)
    try:
        # Load the structure (supports .pdb or .cif if biopython is configured)
        # For simplicity assuming PDB parser, use MMCIFParser if input is .cif
        structure = parser.get_structure("reference", ref_pdb_path)
        atoms = []
        for model in structure:
            for chain in model:
                if chain.id != ref_chain: continue
                for residue in chain:
                    if residue.get_resname().strip().upper() == ligand_resname:
                        for atom in residue:
                            atoms.append(atom.get_coord())

        if not atoms:
            log_message(message_type="error", error_message=f"Reference ligand {ligand_resname} not found in {ref_pdb_path}")
            raise FileNotFoundError("No atoms found.")

        return np.mean(atoms, axis=0)
    except Exception as e:
        log_message(
            message_type="error",
            error_message=f"No atoms found for {ligand_resname} in {ref_pdb_path}",
        )
        raise FileNotFoundError(f"No atoms found for {ligand_resname} in {ref_pdb_path}: {str(e)}")


def get_smiles_from_pdb_block(pdb_block: str) -> str:
    """
    Converts a PDB block to SMILES string using RDKit.

    Args:
        pdb_block: PDB-formatted text block

    Returns:
        SMILES string, or empty string if conversion fails
    """
    try:
        mol = Chem.MolFromPDBBlock(pdb_block)
        if mol:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


def is_residue_ligand(residue, allowed_cofactors: List[str]) -> bool:
    """
    Determines if a residue is a ligand (not protein, water, or cofactor).

    Args:
        residue: BioPython residue object
        allowed_cofactors: List of cofactor names to exclude

    Returns:
        True if residue is a potential ligand, False otherwise
    """
    if residue.id[0] == ' ':
        return False  # Standard amino acid
    if residue.id[0] == 'W':
        return False  # Water
    if residue.get_resname().strip().upper() in allowed_cofactors:
        return False
    if residue.get_resname().strip().upper() in HET_TO_REMOVE:
        return False
    return True

def check_overlap(
    points_obj: np.ndarray,
    box_points: np.ndarray,
    tolerance: float = 2.0,
    use_majority: bool = True
) -> bool:
    """
    Checks if 'points_obj' lies within the bounding box of 'box_points'.

    Args:
        points_obj: Point(s) to check - either single point (1D) or array of points (2D)
        box_points: Points defining the bounding box
        tolerance: Padding around the bounding box in Angstroms
        use_majority: If True, checks if majority of atoms are inside;
                     if False, uses 70% threshold (legacy behavior)

    Returns:
        True if point/points overlap with bounding box, False otherwise
    """
    if len(box_points) == 0:
        return False

    min_b = np.min(box_points, axis=0) - tolerance
    max_b = np.max(box_points, axis=0) + tolerance

    # Single point check
    if points_obj.ndim == 1:
        return bool(np.all(points_obj >= min_b) and np.all(points_obj <= max_b))

    # Multiple points check
    in_bounds = (points_obj >= min_b) & (points_obj <= max_b)
    mask = np.all(in_bounds, axis=1)
    num_inside = np.sum(mask)
    total = len(points_obj)

    if use_majority:
        # Majority logic: more than 50% of atoms inside
        return num_inside > (total * 0.5)
    else:
        # Legacy: >70% threshold
        return num_inside > (total * 0.7)

def create_bounding_box_pocket(
    center: List[float],
    dimensions: List[float],
    tolerance: float = 2.0
) -> np.ndarray:
    """
    Creates a point cloud representing a bounding box pocket.

    Args:
        center: [x, y, z] coordinates of the box center
        dimensions: [dx, dy, dz] dimensions of the box
        tolerance: Additional padding around the box

    Returns:
        Array of corner points defining the bounding box
    """
    center = np.array(center)
    dims = np.array(dimensions) / 2.0 + tolerance

    # Generate 8 corners of the bounding box
    corners = []
    for dx in [-dims[0], dims[0]]:
        for dy in [-dims[1], dims[1]]:
            for dz in [-dims[2], dims[2]]:
                corners.append(center + np.array([dx, dy, dz]))

    return np.array(corners)


def get_interacting_residues_coords(
    pdb_path: Path,
    interactions: List[str],
    padding: float = 5.0
) -> Optional[np.ndarray]:
    """
    Extracts coordinates of atoms from interacting residues.

    Args:
        pdb_path: Path to PDB file
        interactions: List of interaction strings (format: "RES_NUM|InteractionType")
        padding: Padding around interacting residues in Angstroms

    Returns:
        Array of coordinates from interacting residues, or None if parsing fails
    """
    from biopandas.pdb import PandasPdb

    try:
        ppdb = PandasPdb().read_pdb(str(pdb_path))
        atoms = ppdb.df['ATOM']

        # Parse residue identifiers from interactions
        residue_ids = set()
        for interaction in interactions:
            if "|" in interaction:
                res_id = interaction.split("|")[0]
                residue_ids.add(res_id)

        if not residue_ids:
            return None

        # Extract coordinates of atoms in interacting residues
        coords_list = []
        for res_id in residue_ids:
            # Parse residue name and number (format: "RES_NUM")
            parts = res_id.split("_")
            if len(parts) >= 2:
                res_name = parts[0]
                res_num = int(parts[1])

                mask = (atoms['residue_name'] == res_name) & (atoms['residue_number'] == res_num)
                res_coords = atoms[mask][['x_coord', 'y_coord', 'z_coord']].to_numpy()
                if len(res_coords) > 0:
                    coords_list.append(res_coords)

        if not coords_list:
            return None

        all_coords = np.vstack(coords_list)
        return all_coords

    except Exception as e:
        log_message(message_type="error", error_message=f"Failed to extract interacting residues: {e}")
        return None


def crop_pocket_by_interactions(
    pocket_cloud: np.ndarray,
    interacting_coords: np.ndarray,
    padding: float = 5.0
) -> np.ndarray:
    """
    Crops a pocket point cloud based on interacting residue coordinates.

    Args:
        pocket_cloud: Original pocket point cloud coordinates
        interacting_coords: Coordinates of atoms in interacting residues
        padding: Padding around interacting residues in Angstroms

    Returns:
        Cropped pocket point cloud
    """
    from scipy.spatial.distance import cdist

    # Calculate distances from pocket points to interacting residues
    distances = cdist(pocket_cloud, interacting_coords)
    min_distances = np.min(distances, axis=1)

    # Keep only points within padding distance of interacting residues
    mask = min_distances <= padding
    return pocket_cloud[mask]


def compute_pocket_bounding_box(
    pocket_coords: np.ndarray,
    padding: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives an axis-aligned bounding box from a pocket point cloud.

    Args:
        pocket_coords: Point cloud (N × 3) of the pocket cavity.
        padding: Expansion added to every face of the box (Å).

    Returns:
        Tuple of ``(min_corner, max_corner)`` each of shape ``(3,)``.
    """
    min_coords = np.min(pocket_coords, axis=0) - padding
    max_coords = np.max(pocket_coords, axis=0) + padding
    return min_coords, max_coords


def compute_volumetric_overlap(
    cavity_coords: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> int:
    """Counts how many *cavity_coords* points fall inside a bounding box.

    Args:
        cavity_coords: Point cloud to test (N × 3).
        bbox_min: Lower corner of the reference bounding box (3,).
        bbox_max: Upper corner of the reference bounding box (3,).

    Returns:
        Integer count of points that lie strictly inside the box.
    """
    inside = np.all(
        (cavity_coords >= bbox_min) & (cavity_coords <= bbox_max),
        axis=1,
    )
    return int(np.sum(inside))


def extract_pdb_block(residue, chain_id: str) -> str:
    """
    Constructs a PDB block for a single residue.

    Args:
        residue: BioPython residue object
        chain_id: Chain identifier

    Returns:
        PDB-formatted text block for the residue
    """
    lines = []
    for i, atom in enumerate(residue):
        x, y, z = atom.get_coord()
        line = (f"HETATM{i+1:>5}  {atom.name:<4}{residue.resname:>3} {chain_id}{residue.id[1]:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom.element:>2}  ")
        lines.append(line)
    return "\n".join(lines)

def save_to_hdf5_ascii(
    data_dict: Dict[str, List[str]],
    output_path: Path,
    compression: str = "gzip",
) -> None:
    """Saves a string-list dictionary to an HDF5 file with ASCII encoding.

    Args:
        data_dict: Mapping from PDB ID to a list of interaction strings.
        output_path: Destination HDF5 file path.
        compression: HDF5 compression filter — ``"gzip"`` or ``"none"``.
    """
    compress_kwargs: Dict[str, Any] = (
        {"compression": "gzip", "compression_opts": 4}
        if compression == "gzip"
        else {}
    )
    with start_action(
        action_type="save_hdf5",
        path=str(output_path),
        compression=compression,
        n_entries=len(data_dict),
    ):
        with h5py.File(output_path, "a") as hf:
            for k, v in data_dict.items():
                residue_list = sorted(v)
                ascii_residues = [n.encode("ascii", "ignore") for n in residue_list]
                if k in hf:
                    del hf[k]
                hf.create_dataset(k, data=ascii_residues, **compress_kwargs)

def calculate_observed_frequencies(h5_path: str):
    """
    Calculates frequencies for OBSERVED interactions (with subtypes like H-bond, Pi-stack).
    Logic: Weights the specific interaction type by the residue's global presence.
    """
    path_obj = Path(h5_path)
    if not path_obj.exists():
        return

    out_path = path_obj.with_name(f"{path_obj.stem}_freq.csv")

    try:
        structure_count, amino_structure_presence, raw_interactions = _extract_observed_data(path_obj)

        if structure_count == 0:
            log_message(message_type="warning", warning_message= f"No valid data in {h5_path}")
            return

        result_df = _compute_weighted_interaction_matrix(
            raw_interactions,
            amino_structure_presence,
            structure_count
        )

        result_df.to_csv(out_path, index=False)
        log_message(f"Observed frequencies saved to {out_path}")

    except Exception as e:
        log_message(message_type="error", error_message = f"Error processing observed data: {e}")

def calculate_potential_frequencies(h5_path: str):
    """
    Calculates frequencies for POTENTIAL interactions (Geometric contacts only).
    Logic: Simple percentage of structures where a residue is part of the pocket.
    """
    path_obj = Path(h5_path)
    if not path_obj.exists():
        log_message(
            message_type="error", error_message=f"File not found: {h5_path}"
        )
        return

    out_path = path_obj.with_name(f"{path_obj.stem}_freq.csv")

    amino_counts = defaultdict(int)
    total_structures = 0

    try:
        with h5py.File(h5_path, 'r') as hf:
            pdb_ids = list(hf.keys())
            total_structures = len(pdb_ids)

            for pdb_id in pdb_ids:
                dataset = hf[pdb_id][:]
                if len(dataset) == 0: continue

                residues = set(res.decode("ascii") for res in dataset)

                for res in residues:
                    amino_counts[res] += 1

        if total_structures == 0:
            log_message(
                message_type="warning", warning_message="No structures found in potential H5 file."
            )
            return

        data = []
        for amino, count in amino_counts.items():
            freq = (count / total_structures) * 100.0
            data.append({'Residue': amino, 'Frequency (%)': freq, 'Count': count})

        df = pd.DataFrame(data).sort_values('Frequency (%)', ascending=False)
        df.to_csv(out_path, index=False)
        log_message(f"Potential frequencies saved to {out_path}")

    except Exception as e:
        log_message(message_type="error", error_message=f"Error processing potential data: {e}")


def calculate_binding_site_utilization(h5_path_observed: str, h5_path_potential: str):
    """
    Calculates what % of the available geometric pocket is actually used by the ligand.
    """
    path_obs = Path(h5_path_observed)
    path_pot = Path(h5_path_potential)

    if not path_obs.exists() or not path_pot.exists():
        log_message(
            message_type="error", error_message="Missing input files for utilization analysis."
        )
        return

    out_path = path_obs.parent / "binding_pocket_utilization.csv"
    utilization_data = []

    try:
        with h5py.File(path_obs, 'r') as f_obs, h5py.File(path_pot, 'r') as f_pot:
            common_pdbs = set(f_obs.keys()) & set(f_pot.keys())

            for pdb_id in common_pdbs:
                pot_set = set(x.decode("ascii") for x in f_pot[pdb_id][:])
                if not pot_set: continue

                obs_raw = f_obs[pdb_id][:]
                obs_set = set(x.decode("ascii").split("|")[0] for x in obs_raw)

                utilized = obs_set.intersection(pot_set)

                pct = (len(utilized) / len(pot_set)) * 100.0
                utilization_data.append({'pdb_id': pdb_id, 'utilization': pct})

        if utilization_data:
            df = pd.DataFrame(utilization_data).sort_values('utilization', ascending=False)
            df.to_csv(out_path, index=False)
            log_message(f"Utilization saved to {out_path}")

    except Exception as e:
        log_message(
            message_type="error",
            error_message=f"Utilization analysis failed: {e}",
        )

def _extract_observed_data(h5_path: Path) -> Tuple[int, Dict[str, int], List[List[str]]]:
    """Helper specifically for parsing OBSERVED data with '|' separators."""
    amino_presence = defaultdict(int)
    raw_data = []

    with h5py.File(h5_path, 'r') as hf:
        pdb_ids = list(hf.keys())
        total = len(pdb_ids)

        for pid in pdb_ids:
            entries = [r.decode("ascii") for r in hf[pid][:]]
            unique_res_in_struct = set()

            for entry in entries:
                if "|" in entry:
                    res, inter = entry.split("|")
                    raw_data.append([res, inter])
                    unique_res_in_struct.add(res)

            for res in unique_res_in_struct:
                amino_presence[res] += 1

    return total, amino_presence, raw_data

def _compute_weighted_interaction_matrix(raw_data, amino_counts, total) -> pd.DataFrame:
    """Helper to pivot observed interaction types."""
    df = pd.DataFrame(raw_data, columns=['amino', 'inter'])

    global_freqs = pd.Series({
        a: (c / total) * 100 for a, c in amino_counts.items()
    })

    if df.empty: return pd.DataFrame()

    ratios = pd.crosstab(df['amino'], df['inter'], normalize='index')
    weighted = ratios.mul(global_freqs, axis=0)

    return weighted.reset_index()

