from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from eliot import log_message, start_action
from rdkit import Chem
from scipy.spatial.distance import cdist

from qpocket.constants import HET_TO_REMOVE

warnings.simplefilter("ignore", PDBConstructionWarning)


def setup_directory_hierarchy(
    workdir: Optional[str],
    ref_pdb_id: str,
) -> Dict[str, Path]:
    """Creates the directory hierarchy for QPocket pipeline execution.

    Args:
        workdir: Working directory path, defaults to current directory if None.
        ref_pdb_id: Reference PDB identifier for creating project subdirectory.

    Returns:
        Dictionary mapping directory names to their Path objects.
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

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_ref_ligand_centroid(
    ref_pdb_path: Path,
    ligand_resname: str,
    search_chains: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """Calculates the geometric centroid of the reference ligand.

    Search strategy:

    * When *search_chains* is provided, only those chains are examined.
    * When *search_chains* is ``None`` all chains in the structure are searched
      (needed for homodimers / heterodimers where the ligand may reside in a
      chain that is neither the primary nor the declared auxiliary chain).

    Args:
        ref_pdb_path: Path to the aligned reference PDB file.
        ligand_resname: Residue name of the ligand (e.g. ``"LKQ"``).
        search_chains: Chain IDs to restrict the search to, or ``None`` to
            search all chains.

    Returns:
        Centroid as a ``(3,)`` numpy array, or ``None`` if the ligand is not
        found or an error occurs.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("reference", ref_pdb_path)
        atoms: List[np.ndarray] = []
        for model in structure:
            for chain in model:
                if search_chains is not None and chain.id not in search_chains:
                    continue
                for residue in chain:
                    if residue.get_resname().strip().upper() == ligand_resname.upper():
                        for atom in residue:
                            atoms.append(atom.get_coord())

        if not atoms:
            searched = search_chains if search_chains is not None else "all chains"
            log_message(
                message_type="warning",
                stage="get_ref_ligand_centroid",
                warning=(
                    f"ligand {ligand_resname!r} not found in {searched} "
                    f"of {ref_pdb_path}"
                ),
            )
            return None

        return np.mean(atoms, axis=0)

    except Exception as e:
        log_message(
            message_type="error",
            stage="get_ref_ligand_centroid",
            error=str(e),
        )
        return None


def get_smiles_from_pdb_block(pdb_block: str) -> str:
    """Converts a PDB block to a SMILES string using RDKit.

    Args:
        pdb_block: PDB-formatted text block.

    Returns:
        SMILES string, or empty string if conversion fails.
    """
    try:
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=True)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


def is_residue_ligand(residue, allowed_cofactors: List[str]) -> bool:
    """Determines whether a residue is a candidate ligand.

    Returns ``False`` for standard amino acids, water molecules, cofactors in
    *allowed_cofactors*, and entries in the global exclusion list.

    Args:
        residue: BioPython residue object.
        allowed_cofactors: List of cofactor residue names to exclude.

    Returns:
        ``True`` if the residue is a potential drug-like ligand.
    """
    if residue.id[0] == " ":
        return False  # standard amino acid
    if residue.id[0] == "W":
        return False  # water
    resname = residue.get_resname().strip().upper()
    if resname in allowed_cofactors:
        return False
    if resname in HET_TO_REMOVE:
        return False
    return True


def check_overlap(
    points_obj: np.ndarray,
    box_points: np.ndarray,
    tolerance: float = 2.0,
    use_majority: bool = True,
) -> bool:
    """Checks whether *points_obj* lies within the bounding box of *box_points*.

    Args:
        points_obj: Point(s) to test — 1-D for a single point, 2-D for many.
        box_points: Points that define the reference bounding box.
        tolerance: Padding (Å) added to every face of the bounding box.
        use_majority: When ``True`` (default), passes if > 50 % of atoms are
            inside; when ``False``, uses a 70 % threshold (legacy behaviour).

    Returns:
        ``True`` if the overlap criterion is satisfied.
    """
    if len(box_points) == 0:
        return False

    min_b = np.min(box_points, axis=0) - tolerance
    max_b = np.max(box_points, axis=0) + tolerance

    if points_obj.ndim == 1:
        return bool(np.all(points_obj >= min_b) and np.all(points_obj <= max_b))

    in_bounds = (points_obj >= min_b) & (points_obj <= max_b)
    mask = np.all(in_bounds, axis=1)
    num_inside = int(np.sum(mask))
    total = len(points_obj)

    threshold = 0.5 if use_majority else 0.7
    return num_inside > (total * threshold)


def create_bounding_box_pocket(
    center: List[float],
    dimensions: List[float],
    tolerance: float = 2.0,
) -> np.ndarray:
    """Creates an 8-corner point cloud representing a bounding box pocket.

    Args:
        center: ``[x, y, z]`` coordinates of the box centre.
        dimensions: ``[dx, dy, dz]`` half-dimensions of the box.
        tolerance: Additional padding (Å) on every face.

    Returns:
        Array of shape ``(8, 3)`` — the corners of the padded bounding box.
    """
    c = np.array(center)
    half = np.array(dimensions) / 2.0 + tolerance

    corners = [
        c + np.array([sx * half[0], sy * half[1], sz * half[2]])
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    return np.array(corners)


def get_interacting_residues_coords(
    pdb_path: Path,
    interactions: List[str],
    padding: float = 5.0,
) -> Optional[np.ndarray]:
    """Extracts coordinates of atoms from interacting residues.

    Args:
        pdb_path: Path to the aligned PDB file.
        interactions: Interaction strings formatted as
            ``"RESNAME_RESNUM|InteractionType"``.
        padding: Unused here; kept for API consistency.

    Returns:
        ``(N, 3)`` array of atom coordinates from interacting residues, or
        ``None`` if parsing fails or no residues are found.
    """
    try:
        ppdb = PandasPdb().read_pdb(str(pdb_path))
        atoms = ppdb.df["ATOM"]

        residue_ids: set = set()
        for interaction in interactions:
            if "|" in interaction:
                res_id = interaction.split("|", 1)[0]
                residue_ids.add(res_id)

        if not residue_ids:
            return None

        coords_list: List[np.ndarray] = []
        for res_id in residue_ids:
            parts = res_id.split("_")
            # Supported formats:
            #   "RESNAME_RESNUM"          (single-chain / legacy)
            #   "RESNAME_RESNUM_CHAINID"  (multi-chain / chain-tagged)
            if len(parts) < 2:
                continue
            res_name = parts[0]
            try:
                res_num = int(parts[1])
            except ValueError:
                continue
            mask = (atoms["residue_name"] == res_name) & (
                atoms["residue_number"] == res_num
            )
            if len(parts) >= 3:
                # Chain-tagged format: further restrict to the specific chain.
                chain_filter = parts[2]
                mask = mask & (atoms["chain_id"] == chain_filter)
            res_coords = atoms[mask][["x_coord", "y_coord", "z_coord"]].to_numpy()
            if len(res_coords) > 0:
                coords_list.append(res_coords)

        if not coords_list:
            return None

        return np.vstack(coords_list)

    except Exception as e:
        log_message(
            message_type="error",
            stage="get_interacting_residues_coords",
            error=str(e),
        )
        return None


def crop_pocket_by_interactions(
    pocket_cloud: np.ndarray,
    interacting_coords: np.ndarray,
    padding: float = 5.0,
) -> np.ndarray:
    """Crops a pocket point cloud to points within *padding* Å of interacting residues.

    Args:
        pocket_cloud: Full pocket point cloud ``(N, 3)``.
        interacting_coords: Atom coordinates of the interacting residues.
        padding: Distance cutoff in Å.

    Returns:
        Subset of *pocket_cloud* within the cutoff.
    """
    distances = cdist(pocket_cloud, interacting_coords)
    mask = distances.min(axis=1) <= padding
    return pocket_cloud[mask]


def compute_pocket_bounding_box(
    pocket_coords: np.ndarray,
    padding: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives an axis-aligned bounding box from a pocket point cloud.

    Args:
        pocket_coords: Point cloud ``(N, 3)`` of the pocket cavity.
        padding: Expansion (Å) added to every face of the box.

    Returns:
        ``(min_corner, max_corner)`` each of shape ``(3,)``.
    """
    return (
        np.min(pocket_coords, axis=0) - padding,
        np.max(pocket_coords, axis=0) + padding,
    )


def compute_volumetric_overlap(
    cavity_coords: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> int:
    """Counts cavity points that fall inside a reference bounding box.

    Args:
        cavity_coords: Point cloud to test ``(N, 3)``.
        bbox_min: Lower corner of the reference bounding box ``(3,)``.
        bbox_max: Upper corner of the reference bounding box ``(3,)``.

    Returns:
        Integer count of points strictly inside the box.
    """
    inside = np.all(
        (cavity_coords >= bbox_min) & (cavity_coords <= bbox_max),
        axis=1,
    )
    return int(np.sum(inside))


def extract_pdb_block(residue, chain_id: str) -> str:
    """Constructs a PDB-compliant HETATM block for a single residue.

    Follows the PDB fixed-column format (80 chars per record).

    Args:
        residue: BioPython residue object.
        chain_id: Chain identifier character.

    Returns:
        PDB-formatted text block terminated with a newline.
    """
    lines: List[str] = []
    for i, atom in enumerate(residue):
        x, y, z = atom.get_coord()
        element = (atom.element or "").strip()
        name = atom.name.strip()

        # PDB atom name occupies columns 13-16 — always exactly four characters.
        # A 4-character name (common for ligand hydrogens, e.g. "H011") fills the
        # field and must NOT be given a leading space, or every later field on the
        # line shifts by one and strict parsers reject the record. Shorter names
        # get a leading space when the element symbol is 1 character (" C1 ") and
        # start at column 13 when it is 2 ("FE  ").
        if len(name) >= 4:
            atom_str = name[:4]
        elif len(element) == 1:
            atom_str = f" {name:<3}"
        else:
            atom_str = f"{name:<4}"

        res_seq = residue.id[1]

        # Legacy PDB has no room for longer values; truncating keeps the columns
        # aligned, which matters more to downstream parsers than the lost tail.
        resname = residue.resname[:3]

        # Standard HETATM record — 80 columns exactly.
        line = (
            f"HETATM{i + 1:5d} {atom_str} "   # cols 1-17 (record + serial + name + altloc)
            f"{resname:>3} "                    # cols 18-21 (resname + space)
            f"{chain_id}"                       # col  22    (chain)
            f"{res_seq:4d} "                    # cols 23-27 (resseq + icode)
            f"   "                              # cols 28-30 (padding)
            f"{x:8.3f}{y:8.3f}{z:8.3f}"       # cols 31-54 (coordinates)
            f"  1.00  0.00"                     # cols 55-66 (occupancy + B-factor)
            f"          "                       # cols 67-76 (segment ID padding)
            f"{element[:2]:>2}  "               # cols 77-80 (element + charge)
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


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


def calculate_observed_frequencies(h5_path: str) -> None:
    """Calculates weighted frequencies for observed (PLIP) interactions.

    Each interaction type is weighted by the residue's global presence
    across all structures.

    Args:
        h5_path: Path to the observed interactions HDF5 file.
    """
    path_obj = Path(h5_path)
    if not path_obj.exists():
        return

    out_path = path_obj.with_name(f"{path_obj.stem}_freq.csv")

    try:
        structure_count, amino_structure_presence, raw_interactions = _extract_observed_data(
            path_obj
        )

        if structure_count == 0:
            log_message(
                message_type="warning",
                stage="calculate_observed_frequencies",
                warning=f"no valid data in {h5_path}",
            )
            return

        result_df = _compute_weighted_interaction_matrix(
            raw_interactions, amino_structure_presence, structure_count
        )

        result_df.to_csv(out_path, index=False)
        log_message(
            message_type="observed_frequencies_saved",
            path=str(out_path),
        )

    except Exception as e:
        log_message(
            message_type="error",
            stage="calculate_observed_frequencies",
            error=str(e),
        )


def calculate_potential_frequencies(h5_path: str) -> None:
    """Calculates simple percentage frequencies for potential (geometric) interactions.

    For each residue, reports the fraction of structures in which it is
    part of the pocket.

    Args:
        h5_path: Path to the potential interactions HDF5 file.
    """
    path_obj = Path(h5_path)
    if not path_obj.exists():
        log_message(
            message_type="error",
            stage="calculate_potential_frequencies",
            error=f"file not found: {h5_path}",
        )
        return

    out_path = path_obj.with_name(f"{path_obj.stem}_freq.csv")

    amino_counts: Dict[str, int] = defaultdict(int)
    total_structures = 0

    try:
        with h5py.File(h5_path, "r") as hf:
            pdb_ids = list(hf.keys())
            total_structures = len(pdb_ids)

            for pdb_id in pdb_ids:
                dataset = hf[pdb_id][:]
                if len(dataset) == 0:
                    continue
                residues = {res.decode("ascii") for res in dataset}
                for res in residues:
                    amino_counts[res] += 1

        if total_structures == 0:
            log_message(
                message_type="warning",
                stage="calculate_potential_frequencies",
                warning="no structures found in potential HDF5 file",
            )
            return

        data = [
            {"Residue": amino, "Frequency (%)": (count / total_structures) * 100.0, "Count": count}
            for amino, count in amino_counts.items()
        ]

        # Residue as the tie-break: the counts are accumulated by iterating a
        # set, whose order varies with string hash randomisation, so sorting on
        # frequency alone makes the row order differ between identical runs.
        df = pd.DataFrame(data).sort_values(
            ["Frequency (%)", "Residue"], ascending=[False, True]
        )
        df.to_csv(out_path, index=False)
        log_message(
            message_type="potential_frequencies_saved",
            path=str(out_path),
        )

    except Exception as e:
        log_message(
            message_type="error",
            stage="calculate_potential_frequencies",
            error=str(e),
        )


def calculate_binding_site_utilization(
    h5_path_observed: str,
    h5_path_potential: str,
) -> None:
    """Calculates what percentage of the available geometric pocket is used by the ligand.

    Args:
        h5_path_observed: Path to the observed interactions HDF5 file.
        h5_path_potential: Path to the potential interactions HDF5 file.
    """
    path_obs = Path(h5_path_observed)
    path_pot = Path(h5_path_potential)

    if not path_obs.exists() or not path_pot.exists():
        log_message(
            message_type="error",
            stage="calculate_binding_site_utilization",
            error="missing input files for utilization analysis",
        )
        return

    out_path = path_obs.parent / "binding_pocket_utilization.csv"
    utilization_data = []

    try:
        with h5py.File(path_obs, "r") as f_obs, h5py.File(path_pot, "r") as f_pot:
            common_pdbs = set(f_obs.keys()) & set(f_pot.keys())

            for pdb_id in common_pdbs:
                pot_set = {x.decode("ascii") for x in f_pot[pdb_id][:]}
                if not pot_set:
                    continue

                obs_raw = f_obs[pdb_id][:]
                # Observed entries are "RESNAME_NUM|InteractionType" — strip the subtype.
                obs_set = {x.decode("ascii").split("|", 1)[0] for x in obs_raw}

                utilized = obs_set.intersection(pot_set)
                pct = (len(utilized) / len(pot_set)) * 100.0
                utilization_data.append({"pdb_id": pdb_id, "utilization": pct})

        if utilization_data:
            df = pd.DataFrame(utilization_data).sort_values("utilization", ascending=False)
            df.to_csv(out_path, index=False)
            log_message(
                message_type="utilization_saved",
                path=str(out_path),
            )

    except Exception as e:
        log_message(
            message_type="error",
            stage="calculate_binding_site_utilization",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_observed_data(
    h5_path: Path,
) -> Tuple[int, Dict[str, int], List[List[str]]]:
    """Parses OBSERVED interaction HDF5 data (entries have ``|`` separators)."""
    amino_presence: Dict[str, int] = defaultdict(int)
    raw_data: List[List[str]] = []

    with h5py.File(h5_path, "r") as hf:
        pdb_ids = list(hf.keys())
        total = len(pdb_ids)

        for pid in pdb_ids:
            entries = [r.decode("ascii") for r in hf[pid][:]]
            unique_res_in_struct: set = set()

            for entry in entries:
                if "|" in entry:
                    # Use maxsplit=1 to handle subtypes that contain a second "|".
                    res, inter = entry.split("|", 1)
                    raw_data.append([res, inter])
                    unique_res_in_struct.add(res)

            for res in unique_res_in_struct:
                amino_presence[res] += 1

    return total, amino_presence, raw_data


def _compute_weighted_interaction_matrix(
    raw_data: List[List[str]],
    amino_counts: Dict[str, int],
    total: int,
) -> pd.DataFrame:
    """Pivots observed interaction types and weights by global residue frequency."""
    df = pd.DataFrame(raw_data, columns=["amino", "inter"])

    if df.empty:
        return pd.DataFrame()

    global_freqs = pd.Series(
        {a: (c / total) * 100.0 for a, c in amino_counts.items()}
    )

    ratios = pd.crosstab(df["amino"], df["inter"], normalize="index")
    weighted = ratios.mul(global_freqs, axis=0)

    return weighted.reset_index()