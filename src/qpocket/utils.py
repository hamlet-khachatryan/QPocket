"""Run-level helpers kept until the pipeline stages absorb them (Task 8)."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from eliot import log_message

from qpocket.core import interactions, pocket
from qpocket.io import hdf5

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


def save_to_hdf5_ascii(
    data_dict: Dict[str, List[str]],
    output_path: Path,
    compression: str = "gzip",
) -> None:
    """Add or replace structures in an interaction archive.

    Args:
        data_dict: Structure ID to entries.
        output_path: HDF5 file.
        compression: "gzip" or "none".
    """
    hdf5.save_interactions(data_dict, output_path, compression)


def calculate_observed_frequencies(h5_path: str) -> None:
    """Write <archive>_freq.csv for an observed-interaction archive.

    Args:
        h5_path: Observed interactions HDF5.
    """
    path_obj = Path(h5_path)
    if not path_obj.exists():
        return
    out_path = path_obj.with_name(f"{path_obj.stem}_freq.csv")
    try:
        result_df = interactions.observed_frequencies(hdf5.read_interactions(path_obj))
        if result_df is None:
            log_message(
                message_type="warning",
                stage="calculate_observed_frequencies",
                warning=f"no valid data in {h5_path}",
            )
            return
        result_df.to_csv(out_path, index=False)
        log_message(message_type="observed_frequencies_saved", path=str(out_path))
    except Exception as e:
        log_message(message_type="error", stage="calculate_observed_frequencies", error=str(e))


def calculate_potential_frequencies(h5_path: str) -> None:
    """Write <archive>_freq.csv for a potential-interaction archive.

    Args:
        h5_path: Potential interactions HDF5.
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
    try:
        df = interactions.potential_frequencies(hdf5.read_interactions(path_obj))
        if df is None:
            log_message(
                message_type="warning",
                stage="calculate_potential_frequencies",
                warning="no structures found in potential HDF5 file",
            )
            return
        df.to_csv(out_path, index=False)
        log_message(message_type="potential_frequencies_saved", path=str(out_path))
    except Exception as e:
        log_message(message_type="error", stage="calculate_potential_frequencies", error=str(e))


def calculate_binding_site_utilization(
    h5_path_observed: str,
    h5_path_potential: str,
) -> None:
    """Write binding_pocket_utilization.csv beside the observed archive.

    Args:
        h5_path_observed: Observed interactions HDF5.
        h5_path_potential: Potential interactions HDF5.
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
    try:
        df = interactions.utilization(hdf5.read_interactions(path_obs), hdf5.read_interactions(path_pot))
        if not df.empty:
            df.to_csv(out_path, index=False)
            log_message(message_type="utilization_saved", path=str(out_path))
    except Exception as e:
        log_message(message_type="error", stage="calculate_binding_site_utilization", error=str(e))
