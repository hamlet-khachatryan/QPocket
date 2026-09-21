import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from rdkit import Chem

from eliot import log_message

def calculate_box(points: np.ndarray, padding: float = 4.0) -> Dict[str, float]:
    """Calculates min/max and center/size for a set of 3D points."""
    min_coords = np.min(points, axis=0) - padding
    max_coords = np.max(points, axis=0) + padding

    center = (min_coords + max_coords) / 2.0
    size = max_coords - min_coords

    return {
        'center_x': center[0], 'center_y': center[1], 'center_z': center[2],
        'size_x': size[0], 'size_y': size[1], 'size_z': size[2],
        'min_x': min_coords[0], 'min_y': min_coords[1], 'min_z': min_coords[2],
        'max_x': max_coords[0], 'max_y': max_coords[1], 'max_z': max_coords[2]
    }

def extract_vina_box(
        cfg: Dict,
        dirs: Dict[str, Path]
):
    """
    Generates a Vina-compatible config.txt with bounding box coordinates.
    Mode: 'ligand_based' (uses extracted ligands) or 'cavity_based' (uses pocket CSVs).
    """
    mode = cfg.get('box_mode', 'ligand_based') # or 'cavity_based'
    padding = cfg.get('padding', 5.0)

    log_message(message_type="box_extraction_start", mode=mode, padding=padding)

    all_points = []

    if mode == 'ligand_based':
        # Iterate all extracted ligands in metadata/ligands
        ligand_files = list(dirs['ligands'].glob("*.pdb"))
        if not ligand_files:
            log_message(message_type="error", error_message="No ligands found for box calculation.")
            return

        for lig_file in ligand_files:
            try:
                mol = Chem.MolFromPDBFile(str(lig_file))
                if mol:
                    conf = mol.GetConformer()
                    coords = conf.GetPositions()
                    all_points.append(coords)
            except Exception as e:
                log_message(message_type="warning", msg=f"Failed to read {lig_file.name}: {e}")

    elif mode == 'cavity_based':
        # Iterate all cavity CSVs
        # Assumes structure: metadata/cavities/{pdb_id}/*_binding_site.csv
        cavity_files = list(dirs['cavities'].rglob("*_binding_site.csv"))
        if not cavity_files:
            log_message(message_type="error", error_message="No binding site files found.")
            return

        for csv_file in cavity_files:
            try:
                df = pd.read_csv(csv_file)
                # Ensure columns exist
                if {'x', 'y', 'z'}.issubset(df.columns):
                    coords = df[['x', 'y', 'z']].values
                    all_points.append(coords)
            except Exception as e:
                log_message(message_type="warning", msg=f"Failed to read {csv_file.name}: {e}")

    else:
        log_message(message_type="error", error_message=f"Unknown box mode: {mode}")
        return

    if not all_points:
        log_message(message_type="error", error_message="No coordinates collected. Box generation failed.")
        return

    global_points = np.vstack(all_points)
    box = calculate_box(global_points, padding=padding)

    out_path = dirs['screening'] / "vina_box.txt"
    dirs['screening'].mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"center_x = {box['center_x']:.3f}\n")
        f.write(f"center_y = {box['center_y']:.3f}\n")
        f.write(f"center_z = {box['center_z']:.3f}\n")
        f.write(f"size_x = {box['size_x']:.3f}\n")
        f.write(f"size_y = {box['size_y']:.3f}\n")
        f.write(f"size_z = {box['size_z']:.3f}\n")
        f.write("num_modes = 10\n")
        f.write("energy_range = 3\n")
        f.write(f"exhaustiveness = { cfg.get('exhaustiveness', 8.0)}\n") # Optional

    log_message(message_type="box_extraction_complete", file=str(out_path), box=box)
