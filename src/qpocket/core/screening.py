"""Virtual screening: boxes, molecule standardisation, job tables and scripts (pure)."""
from __future__ import annotations

import shlex
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from molvs import Standardizer
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


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


def pdbqt_lines_to_mol(lines: List[str]) -> Optional[Chem.Mol]:
    """Convert PDBQT ATOM/HETATM lines to an RDKit Mol via PDB block."""
    pdb_lines = []
    for line in lines:
        record = line[:6].strip()
        if record in ("ATOM", "HETATM"):
            # PDBQT has extra partial-charge column — truncate to standard PDB width
            pdb_lines.append(line[:60].ljust(60) + "\n")
    if not pdb_lines:
        return None
    pdb_block = "".join(pdb_lines) + "END\n"
    try:
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
        if mol is not None:
            Chem.SanitizeMol(mol, catchErrors=True)
        return mol
    except Exception:
        return None


def charges_ph(molecule_smi: str) -> str:
    """Corrects pH using OpenBabel."""
    if openbabel is None:
        return molecule_smi

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, molecule_smi)
    mol.AddHydrogens()
    mol.CorrectForPH(7.4)
    mol.AddHydrogens()
    return obConversion.WriteString(mol).strip()


def smile_obabel_corrector(smiles_ionized: str) -> str:
    """Manually corrects specific functional group protonations."""
    mol1 = Chem.MolFromSmiles(smiles_ionized, sanitize=False)
    if not mol1:
        return smiles_ionized

    # Ether correction
    p1 = Chem.MolFromSmarts("[#6]-[#8-]-[#6]")
    if mol1.HasSubstructMatch(p1):
        for match in mol1.GetSubstructMatches(p1):
            atom = mol1.GetAtomWithIdx(match[1])
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    try:
        return Chem.MolToSmiles(mol1)
    except Exception:
        return smiles_ionized


def standardize_molecule(smiles: str, standardizer: Standardizer):
    """Runs MolVS standardization + LUDe corrections."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        mol_std = standardizer.fragment_parent(mol)
        mol_std = standardizer.charge_parent(mol_std, skip_standardize=True)
        mol_std = standardizer.isotope_parent(mol_std, skip_standardize=True)

        smi_std = Chem.MolToSmiles(mol_std)
        smi_ph = charges_ph(smi_std)
        smi_final = smile_obabel_corrector(smi_ph)

        return Chem.MolFromSmiles(smi_final)
    except Exception:
        return None


def calc_descriptors(mol):
    """Calculates MW, LogP, RB, HBA, HBD, Charge.

    Returns:
        (pd.Series of properties, heavy atom count)
    """
    props = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Chem.Crippen.MolLogP(mol),
        "Hba": Lipinski.NumHAcceptors(mol),
        "Hbd": Lipinski.NumHDonors(mol),
        "rb": Lipinski.NumRotatableBonds(mol),
        "charge": Chem.rdmolops.GetFormalCharge(mol),
    }
    return pd.Series(props), Descriptors.HeavyAtomCount(mol)


# Known issue (spec §6): not called by generate_decoys_lude.
def calculate_pss(df_decoys_props: pd.DataFrame, query_props: pd.Series, prop_cols) -> list:
    """Calculates Physicochemical Similarity Score (PSS)."""
    distances = {}
    for col in prop_cols:
        d_min = abs(query_props[col] - df_decoys_props[col].min())
        d_max = abs(query_props[col] - df_decoys_props[col].max())
        distances[f"f_{col}"] = max(d_min, d_max)

    pss_scores = []
    for _, row in df_decoys_props.iterrows():
        score_sum = 0.0
        for col in prop_cols:
            denom = distances[f"f_{col}"]
            val = 1.0 if denom == 0 else 1.0 - (abs(query_props[col] - row[col]) / denom)
            score_sum += val
        pss_scores.append(score_sum / len(prop_cols))

    return pss_scores


#: Column order of docking_jobs.csv.
JOB_COLUMNS = (
    "job_id", "protein_pdbqt", "ligand_pdbqt", "output_pdbqt",
    "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
    "exhaustiveness", "num_modes", "energy_range", "molecule_type", "pdb_id", "ligand_id",
)


def docking_jobs(proteins, ligands, box_params, results_dir, docking_cfg) -> pd.DataFrame:
    """One row per protein x ligand docking run.

    Args:
        proteins: Receptor PDBQT paths.
        ligands: {"actives": [...], "decoys": [...]} ligand PDBQT paths.
        box_params: center_x/y/z and size_x/y/z.
        results_dir: Where outputs will be written.
        docking_cfg: Docking settings (exhaustiveness, num_modes, energy_range).

    Returns:
        Jobs table with JOB_COLUMNS.
    """
    rows = []
    job_idx = 0
    for mol_type, lig_list in ligands.items():
        for lig_path in lig_list:
            for prot_path in proteins:
                job_id = f"job_{job_idx:05d}"
                rows.append({
                    "job_id": job_id,
                    "protein_pdbqt": str(prot_path),
                    "ligand_pdbqt": str(lig_path),
                    "output_pdbqt": str(results_dir / f"{job_id}_out.pdbqt"),
                    "center_x": box_params.get("center_x", 0.0),
                    "center_y": box_params.get("center_y", 0.0),
                    "center_z": box_params.get("center_z", 0.0),
                    "size_x": box_params.get("size_x", 20.0),
                    "size_y": box_params.get("size_y", 20.0),
                    "size_z": box_params.get("size_z", 20.0),
                    "exhaustiveness": int(docking_cfg.exhaustiveness),
                    "num_modes": int(docking_cfg.num_modes),
                    "energy_range": docking_cfg.energy_range,
                    "molecule_type": mol_type,
                    "pdb_id": prot_path.stem,
                    "ligand_id": lig_path.stem,
                })
                job_idx += 1
    return pd.DataFrame(rows, columns=list(JOB_COLUMNS))


def vina_argv(job, engine: str, cpus: int) -> List[str]:
    """Command line for one docking job.

    Args:
        job: One jobs-table row (mapping).
        engine: Vina or QuickVina executable.
        cpus: Threads for the run.

    Returns:
        argv list, no shell involved.
    """
    return [
        engine,
        "--receptor", str(job["protein_pdbqt"]),
        "--ligand", str(job["ligand_pdbqt"]),
        "--out", str(job["output_pdbqt"]),
        "--center_x", str(job["center_x"]), "--center_y", str(job["center_y"]), "--center_z", str(job["center_z"]),
        "--size_x", str(job["size_x"]), "--size_y", str(job["size_y"]), "--size_z", str(job["size_z"]),
        "--exhaustiveness", str(job["exhaustiveness"]),
        "--num_modes", str(job["num_modes"]),
        "--energy_range", str(job["energy_range"]),
        "--cpu", str(cpus),
    ]


def local_command(jobs_csv, workers: int) -> str:
    """Shell-independent command that runs every job on this machine.

    Args:
        jobs_csv: docking_jobs.csv path.
        workers: Jobs run at once.

    Returns:
        The command line.
    """
    return f'qpocket screen run-local "{jobs_csv}" --workers {int(workers)}'


def slurm_array_script(jobs_csv, slurm_cfg, offset: int, n_tasks: int, log_dir,
                       chunk: Optional[int] = None, environment: Sequence[str] = ()) -> str:
    """sbatch array script: each task runs one job through `qpocket screen run-task`.

    Args:
        jobs_csv: docking_jobs.csv path.
        slurm_cfg: partition, cpus_per_task, mem, time, max_concurrent.
        offset: First job index this array covers.
        n_tasks: Jobs in this array.
        log_dir: Where SLURM writes .out/.err.
        chunk: Chunk number when the jobs are split across scripts.
        environment: Lines run before the job (module loads, venv activation).

    Returns:
        Script text.
    """
    name = "qpocket_screen" + (f"_{chunk}" if chunk is not None else "")
    command = " ".join([
        "qpocket", "screen", "run-task", shlex.quote(str(jobs_csv)),
        "--index", "$(( SLURM_ARRAY_TASK_ID + %d ))" % offset,
        "--cpus", "\"$SLURM_CPUS_PER_TASK\"",
    ])
    return "\n".join([
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={slurm_cfg.partition}",
        f"#SBATCH --cpus-per-task={slurm_cfg.cpus_per_task}",
        f"#SBATCH --mem={slurm_cfg.mem}",
        f"#SBATCH --time={slurm_cfg.time}",
        f"#SBATCH --array=0-{n_tasks - 1}%{slurm_cfg.max_concurrent}",
        f"#SBATCH --output={log_dir}/dock_%A_%a.out",
        f"#SBATCH --error={log_dir}/dock_%A_%a.err",
        "",
        "# Generated by `qpocket screen prepare` — regenerate rather than editing.",
        "set -euo pipefail",
        "",
        *environment,
        "",
        command,
        "",
    ])
