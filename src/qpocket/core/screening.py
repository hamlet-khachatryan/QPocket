"""Virtual screening: boxes, molecule standardisation, job tables and scripts (pure)."""
from __future__ import annotations

import shlex
import string
from pathlib import Path
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
from molvs import Standardizer
from rdkit import Chem
from scipy.spatial import cKDTree

from qpocket.constants import STANDARD_AA, canonical_residue, element_of
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


class Pose(NamedTuple):
    """One docked pose.

    Attributes:
        rank: 1 for the best-scored pose.
        score: Engine score (kcal/mol for the Vina family).
        mol: RDKit molecule in the receptor frame, or None when unreadable.
    """

    rank: int
    score: float
    mol: Optional[Chem.Mol]


class Engine(NamedTuple):
    """What a docking engine needs from QPocket and gives back.

    Attributes:
        argv: (job row, executable, cpus) -> command line.
        read_poses: Output file -> poses in rank order.
        output_ext: Output file suffix.
        executable: Default binary name.
    """

    argv: Callable[[Mapping, str, int], List[str]]
    read_poses: Callable[[Path], List[Pose]]
    output_ext: str
    executable: str


def vina_poses(text: str) -> List[Pose]:
    """Every pose in a Vina-family PDBQT output, best first.

    Args:
        text: Output file contents.

    Returns:
        Poses with rank, score and molecule.
    """
    poses: List[Pose] = []
    lines: List[str] = []
    score: Optional[float] = None
    for line in text.splitlines(keepends=True):
        if line.startswith("MODEL"):
            lines, score = [], None
        elif line.startswith("REMARK VINA RESULT"):
            score = float(line.split()[3])
        elif line.startswith("ENDMDL"):
            poses.append(Pose(len(poses) + 1, score if score is not None else float("nan"),
                              pdbqt_lines_to_mol(lines)))
        else:
            lines.append(line)
    return poses


#: Fields a custom engine's command template may use.
PLACEHOLDERS = ("exe", "receptor", "ligand", "output", "center_x", "center_y", "center_z",
                "size_x", "size_y", "size_z", "exhaustiveness", "num_modes", "energy_range", "cpus")


def template_fields(command: str) -> set:
    """Placeholder names a command template uses.

    Args:
        command: The template.

    Returns:
        Field names.
    """
    return {name for _, name, _, _ in string.Formatter().parse(command) if name}


def _sdf_poses(score_property: str):
    def read(path) -> List[Pose]:
        if not Path(path).exists():
            return []
        poses = []
        for mol in Chem.SDMolSupplier(str(path), removeHs=False):
            if mol is not None:
                score = float(mol.GetProp(score_property)) if mol.HasProp(score_property) else float("nan")
                poses.append(Pose(len(poses) + 1, score, mol))
        return poses
    return read


def pdbqt_poses(path) -> List[Pose]:
    """Every pose in a Vina-style PDBQT file; none when it is absent.

    Args:
        path: Output file.

    Returns:
        Poses, best first.
    """
    return vina_poses(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else []


def template_engine(spec: Mapping) -> Engine:
    """An Engine built from a config declaration.

    Args:
        spec: executable, command template, output ("pdbqt" or "sdf"), score_property.

    Returns:
        The engine.
    """
    command = str(spec["command"])

    def argv(job: Mapping, executable: str, cpus: int) -> List[str]:
        return shlex.split(command.format(
            exe=executable, receptor=job["protein_pdbqt"], ligand=job["ligand_pdbqt"],
            output=job["output_pdbqt"], cpus=cpus,
            **{k: job[k] for k in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
                                   "exhaustiveness", "num_modes", "energy_range")}))

    sdf = spec.get("output") == "sdf"
    reader = _sdf_poses(spec.get("score_property", "minimizedAffinity")) if sdf else pdbqt_poses
    return Engine(argv, reader, ".sdf" if sdf else ".pdbqt", str(spec["executable"]))


def charges_ph(molecule_smi: str, ph: float = 7.4) -> str:
    """Protonation state of a molecule at a pH.

    Args:
        molecule_smi: SMILES of the molecule.
        ph: pH at which to protonate.

    Returns:
        SMILES of the dominant state, or the input when none could be assigned.
    """
    from qpocket.core.protonation import dominant_state

    return dominant_state(molecule_smi, ph) or molecule_smi


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
    "engine",
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
                    "engine": getattr(docking_cfg, "engine", "vina"),
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
    """Sbatch array script: each task runs one job through `qpocket screen run-task`.

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


_AROMATIC_CARBONS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"},
}
_AD_ELEMENT_TYPES = {"O": "OA", "S": "SA", "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
                     "ZN": "Zn", "MG": "Mg", "CA": "Ca", "FE": "Fe", "MN": "Mn", "CO": "Co", "CU": "Cu", "NI": "Ni"}
_H_BOND_MAX = 1.3  # A; an H further than this from every heavy atom keeps its own line


def receptor_pdbqt(atoms) -> str:
    """AutoDock/Vina receptor PDBQT text from a protonated PQR.

    Args:
        atoms: PQRAtoms from read_pqr (pdb2pqr output with AMBER names).

    Returns:
        PDBQT text: PQR charges, AutoDock types, non-polar hydrogens merged.
    """
    n = len(atoms.coords)
    elements = [element_of(a, r) for a, r in zip(atoms.atom_names, atoms.res_names)]
    heavy = [i for i in range(n) if elements[i] != "H"]
    charges = np.array(atoms.charges, dtype=float)
    parent = {}
    if heavy:
        tree = cKDTree(atoms.coords[heavy])
        for i in (i for i in range(n) if elements[i] == "H"):
            d, j = tree.query(atoms.coords[i])
            if d <= _H_BOND_MAX:
                parent[i] = heavy[j]

    n_hydrogens = {}
    for h, p in parent.items():
        n_hydrogens[p] = n_hydrogens.get(p, 0) + 1
    merged = {h for h, p in parent.items() if elements[p] == "C"}
    for h in merged:
        charges[parent[h]] += charges[h]  # united atom: carbon carries its hydrogens' charge

    lines = []
    serial = 0
    for i in range(n):
        if i in merged:
            continue
        name, res = atoms.atom_names[i], atoms.res_names[i]
        std = canonical_residue(res)
        element = elements[i]
        if element == "H":
            ad_type = "HD" if i in parent else "H"
        elif element == "C":
            ad_type = "A" if name in _AROMATIC_CARBONS.get(std, ()) else "C"
        elif element == "N":
            # An N with no hydrogen accepts: HIE's ND1, HID's NE2, ligand ring N (not proline).
            accepts = n_hydrogens.get(i, 0) == 0 and not (std == "PRO" or name == "N")
            ad_type = "NA" if accepts else "N"
        else:
            ad_type = _AD_ELEMENT_TYPES.get(element, element.capitalize())
        serial += 1
        record = "ATOM" if std in STANDARD_AA else "HETATM"
        label = name[:4] if len(name) >= 4 else f" {name:<3}"
        x, y, z = atoms.coords[i]
        lines.append(
            f"{record:<6}{serial:5d} {label} {res[:3]:>3} {atoms.chain_ids[i][:1]:1}{atoms.res_seqs[i]:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}    {charges[i]:6.3f} {ad_type:<2}"
        )
    return "\n".join(lines) + "\n"
