"""Screening files and tools: Vina output, boxes, ChEMBL, receptor and ligand PDBQT."""
from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from eliot import log_message, start_action
from rdkit import Chem
from rdkit.Chem import AllChem

from qpocket.core.screening import calculate_box, pdbqt_lines_to_mol
from qpocket.screen_config import PreparationConfig


def parse_vina_output(pdbqt_path: Path) -> Tuple[Optional[float], Optional[Chem.Mol]]:
    """Parse an AutoDock Vina output PDBQT file.

    Args:
        pdbqt_path: Path to the docking output PDBQT.

    Returns:
        ``(best_score, best_pose_mol)`` — either may be ``None`` on failure.
    """
    if not pdbqt_path.exists():
        return None, None

    best_score: Optional[float] = None
    model_lines: List[str] = []
    in_first_model = False

    try:
        with open(pdbqt_path) as fh:
            for line in fh:
                # Score lines: "REMARK VINA RESULT:  -8.5   0.000   0.000"
                if line.startswith("REMARK VINA RESULT"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            score = float(parts[3])
                            if best_score is None or score < best_score:
                                best_score = score
                        except ValueError:
                            pass

                if line.startswith("MODEL"):
                    if not in_first_model:
                        in_first_model = True
                        model_lines = []
                elif line.startswith("ENDMDL"):
                    if in_first_model:
                        break
                elif in_first_model:
                    model_lines.append(line)

        mol = pdbqt_lines_to_mol(model_lines)
        return best_score, mol

    except Exception as e:
        log_message(message_type="error", stage="parse_vina_output", path=str(pdbqt_path), error=str(e))
        return None, None


def parse_vina_box(vina_box_txt: Path) -> Dict[str, float]:
    """Parse a Vina-style ``vina_box.txt`` file.

    Args:
        vina_box_txt: Path to the box config file.

    Returns:
        Dict with keys ``center_x/y/z``, ``size_x/y/z``, ``exhaustiveness``, ``num_modes``, ``energy_range``.
    """
    params: Dict[str, float] = {}
    with open(vina_box_txt) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                try:
                    params[key.strip()] = float(val.strip())
                except ValueError:
                    pass
    return params


def extract_vina_box(
        cfg: Dict,
        dirs: Dict[str, Path]
):
    """Generates a Vina-compatible config.txt with bounding box coordinates."""
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
        # Known issue (spec §6): pockets are saved as binding_site.csv, so this glob matches nothing.
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


def download_chembl_dataset(url: str, local_dir: Union[str, Path]) -> Path:
    """Download a ChEMBL dataset from Google Drive (or any direct URL).

    Args:
        url: Google Drive share URL or direct download URL.
        local_dir: Target directory.

    Returns:
        Resolved path to *local_dir*.
    """
    local_dir = Path(local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    existing_txt = list(local_dir.glob("*.txt"))
    if existing_txt:
        log_message(
            message_type="chembl_download_skipped",
            reason="txt files already present",
            count=len(existing_txt),
            local_dir=str(local_dir),
        )
        return local_dir

    import gdown  # lazy import so the rest of the module works without gdown

    log_message(message_type="chembl_download_start", url=url, local_dir=str(local_dir))

    archive_path = local_dir / "chembl_data.zip"
    gdown.download(url, str(archive_path), quiet=False, fuzzy=True)

    if archive_path.exists() and zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(local_dir)
        archive_path.unlink()
        log_message(message_type="chembl_extracted", local_dir=str(local_dir))
    elif not archive_path.exists():
        log_message(
            message_type="warning",
            msg="gdown did not produce an archive — check the URL or download manually",
        )

    return local_dir


def prepare_protein_pdbqt(
    input_pdb: Path,
    output_pdbqt: Path,
    cfg: PreparationConfig,
) -> bool:
    """Convert a protein PDB to PDBQT format for AutoDock Vina.

    Args:
        input_pdb: Path to the cleaned protein PDB file.
        output_pdbqt: Desired output path.
        cfg: :class:`~qpocket.screen_config.PreparationConfig`.

    Returns:
        ``True`` on success, ``False`` if preparation failed.
    """
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    with start_action(action_type="prepare_protein_pdbqt", pdb=str(input_pdb)):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pqr_path = tmp_path / "protonated.pqr"

            # Step 1: pdb2pqr
            pdb2pqr_cmd = [
                "pdb2pqr",
                "--ff", "AMBER",
                "--ffout", "AMBER",
                "--nodebump",
                str(input_pdb),
                str(pqr_path),
            ]
            try:
                result = subprocess.run(
                    pdb2pqr_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    log_message(
                        message_type="warning",
                        stage="pdb2pqr",
                        pdb=str(input_pdb),
                        stderr=result.stderr[:500],
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log_message(message_type="error", stage="pdb2pqr", error=str(e))
                return False

            # Use PQR if produced, else fall back to original PDB
            prep_input = pqr_path if pqr_path.exists() else input_pdb

            # Step 2: mk_prepare_receptor (Meeko)
            meeko_cmd = [
                "mk_prepare_receptor",
                "-i", str(prep_input),
                "-o", str(output_pdbqt),
            ]
            try:
                result = subprocess.run(
                    meeko_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0 and output_pdbqt.exists():
                    log_message(
                        message_type="protein_prepared",
                        pdb=str(input_pdb),
                        pdbqt=str(output_pdbqt),
                    )
                    return True
                log_message(
                    message_type="warning",
                    stage="mk_prepare_receptor",
                    stderr=result.stderr[:500],
                )
            except FileNotFoundError:
                log_message(
                    message_type="warning",
                    stage="mk_prepare_receptor",
                    msg="mk_prepare_receptor not found — using fallback PQR→PDBQT converter",
                )

            # Fallback: manual PQR → PDBQT conversion
            return _pqr_to_pdbqt_fallback(prep_input, output_pdbqt)


def _pqr_to_pdbqt_fallback(input_path: Path, output_pdbqt: Path) -> bool:
    """Minimal PQR/PDB → PDBQT conversion (strips charge/radius columns)."""
    try:
        lines_out = []
        with open(input_path) as fh:
            for line in fh:
                record = line[:6].strip()
                if record not in ("ATOM", "HETATM"):
                    if record in ("END", "ENDMDL", "TER"):
                        lines_out.append(line.rstrip() + "\n")
                    continue

                # PQR cols 55-62 = charge, 63-70 = radius — strip them
                pdb_part = line[:54].ljust(54)
                # Append partial charge column (mandatory in PDBQT)
                lines_out.append(f"{pdb_part}  0.000\n")

        with open(output_pdbqt, "w") as fh:
            fh.writelines(lines_out)

        log_message(
            message_type="protein_prepared_fallback",
            input=str(input_path),
            pdbqt=str(output_pdbqt),
        )
        return True

    except Exception as e:
        log_message(message_type="error", stage="pqr_to_pdbqt_fallback", error=str(e))
        return False


def prepare_ligand_pdbqt(
    smiles: str,
    output_pdbqt: Path,
    cfg: PreparationConfig,
    ligand_id: str = "LIG",
) -> bool:
    """Convert a SMILES string to a 3-D PDBQT file for AutoDock Vina.

    Args:
        smiles: Input SMILES string.
        output_pdbqt: Desired output path.
        cfg: :class:`~qpocket.screen_config.PreparationConfig`.
        ligand_id: Short identifier used in log messages.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    with start_action(action_type="prepare_ligand_pdbqt", ligand_id=ligand_id):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log_message(
                message_type="error",
                stage="ligand_prep",
                ligand_id=ligand_id,
                error="invalid SMILES",
            )
            return False

        mol = Chem.AddHs(mol)

        # Conformer generation — ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=cfg.num_conformers, params=params)

        if not conf_ids:
            # Fallback to ETKDGv2
            params2 = AllChem.ETKDGv2()
            params2.randomSeed = 42
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=cfg.num_conformers, params=params2)

        if not conf_ids:
            log_message(
                message_type="error",
                stage="conformer_generation",
                ligand_id=ligand_id,
                error="no conformers generated",
            )
            return False

        # MMFF94 minimisation — keep best (lowest energy) conformer
        best_conf_id = 0
        best_energy = float("inf")
        for conf_id in conf_ids:
            try:
                res = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=cfg.max_iters)
                if res == 0:  # converged
                    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
                    if ff is not None:
                        energy = ff.CalcEnergy()
                        if energy < best_energy:
                            best_energy = energy
                            best_conf_id = conf_id
            except Exception:
                pass

        # Write the best conformer as PDBQT via Meeko
        return _write_ligand_pdbqt_meeko(mol, best_conf_id, output_pdbqt, ligand_id)


def _write_ligand_pdbqt_meeko(
    mol: Chem.Mol,
    conf_id: int,
    output_pdbqt: Path,
    ligand_id: str,
) -> bool:
    """Use Meeko to write a 3-D RDKit mol to PDBQT."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore

        mol_setup_list = MoleculePreparation().prepare(mol, conformer_id=conf_id)
        if not mol_setup_list:
            raise RuntimeError("MoleculePreparation returned empty setup list")
        molsetup = mol_setup_list[0]
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(molsetup)
        if not is_ok:
            raise RuntimeError(f"PDBQTWriterLegacy failed: {error_msg}")

        with open(output_pdbqt, "w") as fh:
            fh.write(pdbqt_string)

        log_message(
            message_type="ligand_prepared",
            ligand_id=ligand_id,
            pdbqt=str(output_pdbqt),
        )
        return True

    except ImportError:
        log_message(
            message_type="warning",
            stage="meeko",
            msg="meeko not installed — using SDF fallback for ligand PDBQT",
        )
        return _write_ligand_pdbqt_sdf_fallback(mol, conf_id, output_pdbqt, ligand_id)
    except Exception as e:
        log_message(
            message_type="error",
            stage="meeko_ligand",
            ligand_id=ligand_id,
            error=str(e),
        )
        return False


def _write_ligand_pdbqt_sdf_fallback(
    mol: Chem.Mol,
    conf_id: int,
    output_pdbqt: Path,
    ligand_id: str,
) -> bool:
    """Minimal SDF → PDBQT via OpenBabel fallback."""
    import tempfile

    from rdkit.Chem import SDWriter

    try:
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp_sdf:
            sdf_path = Path(tmp_sdf.name)

        writer = SDWriter(str(sdf_path))
        writer.write(mol, confId=conf_id)
        writer.close()

        result = subprocess.run(
            ["obabel", str(sdf_path), "-O", str(output_pdbqt), "--gen3d"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        sdf_path.unlink(missing_ok=True)

        if result.returncode == 0 and output_pdbqt.exists():
            log_message(
                message_type="ligand_prepared_fallback",
                ligand_id=ligand_id,
                pdbqt=str(output_pdbqt),
            )
            return True

        log_message(
            message_type="error",
            stage="obabel_fallback",
            ligand_id=ligand_id,
            stderr=result.stderr[:300],
        )
        return False

    except Exception as e:
        log_message(message_type="error", stage="sdf_fallback", ligand_id=ligand_id, error=str(e))
        return False
