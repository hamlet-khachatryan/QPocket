"""Protein and ligand preparation for AutoDock Vina / QuickVina.

Public API:
- ``prepare_protein_pdbqt(input_pdb, output_pdbqt, cfg)``
- ``prepare_ligand_pdbqt(smiles, output_pdbqt, cfg)``
- ``prepare_all_proteins(screening_dir, representatives_csv, master_csv, cfg)``
- ``prepare_all_ligands(screening_dir, actives_smi, decoys_smi, cfg, mode)``
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from eliot import log_message, start_action
from rdkit import Chem
from rdkit.Chem import AllChem

from qpocket_screen.config import DockingConfig, PreparationConfig, ScreenConfig


# ---------------------------------------------------------------------------
# Protein preparation
# ---------------------------------------------------------------------------

def prepare_protein_pdbqt(
    input_pdb: Path,
    output_pdbqt: Path,
    cfg: PreparationConfig,
) -> bool:
    """Convert a protein PDB to PDBQT format for AutoDock Vina.

    Steps:
    1. ``pdb2pqr`` — add hydrogens and correct protonation at target pH.
    2. ``mk_prepare_receptor`` (Meeko CLI) — convert PQR → PDBQT.
       Falls back to a minimal manual conversion if Meeko is unavailable.

    Args:
        input_pdb: Path to the cleaned protein PDB file.
        output_pdbqt: Desired output path.
        cfg: :class:`~qpocket_screen.config.PreparationConfig`.

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
    """Minimal PQR/PDB → PDBQT conversion (strips charge/radius columns).

    This fallback writes ATOM/HETATM lines in PDBQT format with a generic
    charge of 0.000 and no radius.  It is intended as a last resort when
    the Meeko receptor tool is unavailable.
    """
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


# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------

def prepare_ligand_pdbqt(
    smiles: str,
    output_pdbqt: Path,
    cfg: PreparationConfig,
    ligand_id: str = "LIG",
) -> bool:
    """Convert a SMILES string to a 3-D PDBQT file for AutoDock Vina.

    Uses RDKit ETKDGv3 for conformer generation + MMFF94 minimisation, then
    Meeko ``MoleculePreparation`` / ``PDBQTWriterLegacy`` for the PDBQT.

    Args:
        smiles: Input SMILES string.
        output_pdbqt: Desired output path.
        cfg: :class:`~qpocket_screen.config.PreparationConfig`.
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
    """Minimal SDF → PDBQT via OpenBabel fallback.

    Writes the conformer to a temporary SDF then converts it using
    ``obabel`` CLI.  Requires OpenBabel to be installed as a command.
    """
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


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def prepare_all_proteins(
    screening_dir: Path,
    representatives_csv: Path,
    master_csv: Path,
    cfg: ScreenConfig,
) -> list[str]:
    """Prepare PDBQT files for all representative structures.

    Reads ``selected_representatives.csv`` to identify which PDB IDs to
    prepare, looks up their cleaned file paths in ``master.csv``, then calls
    :func:`prepare_protein_pdbqt` for each.

    Args:
        screening_dir: Root screening directory (``screening/proteins/`` is created here).
        representatives_csv: Path to ``metadata/selected_representatives.csv``.
        master_csv: Path to ``metadata/master.csv``.
        cfg: Full :class:`~qpocket_screen.config.ScreenConfig`.

    Returns:
        List of PDB IDs that were successfully prepared.
    """
    proteins_dir = screening_dir / "proteins"
    proteins_dir.mkdir(parents=True, exist_ok=True)

    df_reps = pd.read_csv(representatives_csv)
    df_master = pd.read_csv(master_csv)

    rep_ids: list[str] = df_reps["pdb_id"].tolist() if "pdb_id" in df_reps.columns else df_reps.iloc[:, 0].tolist()

    prepared: list[str] = []
    for pdb_id in rep_ids:
        row = df_master[df_master["pdb_id"] == pdb_id]
        if row.empty:
            log_message(message_type="warning", msg=f"{pdb_id} not in master.csv, skipping")
            continue

        cleaned_file = row.iloc[0].get("cleaned_file")
        if not cleaned_file or pd.isna(cleaned_file):
            log_message(message_type="warning", msg=f"{pdb_id} has no cleaned_file, skipping")
            continue

        # Resolve relative path (master.csv stores filenames relative to structures/)
        structures_dir = screening_dir.parent / "structures"
        cleaned_path = structures_dir / "cleaned" / cleaned_file
        if not cleaned_path.exists():
            log_message(message_type="warning", msg=f"cleaned file not found: {cleaned_path}")
            continue

        out_pdbqt = proteins_dir / f"{pdb_id}.pdbqt"
        if out_pdbqt.exists():
            log_message(message_type="skip", reason="already prepared", pdb_id=pdb_id)
            prepared.append(pdb_id)
            continue

        ok = prepare_protein_pdbqt(cleaned_path, out_pdbqt, cfg.preparation)
        if ok:
            prepared.append(pdb_id)

    log_message(
        message_type="proteins_prepared",
        total=len(rep_ids),
        prepared=len(prepared),
    )
    return prepared


def prepare_all_ligands(
    screening_dir: Path,
    actives_smi: Path,
    decoys_smi: Optional[Path],
    cfg: ScreenConfig,
    mode: str = "cross_docking",
) -> dict[str, list[str]]:
    """Prepare PDBQT files for all actives (and optionally decoys).

    Args:
        screening_dir: Root screening directory.
        actives_smi: Path to ``screening/actives.smi``.
        decoys_smi: Path to ``screening/decoys.smi``; only used when
            *mode* is ``"virtual_screening"``.
        cfg: Full :class:`~qpocket_screen.config.ScreenConfig`.
        mode: ``"cross_docking"`` or ``"virtual_screening"``.

    Returns:
        Dict with keys ``"actives"`` and ``"decoys"``, each a list of
        prepared ligand IDs.
    """
    actives_dir = screening_dir / "actives"
    decoys_dir = screening_dir / "decoys"
    actives_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[str]] = {"actives": [], "decoys": []}

    # Actives
    if actives_smi.exists():
        df_act = pd.read_csv(actives_smi, sep="\t", header=None, names=["SMILES", "PDB_ID", "Ligand_Name"])
        for _, row in df_act.iterrows():
            smiles = row["SMILES"]
            lig_id = f"{row['PDB_ID']}_{row['Ligand_Name']}"
            out_pdbqt = actives_dir / f"{lig_id}.pdbqt"
            if out_pdbqt.exists():
                result["actives"].append(lig_id)
                continue
            ok = prepare_ligand_pdbqt(smiles, out_pdbqt, cfg.preparation, ligand_id=lig_id)
            if ok:
                result["actives"].append(lig_id)
    else:
        log_message(message_type="warning", msg="actives.smi not found", path=str(actives_smi))

    # Decoys (virtual_screening only)
    if mode == "virtual_screening" and decoys_smi and decoys_smi.exists():
        decoys_dir.mkdir(parents=True, exist_ok=True)
        df_dec = pd.read_csv(decoys_smi, sep="\t", header=None, names=["SMILES", "ID"])
        for _, row in df_dec.iterrows():
            smiles = row["SMILES"]
            lig_id = str(row["ID"])
            out_pdbqt = decoys_dir / f"{lig_id}.pdbqt"
            if out_pdbqt.exists():
                result["decoys"].append(lig_id)
                continue
            ok = prepare_ligand_pdbqt(smiles, out_pdbqt, cfg.preparation, ligand_id=lig_id)
            if ok:
                result["decoys"].append(lig_id)

    log_message(
        message_type="ligands_prepared",
        actives=len(result["actives"]),
        decoys=len(result["decoys"]),
    )
    return result
