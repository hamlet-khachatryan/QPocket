"""Receptor and ligand preparation for every representative and ligand."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from eliot import log_message

from qpocket.io.repair import repair
from qpocket.io.screening import prepare_ligand_pdbqt, prepare_protein_pdbqt, receptor_from_pqr
from qpocket.screen_config import ScreenConfig


def prepare_all_proteins(
    screening_dir: Path,
    representatives_csv: Path,
    master_csv: Path,
    cfg: ScreenConfig,
) -> list[str]:
    """Prepare PDBQT files for all representative structures.

    Args:
        screening_dir: Root screening directory (``screening/proteins/`` is created here).
        representatives_csv: Path to ``metadata/selected_representatives.csv``.
        master_csv: Path to ``metadata/master.csv``.
        cfg: Full :class:`~qpocket.screen_config.ScreenConfig`.

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

        # The featurisation PQR carries the run's protonation states and charges;
        # runs made before it existed fall back to preparing the cleaned PDB.
        pqr = screening_dir.parent / "pockets" / pdb_id / "protein.pqr"
        if pqr.exists():
            ok = receptor_from_pqr(pqr, out_pdbqt)
        else:
            log_message(message_type="warning", pdb_id=pdb_id,
                        msg="no protein.pqr from the run; preparing the receptor from the cleaned PDB")
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
        decoys_smi: Path to ``screening/decoys.smi``; only used when *mode* is ``"virtual_screening"``.
        cfg: Full :class:`~qpocket.screen_config.ScreenConfig`.
        mode: ``"cross_docking"`` or ``"virtual_screening"``.

    Returns:
        Dict with keys ``"actives"`` and ``"decoys"``, each a list of prepared ligand IDs.
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


def _protonate(pdb: Path, pqr: Path, cfg: ScreenConfig) -> Path:
    from qpocket.config import plugin_params
    from qpocket.core.ports import RECEPTORS

    plugin = RECEPTORS.get(cfg.receptor.use)
    return plugin.fn(pdb, pqr, plugin_params(RECEPTORS, cfg.receptor.use, cfg.receptor.options))


def prepare_receptor(cleaned_pdb: Path, pocket: np.ndarray, out_dir: Path, rep_id: str,
                     cfg: ScreenConfig) -> Optional[Path]:
    """Repair, protonate and write one docking receptor.

    Args:
        cleaned_pdb: The representative's cleaned structure.
        pocket: Its pocket points, for deciding which loops to build.
        out_dir: receptors/ of the task folder.
        rep_id: Representative id.
        cfg: Screening configuration.

    Returns:
        The receptor PDBQT, or None when a step failed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb, pqr, pdbqt = out_dir / f"{rep_id}.pdb", out_dir / f"{rep_id}.pqr", out_dir / f"{rep_id}.pdbqt"
    try:
        if cfg.receptor.repair:
            repair(cleaned_pdb, pdb, pocket, cfg.receptor.loop_radius)
        else:
            shutil.copyfile(cleaned_pdb, pdb)
        _protonate(pdb, pqr, cfg)
    except Exception as e:
        log_message(message_type="error", stage="prepare_receptor", pdb_id=rep_id, error=str(e))
        return None
    return pdbqt if receptor_from_pqr(pqr, pdbqt) else None
