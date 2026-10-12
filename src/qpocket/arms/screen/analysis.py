"""Post-docking analysis: scores, pose RMSD, enrichment and interactions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from eliot import log_message, start_action
from rdkit import Chem

from qpocket.io.screening import parse_vina_output
from qpocket.screen_config import AnalysisConfig, ScreenConfig


def collect_docking_scores(
    screening_dir: Path,
    jobs_csv: Path,
) -> pd.DataFrame:
    """Collect docking scores for all jobs listed in *jobs_csv*.

    Args:
        screening_dir: Root screening directory (used to resolve relative paths).
        jobs_csv: Path to ``docking_jobs.csv``.

    Returns:
        DataFrame with columns: job_id, ligand_id, pdb_id, molecule_type, score, output_exists.
    """
    df_jobs = pd.read_csv(jobs_csv)
    records = []

    for _, row in df_jobs.iterrows():
        out_pdbqt = Path(row["output_pdbqt"])
        score, _ = parse_vina_output(out_pdbqt)
        records.append(
            {
                "job_id": row["job_id"],
                "ligand_id": row["ligand_id"],
                "pdb_id": row["pdb_id"],
                "molecule_type": row["molecule_type"],
                "score": score,
                "output_exists": out_pdbqt.exists(),
            }
        )

    df = pd.DataFrame(records)
    log_message(
        message_type="scores_collected",
        total=len(df),
        with_score=int(df["score"].notna().sum()),
    )
    return df


def compute_rmsd_mcs(docked_mol: Chem.Mol, crystal_mol: Chem.Mol) -> Optional[float]:
    """Compute RMSD between a docked pose and the crystal ligand using MCS.

    Args:
        docked_mol: RDKit Mol from a docked pose.
        crystal_mol: RDKit Mol from the crystal structure.

    Returns:
        RMSD in Å, or ``None`` if the calculation fails.
    """
    try:
        from rdkit.Chem import rdFMCS, rdMolAlign

        mcs = rdFMCS.FindMCS(
            [docked_mol, crystal_mol],
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            timeout=10,
        )
        if mcs.numAtoms == 0:
            return None

        query = Chem.MolFromSmarts(mcs.smartsString)
        docked_match = docked_mol.GetSubstructMatch(query)
        crystal_match = crystal_mol.GetSubstructMatch(query)

        if not docked_match or not crystal_match:
            return None

        atom_map = list(zip(docked_match, crystal_match))
        rmsd = rdMolAlign.GetBestRMS(docked_mol, crystal_mol, map=[atom_map])
        return rmsd

    except Exception as e:
        log_message(message_type="error", stage="compute_rmsd_mcs", error=str(e))
        return None


def run_rmsd_analysis(
    scores_df: pd.DataFrame,
    crystal_ligands_dir: Path,
    master_csv: Path,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Compute RMSD for each docked pose vs. crystal ligand.

    Args:
        scores_df: Output of :func:`collect_docking_scores`.
        crystal_ligands_dir: ``structures/extracted_ligands/``.
        master_csv: ``metadata/master.csv`` (for ligand name lookup).
        cfg: Analysis config.

    Returns:
        DataFrame with columns: pdb_id, ligand_id, best_score, rmsd, success.
    """
    df_master = pd.read_csv(master_csv) if master_csv.exists() else pd.DataFrame()

    records = []
    for _, row in scores_df[scores_df["output_exists"]].iterrows():
        pdb_id = row["pdb_id"]

        # Find crystal ligand file for this pdb_id
        crystal_files = list(crystal_ligands_dir.glob(f"{pdb_id}_*.pdb"))
        if not crystal_files:
            records.append(
                {
                    "pdb_id": pdb_id,
                    "ligand_id": row["ligand_id"],
                    "best_score": row["score"],
                    "rmsd": None,
                    "success": False,
                }
            )
            continue

        crystal_path = crystal_files[0]
        crystal_mol = Chem.MolFromPDBFile(str(crystal_path), sanitize=False, removeHs=True)
        if crystal_mol is None:
            records.append(
                {
                    "pdb_id": pdb_id,
                    "ligand_id": row["ligand_id"],
                    "best_score": row["score"],
                    "rmsd": None,
                    "success": False,
                }
            )
            continue

        # Parse docked pose
        # Look up output_pdbqt from jobs_csv if available; fallback to None
        _, docked_mol = parse_vina_output(
            # Known issue (spec §6): placeholder path; RMSD reference lookup is not implemented.
            Path(str(crystal_ligands_dir).replace("extracted_ligands", ".."))  # placeholder
        )

        rmsd = None
        success = False
        if docked_mol is not None:
            rmsd = compute_rmsd_mcs(docked_mol, crystal_mol)
            if rmsd is not None:
                success = rmsd <= cfg.rmsd_threshold

        records.append(
            {
                "pdb_id": pdb_id,
                "ligand_id": row["ligand_id"],
                "best_score": row["score"],
                "rmsd": rmsd,
                "success": success,
            }
        )

    df = pd.DataFrame(records)
    log_message(
        message_type="rmsd_analysis_done",
        total=len(df),
        success=int(df["success"].sum()) if not df.empty else 0,
    )
    return df


def _run_rmsd_analysis_from_jobs(
    jobs_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    crystal_ligands_dir: Path,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Internal: compute RMSD using job DataFrame output PDBQT paths."""
    records = []
    df_merged = scores_df.merge(jobs_df[["job_id", "output_pdbqt"]], on="job_id", how="left")

    for _, row in df_merged[df_merged["output_exists"]].iterrows():
        pdb_id = row["pdb_id"]
        out_pdbqt = Path(row["output_pdbqt"])

        crystal_files = list(crystal_ligands_dir.glob(f"{pdb_id}_*.pdb"))
        crystal_mol = None
        if crystal_files:
            crystal_mol = Chem.MolFromPDBFile(str(crystal_files[0]), sanitize=False, removeHs=True)

        _, docked_mol = parse_vina_output(out_pdbqt)

        rmsd = None
        success = False
        if docked_mol is not None and crystal_mol is not None:
            rmsd = compute_rmsd_mcs(docked_mol, crystal_mol)
            if rmsd is not None:
                success = rmsd <= cfg.rmsd_threshold

        records.append(
            {
                "pdb_id": pdb_id,
                "ligand_id": row["ligand_id"],
                "best_score": row["score"],
                "rmsd": rmsd,
                "success": success,
            }
        )

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["pdb_id", "ligand_id", "best_score", "rmsd", "success"]
    )


def compute_enrichment_metrics(
    scores_df: pd.DataFrame,
    cfg: AnalysisConfig,
) -> Dict:
    """Compute VS enrichment metrics from a scored compound list.

    Args:
        scores_df: DataFrame from :func:`collect_docking_scores` with scores.
        cfg: Analysis config.

    Returns:
        Dict with keys ``roc_auc``, ``bedroc``, and ``ef`` (sub-dict keyed by fraction).
    """
    df = scores_df.dropna(subset=["score"]).copy()
    df["is_active"] = (df["molecule_type"] == "actives").astype(int)
    df = df.sort_values("score", ascending=True)  # lower = better

    if df["is_active"].sum() == 0:
        log_message(message_type="warning", msg="No actives in scores_df — enrichment metrics undefined")
        return {"roc_auc": None, "bedroc": None, "ef": {}}

    metrics: Dict = {}

    # ROC-AUC
    try:
        from sklearn.metrics import roc_auc_score

        # Invert scores so that high value = good (active)
        scores_inverted = -df["score"].values
        metrics["roc_auc"] = float(roc_auc_score(df["is_active"].values, scores_inverted))
    except Exception as e:
        log_message(message_type="warning", stage="roc_auc", error=str(e))
        metrics["roc_auc"] = None

    # BEDROC
    try:
        from rdkit.ML.Scoring.Scoring import CalcBEDROC

        sorted_list = list(zip(df["score"].values, df["is_active"].values))
        metrics["bedroc"] = float(CalcBEDROC(sorted_list, col=1, alpha=cfg.bedroc_alpha))
    except Exception as e:
        log_message(message_type="warning", stage="bedroc", error=str(e))
        metrics["bedroc"] = None

    # Enrichment Factors
    total = len(df)
    total_actives = int(df["is_active"].sum())
    baseline_rate = total_actives / total if total > 0 else 0.0

    ef_dict: Dict[float, Optional[float]] = {}
    for frac in cfg.ef_fractions:
        n_top = max(1, int(np.ceil(frac * total)))
        top_df = df.head(n_top)
        actives_in_top = int(top_df["is_active"].sum())
        rate_in_top = actives_in_top / n_top if n_top > 0 else 0.0
        ef_dict[frac] = (rate_in_top / baseline_rate) if baseline_rate > 0 else None

    metrics["ef"] = ef_dict

    log_message(
        message_type="enrichment_metrics_computed",
        roc_auc=metrics.get("roc_auc"),
        bedroc=metrics.get("bedroc"),
    )
    return metrics


def run_interaction_analysis(
    screening_dir: Path,
    jobs_csv: Path,
    cfg: ScreenConfig,
) -> pd.DataFrame:
    """Run PLIP on active docking results and aggregate interaction stats.

    Args:
        screening_dir: Root screening directory.
        jobs_csv: Path to ``docking_jobs.csv``.
        cfg: Full screen config (not used directly but passed for consistency).

    Returns:
        DataFrame with columns: job_id, pdb_id, ligand_id, interaction_type, count.
    """
    from qpocket.io.interactions import interactions_for

    df_jobs = pd.read_csv(jobs_csv)
    actives_jobs = df_jobs[df_jobs["molecule_type"] == "actives"]

    records = []
    for _, row in actives_jobs.iterrows():
        out_pdbqt = Path(row["output_pdbqt"])
        if not out_pdbqt.exists():
            continue

        # Convert PDBQT to PDB for the detector (write temp file)
        import tempfile
        _, mol = parse_vina_output(out_pdbqt)
        if mol is None:
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
                pdb_block = Chem.MolToPDBBlock(mol)
                if pdb_block:
                    tmp.write(pdb_block)
                    tmp_path = Path(tmp.name)

            # Known issue (spec §6): the pose is analysed without its receptor.
            interactions = interactions_for(tmp_path, "LIG")
            tmp_path.unlink(missing_ok=True)

        except Exception as e:
            log_message(
                message_type="error",
                stage="interaction_analysis",
                job_id=row["job_id"],
                error=str(e),
            )
            continue

        # Count by interaction type
        type_counts: Dict[str, int] = {}
        for inter in interactions:
            if "|" in inter:
                itype = inter.split("|")[1]
                type_counts[itype] = type_counts.get(itype, 0) + 1

        for itype, count in type_counts.items():
            records.append(
                {
                    "job_id": row["job_id"],
                    "pdb_id": row["pdb_id"],
                    "ligand_id": row["ligand_id"],
                    "interaction_type": itype,
                    "count": count,
                }
            )

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["job_id", "pdb_id", "ligand_id", "interaction_type", "count"]
    )
    log_message(message_type="interaction_analysis_done", n_interactions=len(df))
    return df


def run_full_analysis(
    screening_dir: Path,
    pipeline_dir: Path,
    cfg: ScreenConfig,
) -> None:
    """Run the complete post-docking analysis pipeline.

    Args:
        screening_dir: Root screening directory.
        pipeline_dir: Root pipeline output directory (for crystal ligands etc.).
        cfg: Full screen config.
    """
    analysis_dir = screening_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    jobs_csv = screening_dir / "docking_jobs.csv"
    if not jobs_csv.exists():
        log_message(message_type="error", error="docking_jobs.csv not found", path=str(jobs_csv))
        return

    jobs_df = pd.read_csv(jobs_csv)

    # 1. Collect scores
    with start_action(action_type="collect_scores"):
        scores_df = collect_docking_scores(screening_dir, jobs_csv)
        scores_df.to_csv(analysis_dir / "docking_scores.csv", index=False)

    # 2. RMSD analysis
    crystal_ligands_dir = pipeline_dir / "structures" / "extracted_ligands"
    master_csv = pipeline_dir / "metadata" / "master.csv"

    with start_action(action_type="rmsd_analysis"):
        rmsd_df = _run_rmsd_analysis_from_jobs(jobs_df, scores_df, crystal_ligands_dir, cfg.analysis)
        rmsd_df.to_csv(analysis_dir / "rmsd_analysis.csv", index=False)

    # 3. Enrichment metrics (virtual_screening only)
    if cfg.run_mode == "virtual_screening":
        with start_action(action_type="enrichment_metrics"):
            metrics = compute_enrichment_metrics(scores_df, cfg.analysis)
            ef_rows = [
                {"fraction": k, "ef": v} for k, v in metrics.get("ef", {}).items()
            ]
            metrics_df = pd.DataFrame(
                [
                    {
                        "metric": "roc_auc",
                        "value": metrics.get("roc_auc"),
                        "fraction": None,
                    },
                    {
                        "metric": "bedroc",
                        "value": metrics.get("bedroc"),
                        "fraction": None,
                    },
                ]
                + [{"metric": "ef", "value": row["ef"], "fraction": row["fraction"]} for row in ef_rows]
            )
            metrics_df.to_csv(analysis_dir / "enrichment_metrics.csv", index=False)

    # 4. Interaction statistics
    with start_action(action_type="interaction_analysis"):
        inter_df = run_interaction_analysis(screening_dir, jobs_csv, cfg)
        inter_df.to_csv(analysis_dir / "interaction_stats.csv", index=False)

    log_message(
        message_type="full_analysis_complete",
        analysis_dir=str(analysis_dir),
    )
