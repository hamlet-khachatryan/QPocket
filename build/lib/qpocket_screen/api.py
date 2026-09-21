"""High-level API for the QPocket screening workflow.

Two entry-points:
- :func:`run_qpocket_screen_prepare` — decoys, protein/ligand prep, box, scripts
- :func:`run_qpocket_screen_analyze` — post-docking analysis
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from eliot import log_message, start_action

from qpocket_screen.config import ScreenConfig, load_screen_config, validate_screen_config


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def run_qpocket_screen_prepare(
    pipeline_workdir: str,
    screen_config_path: Optional[str] = None,
    **overrides,
) -> None:
    """Prepare all inputs for a docking campaign.

    Steps:
    1. Load and validate :class:`~qpocket_screen.config.ScreenConfig`.
    2. Resolve directory layout from *pipeline_workdir*.
    3. Generate ``vina_box.txt``.
    4. If ``virtual_screening``: download ChEMBL data + generate decoys.
    5. Prepare protein PDBQT files (representative structures only).
    6. Prepare ligand PDBQT files (actives; decoys if ``virtual_screening``).
    7. Build ``docking_jobs.csv``.
    8. Write ``run_docking.sh`` and ``run_docking.slurm``.
    9. Log a summary.

    Args:
        pipeline_workdir: Root directory of a finished QPocket pipeline run.
        screen_config_path: Optional path to a ``screen_config.yaml`` file.
        **overrides: Keyword overrides applied on top of the loaded config
            (e.g. ``run_mode="virtual_screening"``).
    """
    with start_action(action_type="qpocket_screen_prepare", workdir=pipeline_workdir):
        # 1. Config
        cfg: ScreenConfig = load_screen_config(screen_config_path, **overrides)
        validate_screen_config(cfg)

        # 2. Resolve dirs
        base = Path(pipeline_workdir).resolve()
        screening_dir = base / "screening"
        metadata_dir = base / "metadata"
        structures_dir = base / "structures"

        screening_dir.mkdir(parents=True, exist_ok=True)

        representatives_csv = metadata_dir / "selected_representatives.csv"
        master_csv = metadata_dir / "master.csv"
        actives_smi = screening_dir / "actives.smi"

        # 3. Vina box
        from qpocket_screen.binding_box import extract_vina_box

        box_dirs = {
            "ligands": structures_dir / "extracted_ligands",
            "cavities": base / "pockets",
            "screening": screening_dir,
        }
        extract_vina_box(
            {"box_mode": cfg.box.mode, "padding": cfg.box.padding, "exhaustiveness": cfg.docking.exhaustiveness},
            box_dirs,
        )

        vina_box_txt = screening_dir / "vina_box.txt"

        # 4. Decoys (virtual_screening only)
        decoys_smi: Optional[Path] = None
        if cfg.run_mode == "virtual_screening":
            from qpocket_screen.generate_decoys import download_chembl_dataset, generate_decoys_lude

            chembl_dir = download_chembl_dataset(cfg.chembl.url, cfg.chembl.local_dir)
            generate_decoys_lude(actives_smi, chembl_dir, screening_dir, cfg.decoy)
            candidate = screening_dir / "decoys.smi"
            decoys_smi = candidate if candidate.exists() else None

        # 5. Prepare proteins
        from qpocket_screen.preparation import prepare_all_proteins

        prepared_pdb_ids = prepare_all_proteins(screening_dir, representatives_csv, master_csv, cfg)
        proteins_dir = screening_dir / "proteins"
        protein_pdbqts = [proteins_dir / f"{pdb_id}.pdbqt" for pdb_id in prepared_pdb_ids]

        # 6. Prepare ligands
        from qpocket_screen.preparation import prepare_all_ligands

        lig_result = prepare_all_ligands(
            screening_dir, actives_smi, decoys_smi, cfg, mode=cfg.run_mode
        )
        actives_dir = screening_dir / "actives"
        decoys_dir = screening_dir / "decoys"

        ligands: dict[str, list[Path]] = {
            "actives": [actives_dir / f"{lid}.pdbqt" for lid in lig_result["actives"]],
        }
        if lig_result["decoys"]:
            ligands["decoys"] = [decoys_dir / f"{lid}.pdbqt" for lid in lig_result["decoys"]]

        # 7. Build docking_jobs.csv
        from qpocket_screen.docking import generate_docking_jobs_csv, parse_vina_box

        box_params = parse_vina_box(vina_box_txt) if vina_box_txt.exists() else {}
        results_dir = screening_dir / "results"
        jobs_df = generate_docking_jobs_csv(
            protein_pdbqts, ligands, box_params, results_dir, cfg.docking
        )
        jobs_csv_path = screening_dir / "docking_jobs.csv"
        jobs_df.to_csv(jobs_csv_path, index=False)

        # 8. Submission scripts
        from qpocket_screen.docking import write_local_script, write_slurm_script

        write_local_script(jobs_csv_path, cfg.docking, screening_dir / "run_docking.sh")
        write_slurm_script(jobs_csv_path, cfg.docking, cfg.slurm, screening_dir / "run_docking.slurm")

        # 9. Summary
        log_message(
            message_type="prepare_complete",
            workdir=str(base),
            run_mode=cfg.run_mode,
            n_jobs=len(jobs_df),
            n_proteins=len(protein_pdbqts),
            n_actives=len(lig_result["actives"]),
            n_decoys=len(lig_result["decoys"]),
        )

        print(
            f"\n[qpocket-screen prepare] Done.\n"
            f"  Mode          : {cfg.run_mode}\n"
            f"  Proteins      : {len(protein_pdbqts)}\n"
            f"  Actives       : {len(lig_result['actives'])}\n"
            f"  Decoys        : {len(lig_result['decoys'])}\n"
            f"  Docking jobs  : {len(jobs_df)}\n"
            f"  Submit locally: bash {screening_dir / 'run_docking.sh'}\n"
            f"  Submit SLURM  : sbatch {screening_dir / 'run_docking.slurm'}\n"
        )


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def run_qpocket_screen_analyze(
    pipeline_workdir: str,
    screen_config_path: Optional[str] = None,
    **overrides,
) -> None:
    """Run post-docking analysis on a completed screening campaign.

    Steps:
    1. Load and validate :class:`~qpocket_screen.config.ScreenConfig`.
    2. Collect docking scores from ``screening/results/``.
    3. RMSD analysis vs. crystal ligands.
    4. Enrichment metrics (``virtual_screening`` mode only).
    5. Interaction statistics via PLIP.
    6. Write CSV reports to ``screening/analysis/``.

    Args:
        pipeline_workdir: Root directory of the pipeline run.
        screen_config_path: Optional path to a ``screen_config.yaml`` file.
        **overrides: Keyword overrides applied on top of the loaded config.
    """
    with start_action(action_type="qpocket_screen_analyze", workdir=pipeline_workdir):
        cfg: ScreenConfig = load_screen_config(screen_config_path, **overrides)
        validate_screen_config(cfg)

        base = Path(pipeline_workdir).resolve()
        screening_dir = base / "screening"

        from qpocket_screen.analysis import run_full_analysis

        run_full_analysis(screening_dir, base, cfg)

        analysis_dir = screening_dir / "analysis"
        print(
            f"\n[qpocket-screen analyze] Done.\n"
            f"  Reports written to: {analysis_dir}\n"
        )
