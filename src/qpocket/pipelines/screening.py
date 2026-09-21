"""Virtual screening: prepare a docking campaign, run its jobs, analyse the results."""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional

import pandas as pd
from eliot import log_message, start_action

from qpocket.core import screening as scr
from qpocket.io import screening as sio_screen
from qpocket.pipelines.slurm import chunk_bounds
from qpocket.screen_config import ScreenConfig, load_screen_config, validate_screen_config

Submit = Literal["slurm", "local"]


class ScreeningPlan(NamedTuple):
    """What prepare produced.

    Attributes:
        jobs_csv: The task list.
        scripts: SLURM scripts (one per array chunk), or the local shell script.
        command: What to run next.
    """

    jobs_csv: Path
    scripts: List[Path]
    command: str


def write_submission(jobs_csv: Path, n_jobs: int, cfg: ScreenConfig, submit: Submit) -> ScreeningPlan:
    """Write the submission files for a task list.

    Args:
        jobs_csv: docking_jobs.csv.
        n_jobs: Rows in it.
        cfg: Screening config.
        submit: "slurm" for array scripts, "local" for a local runner.

    Returns:
        The plan.
    """
    screening_dir = jobs_csv.parent
    if submit == "slurm":
        log_dir = screening_dir / "slurm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        chunks = chunk_bounds(n_jobs, cfg.slurm.max_array_size)
        scripts = []
        for i, (offset, size) in enumerate(chunks):
            path = screening_dir / (f"run_docking_{i}.sbatch" if len(chunks) > 1 else "run_docking.sbatch")
            text = scr.slurm_array_script(jobs_csv, cfg.slurm, offset, size, log_dir,
                                          chunk=i if len(chunks) > 1 else None,
                                          environment=cfg.slurm.environment)
            path.write_text(text, encoding="utf-8")
            scripts.append(path)
        command = "\n".join(f"sbatch {s}" for s in scripts)
    else:
        command = scr.local_command(jobs_csv, cfg.docking.cpus)
        script = screening_dir / "run_docking.sh"
        script.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n", encoding="utf-8")
        script.chmod(0o755)
        scripts = [script]
    return ScreeningPlan(jobs_csv, scripts, command)


def prepare(pipeline_workdir, config_path: Optional[str] = None, submit: Optional[Submit] = None,
            **overrides) -> ScreeningPlan:
    """Prepare every input of a docking campaign from a finished pipeline run.

    Args:
        pipeline_workdir: Run root (holds metadata/, structures/, pockets/).
        config_path: screen_config.yaml, or None for defaults.
        submit: "slurm" or "local"; defaults to slurm when slurm.enabled.
        **overrides: Config overrides.

    Returns:
        The task list, submission files and next command.
    """
    with start_action(action_type="qpocket_screen_prepare", workdir=str(pipeline_workdir)):
        cfg: ScreenConfig = load_screen_config(config_path, **overrides)
        validate_screen_config(cfg)
        submit = submit or ("slurm" if cfg.slurm.enabled else "local")

        base = Path(pipeline_workdir).resolve()
        screening_dir = base / "screening"
        metadata_dir = base / "metadata"
        screening_dir.mkdir(parents=True, exist_ok=True)
        actives_smi = screening_dir / "actives.smi"

        # 1. Box. Known issue (spec §6): its glob of cavity files never matches.
        sio_screen.extract_vina_box(
            {"box_mode": cfg.box.mode, "padding": cfg.box.padding, "exhaustiveness": cfg.docking.exhaustiveness},
            {"ligands": base / "structures" / "extracted_ligands", "cavities": base / "pockets",
             "screening": screening_dir},
        )
        vina_box_txt = screening_dir / "vina_box.txt"

        # 2. Decoys, virtual screening only.
        decoys_smi: Optional[Path] = None
        if cfg.run_mode == "virtual_screening":
            from qpocket.pipelines.decoys import generate_decoys_lude

            chembl_dir = sio_screen.download_chembl_dataset(cfg.chembl.url, cfg.chembl.local_dir)
            generate_decoys_lude(actives_smi, chembl_dir, screening_dir, cfg.decoy)
            candidate = screening_dir / "decoys.smi"
            decoys_smi = candidate if candidate.exists() else None

        # 3. Receptors and ligands.
        from qpocket.pipelines.screening_prep import prepare_all_ligands, prepare_all_proteins

        prepared = prepare_all_proteins(screening_dir, metadata_dir / "selected_representatives.csv",
                                        metadata_dir / "master.csv", cfg)
        proteins = [screening_dir / "proteins" / f"{pdb_id}.pdbqt" for pdb_id in prepared]
        lig_result = prepare_all_ligands(screening_dir, actives_smi, decoys_smi, cfg, mode=cfg.run_mode)
        ligands = {"actives": [screening_dir / "actives" / f"{lid}.pdbqt" for lid in lig_result["actives"]]}
        if lig_result["decoys"]:
            ligands["decoys"] = [screening_dir / "decoys" / f"{lid}.pdbqt" for lid in lig_result["decoys"]]

        # 4. Task list.
        results_dir = screening_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        box = sio_screen.parse_vina_box(vina_box_txt) if vina_box_txt.exists() else {}
        jobs = scr.docking_jobs(proteins, ligands, box, results_dir, cfg.docking)
        jobs_csv = screening_dir / "docking_jobs.csv"
        jobs.to_csv(jobs_csv, index=False)
        log_message(message_type="docking_jobs_created", n_jobs=len(jobs))

        # 5. Submission.
        plan = write_submission(jobs_csv, len(jobs), cfg, submit)
        log_message(message_type="prepare_complete", workdir=str(base), run_mode=cfg.run_mode,
                    n_jobs=len(jobs), n_proteins=len(proteins), n_actives=len(lig_result["actives"]),
                    n_decoys=len(lig_result["decoys"]), submit=submit)
        return plan


def _read_jobs(jobs_csv: Path) -> pd.DataFrame:
    return pd.read_csv(jobs_csv, dtype=str, keep_default_na=False)


def run_job(jobs_csv: Path, index: int, cpus: int, engine: Optional[str] = None) -> int:
    """Run one row of the task list.

    Args:
        jobs_csv: docking_jobs.csv.
        index: 0-based row.
        cpus: Threads for the docking engine.
        engine: Docking executable; defaults to the screening config's engine_path.

    Returns:
        The engine's exit code.
    """
    job = _read_jobs(jobs_csv).iloc[index]
    Path(job["output_pdbqt"]).parent.mkdir(parents=True, exist_ok=True)
    argv = scr.vina_argv(job, engine or load_screen_config().docking.engine_path, cpus)
    return subprocess.run(argv).returncode


def run_local(jobs_csv: Path, workers: int = 1, cpus: int = 1, engine: Optional[str] = None) -> int:
    """Run every job on this machine, a few at a time.

    Args:
        jobs_csv: docking_jobs.csv.
        workers: Jobs in parallel.
        cpus: Threads per job.
        engine: Docking executable; defaults to the screening config's engine_path.

    Returns:
        Number of failed jobs.
    """
    jobs = _read_jobs(jobs_csv)
    engine = engine or load_screen_config().docking.engine_path

    def one(i: int) -> int:
        job = jobs.iloc[i]
        Path(job["output_pdbqt"]).parent.mkdir(parents=True, exist_ok=True)
        return subprocess.run(scr.vina_argv(job, engine, cpus), capture_output=True, text=True).returncode

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        codes = list(pool.map(one, range(len(jobs))))
    failed = sum(1 for c in codes if c != 0)
    log_message(message_type="local_docking_done", n_jobs=len(codes), n_failed=failed)
    return failed


def analyze(pipeline_workdir, config_path: Optional[str] = None, **overrides) -> Path:
    """Post-docking analysis: scores, pose RMSD, enrichment, interactions.

    Args:
        pipeline_workdir: Run root.
        config_path: screen_config.yaml, or None for defaults.
        **overrides: Config overrides.

    Returns:
        The analysis directory.
    """
    with start_action(action_type="qpocket_screen_analyze", workdir=str(pipeline_workdir)):
        cfg: ScreenConfig = load_screen_config(config_path, **overrides)
        validate_screen_config(cfg)
        base = Path(pipeline_workdir).resolve()
        from qpocket.pipelines.screening_analysis import run_full_analysis

        run_full_analysis(base / "screening", base, cfg)
        return base / "screening" / "analysis"
