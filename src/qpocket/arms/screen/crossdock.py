"""Cross-docking: every distinct active into every cluster representative."""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from eliot import log_message, start_action

from qpocket.arms.screen.campaign import ScreeningPlan
from qpocket.arms.screen.ligands import distinct_actives, prepare_ligands
from qpocket.arms.screen.receptors import prepare_receptor
from qpocket.core.crossdock import crossdock_jobs
from qpocket.core.screening import calculate_box
from qpocket.io import layout
from qpocket.screen_config import ScreenConfig, load_screen_config, validate_screen_config

CROSSDOCK_DIR = "crossdock"


def _representatives(ws) -> List[str]:
    reps = pd.read_csv(ws.metadata / "selected_representatives.csv")
    column = "pdb_id" if "pdb_id" in reps.columns else reps.columns[0]
    return [str(v) for v in reps[column]]


def _pocket(ws, rep_id: str) -> np.ndarray:
    return pd.read_csv(ws.pockets / rep_id / "binding_site.csv")[["x", "y", "z"]].to_numpy(float)


def shared_box(ws, rep_ids: List[str], mode: str, padding: float) -> Dict[str, float]:
    """The one search box every docking uses.

    Args:
        ws: The run.
        rep_ids: Representatives.
        mode: "representative_pockets" or "interacting_residues".
        padding: Added on every side, Å.

    Returns:
        calculate_box output (center/size/min/max per axis).

    Raises:
        ValueError: If no coordinates are found for the mode.
    """
    if mode == "representative_pockets":
        points = [_pocket(ws, r) for r in rep_ids if (ws.pockets / r / "binding_site.csv").exists()]
    else:
        from biopandas.pdb import PandasPdb

        from qpocket.core.pocket import interacting_residue_coords
        from qpocket.io.hdf5 import read_interactions

        observed = read_interactions(ws.metadata / "observed_interactions.h5", rep_ids)
        aligned = pd.read_csv(ws.metadata / "master.csv").set_index("pdb_id")["aligned_file"].to_dict()
        points = []
        for rep in rep_ids:
            entries = [e for e in observed.get(rep, []) if "|" in e]
            if entries and Path(str(aligned.get(rep))).exists():
                xyz = interacting_residue_coords(PandasPdb().read_pdb(str(aligned[rep])).df["ATOM"], entries)
                if xyz is not None:
                    points.append(xyz)
    if not points:
        raise ValueError(f"no coordinates for box mode {mode!r} among {rep_ids}")
    return calculate_box(np.vstack(points), padding=padding)


def _write_runners(folder: Path, n_jobs: int, cfg: ScreenConfig) -> str:
    cpus = int(cfg.docking.cpus)
    (folder / "tasks.txt").write_text("".join(
        f"qpocket screen run-task docking_jobs.csv --index {i} --cpus {cpus}\n" for i in range(n_jobs)),
        encoding="utf-8")
    sl = cfg.slurm
    script = "\n".join([
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=qpocket_crossdock",
        f"#SBATCH --partition={sl.partition}",
        f"#SBATCH --cpus-per-task={sl.cpus_per_task}",
        f"#SBATCH --mem={sl.mem}",
        f"#SBATCH --time={sl.time}",
        f"#SBATCH --array=0-{max(n_jobs - 1, 0)}%{sl.max_concurrent}",
        "#SBATCH --output=slurm_logs/dock_%A_%a.out",
        "#SBATCH --error=slurm_logs/dock_%A_%a.err",
        "",
        "# Submit from inside this folder: sbatch run_docking.sbatch",
        "set -euo pipefail",
        'cd "${SLURM_SUBMIT_DIR}"',
        *sl.environment,
        'qpocket screen run-task docking_jobs.csv --index "${SLURM_ARRAY_TASK_ID}" --cpus "${SLURM_CPUS_PER_TASK}"',
        "",
    ])
    (folder / "slurm_logs").mkdir(exist_ok=True)
    (folder / "run_docking.sbatch").write_text(script, encoding="utf-8")
    return (f"cd {shlex.quote(str(folder))} && sbatch run_docking.sbatch   # cluster\n"
            f"cd {shlex.quote(str(folder))} && qpocket screen run-local docking_jobs.csv --workers 0   # this machine")


def prepare(run_dir, config_path: Optional[str] = None, **overrides) -> ScreeningPlan:
    """Build the cross-docking task folder for a finished run.

    Args:
        run_dir: Run root.
        config_path: screen_config.yaml, or None for the defaults.
        **overrides: Config overrides.

    Returns:
        The plan: task list, scripts, and how to run them.
    """
    from qpocket.arms.screen.campaign import engine_for

    cfg = load_screen_config(config_path, **overrides)
    validate_screen_config(cfg)
    ws = layout.workspace(Path(run_dir).resolve())
    folder = ws.screening / CROSSDOCK_DIR
    with start_action(action_type="crossdock_prepare", run=str(ws.root)):
        rep_ids = _representatives(ws)
        cleaned = pd.read_csv(ws.metadata / "master.csv").set_index("pdb_id")["cleaned_file"].to_dict()
        receptors = {}
        for rep in rep_ids:
            path = prepare_receptor(Path(cleaned[rep]), _pocket(ws, rep), folder / "receptors", rep, cfg)
            if path is not None:
                receptors[rep] = path.relative_to(folder).as_posix()
        actives = distinct_actives(ws.screening / "actives.smi", ws.metadata / "master.csv")
        actives.to_csv(folder / "ligands.csv", index=False)
        ligands = {a: p.relative_to(folder).as_posix()
                   for a, p in prepare_ligands(actives, folder / "ligands", cfg).items()}
        box = shared_box(ws, list(receptors), cfg.crossdock.box_mode, cfg.crossdock.box_padding)
        (folder / "box.json").write_text(json.dumps({"mode": cfg.crossdock.box_mode,
                                                      "padding": cfg.crossdock.box_padding,
                                                      **{k: float(v) for k, v in box.items()}}, indent=2),
                                         encoding="utf-8")
        (folder / "engines.json").write_text(json.dumps(dict(cfg.docking.custom_engines or {}), indent=2),
                                             encoding="utf-8")
        engine = engine_for({"engine": cfg.docking.engine}, folder)
        jobs = crossdock_jobs(receptors, ligands, box, cfg.docking.engine, engine.output_ext,
                              cfg.docking.exhaustiveness, cfg.docking.num_modes, cfg.docking.energy_range)
        (folder / "poses").mkdir(parents=True, exist_ok=True)
        jobs_csv = folder / "docking_jobs.csv"
        jobs.to_csv(jobs_csv, index=False)
        command = _write_runners(folder, len(jobs), cfg)
        log_message(message_type="crossdock_prepared", receptors=len(receptors), actives=len(ligands),
                    jobs=len(jobs))
        return ScreeningPlan(jobs_csv, [folder / "run_docking.sbatch"], command)


class CrossdockResult(NamedTuple):
    """What analyze produced.

    Attributes:
        poses: Every pose with score and RMSD.
        pairs: One row per docking.
        success: Rates per representative, per active and overall.
        folder: The task folder.
    """

    poses: pd.DataFrame
    pairs: pd.DataFrame
    success: pd.DataFrame
    folder: Path


def _engine(name: str, folder: Optional[Path] = None):
    from qpocket.arms.screen.campaign import engine_for

    return engine_for({"engine": name}, folder)


def pose_table(folder: Path, jobs: pd.DataFrame, actives: pd.DataFrame) -> pd.DataFrame:
    """Every docked pose, scored against the closest crystal copy of its molecule.

    Args:
        folder: The task folder.
        jobs: docking_jobs.csv.
        actives: ligands.csv.

    Returns:
        job_id, representative, active_id, rank, score, rmsd.
    """
    from rdkit import Chem

    from qpocket.core.crossdock import pose_rmsd

    by_id = actives.set_index("active_id")
    crystals: Dict[str, list] = {}
    for active_id, row in by_id.iterrows():
        files = [f for f in str(row["crystal_files"] if pd.notna(row["crystal_files"]) else "").split(";") if f]
        crystals[active_id] = [m for m in (Chem.MolFromPDBFile(f, sanitize=False, removeHs=True) for f in files)
                               if m is not None]
    rows = []
    for job in jobs.itertuples():
        out = Path(job.output_pdbqt)
        out = out if out.is_absolute() else Path(folder) / out
        if not out.exists():
            continue
        smiles = by_id.loc[job.active_id, "canonical"] if job.active_id in by_id.index else None
        for pose in _engine(str(job.engine or "vina"), folder).read_poses(out):
            values = [pose_rmsd(pose.mol, c, smiles) for c in crystals.get(job.active_id, [])]
            values = [v for v in values if v is not None]
            rows.append({"job_id": job.job_id, "representative": job.representative,
                         "active_id": job.active_id, "rank": pose.rank, "score": pose.score,
                         "rmsd": min(values) if values else float("nan")})
    return pd.DataFrame(rows, columns=["job_id", "representative", "active_id", "rank", "score", "rmsd"])


def analyze(run_dir, config_path: Optional[str] = None, **overrides) -> CrossdockResult:
    """Score a finished cross-docking campaign.

    Args:
        run_dir: Run root.
        config_path: screen_config.yaml, or None for the defaults.
        **overrides: Config overrides.

    Returns:
        Poses, pairs and success rates; also written as CSVs in the task folder.
    """
    from qpocket.core.crossdock import pair_table, success_table

    cfg = load_screen_config(config_path, **overrides)
    folder = layout.workspace(Path(run_dir).resolve()).screening / CROSSDOCK_DIR
    with start_action(action_type="crossdock_analyze", folder=str(folder)):
        jobs = pd.read_csv(folder / "docking_jobs.csv", dtype={"engine": str})
        actives = pd.read_csv(folder / "ligands.csv", keep_default_na=False)
        poses = pose_table(folder, jobs, actives)
        pairs = pair_table(poses, cfg.analysis.rmsd_threshold, cfg.analysis.top_k)
        success = success_table(pairs)
        poses.to_csv(folder / "crossdock_poses.csv", index=False)
        pairs.to_csv(folder / "crossdock_pairs.csv", index=False)
        success.to_csv(folder / "crossdock_success.csv", index=False)
        log_message(message_type="crossdock_analysed", jobs=len(jobs), docked=len(pairs),
                    overall=success.iloc[-1].to_dict() if len(success) else {})
        return CrossdockResult(poses, pairs, success, folder)
