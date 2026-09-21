"""Single-stage entry points, one per SLURM array task.

``qpocket slurm generate`` writes scripts whose body is one ``qpocket task
<stage>`` call. These are those stages. Each is a thin driver over the same
functions the serial pipeline calls, which is deliberate: there is one
implementation of the science and two ways of scheduling it, not two
implementations.

Three properties every task here holds to, because an array job makes all three
load-bearing:

*Idempotent.* A task whose output already exists does nothing and exits 0. A
resubmitted array therefore re-runs only what is missing, and finishes in
seconds when nothing is. ``--force`` overrides.

*Isolated.* A per-structure task writes only inside ``pockets/<pdb_id>/``,
``structures/extracted_ligands/`` and its own JSON sidecar. It never touches
``master.csv``, the HDF5 files or ``actives.smi`` — those belong to ``collect``,
which runs once.

*Loud.* A task that produces no output raises. Several of the underlying
functions return early and log on failure, which is right for a serial run that
carries on to the next structure and wrong for an array task, where a silent
success is indistinguishable from a real one until the next stage fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from eliot import log_message, start_action

from qpocket_pipeline import api, utils
from qpocket_pipeline.config import QPocketConfig
from qpocket_pipeline.distance_logic.spectral_distance import (
    compute_geodesic,
    compute_spectral_distance_matrix,
    pocket_manifest,
)

#: Stages ``run_task`` accepts, in pipeline order.
TASK_NAMES = ("prepare", "apbs", "featurize", "collect", "geodesic", "distance")

#: Stages that take a structure, and so need a manifest and an index.
PER_STRUCTURE = ("apbs", "featurize", "geodesic")


class TaskError(RuntimeError):
    """A task could not produce the output the next stage needs."""


def resolve_pdb_id(manifest: Path, index: int) -> str:
    """The structure an array task should process.

    Args:
        manifest: One PDB ID per line, written by ``qpocket slurm generate``.
        index: ``SLURM_ARRAY_TASK_ID`` plus the script's offset.

    Returns:
        The PDB ID on line ``index``, zero-based.

    Raises:
        TaskError: If the manifest is missing or the index is out of range.
            Both mean the scripts and the manifest have drifted apart, which is
            worth stopping for rather than silently skipping a structure.
    """
    path = Path(manifest)
    if not path.exists():
        raise TaskError(f"manifest not found: {path}")

    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not 0 <= index < len(entries):
        raise TaskError(
            f"index {index} is outside manifest {path.name}, which has "
            f"{len(entries)} entries — regenerate the scripts"
        )
    return entries[index]


def _master_row(dirs: Dict[str, Path], pdb_id: str) -> pd.Series:
    """One structure's row of ``master.csv``.

    Raises:
        TaskError: If ``master.csv`` is absent or has no such structure.
    """
    csv_path = dirs["metadata"] / "master.csv"
    if not csv_path.exists():
        raise TaskError(f"{csv_path} not found — run the prepare stage first")

    df = pd.read_csv(csv_path)
    match = df[df["pdb_id"] == pdb_id]
    if match.empty:
        raise TaskError(f"{pdb_id} is not in master.csv")
    return match.iloc[0]


def task_prepare(cfg: QPocketConfig, dirs: Dict[str, Path], force: bool = False) -> None:
    """Fetch, align and clean every structure, then locate the reference pocket.

    Serial by necessity. PyMOL's ``cmd`` is a module-level singleton holding one
    loaded reference object, so alignment cannot be shared between processes, and
    the reference pocket bounding box constrains every other structure's cavity
    selection.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        force: Rebuild the extraction context even if one is already saved.

    Raises:
        TaskError: If no ``master.csv`` or no reference pocket results.
    """
    with start_action(action_type="task_prepare"):
        api.fetch_structures(cfg, dirs["original"])
        api.align_structures(cfg, dirs)
        api.clean_structures(cfg, dirs)

        csv_path = dirs["metadata"] / "master.csv"
        if not csv_path.exists():
            raise TaskError("prepare produced no master.csv — nothing was aligned")

        context_path = dirs["metadata"] / "extraction_context.json"
        if context_path.exists() and not force:
            log_message(message_type="skip", stage="prepare", reason="context exists")
            return

        context = api.build_extraction_context(cfg, dirs, pd.read_csv(csv_path))
        if context is None:
            raise TaskError(
                "could not locate the reference pocket; every per-structure task "
                "depends on it, so the run cannot continue"
            )
        api.save_extraction_context(dirs, context)


def task_featurize(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    pdb_id: str,
    force: bool = False,
) -> None:
    """Extract, featurise and analyse one structure.

    This is where APBS runs — 5.3s of a 6.8s featurisation, and the reason the
    pipeline is worth distributing at all.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        pdb_id: Structure to process.
        force: Recompute even when the sidecar already exists.

    Raises:
        TaskError: If the task leaves no sidecar behind.
    """
    with start_action(action_type="task_featurize", pdb_id=pdb_id):
        part = api.parts_dir(dirs) / f"{pdb_id}.json"
        if part.exists() and not force:
            log_message(message_type="skip", stage="featurize", pdb_id=pdb_id,
                        reason="sidecar exists")
            return

        context = api.load_extraction_context(dirs)
        row = _master_row(dirs, pdb_id)
        api.process_one_structure(cfg, dirs, row, context)

        if not part.exists():
            raise TaskError(f"featurisation of {pdb_id} wrote no sidecar at {part}")


def task_apbs(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    pdb_id: str,
    force: bool = False,
) -> None:
    """Reserved for the split APBS stage.

    Splitting APBS out needs the potential grid cached per structure, so that
    featurisation can read it back instead of recomputing it. That cache is not
    implemented yet, so this refuses rather than quietly doing the same work
    twice.

    Raises:
        TaskError: Always.
    """
    raise TaskError(
        "slurm.split_apbs_stage is not implemented yet: it requires caching the "
        "APBS .dx grid per structure. Set it to false — APBS still runs as one "
        "task per structure inside the featurize stage."
    )


def task_collect(cfg: QPocketConfig, dirs: Dict[str, Path], force: bool = False) -> None:
    """Merge the per-structure sidecars into the run-wide tables.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        force: Unused; collect always rebuilds from whatever sidecars exist.

    Raises:
        TaskError: If no sidecars were written, which means every featurisation
            task failed and going further would produce empty tables.
    """
    with start_action(action_type="task_collect"):
        parts = list(api.parts_dir(dirs).glob("*.json"))
        if not parts:
            raise TaskError(
                "no per-structure sidecars found — the featurize stage produced "
                "nothing, so there is nothing to collect"
            )
        api.collect_structure_results(cfg, dirs)

        obs_h5 = dirs["metadata"] / "observed_interactions.h5"
        if not obs_h5.exists():
            raise TaskError(f"collect wrote no {obs_h5.name}")


def task_geodesic(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    pdb_id: str,
    force: bool = False,
) -> None:
    """Build one pocket's geodesic graph and cache its distances.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        pdb_id: Pocket to process.
        force: Delete an existing cache and rebuild it.

    Raises:
        TaskError: If the pocket has no ``featurized.csv``.
    """
    with start_action(action_type="task_geodesic", pdb_id=pdb_id):
        pocket_csv = dirs["pockets"] / pdb_id / "featurized.csv"
        if not pocket_csv.exists():
            raise TaskError(
                f"{pocket_csv} not found — {pdb_id} was not featurised successfully"
            )

        cache = pocket_csv.with_name("geospectral_cache.npz")
        if force and cache.exists():
            cache.unlink()

        result = compute_geodesic(
            pocket_csv,
            neighbor_radius=cfg.distance.geo_graph_neighbor_radius,
        )
        # A pocket under five points legitimately has no graph; the distance
        # stage already skips those, so this is not an error.
        if result is None:
            log_message(
                message_type="skip", stage="geodesic", pdb_id=pdb_id,
                reason="no geodesic graph (too few points or no edges)",
            )


def task_distance(cfg: QPocketConfig, dirs: Dict[str, Path], force: bool = False) -> None:
    """Fingerprint every pocket, build the distance matrices and cluster.

    One job rather than an array: measured at 4.2s for 21 structures, with
    ``joblib`` already parallelising the per-pocket work across the cores the
    stage is given.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        force: Unused; the stage always recomputes from the cached geodesics.

    Raises:
        TaskError: If no distance matrix results.
    """
    with start_action(action_type="task_distance"):
        compute_spectral_distance_matrix(cfg, dirs)

        matrix = dirs["metadata"] / "spectral_distance_matrix.csv"
        if not matrix.exists():
            raise TaskError(f"the distance stage wrote no {matrix.name}")


def manifest_ids(cfg: QPocketConfig, dirs: Dict[str, Path], stage: str) -> List[str]:
    """Structures a stage should cover, read from what is already on disk.

    ``geodesic`` works from the pockets that were actually featurised, so a
    structure that failed earlier is simply absent rather than becoming a task
    that is certain to fail.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        stage: Stage name.

    Returns:
        PDB IDs, in ``master.csv`` order.
    """
    csv_path = dirs["metadata"] / "master.csv"
    if stage == "geodesic" and csv_path.exists():
        _, ids = pocket_manifest(cfg, dirs)
        return ids
    if csv_path.exists():
        return [str(v) for v in pd.read_csv(csv_path)["pdb_id"]]
    return []


def run_task(
    name: str,
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    manifest: Optional[Path] = None,
    index: Optional[int] = None,
    force: bool = False,
) -> None:
    """Dispatch one stage, resolving the structure for a per-structure stage.

    Args:
        name: One of :data:`TASK_NAMES`.
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        manifest: Manifest path, required for a per-structure stage.
        index: Row within the manifest, required for a per-structure stage.
        force: Recompute rather than skipping completed work.

    Raises:
        TaskError: On an unknown stage, or a per-structure stage without a
            manifest and index.
    """
    if name not in TASK_NAMES:
        raise TaskError(f"unknown stage {name!r}; expected one of {TASK_NAMES}")

    if name in PER_STRUCTURE:
        if manifest is None or index is None:
            raise TaskError(f"stage {name!r} needs --manifest and --index")
        pdb_id = resolve_pdb_id(manifest, index)
        worker = {"apbs": task_apbs, "featurize": task_featurize, "geodesic": task_geodesic}[name]
        worker(cfg, dirs, pdb_id, force=force)
        return

    {"prepare": task_prepare, "collect": task_collect, "distance": task_distance}[name](
        cfg, dirs, force=force
    )
