"""Single-stage entry points, one per SLURM array task."""

# ``qpocket slurm generate`` writes scripts whose body is one ``qpocket task
# <stage>`` call. These are those stages. Each is a thin driver over the same
# functions the serial pipeline calls, which is deliberate: there is one
# implementation of the science and two ways of scheduling it, not two
# implementations.
#
# Three properties every task here holds to, because an array job makes all three
# load-bearing:
#
# *Idempotent.* A task whose output already exists does nothing and exits 0. A
# resubmitted array therefore re-runs only what is missing, and finishes in
# seconds when nothing is. ``--force`` overrides.
#
# *Isolated.* A per-structure task writes only inside ``pockets/<pdb_id>/``,
# ``structures/extracted_ligands/`` and its own JSON sidecar. It never touches
# ``master.csv``, the HDF5 files or ``actives.smi`` — those belong to ``collect``,
# which runs once.
#
# *Loud.* A task that produces no output raises. Several of the underlying
# functions return early and log on failure, which is right for a serial run that
# carries on to the next structure and wrong for an array task, where a silent
# success is indistinguishable from a real one until the next stage fails.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from eliot import log_message, start_action

from qpocket.io import layout
from qpocket.pipelines import collect as collecting
from qpocket.pipelines.reference import build_reference, load_reference
from qpocket.pipelines.structure import process, structure_params
from qpocket.config import QPocketConfig
from qpocket.pipelines.distance import (
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
    """
    path = Path(manifest)
    if not path.exists():
        raise TaskError(f"manifest not found: {path}")

    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= index < len(entries):
        raise TaskError(
            f"index {index} is outside manifest {path.name}, which has "
            f"{len(entries)} entries — regenerate the scripts"
        )
    return entries[index]


def task_prepare(cfg: QPocketConfig, dirs: Dict[str, Path], force: bool = False) -> None:
    """Ingest the structures and build the reference every task aligns to.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        force: Rebuild the reference even if one is already saved.

    Raises:
        TaskError: If nothing is ingested or the reference cannot be built.
    """
    with start_action(action_type="task_prepare"):
        if (dirs.metadata / layout.REFERENCE_RECORD).exists() and not force:
            log_message(message_type="skip", stage="prepare", reason="reference exists")
            return
        from qpocket.pipelines.run import ingest

        try:
            build_reference(cfg, dirs, ingest(cfg, dirs).reference)
        except (ValueError, FileNotFoundError) as e:
            raise TaskError(f"prepare failed: {e}; every per-structure task depends on it") from e


def task_featurize(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    pdb_id: str,
    force: bool = False,
) -> None:
    """Extract, featurise and analyse one structure.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        pdb_id: Structure to process.
        force: Recompute even when the sidecar already exists.

    Raises:
        TaskError: If the task leaves no sidecar behind.
    """
    with start_action(action_type="task_featurize", pdb_id=pdb_id):
        part = layout.parts_dir(dirs) / f"{pdb_id}.json"
        if part.exists() and not force:
            log_message(message_type="skip", stage="featurize", pdb_id=pdb_id,
                        reason="sidecar exists")
            return
        try:
            ref = load_reference(dirs)
            raw = layout.raw_structure(dirs, pdb_id)
        except FileNotFoundError as e:
            raise TaskError(str(e)) from e
        process(ref, structure_params(cfg), dirs, raw)
        if not part.exists():
            raise TaskError(f"featurisation of {pdb_id} wrote no sidecar at {part}")


def task_apbs(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    pdb_id: str,
    force: bool = False,
) -> None:
    """Reserved for the split APBS stage.

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
        TaskError: If no sidecars were written, which means every featurisation task failed and going further would produce empty tables.
    """
    with start_action(action_type="task_collect"):
        if not any(layout.parts_dir(dirs).glob("*.json")):
            raise TaskError(
                "no per-structure sidecars found — the featurize stage produced "
                "nothing, so there is nothing to collect"
            )
        ids = [p.stem for p in layout.structure_files(dirs)]
        collecting.collect_from_parts(dirs, ids, cfg.features.enabled, cfg.io.compression)

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

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        stage: Stage name.

    Returns:
        PDB IDs, in input order (``master.csv`` order for geodesic).
    """
    if stage == "featurize":
        return [p.stem for p in layout.structure_files(dirs)]
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
        TaskError: On an unknown stage, or a per-structure stage without a manifest and index.
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
