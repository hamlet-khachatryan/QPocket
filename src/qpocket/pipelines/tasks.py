"""Single-stage entry points, one per SLURM array task."""

# ``qpocket slurm generate`` writes scripts whose body is one ``qpocket task
# <stage>`` call. These are those stages. Each is a thin driver over the same
# functions the serial pipeline and ``qpocket site`` call, which is deliberate:
# there is one implementation of the science and two ways of scheduling it, not
# two implementations.
#
# Three properties every task here holds to, because an array job makes all three
# load-bearing:
#
# *Idempotent.* A task whose output already exists does nothing and exits 0. A
# resubmitted array therefore re-runs only what is missing, and finishes in
# seconds when nothing is. ``--force`` overrides.
#
# *Isolated.* A ``structure`` task writes only its own structure files and JSON
# sidecar; a ``site_featurize`` task writes only inside its own
# ``sites/<site>/pockets/<pdb_id>/``. The run- and site-wide tables belong to
# ``sites`` and ``site_cluster``, which run once.
#
# *Loud.* A task that produces no output raises. Several of the underlying
# functions return early and log on failure, which is right for a serial run that
# carries on to the next structure and wrong for an array task, where a silent
# success is indistinguishable from a real one until the next stage fails.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from eliot import log_message, start_action

from qpocket.config import QPocketConfig, save_run_config
from qpocket.io import layout
from qpocket.pipelines import collect as collecting
from qpocket.pipelines.distance import pocket_manifest
from qpocket.pipelines.reference import build_reference, load_reference
from qpocket.pipelines.sites import extract_all_sites, sync_sites
from qpocket.pipelines.structure import process, structure_params

#: Stages of the run, which ``run_task`` accepts, in pipeline order.
RUN_TASKS = ("prepare", "structure", "sites")

#: Stages of one binding site, run by ``qpocket.arms.sites.run_site_task``: the core run imports no arm.
SITE_TASKS = ("site_featurize", "site_cluster")

#: Every stage ``qpocket task`` accepts.
TASK_NAMES = RUN_TASKS + SITE_TASKS


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


def task_structure(
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    pdb_id: str,
    force: bool = False,
) -> None:
    """Align, clean, protonate and read the ligands, contacts and cavities of one structure.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        pdb_id: Structure to process.
        force: Recompute even when the sidecar already exists.

    Raises:
        TaskError: If the task leaves no sidecar behind.
    """
    with start_action(action_type="task_structure", pdb_id=pdb_id):
        part = layout.parts_dir(dirs) / f"{pdb_id}.json"
        if part.exists() and not force:
            log_message(message_type="skip", stage="structure", pdb_id=pdb_id,
                        reason="sidecar exists")
            return
        try:
            ref = load_reference(dirs)
            raw = layout.raw_structure(dirs, pdb_id)
        except FileNotFoundError as e:
            raise TaskError(str(e)) from e
        process(ref, structure_params(cfg), dirs, raw)
        if not part.exists():
            raise TaskError(f"processing {pdb_id} wrote no sidecar at {part}")


def task_sites(cfg: QPocketConfig, dirs: Dict[str, Path], force: bool = False) -> None:
    """Merge the sidecars, discover the binding sites, cut every site's clouds and snapshot the config.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        force: Unused; the stage always rebuilds from whatever sidecars exist.

    Raises:
        TaskError: If no sidecars were written, which means every structure task failed, or no site table results.
    """
    with start_action(action_type="task_sites"):
        if not any(layout.parts_dir(dirs).glob("*.json")):
            raise TaskError(
                "no per-structure sidecars found — the structure stage produced "
                "nothing, so there is nothing to collect"
            )
        ids = [p.stem for p in layout.structure_files(dirs)]
        records, _ = collecting.collect_from_parts(dirs, ids)
        extract_all_sites(cfg, dirs, sync_sites(cfg, dirs, records)[0], records)
        save_run_config(cfg, dirs)

        table = dirs.sites / "sites.csv"
        if not table.exists():
            raise TaskError(f"the sites stage wrote no {table}")


def manifest_ids(cfg: QPocketConfig, dirs: Dict[str, Path], stage: str,
                 site: Optional[str] = None) -> List[str]:
    """Structures an array stage should cover, read from what is already on disk.

    Args:
        cfg: QPocket configuration.
        dirs: Directory path dictionary of the run.
        stage: ``structure`` or ``site_featurize``.
        site: The site, for ``site_featurize``.

    Returns:
        PDB IDs: every ingested structure, or the site's ``master.csv`` structures that have a cloud there.
    """
    if stage == "structure":
        return [p.stem for p in layout.structure_files(dirs)]
    site_ws = layout.workspace(dirs.sites / str(site))
    if not (site_ws.metadata / "master.csv").exists():
        return []
    return [str(pid) for pid in pocket_manifest(cfg, site_ws, require_features=False)[1]]


def run_task(
    name: str,
    cfg: QPocketConfig,
    dirs: Dict[str, Path],
    manifest: Optional[Path] = None,
    index: Optional[int] = None,
    force: bool = False,
) -> None:
    """Dispatch one run stage, resolving the structure for the per-structure one.

    Args:
        name: One of :data:`RUN_TASKS`.
        cfg: QPocket configuration.
        dirs: Directory path dictionary.
        manifest: Manifest path, required for ``structure``.
        index: Row within the manifest, required for ``structure``.
        force: Recompute rather than skipping completed work.

    Raises:
        TaskError: On an unknown stage, or ``structure`` without a manifest and index.
    """
    if name not in RUN_TASKS:
        raise TaskError(f"unknown stage {name!r}; expected one of {RUN_TASKS}")

    if name == "structure":
        if manifest is None or index is None:
            raise TaskError(f"stage {name!r} needs --manifest and --index")
        task_structure(cfg, dirs, resolve_pdb_id(manifest, index), force=force)
        return

    {"prepare": task_prepare, "sites": task_sites}[name](cfg, dirs, force=force)
