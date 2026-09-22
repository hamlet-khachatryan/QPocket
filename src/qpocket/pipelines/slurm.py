"""Turning the pipeline into SLURM job arrays."""

# The pipeline is one serial process, and profiling says almost all of it is one
# call: on a 21-structure run, ``calculate_apbs_potential`` accounts for 216s of a
# 366s wall clock, invoked once per structure inside a ``for`` loop. Nothing about
# that loop is sequential — each structure reads its own cleaned PDB and writes its
# own pocket directory — so the fix is to hand each structure to its own array task.
#
# Three stages split that way: featurisation (optionally with APBS broken out into
# a stage of its own), and the geodesic graph. The rest cannot:
#
# ``prepare``
#     PyMOL's ``cmd`` is a module-level singleton holding one loaded reference
#     object, so alignment cannot share a process. The reference pocket bounding
#     box is also a hard dependency of every other structure's cavity selection.
#
# ``collect``
#     The HDF5 files, ``master.csv`` and the frequency tables each need every
#     structure's result, and none of them tolerates concurrent writers.
#
# ``distance``
#     The feature scaler is fitted across all pockets at once, and the distance
#     matrix is N x N. Measured at 4.2s for 21 structures, so it is given cores
#     rather than tasks.
#
# What this module writes is plain sbatch text — no scheduler library, no daemon.
# You can read every script before spending an hour of cluster time on it, which is
# the point.

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from eliot import log_message, start_action

#: Name of the file recording job IDs from the last submission.
SUBMISSION_RECORD = "submitted.json"


class StageSpec(NamedTuple):
    """One step of the pipeline, and how it maps onto SLURM.

    Attributes:
        name: Stage name, used for the config key, manifest and log prefix.
        order: Two-digit prefix that sorts the generated scripts.
        array: Whether this stage runs one task per structure.
        artifact: Path of the file a finished task leaves behind, relative to the run root.
        summary: One line describing the stage, written into the script.
    """

    name: str
    order: int
    array: bool
    artifact: str
    summary: str


#: The pipeline as SLURM sees it, in dependency order.
STAGES: Tuple[StageSpec, ...] = (
    StageSpec("prepare", 0, False, "metadata/master.csv",
              "fetch, align, clean, and locate the reference pocket"),
    StageSpec("apbs", 10, True, "pockets/{pdb_id}/apbs/mol.dx",
              "Poisson-Boltzmann potential grid, one task per structure"),
    StageSpec("featurize", 20, True, "pockets/{pdb_id}/featurized.csv",
              "cavity detection, featurisation and PLIP, one task per structure"),
    StageSpec("collect", 30, False, "metadata/observed_interactions.h5",
              "merge per-structure results into master.csv, HDF5 and frequencies"),
    StageSpec("geodesic", 40, True, "pockets/{pdb_id}/geospectral_cache.npz",
              "geodesic graph and cached distances, one task per pocket"),
    StageSpec("distance", 50, False, "metadata/spectral_distance_matrix.csv",
              "spectral fingerprints, distance matrices and clustering"),
)

STAGES_BY_NAME: Dict[str, StageSpec] = {s.name: s for s in STAGES}


def active_stages(cfg) -> Tuple[StageSpec, ...]:
    """The stages to generate, honouring ``slurm.split_apbs_stage``.

    Args:
        cfg: A loaded :class:`~qpocket.config.QPocketConfig`.

    Returns:
        Stage specs in dependency order.
    """
    split = bool(getattr(cfg.slurm, "split_apbs_stage", False))
    return tuple(s for s in STAGES if split or s.name != "apbs")


def stage_resources(cfg, stage_name: str):
    """Resource request for a stage, falling back to ``slurm.defaults``.

    Args:
        cfg: A loaded config.
        stage_name: One of the names in :data:`STAGES`.

    Returns:
        An object carrying ``cpus``, ``mem`` and ``time``.
    """
    overrides = getattr(cfg.slurm, "stages", None) or {}
    # DictConfig and dict both support __contains__ / __getitem__.
    if stage_name in overrides:
        return overrides[stage_name]
    return cfg.slurm.defaults


def slurm_dir(dirs: Dict[str, Path]) -> Path:
    """The ``slurm/`` tree inside a run, created if absent."""
    root = Path(dirs["root"]) / "slurm"
    for sub in ("", "manifests", "logs", "scratch"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_manifest(dirs: Dict[str, Path], stage: StageSpec, pdb_ids: Sequence[str]) -> Path:
    """Write one PDB ID per line for an array stage to index into.

    Args:
        dirs: Directory map from ``setup_directory_hierarchy``.
        stage: The stage this manifest belongs to.
        pdb_ids: Structures to process, in order.

    Returns:
        Path to the manifest.
    """
    path = slurm_dir(dirs) / "manifests" / f"{stage.name}.txt"
    path.write_text("".join(f"{pid}\n" for pid in pdb_ids), encoding="utf-8")
    return path


def read_manifest(path: Path) -> List[str]:
    """Read a manifest back into a list of PDB IDs, ignoring blank lines."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_bounds(n_tasks: int, max_array_size: int) -> List[Tuple[int, int]]:
    """Split ``n_tasks`` into ``(offset, size)`` chunks under the array cap.

    Args:
        n_tasks: Total tasks wanted.
        max_array_size: Largest array the cluster accepts.

    Returns:
        One ``(offset, size)`` per script.
    """
    if n_tasks <= 0:
        return []
    size = max(1, int(max_array_size))
    return [(off, min(size, n_tasks - off)) for off in range(0, n_tasks, size)]


def _directives(cfg, stage: StageSpec, log_dir: Path, job_name: str,
                n_tasks: Optional[int]) -> List[str]:
    """The ``#SBATCH`` block for one script."""
    res = stage_resources(cfg, stage.name)
    sl = cfg.slurm
    pattern = "%A_%a" if n_tasks is not None else "%j"

    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={sl.partition}",
        f"#SBATCH --cpus-per-task={res.cpus}",
        f"#SBATCH --mem={res.mem}",
        f"#SBATCH --time={res.time}",
        # Absolute, so logs land beside the run and not in the submit directory.
        f"#SBATCH --output={log_dir}/{stage.name}_{pattern}.out",
        f"#SBATCH --error={log_dir}/{stage.name}_{pattern}.err",
    ]
    if sl.account:
        lines.insert(1, f"#SBATCH --account={sl.account}")
    if sl.qos:
        lines.insert(1, f"#SBATCH --qos={sl.qos}")
    if n_tasks is not None:
        lines.append(f"#SBATCH --array=0-{n_tasks - 1}%{sl.max_concurrent}")
    return lines


def render_sbatch(
    cfg,
    dirs: Dict[str, Path],
    stage: StageSpec,
    config_path: Path,
    n_tasks: Optional[int] = None,
    offset: int = 0,
    manifest: Optional[Path] = None,
    chunk: Optional[int] = None,
) -> str:
    """Render one sbatch script.

    Args:
        cfg: A loaded config.
        dirs: Directory map for the run.
        stage: Stage to render.
        config_path: Config file the worker should load, made absolute.
        n_tasks: Tasks in this array, or ``None`` for a single job.
        offset: Added to ``SLURM_ARRAY_TASK_ID`` to index the manifest, for a stage split across several scripts.
        manifest: Manifest path, required for an array stage.
        chunk: Chunk number, appended to the job name when a stage is split.

    Returns:
        The script text.

    Raises:
        ValueError: If an array stage is rendered without a manifest.
    """
    if n_tasks is not None and manifest is None:
        raise ValueError(f"array stage {stage.name!r} needs a manifest")

    root = slurm_dir(dirs)
    job_name = f"qp_{stage.name}" + (f"_{chunk}" if chunk is not None else "")

    command = [
        "qpocket", "task", stage.name,
        "--config", str(Path(config_path).resolve()),
        "--workdir", str(Path(dirs["root"]).resolve().parent),
    ]
    if n_tasks is not None:
        command += [
            "--manifest", str(manifest),
            "--index", "$(( SLURM_ARRAY_TASK_ID + %d ))" % offset,
        ]

    # shlex.quote every literal, then re-insert the one shell expansion we want.
    rendered = " ".join(
        part if part.startswith("$((") else shlex.quote(part) for part in command
    )

    env_lines = list(getattr(cfg.slurm, "environment", None) or [])
    env_block = (
        "# Environment, verbatim from slurm.environment in the config.\n"
        + "\n".join(env_lines)
        if env_lines
        else "# No slurm.environment lines configured. If qpocket is not already on\n"
             "# PATH on the compute nodes, add them to the config and regenerate."
    )

    return "\n".join([
        "#!/usr/bin/env bash",
        *_directives(cfg, stage, root / "logs", job_name, n_tasks),
        # --chdir keeps stray writes (APBS drops an io.mc beside the process) out
        # of the directory you submitted from.
        f"#SBATCH --chdir={root / 'scratch'}",
        "",
        f"# Q-Pocket stage '{stage.name}': {stage.summary}",
        "# Generated by `qpocket slurm generate` — regenerate rather than editing.",
        "",
        "set -euo pipefail",
        "",
        env_block,
        "",
        rendered,
        "",
    ])


def write_scripts(
    cfg,
    dirs: Dict[str, Path],
    config_path: Path,
    pdb_ids: Sequence[str],
    stages: Optional[Sequence[str]] = None,
) -> Dict[str, List[Path]]:
    """Write every stage's sbatch script and manifest.

    Args:
        cfg: A loaded config.
        dirs: Directory map for the run.
        config_path: Config the workers should load.
        pdb_ids: Structures to process in the array stages.
        stages: Restrict to these stage names.

    Returns:
        Stage name to the scripts written for it, in submission order.
    """
    root = slurm_dir(dirs)
    wanted = None if stages is None else set(stages)
    written: Dict[str, List[Path]] = {}

    for stage in active_stages(cfg):
        if wanted is not None and stage.name not in wanted:
            continue

        if not stage.array:
            path = root / f"{stage.order:02d}_{stage.name}.sbatch"
            path.write_text(render_sbatch(cfg, dirs, stage, config_path), encoding="utf-8")
            path.chmod(0o755)
            written[stage.name] = [path]
            continue

        manifest = write_manifest(dirs, stage, pdb_ids)
        bounds = chunk_bounds(len(pdb_ids), cfg.slurm.max_array_size)
        paths: List[Path] = []
        for index, (offset, size) in enumerate(bounds):
            suffix = "" if len(bounds) == 1 else f"_{index}"
            path = root / f"{stage.order:02d}_{stage.name}{suffix}.sbatch"
            path.write_text(
                render_sbatch(
                    cfg, dirs, stage, config_path,
                    n_tasks=size, offset=offset, manifest=manifest,
                    chunk=None if len(bounds) == 1 else index,
                )
            , encoding="utf-8")
            path.chmod(0o755)
            paths.append(path)
        written[stage.name] = paths

    log_message(
        message_type="slurm_scripts_written",
        directory=str(root),
        n_structures=len(pdb_ids),
        stages={k: len(v) for k, v in written.items()},
    )
    return written


def write_submit_all(cfg, dirs: Dict[str, Path], scripts: Dict[str, List[Path]]) -> Path:
    """Write the driver that submits every stage with the right dependencies.

    Args:
        cfg: A loaded config.
        dirs: Directory map for the run.
        scripts: Output of :func:`write_scripts`.

    Returns:
        Path to ``submit_all.sh``.
    """
    root = slurm_dir(dirs)
    dependency = cfg.slurm.dependency

    lines = [
        "#!/usr/bin/env bash",
        "# Submit the Q-Pocket pipeline as a dependency-chained set of SLURM jobs.",
        "# Generated by `qpocket slurm generate` — regenerate rather than editing.",
        "set -euo pipefail",
        "",
        'PREV=""',
        "",
    ]

    for stage in active_stages(cfg):
        paths = scripts.get(stage.name) or []
        if not paths:
            lines += [f"# stage '{stage.name}': nothing to submit", ""]
            continue

        lines.append(f"# --- {stage.name}: {stage.summary}")
        lines.append('IDS=""')
        for path in paths:
            lines.append(
                'ID=$(sbatch --parsable ${PREV:+--dependency='
                + dependency
                + '$PREV} '
                + shlex.quote(str(path))
                + ")"
            )
            lines.append(f'echo "  {stage.name}: submitted $ID"')
            lines.append('IDS="$IDS:$ID"')
        lines += ['PREV="$IDS"', ""]

    lines += ['echo "All stages submitted."', ""]

    path = root / "submit_all.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def submit(dirs: Dict[str, Path], scripts: Dict[str, List[Path]],
           dependency: str = "afterany") -> Dict[str, List[str]]:
    """Submit the stages with ``sbatch --parsable`` and record the job IDs.

    Args:
        dirs: Directory map for the run.
        scripts: Output of :func:`write_scripts`.
        dependency: ``afterany`` or ``afterok``.

    Returns:
        Stage name to the job IDs submitted for it.

    Raises:
        RuntimeError: If ``sbatch`` is unavailable or rejects a script.
    """
    submitted: Dict[str, List[str]] = {}
    previous: List[str] = []

    with start_action(action_type="slurm_submit", dependency=dependency):
        for stage_name, paths in scripts.items():
            ids: List[str] = []
            for path in paths:
                command = ["sbatch", "--parsable"]
                if previous:
                    command.append(f"--dependency={dependency}:{':'.join(previous)}")
                command.append(str(path))
                try:
                    done = subprocess.run(
                        command, capture_output=True, text=True, check=True
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        "sbatch not found — generate the scripts on the login node, "
                        "or run without --submit and use submit_all.sh"
                    ) from exc
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"sbatch rejected {path.name}: {exc.stderr.strip()}"
                    ) from exc
                job_id = done.stdout.strip().split(";")[0]
                ids.append(job_id)
                log_message(
                    message_type="slurm_job_submitted",
                    stage=stage_name, script=path.name, job_id=job_id,
                )
            submitted[stage_name] = ids
            if ids:
                previous = ids

    record = slurm_dir(dirs) / SUBMISSION_RECORD
    record.write_text(json.dumps(submitted, indent=2) + "\n", encoding="utf-8")
    return submitted


def stage_progress(cfg, dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    """How far each stage has got, judged by the files on disk.

    Args:
        cfg: A loaded config.
        dirs: Directory map for the run.

    Returns:
        One record per stage with ``stage``, ``expected``, ``complete`` and ``missing`` (the first few incomplete PDB IDs).
    """
    root = Path(dirs["root"])
    manifest_dir = slurm_dir(dirs) / "manifests"
    out: List[Dict[str, object]] = []

    for stage in active_stages(cfg):
        if not stage.array:
            done = (root / stage.artifact).exists()
            out.append({
                "stage": stage.name, "expected": 1,
                "complete": int(done), "missing": [] if done else [stage.artifact],
            })
            continue

        manifest = manifest_dir / f"{stage.name}.txt"
        pdb_ids = read_manifest(manifest) if manifest.exists() else []
        missing = [
            pid for pid in pdb_ids
            if not (root / stage.artifact.format(pdb_id=pid)).exists()
        ]
        out.append({
            "stage": stage.name,
            "expected": len(pdb_ids),
            "complete": len(pdb_ids) - len(missing),
            "missing": missing[:10],
        })
    return out
