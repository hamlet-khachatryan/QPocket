"""Turning the pipeline into SLURM job arrays."""

# The pipeline is one serial process, and almost all of its time is spent per
# structure: protonation, contacts and cavities in the run, and the APBS potential
# when a site is featurised (216s of a 366s 21-structure run). Nothing about those
# loops is sequential — each structure reads its own files and writes its own
# results — so each structure is handed to its own array task.
#
# A run is three stages, and each site is two more, generated per site once the
# run has found its sites (``qpocket slurm generate --site site_01``):
#
# ``prepare``
#     PyMOL's ``cmd`` is a module-level singleton holding one loaded reference
#     object, so alignment cannot share a process, and every structure aligns to
#     the reference.
#
# ``structure``
#     One task per structure, writing its own sidecar.
#
# ``sites``
#     Discovering the sites needs every structure's ligands, and master.csv, the
#     site tables and each site's archives tolerate no concurrent writers.
#
# ``site_featurize``
#     One task per cloud of the site; APBS runs here.
#
# ``site_cluster``
#     The feature scaler is fitted across all of a site's pockets at once and the
#     distance matrix is N x N, so it is given cores rather than tasks.
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
        artifact: Path of the file a finished task leaves behind, relative to the run root, with ``{pdb_id}`` and ``{site}`` placeholders.
        summary: One line describing the stage, written into the script.
    """

    name: str
    order: int
    array: bool
    artifact: str
    summary: str


#: The run as SLURM sees it, in dependency order.
STAGES: Tuple[StageSpec, ...] = (
    StageSpec("prepare", 0, False, "metadata/reference.json",
              "fetch the structures and build the reference"),
    StageSpec("structure", 10, True, "structures/cavities/{pdb_id}.csv",
              "align, clean, protonate, ligands, contacts and cavities, one task per structure"),
    StageSpec("sites", 20, False, "sites/sites.csv",
              "collect the structures, discover the sites, cut every site's clouds and snapshot the config"),
)

#: One binding site's stages, generated with ``--site``, in dependency order.
SITE_STAGES: Tuple[StageSpec, ...] = (
    StageSpec("site_featurize", 30, True, "sites/{site}/pockets/{pdb_id}/featurized.csv",
              "features and APBS potential, one task per cloud of the site"),
    StageSpec("site_cluster", 40, False, "sites/{site}/metadata/spectral_distance_matrix.csv",
              "spectral fingerprints, distance matrices and two-stage clustering of the site"),
)

STAGES_BY_NAME: Dict[str, StageSpec] = {s.name: s for s in STAGES + SITE_STAGES}


def stages_for(site: Optional[str]) -> Tuple[StageSpec, ...]:
    """The stages to generate: the run's, or one site's.

    Args:
        site: The site, such as ``site_01``, or None for the run.

    Returns:
        Stage specs in dependency order.
    """
    return SITE_STAGES if site else STAGES


def _prefix(site: Optional[str]) -> str:
    """File-name prefix keeping a site's scripts, manifests and logs apart from the run's."""
    return f"{site}_" if site else ""


class _Request(NamedTuple):
    cpus: int
    mem: str
    time: str
    gpu: int


def _mem_gb(text: str) -> float:
    """A SLURM memory string in GB; a bare number is MB, as SLURM reads it."""
    value = str(text).strip().upper()
    scale = {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    if value and value[-1] in scale:
        return float(value[:-1]) * scale[value[-1]]
    return float(value) / 1024


def _minutes(text: str) -> int:
    """A SLURM walltime (HH:MM:SS, MM:SS or D-HH:MM:SS) in whole minutes, rounded up."""
    days, _, clock = str(text).rpartition("-")
    parts = [int(p) for p in clock.split(":")]
    hours, minutes, seconds = [0] * (3 - len(parts)) + parts
    if len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    return (int(days or 0) * 24 + hours) * 60 + minutes + (1 if seconds else 0)


def _clock(minutes: int) -> str:
    days, rest = divmod(int(minutes), 24 * 60)
    text = f"{rest // 60:02d}:{rest % 60:02d}:00"
    return f"{days}-{text}" if days else text


def _plugin_resources(cfg, stage_name: str):
    """Resources the plug-ins running inside a stage declare: contacts in structure, features and charges in site_featurize."""
    from qpocket.core.ports import CHARGES, CONTACTS, FEATURES

    if stage_name == "structure":
        plugins = list(CONTACTS.resolve(list(cfg.contacts.use)))
    elif stage_name == "site_featurize":
        plugins = list(FEATURES.resolve(list(cfg.features.enabled))) + [CHARGES.get(cfg.charges.use)]
    else:
        plugins = []
    return [p.resources for p in plugins]


def stage_resources(cfg, stage_name: str):
    """Resource request for a stage: its override, else the defaults raised to the plug-ins' needs.

    Args:
        cfg: A loaded config.
        stage_name: One of the names in :data:`STAGES_BY_NAME`.

    Returns:
        An object carrying ``cpus``, ``mem``, ``time`` and ``gpu``.
    """
    overrides = getattr(cfg.slurm, "stages", None) or {}
    base = overrides[stage_name] if stage_name in overrides else cfg.slurm.defaults
    request = _Request(int(base.cpus), str(base.mem), str(base.time),
                       int(getattr(base, "gpu", 0) or 0))
    needs = [] if stage_name in overrides else _plugin_resources(cfg, stage_name)
    if not needs:
        return request
    mem = max([_mem_gb(request.mem)] + [r.mem_gb for r in needs])
    minutes = max([_minutes(request.time)] + [r.walltime_min for r in needs])
    return _Request(
        cpus=max([request.cpus] + [r.cpus for r in needs]),
        mem=request.mem if mem <= _mem_gb(request.mem) else f"{int(-(-mem // 1))}G",
        time=request.time if minutes <= _minutes(request.time) else _clock(minutes),
        gpu=max([request.gpu] + [r.gpu for r in needs]),
    )


def slurm_dir(dirs: Dict[str, Path]) -> Path:
    """The ``slurm/`` tree inside a run, created if absent."""
    root = Path(dirs["root"]) / "slurm"
    for sub in ("", "manifests", "logs", "scratch"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def manifest_path(dirs: Dict[str, Path], stage: StageSpec, site: Optional[str] = None) -> Path:
    """Where a stage's manifest lives; a site's carries the site id in its name."""
    return slurm_dir(dirs) / "manifests" / f"{_prefix(site)}{stage.name}.txt"


def write_manifest(dirs: Dict[str, Path], stage: StageSpec, pdb_ids: Sequence[str],
                   site: Optional[str] = None) -> Path:
    """Write one PDB ID per line for an array stage to index into.

    Args:
        dirs: Directory map from ``layout.create``.
        stage: The stage this manifest belongs to.
        pdb_ids: Structures to process, in order.
        site: The site, for a per-site stage.

    Returns:
        Path to the manifest.
    """
    path = manifest_path(dirs, stage, site)
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
                n_tasks: Optional[int], site: Optional[str] = None) -> List[str]:
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
        *([f"#SBATCH --gres=gpu:{res.gpu}"] if res.gpu else []),
        # Absolute, so logs land beside the run and not in the submit directory.
        f"#SBATCH --output={log_dir}/{_prefix(site)}{stage.name}_{pattern}.out",
        f"#SBATCH --error={log_dir}/{_prefix(site)}{stage.name}_{pattern}.err",
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
    site: Optional[str] = None,
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
        site: The site a per-site stage works on, passed to the task as ``--site``.

    Returns:
        The script text.

    Raises:
        ValueError: If an array stage is rendered without a manifest.
    """
    if n_tasks is not None and manifest is None:
        raise ValueError(f"array stage {stage.name!r} needs a manifest")

    root = slurm_dir(dirs)
    job_name = f"qp_{_prefix(site)}{stage.name}" + (f"_{chunk}" if chunk is not None else "")

    command = [
        "qpocket", "task", stage.name,
        "--config", str(Path(config_path).resolve()),
        "--workdir", str(Path(dirs["root"]).resolve().parent),
        *(["--site", site] if site else []),
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
        *_directives(cfg, stage, root / "logs", job_name, n_tasks, site),
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
    site: Optional[str] = None,
) -> Dict[str, List[Path]]:
    """Write every stage's sbatch script and manifest.

    Args:
        cfg: A loaded config.
        dirs: Directory map for the run.
        config_path: Config the workers should load.
        pdb_ids: Structures to process in the array stage.
        stages: Restrict to these stage names.
        site: Write this site's stages instead of the run's.

    Returns:
        Stage name to the scripts written for it, in submission order.
    """
    root = slurm_dir(dirs)
    wanted = None if stages is None else set(stages)
    written: Dict[str, List[Path]] = {}

    for stage in stages_for(site):
        if wanted is not None and stage.name not in wanted:
            continue
        stem = f"{_prefix(site)}{stage.order:02d}_{stage.name}"

        if not stage.array:
            path = root / f"{stem}.sbatch"
            path.write_text(render_sbatch(cfg, dirs, stage, config_path, site=site), encoding="utf-8")
            path.chmod(0o755)
            written[stage.name] = [path]
            continue

        manifest = write_manifest(dirs, stage, pdb_ids, site)
        bounds = chunk_bounds(len(pdb_ids), cfg.slurm.max_array_size)
        paths: List[Path] = []
        for index, (offset, size) in enumerate(bounds):
            suffix = "" if len(bounds) == 1 else f"_{index}"
            path = root / f"{stem}{suffix}.sbatch"
            path.write_text(
                render_sbatch(
                    cfg, dirs, stage, config_path,
                    n_tasks=size, offset=offset, manifest=manifest,
                    chunk=None if len(bounds) == 1 else index, site=site,
                )
            , encoding="utf-8")
            path.chmod(0o755)
            paths.append(path)
        written[stage.name] = paths

    log_message(
        message_type="slurm_scripts_written",
        directory=str(root),
        site=site,
        n_structures=len(pdb_ids),
        stages={k: len(v) for k, v in written.items()},
    )
    return written


def write_submit_all(cfg, dirs: Dict[str, Path], scripts: Dict[str, List[Path]],
                     site: Optional[str] = None) -> Path:
    """Write the driver that submits every stage with the right dependencies.

    Args:
        cfg: A loaded config.
        dirs: Directory map for the run.
        scripts: Output of :func:`write_scripts`.
        site: The site the scripts belong to, or None for the run.

    Returns:
        Path to ``submit_all.sh``, or ``submit_<site>.sh`` for a site.
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

    for stage in stages_for(site):
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

    path = root / (f"submit_{site}.sh" if site else "submit_all.sh")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def submit(dirs: Dict[str, Path], scripts: Dict[str, List[Path]],
           dependency: str = "afterany", site: Optional[str] = None) -> Dict[str, List[str]]:
    """Submit the stages with ``sbatch --parsable`` and record the job IDs.

    Args:
        dirs: Directory map for the run.
        scripts: Output of :func:`write_scripts`.
        dependency: ``afterany`` or ``afterok``.
        site: The site the scripts belong to, which names the record.

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

    record = slurm_dir(dirs) / f"{_prefix(site)}{SUBMISSION_RECORD}"
    record.write_text(json.dumps(submitted, indent=2) + "\n", encoding="utf-8")
    return submitted


def stage_progress(cfg, dirs: Dict[str, Path], site: Optional[str] = None) -> List[Dict[str, object]]:
    """How far each stage has got, judged by the files on disk.

    Args:
        cfg: A loaded config.
        dirs: Directory map for the run.
        site: Report this site's stages instead of the run's.

    Returns:
        One record per stage with ``stage``, ``expected``, ``complete`` and ``missing`` (the first few incomplete PDB IDs).
    """
    root = Path(dirs["root"])
    out: List[Dict[str, object]] = []

    for stage in stages_for(site):
        if not stage.array:
            artifact = stage.artifact.format(site=site)
            done = (root / artifact).exists()
            out.append({
                "stage": stage.name, "expected": 1,
                "complete": int(done), "missing": [] if done else [artifact],
            })
            continue

        manifest = manifest_path(dirs, stage, site)
        pdb_ids = read_manifest(manifest) if manifest.exists() else []
        missing = [
            pid for pid in pdb_ids
            if not (root / stage.artifact.format(pdb_id=pid, site=site)).exists()
        ]
        out.append({
            "stage": stage.name,
            "expected": len(pdb_ids),
            "complete": len(pdb_ids) - len(missing),
            "missing": missing[:10],
        })
    return out
