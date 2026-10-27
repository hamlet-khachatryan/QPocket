"""Submitting a cross-docking: pair counts, a background run on this machine, and commands for a VM over SSH."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from qpocket.io import final as io_final
from qpocket.io.engines import resolve_executable
from qpocket.screen_config import RemoteConfig, ScreenConfig

LOCAL_RUN = "local_run.json"


def pair_count(run_dir: Path, site_id: str) -> Tuple[int, int, int]:
    """How many dockings a cross-docking of a finalised site makes.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        (representatives, distinct compounds, pairs).

    Raises:
        FileNotFoundError: If the site has not been finalised.
    """
    final = io_final.read_final(Path(run_dir) / "sites" / site_id)
    clusters, ligands = final["clusters"], final["ligands"]
    reps = int((clusters["is_representative"].astype(str).str.lower() == "true").sum())
    compounds = int(ligands.loc[ligands["smiles"].astype(str).str.strip() != "", "smiles"].nunique())
    return reps, compounds, reps * compounds


def engines(cfg: ScreenConfig) -> Dict[str, Optional[str]]:
    """Every docking engine Q-Pocket knows, with its executable, or None when it is not installed.

    Args:
        cfg: Screening config (for custom engines).

    Returns:
        Engine name to executable path or None.
    """
    from qpocket.core.ports import DOCKING

    found: Dict[str, Optional[str]] = {}
    for name in DOCKING.available():
        try:
            found[name] = str(resolve_executable(DOCKING.get(name).fn().executable))
        except Exception:  # not installed: shown greyed, never chosen
            found[name] = None
    for name, spec in (cfg.docking.custom_engines or {}).items():
        command = str(dict(spec).get("executable") or str(dict(spec).get("command", "")).split(" ")[0])
        found[name] = shutil.which(command) if command else None
    return found


def _alive(pid: int) -> bool:
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)  # reaps our own finished child, which kill(0) would call alive
        return done == 0
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _record(folder: Path) -> Optional[dict]:
    path = Path(folder) / LOCAL_RUN
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def start_local(folder: Path, workers: int, cpus: int) -> dict:
    """Start every job of a task folder on this machine, in the background.

    Args:
        folder: The cross-docking task folder.
        workers: Jobs in parallel; 0 uses half the cores.
        cpus: Threads per job.

    Returns:
        What was started: pid, started_at, n_jobs, command (also written to local_run.json).

    Raises:
        RuntimeError: If a run started here is still going.
    """
    folder = Path(folder)
    previous = _record(folder)
    if previous and _alive(int(previous["pid"])):
        raise RuntimeError(f"a run is already running here (pid {previous['pid']})")
    qpocket = Path(sys.executable).with_name("qpocket")
    argv = [str(qpocket) if qpocket.exists() else "qpocket", "screen", "run-local", "docking_jobs.csv",
            "--workers", str(int(workers)), "--cpus", str(int(cpus))]
    with open(folder / "local_run.log", "a", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=str(folder), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    record = {"pid": process.pid, "started_at": datetime.now().replace(microsecond=0).isoformat(),
              "n_jobs": len(pd.read_csv(folder / "docking_jobs.csv")), "command": " ".join(argv)}
    (folder / LOCAL_RUN).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def local_progress(folder: Path) -> Dict[str, object]:
    """How far a cross-docking of this task folder has got, counted from its pose files.

    Args:
        folder: The task folder.

    Returns:
        done, total, running (a run started here is alive), started_at (None if none was started here).
    """
    folder = Path(folder)
    jobs = pd.read_csv(folder / "docking_jobs.csv")
    done = sum(((folder / p) if not Path(p).is_absolute() else Path(p)).exists() for p in jobs["output_pdbqt"])
    record = _record(folder)
    return {"done": int(done), "total": len(jobs), "running": bool(record and _alive(int(record["pid"]))),
            "started_at": record.get("started_at") if record else None}


def _session(site_id: str) -> str:
    return "qpocket_" + re.sub(r"[^A-Za-z0-9_-]", "_", site_id)


_SAFE = re.compile(r"^[A-Za-z0-9._/@+=,-]+$")
SSH_OPTIONS = ["-o", "BatchMode=yes"]   # never wait on a password or host-key prompt


def _remote_dir(remote: RemoteConfig, site_id: str) -> Tuple[str, str]:
    """(path for the remote shell, path for rsync) of the site's remote task folder.

    rsync takes a home-relative path for ``~/…`` (it starts in the remote home); the shell gets ``~/`` unquoted
    so it expands it. Only characters that need no quoting are accepted, so both read the same folder.
    """
    directory = remote.directory.rstrip("/") or "."
    home = directory == "~" or directory.startswith("~/")
    rest = directory[2:] if directory.startswith("~/") else ("" if directory == "~" else directory)
    path = f"{rest}/{site_id}" if rest else site_id
    for part in (path, site_id):
        if not _SAFE.match(part):
            raise ValueError(f"remote folder {path!r}: use letters, digits and . _ / @ + = , - only "
                             "(rsync and the remote shell must read it the same way)")
    return (f"~/{path}" if home else path), path


def remote_commands(folder: Path, site_id: str, remote: RemoteConfig, workers: int,
                    cpus: int) -> Dict[str, List[str]]:
    """The commands that run a task folder on a remote machine: push, start, poll, pull.

    Args:
        folder: The local task folder.
        site_id: The site, naming the remote folder and the tmux session.
        remote: Where to run it; ``directory`` may start with ``~/``.
        workers: Jobs in parallel there; 0 uses half its cores.
        cpus: Threads per job.

    Returns:
        Key to an argv list for subprocess.run. Push replaces the remote task (and its old poses); a failed
        setup line stops the run; the run's output goes to run_local.log there.

    Raises:
        ValueError: If the remote folder or site id holds characters rsync and the shell would read differently.
    """
    shell_dir, rsync_dir = _remote_dir(remote, site_id)
    inner = " && ".join([*remote.setup, f"cd {shell_dir}",
                         f"qpocket screen run-local docking_jobs.csv --workers {int(workers)} --cpus {int(cpus)}"
                         " > run_local.log 2>&1"])
    rsync = ["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTIONS)]
    host = str(remote.host)
    return {
        "push": [*rsync, "--delete", "--delete-excluded", "--exclude", "poses/", "--exclude", "poses_previous/",
                 "--exclude", LOCAL_RUN, "--exclude", "local_run.log",
                 f"--rsync-path=mkdir -p {rsync_dir} && rsync", f"{folder}/", f"{host}:{rsync_dir}/"],
        "start": ["ssh", *SSH_OPTIONS, host, f"tmux new -d -s {_session(site_id)} {shlex.quote(inner)}"],
        "poll": ["ssh", *SSH_OPTIONS, host,
                 f"ls {shell_dir}/poses 2>/dev/null | wc -l; tmux ls 2>/dev/null || true"],
        "pull": [*rsync, f"{host}:{rsync_dir}/poses/", f"{folder}/poses/"],
    }


def screen_config_path(run_dir: Path, site_id: str) -> Optional[Path]:
    """The screening config the page and preparation use: the site's, else the run's, else None (defaults).

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        ``sites/<id>/screening/screen_config.yaml``, else ``<run>/screen_config.yaml``, when present.
    """
    for path in (Path(run_dir) / "sites" / site_id / "screening" / "screen_config.yaml",
                 Path(run_dir) / "screen_config.yaml"):
        if path.is_file():
            return path
    return None


def remote_status(poll_output: str, site_id: str) -> Dict[str, object]:
    """Read what the poll command printed.

    Args:
        poll_output: Its stdout: the pose count, then ``tmux ls`` lines.
        site_id: The site.

    Returns:
        poses, own_session (our tmux session is running), other_jobs (every other tmux session).
    """
    lines = [line.strip() for line in poll_output.splitlines() if line.strip()]
    poses = int(lines[0]) if lines and lines[0].isdigit() else 0
    sessions = [line.split(":", 1)[0] for line in lines[1:] if ":" in line]
    own = _session(site_id)
    return {"poses": poses, "own_session": own in sessions, "other_jobs": [s for s in sessions if s != own]}


def vm_power_state(remote: RemoteConfig) -> Optional[str]:
    """The Azure VM's power state, e.g. "VM running", or None when it cannot be asked; never changes the VM."""
    if not (remote.azure_vm and remote.azure_resource_group and shutil.which("az")):
        return None
    try:
        out = subprocess.run(["az", "vm", "get-instance-view", "-n", remote.azure_vm, "-g", remote.azure_resource_group,
                              "--query", "instanceView.statuses[?starts_with(code,'PowerState/')].displayStatus",
                              "-o", "tsv"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None
