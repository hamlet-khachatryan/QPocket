"""Finding Q-Pocket runs on disk, including somewhere the notebook did not look."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence

#: Environment variable listing extra places to look, ``:``-separated.
RUNS_ENV = "QPOCKET_RUNS"

#: Directories inside a run, or never containing one. Skipped outright.
PRUNE = frozenset({
    "pockets", "structures", "metadata", "screening", "slurm", "sites",
    "__pycache__", "__marimo__", "node_modules", ".git", ".venv", "venv",
})

#: How deep below a search root a run may sit.
MAX_DEPTH = 4


class Location(NamedTuple):
    """What a typed-in path turned out to be.

    Attributes:
        runs: Runs found there: the path itself when it is a run, or those beneath it when it is a directory of runs.
        message: Why nothing was found, or ``None`` when something was.
    """

    runs: List[Path]
    message: Optional[str]


def is_run(path: Path) -> bool:
    """Whether ``path`` is a Q-Pocket run directory in the current layout."""
    from qpocket.io.layout import LAYOUT_VERSION, read_run_record

    p = Path(path)
    record = read_run_record(p)
    return (record is not None and record.get("layout_version") == LAYOUT_VERSION
            and (p / "metadata" / "master.csv").is_file() and (p / "sites").is_dir())


def is_complete(path: Path) -> bool:
    """Whether a run got as far as its site table."""
    return (Path(path) / "sites" / "sites.csv").is_file()


def site_link_cutoff(run_dir: Path, lo: float, hi: float) -> float:
    """The run's ``sites.link_cutoff``, clamped into a control's range.

    Args:
        run_dir: The run directory.
        lo: Smallest value the control takes.
        hi: Largest value the control takes.

    Returns:
        The cutoff, or the default one when the run's config cannot be read.
    """
    from qpocket.config import QPocketConfig, load_run_config

    try:
        cutoff = float(load_run_config(run_dir).sites.link_cutoff)
    except Exception:  # a broken config must not hide the site table
        cutoff = QPocketConfig().sites.link_cutoff
    return min(max(cutoff, lo), hi)


def find_runs(roots: Iterable[Path], max_depth: int = MAX_DEPTH) -> List[Path]:
    """Every run at or below ``roots``, without walking into runs themselves.

    Args:
        roots: Directories to search.
        max_depth: Levels below each root to descend.

    Returns:
        Resolved run directories, sorted, without duplicates.
    """
    found = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        base_depth = len(root.resolve().parts)
        for current, dirnames, _ in os.walk(root.resolve(), followlinks=False):
            here = Path(current)
            if is_run(here):
                found.add(here)
                dirnames[:] = []  # a run holds no further runs
                continue
            if len(here.parts) - base_depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames if d not in PRUNE and not d.startswith(".")
            ]
    return sorted(found, key=str)


def default_roots(notebook_dir: Optional[Path] = None,
                  extra: Sequence[str] = ()) -> List[Path]:
    """Where to look before the user types anything."""
    base = Path(notebook_dir) if notebook_dir else Path.cwd()
    roots = [base, base.parent / "experiments", Path.cwd() / "experiments"]
    roots += [Path(p) for p in os.environ.get(RUNS_ENV, "").split(os.pathsep) if p]
    roots += [Path(p) for p in extra if p]
    unique, seen = [], set()
    for r in roots:
        key = str(Path(r).expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(Path(key))
    return unique


def resolve_location(text: str, max_depth: int = MAX_DEPTH) -> Location:
    """Turn whatever was typed into the runs it names.

    Args:
        text: A run directory, a directory containing runs, or empty.
        max_depth: Passed to :func:`find_runs`.

    Returns:
        The runs found, or a message saying why there are none.
    """
    raw = (text or "").strip()
    if not raw:
        return Location([], None)

    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.exists():
        return Location([], f"`{path}` does not exist")
    if not path.is_dir():
        return Location([], f"`{path}` is a file; give its run directory")
    if is_run(path):
        return Location([path.resolve()], None)

    runs = find_runs([path], max_depth=max_depth)
    if not runs:
        return Location(
            [], f"no run under `{path}` (looked {max_depth} levels down for "
                f"`metadata/master.csv` beside `sites/`)"
        )
    return Location(runs, None)


def run_label(path: Path) -> str:
    """``parent / name`` for the dropdown, flagged when the run is unfinished."""
    p = Path(path)
    tag = "" if is_complete(p) else "  (incomplete)"
    return f"{p.parent.name} / {p.name}{tag}"
