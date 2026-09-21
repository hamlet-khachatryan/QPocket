"""Finding Q-Pocket runs on disk, including somewhere the notebook did not look.

A run is a directory holding ``metadata/master.csv`` and ``pockets/``. The
notebook used to find them with ``glob("**/metadata/master.csv")``, which visits
every file underneath the search root -- including each run's thousands of pocket
and structure files. On a laptop that is instant; on a cluster's parallel
filesystem it can take minutes, and pointing it at a scratch directory with many
runs made the notebook look hung.

So the walk here prunes the directories a run is made of, never descends into a
run once it has found one, and stops at a fixed depth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence

#: Environment variable listing extra places to look, ``:``-separated.
RUNS_ENV = "QPOCKET_RUNS"

#: Directories inside a run, or never containing one. Skipped outright.
PRUNE = frozenset({
    "pockets", "structures", "metadata", "screening", "slurm",
    "__pycache__", "__marimo__", "node_modules", ".git", ".venv", "venv",
})

#: How deep below a search root a run may sit.
MAX_DEPTH = 4


class Location(NamedTuple):
    """What a typed-in path turned out to be.

    Attributes:
        runs: Runs found there: the path itself when it is a run, or those
            beneath it when it is a directory of runs.
        message: Why nothing was found, or ``None`` when something was.
    """

    runs: List[Path]
    message: Optional[str]


def is_run(path: Path) -> bool:
    """Whether ``path`` is a Q-Pocket run directory."""
    p = Path(path)
    return (p / "metadata" / "master.csv").is_file() and (p / "pockets").is_dir()


def is_complete(path: Path) -> bool:
    """Whether a run got as far as the distance matrix.

    Without it there is no landscape and no clustering to show.
    """
    return (Path(path) / "metadata" / "spectral_distance_matrix.csv").is_file()


def find_runs(roots: Iterable[Path], max_depth: int = MAX_DEPTH) -> List[Path]:
    """Every run at or below ``roots``, without walking into runs themselves.

    Args:
        roots: Directories to search. Missing ones are skipped.
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
    """Where to look before the user types anything.

    The notebook's own directory and ``experiments/``, anything in
    ``$QPOCKET_RUNS``, and anything passed as ``extra`` -- the notebook forwards
    ``marimo run ... -- --runs <dir>`` here.
    """
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
        text: A run directory, a directory containing runs, or empty. ``~`` and
            ``$VARIABLES`` are expanded.
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
                f"`metadata/master.csv` beside `pockets/`)"
        )
    return Location(runs, None)


def run_label(path: Path) -> str:
    """``parent / name`` for the dropdown, flagged when the run is unfinished."""
    p = Path(path)
    tag = "" if is_complete(p) else "  (incomplete)"
    return f"{p.parent.name} / {p.name}{tag}"
