"""Where everything in a run lives; the one module that spells run paths."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple, Optional, Union

#: Bumped whenever a run's files move; older runs must be re-run.
LAYOUT_VERSION = 2

RUN_RECORD = "run.json"


class Workspace(NamedTuple):
    """Every directory of one run.

    Attributes:
        root: The run directory.
        structures: Parent of the structure folders.
        original: Input structures as ingested.
        aligned: Structures superposed onto the reference.
        cleaned: Aligned structures stripped to protein and allowed cofactors.
        extracted_ligands: Ligands cut out of each structure.
        pockets: One folder per structure with its pocket files.
        metadata: Run-wide tables.
        screening: Screening inputs and outputs.
        decoys: Decoy sets.
        actives: Active sets.
    """

    root: Path
    structures: Path
    original: Path
    aligned: Path
    cleaned: Path
    extracted_ligands: Path
    pockets: Path
    metadata: Path
    screening: Path
    decoys: Path
    actives: Path

    def __getitem__(self, key):
        # shortcut: dict-style access keeps stages.py's dirs["x"] working; remove with stages.py (plan 2).
        return getattr(self, key) if isinstance(key, str) else tuple.__getitem__(self, key)


def workspace(root: Union[str, Path]) -> Workspace:
    """Paths of a run, without touching the disk.

    Args:
        root: The run directory.

    Returns:
        Its Workspace.
    """
    root = Path(root)
    structures = root / "structures"
    screening = root / "screening"
    return Workspace(root, structures, structures / "original", structures / "aligned",
                     structures / "cleaned", structures / "extracted_ligands", root / "pockets",
                     root / "metadata", screening, screening / "decoys", screening / "actives")


def _qpocket_version() -> str:
    try:
        return version("QPocket")
    except PackageNotFoundError:
        return "unknown"


def create(workdir: Optional[str], run_id: str) -> Workspace:
    """Make a run's directories and record its layout version.

    Args:
        workdir: Parent directory; the current one when None.
        run_id: Name of the run directory.

    Returns:
        The run's Workspace.
    """
    ws = workspace((Path(workdir) if workdir is not None else Path.cwd()) / run_id)
    for path in ws:
        path.mkdir(parents=True, exist_ok=True)
    record = {"layout_version": LAYOUT_VERSION, "qpocket_version": _qpocket_version()}
    (ws.metadata / RUN_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return ws


def read_run_record(root: Union[str, Path]) -> Optional[dict]:
    """The run record, or None when the run has none.

    Args:
        root: The run directory.

    Returns:
        The parsed run.json, or None.
    """
    path = workspace(root).metadata / RUN_RECORD
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
