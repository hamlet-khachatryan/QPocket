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


#: Structures accepted as input.
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".ent")

REFERENCE_RECORD = "reference.json"


def structure_files(ws: Workspace) -> list:
    """Every ingested structure file, sorted by path.

    Args:
        ws: The run.

    Returns:
        Paths; this order decides master.csv order.
    """
    return sorted(p for sfx in STRUCTURE_SUFFIXES for p in ws.original.glob(f"*{sfx}"))


def raw_structure(ws: Workspace, pdb_id: str):
    """The ingested file of one structure.

    Args:
        ws: The run.
        pdb_id: Its identifier.

    Returns:
        Its RawStructure.

    Raises:
        FileNotFoundError: If nothing was ingested under that name.
    """
    from qpocket.core.types import RawStructure

    path = next((p for p in structure_files(ws) if p.stem == pdb_id), None)
    if path is None:
        raise FileNotFoundError(f"no ingested structure named {pdb_id!r} in {ws.original}")
    return RawStructure(pdb_id, path)


def parts_dir(ws: Workspace) -> Path:
    """Directory of the per-structure sidecars, created if absent."""
    path = ws.metadata / "parts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_part(ws: Workspace, pdb_id: str, payload: dict) -> Path:
    """Write one structure's sidecar.

    Args:
        ws: The run.
        pdb_id: The structure.
        payload: Its JSON form.

    Returns:
        The sidecar path.
    """
    path = parts_dir(ws) / f"{pdb_id}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_part(ws: Workspace, pdb_id: str) -> Optional[dict]:
    """One structure's sidecar, or None when it has none.

    Raises:
        ValueError: If the sidecar is not valid JSON.
    """
    path = ws.metadata / "parts" / f"{pdb_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_reference(ws: Workspace, payload: dict) -> Path:
    """Persist the reference, so array tasks can reuse it."""
    path = ws.metadata / REFERENCE_RECORD
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_reference(ws: Workspace) -> dict:
    """The persisted reference.

    Raises:
        FileNotFoundError: If the prepare stage has not run.
    """
    path = ws.metadata / REFERENCE_RECORD
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — run the prepare stage before per-structure tasks")
    return json.loads(path.read_text(encoding="utf-8"))
