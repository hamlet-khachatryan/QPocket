"""Local structure files around one reference (fragment screens)."""
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from eliot import log_message

from qpocket.config import sanitise_id
from qpocket.core.ports import SOURCES
from qpocket.core.types import RawStructure, Sources
from qpocket.io import layout
from qpocket.io.layout import STRUCTURE_SUFFIXES


@dataclass(frozen=True)
class LocalParams:
    """Which files to ingest.

    Attributes:
        input_type: "paths" or "csv".
        csv_path: CSV with a pdb_path column, for input_type "csv".
        structure_paths: Files, for input_type "paths".
        reference_path: Reference file; need not be among the others.
        reference_index: Fallback reference, by ingest position.
    """

    input_type: str
    csv_path: Optional[str]
    structure_paths: Tuple[str, ...]
    reference_path: Optional[str]
    reference_index: int


def input_paths(params: LocalParams) -> List[Path]:
    """Files named by the parameters, reference first, without duplicates.

    Args:
        params: The source's parameters.

    Returns:
        Absolute paths in ingest order.
    """
    listed: List[str] = []
    if params.input_type == "csv":
        if not params.csv_path:
            log_message(message_type="error", stage="load_fragment_screen", error="csv_path not provided")
            return []
        csv_p = Path(params.csv_path)
        if not csv_p.exists():
            log_message(message_type="error", stage="load_fragment_screen",
                        error="CSV file not found", path=str(csv_p))
            return []
        df = pd.read_csv(csv_p)
        if "pdb_path" not in df.columns:
            log_message(message_type="error", stage="load_fragment_screen",
                        error="CSV missing 'pdb_path' column")
            return []
        listed = [str(v) for v in df["pdb_path"].tolist()]
    elif params.input_type == "paths":
        if not params.structure_paths:
            log_message(message_type="error", stage="load_fragment_screen",
                        error="structure_paths not provided")
            return []
        listed = [str(v) for v in params.structure_paths]

    ordered: List[Path] = []
    seen: set = set()
    if params.reference_path:
        resolved = Path(params.reference_path).expanduser().resolve()
        ordered.append(resolved)
        seen.add(resolved)
    for raw in listed:
        resolved = Path(raw).expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def copy_inputs(params: LocalParams, ws: layout.Workspace) -> List[Dict[str, object]]:
    """Copy the files into the run and record where each came from.

    Args:
        params: The source's parameters.
        ws: The run.

    Returns:
        One record per ingested file (pdb_id, source_path, is_reference), in ingest order.
    """
    ref_path = Path(params.reference_path).expanduser().resolve() if params.reference_path else None
    records: List[Dict[str, object]] = []
    used: Dict[str, int] = {}
    for src in input_paths(params):
        if not src.exists():
            log_message(message_type="warning", stage="load_fragment_screen",
                        warning=f"structure not found: {src}")
            continue
        if src.suffix.lower() not in STRUCTURE_SUFFIXES:
            log_message(message_type="warning", stage="load_fragment_screen",
                        warning=f"unrecognised structure format, skipping: {src.name}")
            continue
        # Two directories can hold the same filename; disambiguate rather than overwrite.
        base = sanitise_id(src.stem)
        count = used.get(base, 0)
        used[base] = count + 1
        struct_id = base if count == 0 else f"{base}_{count + 1}"
        shutil.copy2(src, ws.original / f"{struct_id}{src.suffix.lower()}")
        records.append({"pdb_id": struct_id, "source_path": str(src),
                        "is_reference": bool(ref_path is not None and src == ref_path)})
        log_message(message_type="fragment_loaded", source=src.name, id=struct_id)

    if records and not any(r["is_reference"] for r in records):
        # No explicit reference file: the configured index picks one.
        records[min(int(params.reference_index), len(records) - 1)]["is_reference"] = True
    if records:
        ws.metadata.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(ws.metadata / "inputs.csv", index=False)
    log_message(message_type="fragment_screen_loaded", n_structures=len(records),
                reference=next((r["pdb_id"] for r in records if r["is_reference"]), None))
    return records


@SOURCES.register("local", params=LocalParams)
def local(params: LocalParams, ws: layout.Workspace) -> Sources:
    """Ingest local files.

    Args:
        params: Which files.
        ws: The run.

    Returns:
        The ingested structures.

    Raises:
        ValueError: If nothing could be ingested.
    """
    records = copy_inputs(params, ws)
    if not records:
        raise ValueError("fragment_screen ingested no structures — check "
                         "fragment_screen.structure_paths / csv_path and reference_path")
    reference_id = next(str(r["pdb_id"]) for r in records if r["is_reference"])
    structures = tuple(RawStructure(p.stem, p) for p in layout.structure_files(ws))
    return Sources(next(s for s in structures if s.pdb_id == reference_id), structures)
