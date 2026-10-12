"""Pocket from detected cavities: the one at the reference ligand, or overlapping its box."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from eliot import log_message

from qpocket.core.pocket import select_cavity_by_bbox, select_cavity_by_centroid
from qpocket.core.ports import POCKETS
from qpocket.core.types import Reference, Structure
from qpocket.io.tools import run_cavfind


def _best_by_box(cavities, lower: np.ndarray, upper: np.ndarray, pdb_id: str) -> Optional[np.ndarray]:
    best, overlap = select_cavity_by_bbox(cavities, lower, upper)
    if best is None or overlap == 0:
        log_message(message_type="warning", stage="cavity_selection", pdb_id=pdb_id,
                    warning="no cavity overlaps reference bounding box")
        return None
    log_message(message_type="cavity_selected", pdb_id=pdb_id, volumetric_overlap=overlap,
                n_points=len(best))
    return best


@POCKETS.register("cavities")
def cavities(record: Structure, ref: Reference, params) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """The structure's pocket among its detected cavities.

    Args:
        record: The structure; its cleaned file is searched.
        ref: The reference: its ligand centroid, or its pocket box for other structures.
        params: None needed.

    Returns:
        (points, None), or (None, FAILED / NO_ANY_CAVITY_FOUND / CAVITY_CLOSED).
    """
    try:
        found = run_cavfind(Path(record.cleaned_file), record.pdb_id)
    except Exception as e:
        log_message(message_type="warning", stage="extract_pocket", pdb_id=record.pdb_id,
                    warning=f"cavity detection failed: {e}")
        return None, "FAILED"
    if not found:
        return None, "NO_ANY_CAVITY_FOUND"
    is_reference = record.pdb_id == ref.pdb_id
    if not is_reference and ref.box is not None:
        points = _best_by_box(found, ref.box[0], ref.box[1], record.pdb_id)
    else:
        points = select_cavity_by_centroid(found, ref.centroid if is_reference else None)
    return (points, None) if points is not None else (None, "CAVITY_CLOSED")
