"""Pocket as a fixed, user-given box."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from eliot import log_message

from qpocket.core.geometry import create_bounding_box_pocket
from qpocket.core.ports import POCKETS
from qpocket.core.types import Reference, Structure


@dataclass(frozen=True)
class BoxParams:
    """The box.

    Attributes:
        center: Box centre (x, y, z).
        dimensions: Box size along x, y and z.
    """

    center: Optional[Tuple[float, ...]] = None
    dimensions: Optional[Tuple[float, ...]] = None


@POCKETS.register("box", params=BoxParams)
def box(record: Structure, ref: Reference, params: BoxParams) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Grid points filling the box.

    Args:
        record: The structure (only its id is used).
        ref: Unused.
        params: The box.

    Returns:
        (points, None), or (None, "FAILED") when the box is not given.
    """
    if not params.center or not params.dimensions:
        log_message(message_type="error", stage="extract_pocket", pdb_id=record.pdb_id,
                    error="bounding_box parameters not provided")
        return None, "FAILED"
    return create_bounding_box_pocket(list(params.center), list(params.dimensions), tolerance=2.0), None
