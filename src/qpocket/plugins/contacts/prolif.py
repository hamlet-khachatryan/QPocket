"""Observed protein-ligand interactions from ProLIF."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from qpocket.core.ports import CONTACTS
from qpocket.io.interactions import interactions_for


@dataclass(frozen=True)
class ProlifParams:
    """What ProLIF looks for.

    Attributes:
        enabled: Interaction names; None for ProLIF's defaults.
        parameters: Per-interaction geometry overrides.
    """

    enabled: Optional[Tuple[str, ...]] = None
    parameters: Optional[Mapping[str, Mapping]] = None


@CONTACTS.register("prolif", params=ProlifParams)
def prolif(complex_path: Path, ligand_id: str, params: ProlifParams) -> List[str]:
    """Interactions between the ligand and its protein.

    Args:
        complex_path: Protonated complex.
        ligand_id: Ligand component ID.
        params: What to look for.

    Returns:
        RESIDUE|InteractionType strings, or a sentinel.
    """
    enabled = None if params.enabled is None else list(params.enabled)
    return interactions_for(complex_path, ligand_id, enabled=enabled, parameters=params.parameters)
