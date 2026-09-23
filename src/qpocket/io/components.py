"""Chemical-component definitions from the PDB, bundled and cached on disk."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Optional

ComponentBonds = Dict[FrozenSet[str], str]

#: Definitions shipped with the package, so an apo run needs no network.
BUNDLED = Path(__file__).resolve().parent.parent / "data" / "ccd"

#: Bond orders the CCD uses; any other value is skipped.
ORDERS = ("SING", "DOUB", "TRIP")

URL = "https://files.rcsb.org/ligands/download/{comp_id}.cif"


class MissingComponentError(Exception):
    """A chemical component is neither bundled, cached, nor reachable."""

    def __init__(self, comp_id: str):
        """Record which component is missing.

        Args:
            comp_id: The chemical-component ID.
        """
        super().__init__(
            f"no definition for chemical component {comp_id!r}: not bundled, not "
            f"cached, and not downloadable from {URL.format(comp_id=comp_id)}"
        )
        self.comp_id = comp_id


def cache_dir() -> Path:
    """Directory holding downloaded component definitions.

    Returns:
        The path named by QPOCKET_CCD_CACHE, or a per-user cache directory.
    """
    override = os.environ.get("QPOCKET_CCD_CACHE")
    return Path(override) if override else Path.home() / ".cache" / "qpocket" / "ccd"


def parse_component_cif(text: str, comp_id: str) -> ComponentBonds:
    """Read one component's bond table from its CCD entry.

    Args:
        text: Contents of the component CIF.
        comp_id: The component whose rows to keep.

    Returns:
        Bond order keyed by the pair of atom names.
    """
    bonds: ComponentBonds = {}
    for line in text.splitlines():
        parts = line.split()
        # comp_id atom_id_1 atom_id_2 value_order aromatic_flag stereo ordinal
        if len(parts) >= 6 and parts[0] == comp_id and parts[3] in ORDERS:
            bonds[frozenset((parts[1].strip('"'), parts[2].strip('"')))] = parts[3]
    return bonds


def _download(comp_id: str, path: Path) -> bool:
    """Fetch one component definition.

    Args:
        comp_id: The chemical-component ID.
        path: Destination file.

    Returns:
        True when the file was written.
    """
    try:
        with urllib.request.urlopen(URL.format(comp_id=comp_id), timeout=30) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return True


@lru_cache(maxsize=512)
def component_bonds(comp_id: str,
                    cache_dir_override: Optional[Path] = None) -> ComponentBonds:
    """Bond table for one chemical component.

    Args:
        comp_id: The chemical-component ID, e.g. WDG.
        cache_dir_override: Directory for downloads, used by tests.

    Returns:
        Bond order keyed by the pair of atom names.

    Raises:
        MissingComponentError: If no definition can be obtained.
    """
    comp_id = comp_id.strip().upper()
    bundled = BUNDLED / f"{comp_id}.cif"
    if bundled.exists():
        return parse_component_cif(bundled.read_text(encoding="utf-8"), comp_id)

    directory = Path(cache_dir_override) if cache_dir_override else cache_dir()
    cached = directory / f"{comp_id}.cif"
    if not cached.exists() and not _download(comp_id, cached):
        raise MissingComponentError(comp_id)
    return parse_component_cif(cached.read_text(encoding="utf-8"), comp_id)
