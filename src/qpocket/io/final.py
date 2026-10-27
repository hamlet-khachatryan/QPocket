"""Reading a site's final metadata (``sites/<id>/final/``), for any arm that works from it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from qpocket.io.tables import file_hash

#: What final/ holds, besides history/.
FINAL_FILES = ("clusters.csv", "settings.json", "residue_profile.csv", "ligands.csv")


def final_status(site_dir: Path) -> str:
    """Whether a site's final metadata exists and still matches its applied clustering.

    Args:
        site_dir: The site folder.

    Returns:
        "missing", "stale" (the clustering changed since), or "current".
    """
    folder = Path(site_dir) / "final"
    if not all((folder / name).exists() for name in FINAL_FILES):
        return "missing"
    applied = Path(site_dir) / "metadata" / "clustering_metadata.csv"
    saved = json.loads((folder / "settings.json").read_text(encoding="utf-8")).get("clustering_hash")
    return "current" if applied.exists() and file_hash(applied) == saved else "stale"


def read_final(site_dir: Path) -> Dict[str, object]:
    """A site's final metadata: ``clusters``, ``residue_profile``, ``ligands`` frames and ``settings``.

    Args:
        site_dir: The site folder.

    Returns:
        The four parts.

    Raises:
        FileNotFoundError: If the site has not been finalised.
    """
    folder = Path(site_dir) / "final"
    if final_status(site_dir) == "missing":
        raise FileNotFoundError(f"no final metadata in {folder}")
    read = {name[:-4]: pd.read_csv(folder / name, keep_default_na=False, dtype={"pdb_id": str})
            for name in FINAL_FILES if name.endswith(".csv")}
    return {**read, "settings": json.loads((folder / "settings.json").read_text(encoding="utf-8"))}
