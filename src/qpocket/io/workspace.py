"""Run directory layout."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def setup_directory_hierarchy(
    workdir: Optional[str],
    ref_pdb_id: str,
) -> Dict[str, Path]:
    """Creates the directory hierarchy for QPocket pipeline execution.

    Args:
        workdir: Working directory path, defaults to current directory if None.
        ref_pdb_id: Reference PDB identifier for creating project subdirectory.

    Returns:
        Dictionary mapping directory names to their Path objects.
    """
    base_root = Path(workdir) if workdir is not None else Path.cwd()
    project_path = base_root / ref_pdb_id

    dirs = {
        "root": project_path,
        "structures": project_path / "structures",
        "original": project_path / "structures" / "original",
        "aligned": project_path / "structures" / "aligned",
        "cleaned": project_path / "structures" / "cleaned",
        "extracted_ligands": project_path / "structures" / "extracted_ligands",
        "pockets": project_path / "pockets",
        "metadata": project_path / "metadata",
        "screening": project_path / "screening",
        "decoys": project_path / "screening" / "decoys",
        "actives": project_path / "screening" / "actives",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs
