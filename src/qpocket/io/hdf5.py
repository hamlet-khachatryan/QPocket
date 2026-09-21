"""Interaction archives: one HDF5 dataset of ASCII entries per structure."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
from eliot import start_action

from qpocket.core.interactions import Profile, profile


def save_interactions(
    data_dict: Dict[str, List[str]],
    output_path: Path,
    compression: str = "gzip",
) -> None:
    """Add or replace structures in an interaction archive.

    Args:
        data_dict: Structure ID to entries (stored sorted).
        output_path: HDF5 file, created if missing.
        compression: "gzip" or "none".
    """
    compress_kwargs: Dict[str, Any] = (
        {"compression": "gzip", "compression_opts": 4}
        if compression == "gzip"
        else {}
    )
    with start_action(
        action_type="save_hdf5",
        path=str(output_path),
        compression=compression,
        n_entries=len(data_dict),
    ):
        with h5py.File(output_path, "a") as hf:
            for k, v in data_dict.items():
                residue_list = sorted(v)
                ascii_residues = [n.encode("ascii", "ignore") for n in residue_list]
                if k in hf:
                    del hf[k]
                hf.create_dataset(k, data=ascii_residues, **compress_kwargs)


def read_interactions(
    h5_path: Path, pdb_ids: Optional[Iterable[str]] = None
) -> Dict[str, List[str]]:
    """Decode an interaction archive.

    Args:
        h5_path: HDF5 file.
        pdb_ids: Only these structures; unknown IDs are ignored.

    Returns:
        Structure ID to entries, in sorted ID order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"interaction file not found: {path}")

    wanted = None if pdb_ids is None else set(pdb_ids)
    out: Dict[str, List[str]] = {}
    with h5py.File(path, "r") as hf:
        for pid in sorted(hf.keys()):
            if wanted is not None and pid not in wanted:
                continue
            out[pid] = [r.decode("ascii") for r in hf[pid][:]]
    return out


def build_profile(h5_path: Path, pdb_ids: Optional[Iterable[str]] = None) -> Profile:
    """Engagement profile of an interaction archive.

    Args:
        h5_path: HDF5 file.
        pdb_ids: Only these structures.

    Returns:
        The profile.
    """
    return profile(read_interactions(h5_path, pdb_ids))
