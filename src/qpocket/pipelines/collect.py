"""Collect stage: interaction frequency and utilisation tables for a run."""
from __future__ import annotations

from pathlib import Path

from eliot import log_message

from qpocket.core import interactions
from qpocket.io import hdf5


def calculate_observed_frequencies(h5_path: str) -> None:
    """Write <archive>_freq.csv for an observed-interaction archive.

    Args:
        h5_path: Observed interactions HDF5.
    """
    path_obj = Path(h5_path)
    if not path_obj.exists():
        return
    out_path = path_obj.with_name(f"{path_obj.stem}_freq.csv")
    try:
        result_df = interactions.observed_frequencies(hdf5.read_interactions(path_obj))
        if result_df is None:
            log_message(
                message_type="warning",
                stage="calculate_observed_frequencies",
                warning=f"no valid data in {h5_path}",
            )
            return
        result_df.to_csv(out_path, index=False)
        log_message(message_type="observed_frequencies_saved", path=str(out_path))
    except Exception as e:
        log_message(message_type="error", stage="calculate_observed_frequencies", error=str(e))


def calculate_potential_frequencies(h5_path: str) -> None:
    """Write <archive>_freq.csv for a potential-interaction archive.

    Args:
        h5_path: Potential interactions HDF5.
    """
    path_obj = Path(h5_path)
    if not path_obj.exists():
        log_message(
            message_type="error",
            stage="calculate_potential_frequencies",
            error=f"file not found: {h5_path}",
        )
        return
    out_path = path_obj.with_name(f"{path_obj.stem}_freq.csv")
    try:
        df = interactions.potential_frequencies(hdf5.read_interactions(path_obj))
        if df is None:
            log_message(
                message_type="warning",
                stage="calculate_potential_frequencies",
                warning="no structures found in potential HDF5 file",
            )
            return
        df.to_csv(out_path, index=False)
        log_message(message_type="potential_frequencies_saved", path=str(out_path))
    except Exception as e:
        log_message(message_type="error", stage="calculate_potential_frequencies", error=str(e))


def calculate_binding_site_utilization(
    h5_path_observed: str,
    h5_path_potential: str,
) -> None:
    """Write binding_pocket_utilization.csv beside the observed archive.

    Args:
        h5_path_observed: Observed interactions HDF5.
        h5_path_potential: Potential interactions HDF5.
    """
    path_obs = Path(h5_path_observed)
    path_pot = Path(h5_path_potential)
    if not path_obs.exists() or not path_pot.exists():
        log_message(
            message_type="error",
            stage="calculate_binding_site_utilization",
            error="missing input files for utilization analysis",
        )
        return
    out_path = path_obs.parent / "binding_pocket_utilization.csv"
    try:
        df = interactions.utilization(hdf5.read_interactions(path_obs), hdf5.read_interactions(path_pot))
        if not df.empty:
            df.to_csv(out_path, index=False)
            log_message(message_type="utilization_saved", path=str(out_path))
    except Exception as e:
        log_message(message_type="error", stage="calculate_binding_site_utilization", error=str(e))
