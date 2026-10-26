"""Collect stage: the run's structure tables, and each site's interaction tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import pandas as pd
from eliot import log_message, start_action

from qpocket.core import interactions
from qpocket.core.result import Failure
from qpocket.core.types import Structure
from qpocket.io import hdf5, layout


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


def write_feature_record(meta_dir: Path, enabled: Sequence[str]) -> Path:
    """Record the feature set a run used, for readers of its pockets.

    Args:
        meta_dir: The run's metadata directory.
        enabled: Feature plug-in names, in config order.

    Returns:
        Path of features.json.
    """
    from qpocket.core.features import columns_for
    from qpocket.core.ports import FEATURES

    payload = {"enabled": list(enabled),
               "columns": list(columns_for(FEATURES.resolve(list(enabled))))}
    path = Path(meta_dir) / "features.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def collect_structures(ws: "layout.Workspace", records: Sequence[Any], failures: Sequence[Failure]) -> None:
    """Write master.csv and failures.csv, the run-level tables.

    Args:
        ws: The run or the site.
        records: Records with master_row(), in input order.
        failures: Structures that dropped out.
    """
    master_csv = ws.metadata / "master.csv"
    pd.DataFrame([r.master_row() for r in records]).to_csv(master_csv, index=False)
    log_message(message_type="master_csv_saved", path=str(master_csv), n_structures=len(records))
    pd.DataFrame([f._asdict() for f in failures], columns=list(Failure._fields)).to_csv(
        ws.metadata / "failures.csv", index=False)


def collect_run(ws: "layout.Workspace", records: Sequence[Any], failures: Sequence[Failure],
                compression: str) -> None:
    """Write the master, actives and interaction tables of a site.

    Args:
        ws: The site.
        records: Records with pdb_id, master_row(), potential, observed and actives, in input order.
        failures: Structures that dropped out.
        compression: HDF5 compression.
    """
    with start_action(action_type="collect_structure_results"):
        collect_structures(ws, records, failures)

        with open(ws.screening / "actives.smi", "w", encoding="utf-8") as f:
            f.write("SMILES\tPDB_ID\tLigand_Name\tLigand_File\n")
            for r in records:
                for smiles, name, ligand_file in r.actives:
                    if smiles:
                        f.write(f"{smiles}\t{r.pdb_id}\t{name}\t{ligand_file}\n")

        obs_h5 = ws.metadata / "observed_interactions.h5"
        pot_h5 = ws.metadata / "potential_interactions.h5"
        hdf5.save_interactions({r.pdb_id: list(r.potential) for r in records}, pot_h5,
                               compression=compression)
        hdf5.save_interactions({r.pdb_id: list(r.observed) for r in records}, obs_h5,
                               compression=compression)
        log_message(message_type="interaction_data_saved", n_structures=len(records))
        try:
            with start_action(action_type="frequency_analysis"):
                calculate_observed_frequencies(str(obs_h5))
                calculate_potential_frequencies(str(pot_h5))
                calculate_binding_site_utilization(str(obs_h5), str(pot_h5))
                log_message(message_type="frequency_analysis_complete")
        except Exception as e:
            log_message(message_type="error", stage="frequency_analysis", error=str(e))


def collect_from_parts(ws: "layout.Workspace", pdb_ids: Sequence[str]) -> Tuple[List[Structure], List[Failure]]:
    """Collect a run whose structures were processed as separate tasks.

    Args:
        ws: The run.
        pdb_ids: Every ingested structure, in input order.

    Returns:
        (records, failures) as written.
    """
    records: List[Structure] = []
    failures: List[Failure] = []
    for pid in pdb_ids:
        try:
            payload = layout.read_part(ws, pid)
        except ValueError as e:
            log_message(message_type="warning", stage="collect_structure_results",
                        warning=f"unreadable sidecar {pid}.json: {e}")
            failures.append(Failure(pid, "featurize", "unreadable sidecar"))
            continue
        if payload is None:
            failures.append(Failure(pid, "featurize", "no sidecar"))
        elif "failure" in payload:
            failures.append(Failure.from_json(payload))
        else:
            records.append(Structure.from_json(payload))
    if any(f.reason in ("no sidecar", "unreadable sidecar") for f in failures):
        log_message(message_type="warning", stage="collect_structure_results",
                    warning="structures with no sidecar",
                    pdb_ids=[f.pdb_id for f in failures][:20], n_missing=len(failures))
    collect_structures(ws, records, failures)
    return records, failures
