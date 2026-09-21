"""Pipeline stages: compose core computations with io adapters."""
from __future__ import annotations

from pathlib import Path

from eliot import log_message, start_action

from qpocket.core.features import pocket_features
from qpocket.io import tables, tools


def featurize(csv_path: Path, pdb_path: Path) -> None:
    """Write featurized.csv beside a pocket's point cloud.

    Args:
        csv_path: Pocket point cloud CSV (x, y, z).
        pdb_path: Cleaned protein PDB.
    """
    with start_action(action_type="process_point_cloud"):
        if not csv_path.exists():
            log_message(message_type="error", error="CSV missing", path=str(csv_path))
            return

        try:
            table, points = tables.read_point_table(csv_path)
        except Exception as e:
            log_message(message_type="error", error=f"CSV read error: {e}")
            return

        try:
            data = tools.structure_data(pdb_path)
        except Exception as e:
            log_message(message_type="fatal_error", error=str(e))
            return

        out_path = csv_path.with_name("featurized.csv")
        log_message(message_type="status", msg="Running APBS")
        apbs = tools.apbs_potential(pdb_path, points)
        log_message(message_type="status", msg="Running CP/LP and geometry")
        tables.write_featurized(out_path, table, pocket_features(points, data, apbs))
        log_message(message_type="success", output=str(out_path))
