"""Site stage: group every ligand of the run into binding sites and describe them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from eliot import log_message, start_action
from scipy.spatial.distance import pdist

from qpocket.config import QPocketConfig
from qpocket.core.sites import element_symbol, group_ligands, site_ids
from qpocket.core.types import Ligand, Structure
from qpocket.io.layout import Workspace

SITE_COLUMNS = ["site_id", "n_ligands", "n_holo_structures", "centre_x", "centre_y", "centre_z",
                "extent", "resnames", "contact_residues"]
LIGAND_COLUMNS = list(Ligand._fields) + ["site_id"]
NO_LIGAND = "no_ligand_found"


def _heavy_coords(ligand: Ligand) -> np.ndarray:
    """Heavy-atom coordinates of a ligand's extracted file.

    Args:
        ligand: A ligand with its file written.

    Returns:
        ``(n, 3)`` array.
    """
    df = PandasPdb().read_pdb(ligand.file).df["HETATM"]
    heavy = [element_symbol(n, e) not in ("H", "D") for n, e in zip(df["atom_name"], df["element_symbol"])]
    return df.loc[heavy, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=float)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sites(ws: Workspace) -> pd.DataFrame:
    """The run's site table.

    Args:
        ws: The run.

    Returns:
        Contents of ``sites/sites.csv``.
    """
    return pd.read_csv(ws.sites / "sites.csv", keep_default_na=False)


def discover_sites(cfg: QPocketConfig, ws: Workspace, records: Sequence[Structure]) -> pd.DataFrame:
    """Group all ligands of the run into sites and write the site tables.

    Args:
        cfg: Run configuration (``sites.link_cutoff``, ``sites.box_padding``).
        ws: The run.
        records: Every structure of the run.

    Returns:
        The site table, also written to ``sites/sites.csv``; skipped when the inputs match ``grouping.json``.
    """
    ligands = [lig for rec in records for lig in rec.ligands]
    contacts = {key: entries for rec in records for key, entries in rec.contacts}
    ligands_csv, sites_csv, grouping = ws.structures / "ligands.csv", ws.sites / "sites.csv", ws.sites / "grouping.json"
    cutoff, padding = cfg.sites.link_cutoff, cfg.sites.box_padding

    with start_action(action_type="discover_sites", n_ligands=len(ligands)) as action:
        coords = [_heavy_coords(lig) for lig in ligands]
        ids = site_ids(group_ligands(coords, cutoff), [lig.pdb_id for lig in ligands])
        ligand_table = pd.DataFrame([{**lig.to_json(), "site_id": sid} for lig, sid in zip(ligands, ids)],
                                    columns=LIGAND_COLUMNS)
        ligand_text = ligand_table.to_csv(index=False)
        stamp = {"link_cutoff": cutoff, "box_padding": padding,
                 "ligands_hash": hashlib.sha256(ligand_text.encode()).hexdigest()}
        if sites_csv.is_file() and grouping.is_file() and ligands_csv.is_file() \
                and json.loads(grouping.read_text(encoding="utf-8")) == stamp \
                and _hash(ligands_csv) == stamp["ligands_hash"]:
            log_message(message_type="sites_skipped")
            return read_sites(ws)

        rows: List[Dict[str, object]] = []
        by_site: Dict[str, List[int]] = {}
        for i, sid in enumerate(ids):
            by_site.setdefault(sid, []).append(i)
        for sid in sorted(by_site):
            idx = by_site[sid]
            xyz = np.vstack([coords[i] for i in idx])
            members = [ligands[i] for i in idx]
            residues = {e.split("|")[0] for m in members for e in contacts.get(m.key, ()) if e != NO_LIGAND}
            centre = xyz.mean(axis=0)
            rows.append({"site_id": sid, "n_ligands": len(members),
                         "n_holo_structures": len({m.pdb_id for m in members}),
                         "centre_x": centre[0], "centre_y": centre[1], "centre_z": centre[2],
                         "extent": float(pdist(xyz).max()) if len(xyz) > 1 else 0.0,
                         "resnames": ";".join(sorted({m.resname for m in members})),
                         "contact_residues": ";".join(sorted(residues))})
            site_dir = ws.sites / sid
            site_dir.mkdir(parents=True, exist_ok=True)
            (site_dir / "ligands.pdb").write_text(
                "".join(Path(m.file).read_text(encoding="utf-8").replace("END\n", "") for m in members) + "END\n",
                encoding="utf-8")
        table = pd.DataFrame(rows, columns=SITE_COLUMNS)

        for stale in ws.sites.glob("site_*/ligands.pdb"):
            if stale.parent.name not in by_site:
                stale.unlink()
        ligands_csv.write_text(ligand_text, encoding="utf-8")
        table.to_csv(sites_csv, index=False)
        grouping.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        action.add_success_fields(n_sites=len(table))
        return table
