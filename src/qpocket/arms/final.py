"""Finalising a site: its clustering, representatives, residue profile and actives, frozen into final/."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from eliot import log_message

from qpocket.arms.sites import SiteError, _site_ws
from qpocket.core.types import Ligand
from qpocket.io.tables import file_hash

#: What final/ holds, besides history/.
FINAL_FILES = ("clusters.csv", "settings.json", "residue_profile.csv", "ligands.csv")
MATRICES = ("shape_distance_matrix.csv", "feature_distance_matrix.csv", "spectral_distance_matrix.csv")


def _park(folder: Path, stamp: str) -> None:
    """Move the current final (not its history) to history/<stamp>, -2, -3 … on a collision."""
    if not any((folder / name).exists() for name in FINAL_FILES):
        return
    history = folder / "history"
    target, n = history / stamp, 2
    while target.exists():
        target, n = history / f"{stamp}-{n}", n + 1
    target.mkdir(parents=True)
    for name in FINAL_FILES:
        if (folder / name).exists():
            shutil.move(str(folder / name), str(target / name))


def _ligands(run_dir: Path, site_id: str, cluster_of: Dict[str, str]) -> pd.DataFrame:
    path = Path(run_dir) / "structures" / "ligands.csv"
    columns = ["cluster", "pdb_id", "ligand_key", "resname", "smiles"]
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for row in table.itertuples(index=False):
        if site_id in row.site_ids.split(";") and row.pdb_id in cluster_of:
            key = Ligand(*[getattr(row, f) for f in Ligand._fields]).key
            rows.append({"cluster": cluster_of[row.pdb_id], "pdb_id": row.pdb_id, "ligand_key": key,
                         "resname": row.resname, "smiles": row.smiles})
    return pd.DataFrame(rows, columns=columns)


def finalise_site(run_dir: Path, site_id: str, now: Optional[datetime] = None) -> Path:
    """Freeze the site's applied clustering into ``final/``; an earlier final moves to ``final/history/``.

    Args:
        run_dir: The run directory.
        site_id: The site.
        now: The time to record; the current time when None.

    Returns:
        The final/ folder.

    Raises:
        SiteError: If the site is unknown or holds no clustering.
    """
    from qpocket.viz.clusters import cluster_labels_of
    from qpocket.viz.interactions import load_run_interactions
    from qpocket.viz.pages.exploration import cluster_members, residue_profile

    ws = _site_ws(run_dir, site_id)
    meta = ws.metadata
    if not (meta / "clustering_metadata.csv").exists():
        raise SiteError(f"cluster this site first: {site_id} has no clustering to finalise")
    when = (now or datetime.now()).replace(microsecond=0)
    table = pd.read_csv(meta / "clustering_metadata.csv")
    table["pdb_id"] = table["pdb_id"].astype(str)
    master = pd.read_csv(meta / "master.csv", dtype={"pdb_id": str})
    holo = set(master.loc[master["protein_state"] == "BOUND_STATE", "pdb_id"])
    names = cluster_labels_of(table)
    table["cluster"] = [names[int(c)] for c in table["cluster_id"]]
    clusters = pd.DataFrame({"pdb_id": table["pdb_id"], "cluster": table["cluster"],
                             "cluster_id": table["cluster_id"].astype(int),
                             "is_representative": table["is_representative"].astype(bool),
                             "dist_to_representative": table["dist_to_exemplar"].astype(float),
                             "holo": table["pdb_id"].isin(holo)})
    params = json.loads((meta / "clustering_params.json").read_text(encoding="utf-8")) \
        if (meta / "clustering_params.json").exists() else {}
    features = json.loads((meta / "features.json").read_text(encoding="utf-8")).get("columns", []) \
        if (meta / "features.json").exists() else []
    shape_only = params.get("mode") == "shape_only" or not (meta / "feature_distance_matrix.csv").exists()
    profile = residue_profile(load_run_interactions(ws.root), cluster_members(table, names))

    folder = ws.root / "final"
    folder.mkdir(parents=True, exist_ok=True)
    _park(folder, when.isoformat().replace(":", "-"))
    clusters.to_csv(folder / "clusters.csv", index=False)
    profile.drop(columns=["n_members", "n_holo"]).to_csv(folder / "residue_profile.csv", index=False)
    _ligands(run_dir, site_id, dict(zip(clusters["pdb_id"], clusters["cluster"]))).to_csv(
        folder / "ligands.csv", index=False)
    settings = {"site_id": site_id, "finalised_at": when.isoformat(), "shape_only": shape_only,
                "features": [] if shape_only else list(features), "clustering": params,
                "n_structures": len(clusters), "n_clusters": int(clusters["cluster_id"].nunique()),
                "representatives": clusters.loc[clusters["is_representative"], "pdb_id"].tolist(),
                "matrices": {m: file_hash(meta / m) for m in MATRICES if (meta / m).exists()},
                "clustering_hash": file_hash(meta / "clustering_metadata.csv")}
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    log_message(message_type="site_finalised", site_id=site_id, n_clusters=settings["n_clusters"],
                n_representatives=len(settings["representatives"]))
    return folder


def final_status(run_dir: Path, site_id: str) -> str:
    """Whether the site's final metadata exists and still matches its applied clustering.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        "missing", "stale" (the clustering changed since), or "current".
    """
    folder = Path(run_dir) / "sites" / site_id / "final"
    if not all((folder / name).exists() for name in FINAL_FILES):
        return "missing"
    applied = Path(run_dir) / "sites" / site_id / "metadata" / "clustering_metadata.csv"
    saved = json.loads((folder / "settings.json").read_text(encoding="utf-8")).get("clustering_hash")
    return "current" if applied.exists() and file_hash(applied) == saved else "stale"


def read_final(run_dir: Path, site_id: str) -> Dict[str, object]:
    """The site's final metadata: ``clusters``, ``residue_profile``, ``ligands`` frames and ``settings``.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        The four parts.

    Raises:
        SiteError: If the site has not been finalised.
    """
    folder = Path(run_dir) / "sites" / site_id / "final"
    if final_status(run_dir, site_id) == "missing":
        raise SiteError(f"finalise {site_id} first (no {folder})")
    read = {name[:-4]: pd.read_csv(folder / name, keep_default_na=False, dtype={"pdb_id": str})
            for name in FINAL_FILES if name.endswith(".csv")}
    return {**read, "settings": json.loads((folder / "settings.json").read_text(encoding="utf-8"))}
