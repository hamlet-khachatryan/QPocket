"""Per-site analysis steps on a finished run: featurise, shape clustering, two-stage, regroup."""
from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

from qpocket.config import QPocketConfig, evolve, load_run_config, plugin_params, save_run_config
from qpocket.core.ports import CHARGES
from qpocket.core.result import Failure, guarded
from qpocket.core.types import Ligand, Structure
from qpocket.io import layout
from qpocket.io.layout import Workspace
from qpocket.pipelines import parallel, structure
from qpocket.pipelines.clustering import archive_clustering, cluster_shape_only, write_clustering
from qpocket.pipelines.distance import compute_shape_matrix, compute_spectral_distance_matrix, pocket_manifest
from qpocket.pipelines.sites import discover_sites, extract_all_sites, read_sites

PARKED = ".regroup_"
NEEDS_FEATURES = "two-stage clustering needs featurised pockets; run Featurise first"


class SiteError(Exception):
    """A site step cannot run for a reason the user can fix: not a run, unknown site, pockets not featurised."""


def site_table(run_dir: Path) -> pd.DataFrame:
    """The run's site table.

    Args:
        run_dir: The run directory.

    Returns:
        Contents of ``sites/sites.csv``.

    Raises:
        SiteError: If the directory is not a QPocket run.
    """
    ws = layout.workspace(run_dir)
    if not (ws.sites / "sites.csv").is_file():
        raise SiteError(f"{run_dir} is not a QPocket run (no sites/sites.csv)")
    return read_sites(ws)


def _site_ws(run_dir: Path, site_id: str) -> Workspace:
    known = list(site_table(run_dir)["site_id"])
    if site_id not in known:
        raise SiteError(f"unknown site {site_id!r}; known: {', '.join(known) or 'none'}")
    ws = layout.workspace(run_dir)
    return layout.site_workspace(ws, site_id)


def _site(run_dir: Path, site_id: str) -> Tuple[QPocketConfig, Workspace]:
    ws = _site_ws(run_dir, site_id)
    return load_run_config(run_dir), ws


def _featurize_pocket(cfg: QPocketConfig, job: Tuple[Path, str]) -> str:
    csv_path, cleaned_file = job
    structure.featurize(csv_path, Path(cleaned_file), cfg.features.enabled, cfg.charges.use, cfg.features.options,
                        plugin_params(CHARGES, cfg.charges.use, cfg.charges.options))
    return csv_path.parent.name


def _job_pdb(job: Tuple[Path, str]) -> str:
    return job[0].parent.name


def featurize_site(run_dir: Path, site_id: str, workers: Optional[int] = None) -> List[str]:
    """Featurise every cloud of a site that is not already featurised.

    Args:
        run_dir: The run directory.
        site_id: The site, such as ``site_01``.
        workers: Parallel workers; the run's ``compute.workers`` when None.

    Returns:
        The pdb_ids featurised now; a cloud whose featurized.csv is newer than it is skipped.

    Raises:
        SiteError: If the directory is not a run or the site is not in sites.csv.
    """
    cfg, ws = _site(run_dir, site_id)
    cleaned = dict(pd.read_csv(ws.metadata / "master.csv", dtype=str, keep_default_na=False)[["pdb_id", "cleaned_file"]].values)
    clouds, _ = pocket_manifest(cfg, ws, require_features=False)
    done = [p.with_name("featurized.csv") for p in clouds]
    jobs = [(p, cleaned[p.parent.name]) for p, f in zip(clouds, done)
            if not f.exists() or f.stat().st_mtime < p.stat().st_mtime]
    results = parallel.run_map(
        guarded("featurize", partial(_featurize_pocket, cfg), _job_pdb), jobs,
        workers=cfg.compute.workers if workers is None else workers, backend=cfg.compute.backend,
        stage="featurize", log_dir=ws.root / parallel.WORKER_LOG_DIR, label=_job_pdb)
    # featurize logs and returns on a failed pocket, so the file is the evidence.
    return [job[0].parent.name for job, res in zip(jobs, results)
            if not isinstance(res, Failure) and job[0].with_name("featurized.csv").exists()]


def shape_cluster_site(run_dir: Path, site_id: str) -> Path:
    """Cluster a site's pockets on shape alone; any previous clustering moves to history.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        Path of the site's clustering_metadata.csv.

    Raises:
        SiteError: If the directory is not a run or the site is not in sites.csv.
    """
    cfg, ws = _site(run_dir, site_id)
    d_shape = pd.read_csv(compute_shape_matrix(cfg, ws), index_col=0)
    ids = [str(i) for i in d_shape.index]
    matrix = d_shape.to_numpy(float)
    assignment = cluster_shape_only(cfg.distance.clustering.shape, matrix)
    archive_clustering(ws.metadata)
    return write_clustering(ws.metadata, ids, assignment, matrix,
                            {pid: ws.pockets / pid / "binding_site.csv" for pid in ids})


def two_stage_site(run_dir: Path, site_id: str) -> Path:
    """Cluster a site's featurised pockets on shape, then chemistry; any previous clustering moves to history.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        Path of the site's clustering_metadata.csv.

    Raises:
        SiteError: If the run or site is unknown, or fewer than two of the site's pockets are featurised.
    """
    cfg, ws = _site(run_dir, site_id)
    if len(pocket_manifest(cfg, ws, require_features=True)[0]) < 2:
        raise SiteError(NEEDS_FEATURES)
    if cfg.distance.clustering.mode == "shape_only":  # this step is the two-stage one, whatever the run chose
        cfg = evolve(cfg, "distance.clustering.mode", "two_stage")
    archive_clustering(ws.metadata)
    compute_spectral_distance_matrix(cfg, ws)
    return ws.metadata / "clustering_metadata.csv"


def _site_keys(ws: Workspace) -> Dict[str, FrozenSet[str]]:
    path = ws.structures / "ligands.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {sid: frozenset(Ligand(*row).key for row in group[list(Ligand._fields)].itertuples(index=False))
            for sid, group in table.groupby("site_id")}


def _repoint(site_dir: Path) -> None:
    """Point the pocket_file paths of a renamed site's clustering tables at its new folder.

    Args:
        site_dir: The site folder, under its new name.
    """
    for path in (site_dir / "metadata").rglob("*.csv"):
        if "pocket_file" not in pd.read_csv(path, nrows=0).columns:
            continue
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        # pocket_file is <site>/pockets/<pdb_id>/<file>; crossdock files hold folder-relative paths only.
        table["pocket_file"] = [str(site_dir.joinpath(*Path(p).parts[-3:])) if p else p
                                for p in table["pocket_file"]]
        table.to_csv(path, index=False)


def regroup_sites(run_dir: Path, link_cutoff: float) -> pd.DataFrame:
    """Regroup the run's ligands with a new link cutoff; a site whose ligands are unchanged keeps its folder.

    Args:
        run_dir: The run directory.
        link_cutoff: The new ``sites.link_cutoff`` (Å); saved into the run's config.

    Returns:
        The new site table.
    """
    site_table(run_dir)  # refuses a directory that is not a run
    ws = layout.workspace(run_dir)
    cfg = evolve(load_run_config(run_dir), "sites.link_cutoff", float(link_cutoff))
    records = []
    for pid in pd.read_csv(ws.metadata / "master.csv", dtype=str)["pdb_id"]:
        payload = layout.read_part(ws, pid)
        if payload is not None and "failure" not in payload:
            records.append(Structure.from_json(payload))

    # A crashed regroup leaves parked folders: one whose name is free is that old site, the rest are copies.
    for stale in sorted(p for p in ws.sites.glob(f"{PARKED}site_*") if p.is_dir()):
        real = ws.sites / stale.name[len(PARKED):]
        if real.exists():
            shutil.rmtree(stale)
        else:
            stale.rename(real)

    old_keys = _site_keys(ws)
    # discover_sites skips on a matching grouping.json and would then not rewrite the sites' ligands.pdb.
    (ws.sites / "grouping.json").unlink(missing_ok=True)
    parked: Dict[str, Path] = {}
    try:
        # Out of the way first, so the new sites written under the same ids do not land in old folders.
        for folder in sorted(p for p in ws.sites.glob("site_*") if p.is_dir()):
            parked[folder.name] = folder.rename(ws.sites / f"{PARKED}{folder.name}")
        table = discover_sites(cfg, ws, records)
    except Exception:
        for sid, folder in parked.items():
            shutil.rmtree(ws.sites / sid, ignore_errors=True)
            folder.rename(ws.sites / sid)
        raise

    changed = []
    for sid, keys in _site_keys(ws).items():
        # A folder whose extraction never finished has nothing worth keeping.
        match = next((old for old, k in old_keys.items() if k == keys and old in parked
                      and (parked[old] / "metadata" / "master.csv").exists()), None)
        if match is None:
            changed.append(sid)
            continue
        kept, fresh = parked.pop(match), ws.sites / sid
        if (fresh / "ligands.pdb").exists():
            shutil.copy2(fresh / "ligands.pdb", kept / "ligands.pdb")
        shutil.rmtree(fresh, ignore_errors=True)
        kept.rename(fresh)
        if match != sid:
            _repoint(fresh)
    for folder in parked.values():
        shutil.rmtree(folder)

    extract_all_sites(cfg, ws, table[table["site_id"].isin(changed)], records)
    save_run_config(cfg, ws)
    return table


def site_status(run_dir: Path, site_id: str) -> Dict[str, bool]:
    """Which steps a site has been through, read from its files.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        ``clouds, geodesics, shape_clustered, featurised, two_stage, crossdock`` to whether it is done;
        shape_clustered means any clustering, two_stage a two-stage or single one over featurised pockets.

    Raises:
        SiteError: If the directory is not a run or the site is not in sites.csv.
    """
    ws = _site_ws(run_dir, site_id)
    clouds = list(ws.pockets.glob("*/binding_site.csv"))
    params = ws.metadata / "clustering_params.json"
    mode = json.loads(params.read_text(encoding="utf-8")).get("mode") if params.exists() else None
    return {
        "clouds": bool(clouds),
        "geodesics": bool(clouds) and all(c.with_name("geospectral_cache.npz").exists() for c in clouds),
        "shape_clustered": params.exists(),
        "featurised": bool(clouds) and all(c.with_name("featurized.csv").exists() for c in clouds),
        "two_stage": mode in ("two_stage", "single") and (ws.metadata / "feature_distance_matrix.csv").exists(),
        "crossdock": (ws.screening / "crossdock" / "docking_jobs.csv").exists(),
    }
