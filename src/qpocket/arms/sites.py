"""Per-site analysis steps on a finished run: featurise, shape clustering, two-stage, regroup."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from qpocket.config import QPocketConfig, evolve, load_run_config, plugin_params, save_run_config
from qpocket.core.ports import CHARGES
from qpocket.core.result import Failure, guarded
from qpocket.core.types import Structure
from qpocket.io import layout
from qpocket.io.layout import Workspace
from qpocket.io.tables import file_hash
from qpocket.pipelines import parallel, structure
from qpocket.pipelines.clustering import archive_clustering, cluster_shape_only, write_clustering
from qpocket.pipelines.distance import compute_shape_matrix, compute_spectral_distance_matrix, pocket_manifest
from qpocket.pipelines.sites import extract_all_sites, read_sites, sync_sites
from qpocket.pipelines.tasks import TaskError, resolve_pdb_id

#: Beside featurized.csv: the sha256 of the binding_site.csv it was built from.
SOURCE = "featurized.source"
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
    done, source = csv_path.with_name("featurized.csv"), csv_path.with_name(SOURCE)
    # Gone first, so a failed featurisation leaves no earlier result that looks current.
    done.unlink(missing_ok=True)
    source.unlink(missing_ok=True)
    cloud_hash = file_hash(csv_path)
    structure.featurize(csv_path, Path(cleaned_file), cfg.features.enabled, cfg.charges.use, cfg.features.options,
                        plugin_params(CHARGES, cfg.charges.use, cfg.charges.options))
    if done.exists():
        source.write_text(cloud_hash + "\n", encoding="utf-8")
    return csv_path.parent.name


def _job_pdb(job: Tuple[Path, str]) -> str:
    return job[0].parent.name


def _cleaned(ws: Workspace) -> Dict[str, str]:
    return dict(pd.read_csv(ws.metadata / "master.csv", dtype=str, keep_default_na=False)[["pdb_id", "cleaned_file"]].values)


def _stale(cloud: Path) -> bool:
    source = cloud.with_name(SOURCE)
    return not (cloud.with_name("featurized.csv").exists() and source.exists()) \
        or source.read_text(encoding="utf-8").strip() != file_hash(cloud)


def featurize_site(run_dir: Path, site_id: str, workers: Optional[int] = None) -> List[str]:
    """Featurise every cloud of a site that is not already featurised.

    Args:
        run_dir: The run directory.
        site_id: The site, such as ``site_01``.
        workers: Parallel workers; the run's ``compute.workers`` when None.

    Returns:
        The pdb_ids featurised now; a cloud whose featurized.csv was built from its current content is skipped.

    Raises:
        SiteError: If the directory is not a run or the site is not in sites.csv.
    """
    cfg, ws = _site(run_dir, site_id)
    cleaned = _cleaned(ws)
    clouds, _ = pocket_manifest(cfg, ws, require_features=False)
    jobs = [(p, cleaned[p.parent.name]) for p in clouds if _stale(p)]
    results = parallel.run_map(
        guarded("featurize", partial(_featurize_pocket, cfg), _job_pdb), jobs,
        workers=cfg.compute.workers if workers is None else workers, backend=cfg.compute.backend,
        stage="featurize", log_dir=ws.root / parallel.WORKER_LOG_DIR, label=_job_pdb)
    # featurize logs and returns on a failed pocket, so the file is the evidence.
    return [job[0].parent.name for job, res in zip(jobs, results)
            if not isinstance(res, Failure) and job[0].with_name("featurized.csv").exists()]


def featurize_pocket(run_dir: Path, site_id: str, pdb_id: str, force: bool = False) -> bool:
    """Featurise one cloud of a site, the unit of a SLURM array task.

    Args:
        run_dir: The run directory.
        site_id: The site.
        pdb_id: The structure whose cloud to featurise.
        force: Featurise even when featurized.csv was built from the cloud's current content.

    Returns:
        Whether the cloud was featurised now; False when it was already up to date.

    Raises:
        SiteError: If the run or site is unknown, or the structure has no cloud at this site.
    """
    cfg, ws = _site(run_dir, site_id)
    cloud = ws.pockets / pdb_id / "binding_site.csv"
    if not cloud.is_file():
        raise SiteError(f"{pdb_id} has no cloud at {site_id} ({cloud} not found)")
    if not (force or _stale(cloud)):
        return False
    _featurize_pocket(cfg, (cloud, _cleaned(ws)[pdb_id]))
    return True


def run_site_task(name: str, run_dir: Path, site_id: Optional[str], manifest: Optional[Path] = None,
                  index: Optional[int] = None, force: bool = False) -> None:
    """One per-site SLURM stage: featurise one cloud, or cluster the site.

    Args:
        name: ``site_featurize`` or ``site_cluster``.
        run_dir: The run directory.
        site_id: The site.
        manifest: Manifest path, required for ``site_featurize``.
        index: Row within the manifest, required for ``site_featurize``.
        force: Featurise even when featurized.csv is up to date; clustering always reruns.

    Raises:
        TaskError: On a missing site, manifest or index, a site error, or a featurisation that wrote nothing.
    """
    if not site_id:
        raise TaskError(f"stage {name!r} needs --site")
    try:
        if name == "site_cluster":
            two_stage_site(run_dir, site_id)
            matrix = Path(run_dir) / "sites" / site_id / "metadata" / "spectral_distance_matrix.csv"
            if not matrix.exists():  # the distance step returns early on too few pockets
                raise TaskError(f"clustering {site_id} wrote no {matrix}")
            return
        if manifest is None or index is None:
            raise TaskError(f"stage {name!r} needs --manifest and --index")
        pdb_id = resolve_pdb_id(manifest, index)
        featurize_pocket(run_dir, site_id, pdb_id, force=force)
    except SiteError as e:
        raise TaskError(str(e)) from e
    # featurize logs and returns on a failed pocket; an array task must fail loudly instead.
    out = _site_ws(run_dir, site_id).pockets / pdb_id / "featurized.csv"
    if not out.exists():
        raise TaskError(f"featurisation of {pdb_id} at {site_id} wrote no {out}")


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

    table, changed = sync_sites(cfg, ws, records)
    extract_all_sites(cfg, ws, table[table["site_id"].isin(changed)], records)
    save_run_config(cfg, ws)
    return table


def site_status(run_dir: Path, site_id: str) -> Dict[str, Union[bool, int]]:
    """Which steps a site has been through, read from its files.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        ``clouds, geodesics, shape_clustered, featurised, two_stage, crossdock`` to whether it is done;
        shape_clustered means any clustering, two_stage a two-stage or single one over featurised pockets,
        featurised that every cloud is. ``n_featurised`` and ``n_clouds`` count the featurised and all clouds.

    Raises:
        SiteError: If the directory is not a run or the site is not in sites.csv.
    """
    ws = _site_ws(run_dir, site_id)
    clouds = list(ws.pockets.glob("*/binding_site.csv"))
    params = ws.metadata / "clustering_params.json"
    mode = json.loads(params.read_text(encoding="utf-8")).get("mode") if params.exists() else None
    n_featurised = sum(c.with_name("featurized.csv").exists() for c in clouds)
    return {
        "clouds": bool(clouds),
        "geodesics": bool(clouds) and all(c.with_name("geospectral_cache.npz").exists() for c in clouds),
        "shape_clustered": params.exists(),
        "featurised": bool(clouds) and n_featurised == len(clouds),
        "two_stage": mode in ("two_stage", "single") and (ws.metadata / "feature_distance_matrix.csv").exists(),
        "crossdock": (ws.screening / "crossdock" / "docking_jobs.csv").exists(),
        "n_featurised": n_featurised,
        "n_clouds": len(clouds),
    }
