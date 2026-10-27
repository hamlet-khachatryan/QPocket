"""Per-site analysis steps on a finished run: featurise, shape clustering, two-stage, regroup."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from qpocket.config import QPocketConfig, _plain, evolve, load_run_config, plugin_params, save_run_config, validate_config
from qpocket.core.ports import CHARGES
from qpocket.core.result import Failure, guarded
from qpocket.core.types import Structure
from qpocket.io import layout
from qpocket.io.layout import Workspace
from qpocket.io.tables import file_hash
from qpocket.pipelines import parallel, structure
from qpocket.pipelines.collect import write_feature_record
from qpocket.pipelines.clustering import archive_clustering, cluster_shape_only, write_clustering
from qpocket.pipelines.distance import compute_shape_matrix, compute_spectral_distance_matrix, pocket_manifest
from qpocket.pipelines.sites import (_membership, _site_keys, extract_all_sites, group_sites, read_sites,
                                     sync_sites)
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
    if jobs:
        write_feature_record(ws.metadata, cfg.features.enabled)
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
    write_feature_record(ws.metadata, cfg.features.enabled)
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
        SiteError: If the run or site is unknown, or the site has fewer than two pockets.
    """
    cfg, ws = _site(run_dir, site_id)
    try:
        d_shape = pd.read_csv(compute_shape_matrix(cfg, ws), index_col=0)
    except ValueError as e:  # fewer than 2 pockets: the user's to fix, not a crash
        raise SiteError(str(e)) from e
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
        SiteError: If the run or site is unknown, under two pockets are featurised, or no matrix was written.
    """
    cfg, ws = _site(run_dir, site_id)
    if len(pocket_manifest(cfg, ws, require_features=True)[0]) < 2:
        raise SiteError(NEEDS_FEATURES)
    if cfg.distance.clustering.mode == "shape_only":  # this step is the two-stage one, whatever the run chose
        cfg = evolve(cfg, "distance.clustering.mode", "two_stage")
    archive_clustering(ws.metadata)
    matrix = ws.metadata / "spectral_distance_matrix.csv"
    before = matrix.stat().st_mtime_ns if matrix.exists() else None
    compute_spectral_distance_matrix(cfg, ws)  # returns early, writing nothing, on a failure it logs
    if not matrix.exists() or matrix.stat().st_mtime_ns == before:
        raise SiteError(f"clustering {site_id} wrote no {matrix}; see the run log")
    return ws.metadata / "clustering_metadata.csv"


TUNABLE = ("min_cavity_points", "cavity_overlap", "min_structure_fraction", "ligand_fraction", "box_padding")


def regroup_sites(run_dir: Path, **overrides: float) -> pd.DataFrame:
    """Rebuild the run's sites with new ``sites`` parameters; a site whose member cavities are unchanged keeps its folder.

    Args:
        run_dir: The run directory.
        **overrides: New values for any of TUNABLE; saved into the run's config.

    Returns:
        The new site table.

    Raises:
        ValueError: If no parameter, or an unknown one, is given.
    """
    if not overrides:
        raise ValueError(f"regroup needs at least one of: {', '.join(TUNABLE)}")
    unknown = sorted(set(overrides) - set(TUNABLE))
    if unknown:
        raise ValueError(f"unknown site parameter(s) {', '.join(unknown)}; known: {', '.join(TUNABLE)}")
    site_table(run_dir)  # refuses a directory that is not a run
    return _resite(run_dir, _with_params(load_run_config(run_dir), overrides))


def _with_params(cfg: QPocketConfig, params: Dict[str, float]) -> QPocketConfig:
    for key, value in params.items():
        cfg = evolve(cfg, f"sites.{key}", int(value) if key == "min_cavity_points" else float(value))
    return cfg


def _records(ws: Workspace) -> List[Structure]:
    records = []
    for pid in pd.read_csv(ws.metadata / "master.csv", dtype=str)["pdb_id"]:
        payload = layout.read_part(ws, pid)
        if payload is not None and "failure" not in payload:
            records.append(Structure.from_json(payload))
    return records


def _resite(run_dir: Path, cfg: QPocketConfig) -> pd.DataFrame:
    """Rebuild the run's sites under ``cfg``, keep the unchanged sites' folders, and save ``cfg``."""
    validate_config(cfg)
    ws = layout.workspace(run_dir)
    records = _records(ws)
    table, _ = sync_sites(cfg, ws, records)
    # Every site, kept ones too: a kept site's cavities are the same, but its ligands and so its
    # holo/apo states may not be. Unchanged clouds keep their mtime and cached geodesics.
    extract_all_sites(cfg, ws, table, records)
    save_run_config(cfg, ws)
    return table


#: What a site loses when its members change: label, then a glob under the site folder.
RESULTS = (("clustering", "metadata/clustering_metadata.csv"), ("featurised pockets", "pockets/*/featurized.csv"),
           ("final metadata", "final"), ("docking", "screening/crossdock/docking_jobs.csv"))


def site_results(run_dir: Path, site_id: str) -> List[str]:
    """The results a site holds that a change of its members would delete.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        Labels from RESULTS that exist, featurised pockets with their count.
    """
    folder = Path(run_dir) / "sites" / site_id
    found = []
    for label, pattern in RESULTS:
        hits = list(folder.glob(pattern))
        if hits:
            found.append(f"{label} ({len(hits)})" if "*" in pattern else label)
    return found


def preview_changes(run_dir: Path, overrides: Optional[List[Dict[str, Any]]] = None,
                    **params: float) -> Dict[str, List[str]]:
    """Which existing sites a regroup or an override change would alter, and what each would lose.

    Args:
        run_dir: The run directory.
        overrides: The full override list to try; the run's own when None.
        **params: Site parameters to try, as for :func:`regroup_sites`.

    Returns:
        Old site id to its :func:`site_results`, for every site whose member cavities would change; writes nothing.
    """
    site_table(run_dir)
    ws = layout.workspace(run_dir)
    cfg = _with_params(load_run_config(run_dir), params)
    if overrides is not None:
        cfg = evolve(cfg, "sites.overrides", overrides)
    new = set(group_sites(cfg, ws, _records(ws)).site_keys.values())
    return {sid: site_results(run_dir, sid) for sid, keys in _site_keys(ws).items() if keys not in new}


def _cavities_of(run_dir: Path, site_id: str) -> List[List[Any]]:
    _site_ws(run_dir, site_id)  # refuses an unknown site
    per = _membership(layout.workspace(run_dir)).get(site_id, {})
    return [[pdb, int(c)] for pdb, cavs in sorted(per.items()) for c in sorted(cavs)]


def merge_override(run_dir: Path, site_ids: Sequence[str]) -> Dict[str, Any]:
    """The override that merges these sites, not yet applied.

    Args:
        run_dir: The run directory.
        site_ids: Two or more sites.

    Returns:
        ``{"op": "merge", "sites": [...], "label": ...}``.

    Raises:
        SiteError: On fewer than two sites or an unknown one.
    """
    if len(set(site_ids)) < 2:
        raise SiteError("pick at least two sites to merge")
    return {"op": "merge", "sites": [_cavities_of(run_dir, sid) for sid in site_ids], "label": " + ".join(site_ids)}


def drop_override(run_dir: Path, site_id: str) -> Dict[str, Any]:
    """The override that drops this site, not yet applied; its ligands become misses (``site_dropped``)."""
    return {"op": "drop", "cavities": _cavities_of(run_dir, site_id), "label": site_id}


def split_override(run_dir: Path, site_id: str) -> Dict[str, Any]:
    """The override that splits this site into its strict cavity groups, not yet applied.

    Args:
        run_dir: The run directory.
        site_id: The site.

    Returns:
        ``{"op": "split", "cavities": [...], "label": ...}``.

    Raises:
        SiteError: On an unknown site, or one with a single strict cavity group.
    """
    cavities = _cavities_of(run_dir, site_id)
    grouping = Path(run_dir) / "sites" / "grouping.json"
    cores = json.loads(grouping.read_text(encoding="utf-8")).get("cores", {}) if grouping.is_file() else {}
    if len(cores.get(site_id, [])) < 2:
        raise SiteError(f"{site_id} has one strict cavity group; nothing to split")
    return {"op": "split", "cavities": cavities, "label": site_id}


def list_overrides(run_dir: Path) -> List[Dict[str, Any]]:
    """The run's overrides, each with the ``status`` and ``site_ids`` of its last application.

    Args:
        run_dir: The run directory.

    Returns:
        Config entries in order, with ``status`` ("applied", "unmatched", or "pending"), ``site_ids`` and ``parts``.
    """
    overrides = _plain(list(load_run_config(run_dir).sites.overrides or []))
    grouping = Path(run_dir) / "sites" / "grouping.json"
    saved = json.loads(grouping.read_text(encoding="utf-8")) if grouping.is_file() else {}
    applied = saved.get("applied", []) if saved.get("overrides") == overrides else []
    return [{**o, "status": "pending", "site_ids": [], **{k: v for k, v in a.items() if k != "op"}}
            for o, a in zip(overrides, applied + [{}] * (len(overrides) - len(applied)))]


def apply_site_overrides(run_dir: Path, overrides: List[Dict[str, Any]]) -> pd.DataFrame:
    """Rebuild the sites with exactly these overrides, saved into the run's config.

    Args:
        run_dir: The run directory.
        overrides: The full list, in the order they apply.

    Returns:
        The new site table.
    """
    site_table(run_dir)
    return _resite(run_dir, evolve(load_run_config(run_dir), "sites.overrides", overrides))


def add_override(run_dir: Path, override: Dict[str, Any]) -> pd.DataFrame:
    """Append one override and rebuild the sites; see :func:`apply_site_overrides`."""
    current = _plain(list(load_run_config(run_dir).sites.overrides or []))
    return apply_site_overrides(run_dir, current + [override])


def undo_override(run_dir: Path, index: int) -> pd.DataFrame:
    """Remove the override at ``index`` and rebuild the sites.

    Args:
        run_dir: The run directory.
        index: Position in :func:`list_overrides`.

    Returns:
        The new site table.

    Raises:
        SiteError: If there is no override at ``index``.
    """
    current = _plain(list(load_run_config(run_dir).sites.overrides or []))
    if not 0 <= index < len(current):
        raise SiteError(f"no override #{index}; the run has {len(current)}")
    return apply_site_overrides(run_dir, current[:index] + current[index + 1:])


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
