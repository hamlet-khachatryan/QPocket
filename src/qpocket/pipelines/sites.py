"""Site stage: group every structure's CavFind cavities into binding sites, assign ligands, cut each structure's cloud."""
from __future__ import annotations

import hashlib
import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from eliot import log_message, start_action
from scipy.spatial.distance import cdist

from qpocket.config import QPocketConfig
from qpocket.constants import THREE_TO_ONE_MAPPING
from qpocket.core.result import Failure, guarded
from qpocket.core.sites import (box_iou, cavity_cores, element_symbol, grid_spacing, ligand_sites, median_box,
                                merge_sites, site_members, site_name, voxel_box)
from qpocket.core.types import Ligand, Structure
from qpocket.io import layout
from qpocket.io.layout import Workspace
from qpocket.pipelines import parallel
from qpocket.pipelines.collect import collect_run
from qpocket.pipelines.distance import compute_geodesic

SITE_COLUMNS = ["site_id", "n_structures", "n_holo", "n_apo", "n_cavfind_miss", "n_ligands", "n_cavities",
                "centre_x", "centre_y", "centre_z", "box_lo_x", "box_lo_y", "box_lo_z",
                "box_hi_x", "box_hi_y", "box_hi_z", "resnames", "lining_residues", "contact_residues"]
CAVITY_COLUMNS = ["pdb_id", "cavity", "n_points", "lo_x", "lo_y", "lo_z", "hi_x", "hi_y", "hi_z", "site_ids"]
LIGAND_COLUMNS = list(Ligand._fields) + ["site_ids"]
MISS_COLUMNS = ["pdb_id", "ligand_key", "resname", "sites", "reason"]
NO_LIGAND = "no_ligand_found"


def _heavy_coords(path: str) -> np.ndarray:
    """Heavy-atom coordinates of the HETATM records of a ligand file.

    Args:
        path: An extracted ligand, or a site's union ``ligands.pdb``.

    Returns:
        ``(n, 3)`` array.
    """
    df = PandasPdb().read_pdb(str(path)).df["HETATM"]
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


def _cavities(record: Structure) -> pd.DataFrame:
    """The structure's CavFind points (x, y, z, cavity), or an empty table when there are none."""
    if not record.cavities_file or not Path(record.cavities_file).exists():
        return pd.DataFrame(columns=["x", "y", "z", "cavity"])
    try:
        return pd.read_csv(record.cavities_file)[["x", "y", "z", "cavity"]]
    except Exception as e:  # one unreadable file costs that structure its cavities, not the run its sites
        log_message(message_type="warning", stage="discover_sites", pdb_id=record.pdb_id,
                    warning=f"cavities not readable: {e}")
        return pd.DataFrame(columns=["x", "y", "z", "cavity"])


def _reference_id(cfg: QPocketConfig, ws: Workspace, records: Sequence[Structure]) -> Optional[str]:
    """The run's reference structure among the records, when it can be told."""
    ids = [r.pdb_id for r in records]
    inputs = ws.metadata / "inputs.csv"
    if inputs.is_file():
        table = pd.read_csv(inputs)
        if "is_reference" in table:
            ref = table.loc[table["is_reference"].astype(str) == "True", "pdb_id"]
            if len(ref) and ref.iloc[0] in ids:
                return str(ref.iloc[0])
    pdb_id = (cfg.reference.pdb_id or "").upper()
    return next((i for i in ids if pdb_id and i.upper().startswith(pdb_id)), None)


def _nearest_residue(cleaned_file: Optional[str], centre: np.ndarray) -> str:
    """One-letter code and number of the protein residue nearest ``centre``, e.g. H21; "" when unknown."""
    if not cleaned_file or not Path(cleaned_file).exists():
        return ""
    atoms = PandasPdb().read_pdb(str(cleaned_file)).df["ATOM"]
    if atoms.empty:
        return ""
    row = atoms.iloc[int(cdist(atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(float), centre[None]).argmin())]
    return f"{THREE_TO_ONE_MAPPING.get(str(row['residue_name']).upper(), 'X')}{int(row['residue_number'])}"


def _potential(cleaned_file: Optional[str], cloud: np.ndarray) -> Tuple[str, ...]:
    """Residues with an atom within 4 Å of the cloud, as RES_NUM_CHAIN."""
    if not cleaned_file or not Path(cleaned_file).exists() or not len(cloud):
        return ()
    atoms = PandasPdb().read_pdb(str(cleaned_file)).df["ATOM"]
    near = atoms[cdist(atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=float), cloud).min(axis=1) <= 4.0]
    # Chain ID included so potential and observed keys intersect in utilisation.
    return tuple(sorted(set(near["residue_name"] + "_" + near["residue_number"].astype(str)
                            + "_" + near["chain_id"])))


def discover_sites(cfg: QPocketConfig, ws: Workspace, records: Sequence[Structure]) -> pd.DataFrame:
    """Group the run's CavFind cavities into sites, assign ligands, report CavFind misses, write the tables.

    Args:
        cfg: Run configuration (the ``sites`` block).
        ws: The run.
        records: Every structure of the run.

    Returns:
        The site table, also written to ``sites/sites.csv``; skipped when the inputs match ``grouping.json``.
    """
    p = cfg.sites
    by_id = {r.pdb_id: r for r in records}
    with start_action(action_type="discover_sites", n_structures=len(records)) as action:
        all_cavities, candidates = [], []          # candidates: (pdb_id, cavity, points)
        for rec in records:
            for cavity, group in _cavities(rec).groupby("cavity", sort=True):
                pts = group[["x", "y", "z"]].to_numpy(dtype=float)
                all_cavities.append((rec.pdb_id, int(cavity), pts))
                if len(pts) >= p.min_cavity_points:
                    candidates.append((rec.pdb_id, int(cavity), pts))
        ligands = [lig for rec in records for lig in rec.ligands]
        stamp = {"sites": {k: getattr(p, k) for k in ("min_cavity_points", "cavity_overlap", "min_structure_fraction",
                                                       "ligand_fraction", "box_padding")},
                 "inputs": hashlib.sha256(json.dumps(
                     [[c[0], c[1], len(c[2]), float(c[2].sum())] for c in all_cavities]
                     + [lig.key for lig in ligands]).encode()).hexdigest()}
        grouping, sites_csv = ws.sites / "grouping.json", ws.sites / "sites.csv"
        if sites_csv.is_file() and grouping.is_file() and json.loads(grouping.read_text(encoding="utf-8")) == stamp:
            log_message(message_type="sites_skipped")
            return read_sites(ws)

        half = grid_spacing([c[2] for c in candidates]) / 2
        boxes = np.array([voxel_box(c[2], half) for c in candidates]).reshape(-1, 6)
        groups = merge_sites(cavity_cores(boxes, p.cavity_overlap), boxes, p.cavity_overlap)
        site_boxes = np.array([median_box(boxes[g]) for g in groups]).reshape(-1, 6)
        members = site_members(site_boxes, boxes, p.cavity_overlap)
        lig_sites = ligand_sites([_heavy_coords(lig.file) for lig in ligands], site_boxes,
                                 p.ligand_fraction, p.box_padding)

        n_struct = [len({candidates[i][0] for i in m}) for m in members]
        has_ligand = [any(k in s for s in lig_sites) for k in range(len(groups))]
        kept = [k for k in range(len(groups))
                if n_struct[k] >= p.min_structure_fraction * len(records) or has_ligand[k]]
        centres = {k: (site_boxes[k, :3] + site_boxes[k, 3:]) / 2 for k in kept}
        kept.sort(key=lambda k: (-n_struct[k], *centres[k]))
        ref_id = _reference_id(cfg, ws, records)

        names: Dict[int, str] = {}
        member_ids: Dict[int, Dict[str, List[int]]] = {}
        for rank, k in enumerate(kept, start=1):
            per: Dict[str, List[int]] = {}
            for i in members[k]:
                per.setdefault(candidates[i][0], []).append(candidates[i][1])
            member_ids[k] = per
            frame = ref_id if ref_id in per else next(iter(per), ref_id)
            names[k] = site_name(rank, _nearest_residue(by_id[frame].cleaned_file if frame in by_id else None,
                                                        centres[k]))

        rows, miss_rows, lig_rows = [], [], []
        holo_at: Dict[int, Set[str]] = {k: set() for k in kept}
        miss_at: Dict[int, Set[str]] = {k: set() for k in kept}
        for lig, sites in zip(ligands, lig_sites):
            mine = [k for k in sites if k in names]
            lig_rows.append({**lig.to_json(), "site_ids": ";".join(names[k] for k in mine)})
            holo = [k for k in mine if lig.pdb_id in member_ids[k]]
            for k in holo:
                holo_at[k].add(lig.pdb_id)
            if not mine:
                miss_rows.append({"pdb_id": lig.pdb_id, "ligand_key": lig.key, "resname": lig.resname,
                                  "sites": "", "reason": "no_site"})
            elif not holo:
                for k in mine:
                    miss_at[k].add(lig.pdb_id)
                miss_rows.append({"pdb_id": lig.pdb_id, "ligand_key": lig.key, "resname": lig.resname,
                                  "sites": ";".join(names[k] for k in mine), "reason": "no_cavity_at_site"})

        contacts = {key: entries for rec in records for key, entries in rec.contacts}
        for k in kept:
            sid, box, per = names[k], site_boxes[k], member_ids[k]
            site_ligs = [lig for lig, sites in zip(ligands, lig_sites) if k in sites]
            frame = ref_id if ref_id in per else next(iter(per))
            frame_pts = np.vstack([c[2] for c in candidates if c[0] == frame and c[1] in per[frame]])
            observed = {e.split("|")[0] for lig in site_ligs if lig.pdb_id in holo_at[k]
                        for e in contacts.get(lig.key, ()) if e != NO_LIGAND}
            rows.append({"site_id": sid, "n_structures": len(per), "n_holo": len(holo_at[k]),
                         "n_apo": len(per) - len(holo_at[k]), "n_cavfind_miss": len(miss_at[k]),
                         "n_ligands": len(site_ligs), "n_cavities": len(members[k]),
                         "centre_x": centres[k][0], "centre_y": centres[k][1], "centre_z": centres[k][2],
                         **{f"box_{side}_{ax}": box[i + 3 * j] for j, side in enumerate(("lo", "hi"))
                            for i, ax in enumerate("xyz")},
                         "resnames": ";".join(sorted({lig.resname for lig in site_ligs})),
                         "lining_residues": ";".join(_potential(by_id[frame].cleaned_file, frame_pts)),
                         "contact_residues": ";".join(sorted(observed))})
            site_dir = ws.sites / sid
            site_dir.mkdir(parents=True, exist_ok=True)
            (site_dir / "ligands.pdb").write_text(
                "".join(Path(lig.file).read_text(encoding="utf-8").replace("END\n", "") for lig in site_ligs)
                + "END\n", encoding="utf-8")

        site_of = {}
        for k in kept:
            for i in members[k]:
                site_of.setdefault(i, []).append(names[k])
        index = {(c[0], c[1]): i for i, c in enumerate(candidates)}
        cav_rows = []
        for pdb_id, cavity, pts in all_cavities:
            i = index.get((pdb_id, cavity))
            lo, hi = (boxes[i, :3], boxes[i, 3:]) if i is not None else (pts.min(0) - half, pts.max(0) + half)
            cav_rows.append({"pdb_id": pdb_id, "cavity": cavity, "n_points": len(pts),
                             **dict(zip(CAVITY_COLUMNS[3:9], [*lo, *hi])),
                             "site_ids": ";".join(site_of.get(i, [])) if i is not None else ""})

        table = pd.DataFrame(rows, columns=SITE_COLUMNS)
        for stale in ws.sites.glob("*/ligands.pdb"):
            if stale.parent.name not in table["site_id"].values:
                stale.unlink()
        pd.DataFrame(cav_rows, columns=CAVITY_COLUMNS).to_csv(ws.sites / "cavities.csv", index=False)
        pd.DataFrame(lig_rows, columns=LIGAND_COLUMNS).to_csv(ws.structures / "ligands.csv", index=False)
        pd.DataFrame(miss_rows, columns=MISS_COLUMNS).to_csv(ws.sites / "cavfind_misses.csv", index=False)
        table.to_csv(sites_csv, index=False)
        grouping.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        action.add_success_fields(n_sites=len(table), n_candidates=len(candidates), n_misses=len(miss_rows))
        return table


PARKED = ".regroup_"


def _membership(ws: Workspace) -> Dict[str, Dict[str, List[int]]]:
    """Per site, the member cavities of each structure, read from ``sites/cavities.csv``."""
    path = ws.sites / "cavities.csv"
    out: Dict[str, Dict[str, List[int]]] = {}
    if not path.is_file():
        return out
    table = pd.read_csv(path, dtype={"pdb_id": str}, keep_default_na=False)
    for row in table.itertuples(index=False):
        for sid in filter(None, str(row.site_ids).split(";")):
            out.setdefault(sid, {}).setdefault(row.pdb_id, []).append(int(row.cavity))
    return out


def _site_ligand_keys(ws: Workspace) -> Dict[str, Dict[str, Set[str]]]:
    """Per site, the keys of each structure's ligands assigned to it, read from ``structures/ligands.csv``."""
    path = ws.structures / "ligands.csv"
    out: Dict[str, Dict[str, Set[str]]] = {}
    if not path.is_file():
        return out
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for row in table.itertuples(index=False):
        key = Ligand(*[getattr(row, f) for f in Ligand._fields]).key
        for sid in filter(None, row.site_ids.split(";")):
            out.setdefault(sid, {}).setdefault(row.pdb_id, set()).add(key)
    return out


def _site_keys(ws: Workspace) -> Dict[str, FrozenSet[str]]:
    return {sid: frozenset(f"{pdb}:{c}" for pdb, cavs in per.items() for c in cavs)
            for sid, per in _membership(ws).items()}


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


def _site_folders(ws: Workspace) -> List[Path]:
    """Site folders: every directory under ``sites/`` that is not a parked copy."""
    return sorted(p for p in ws.sites.iterdir() if p.is_dir() and not p.name.startswith(PARKED)) \
        if ws.sites.is_dir() else []


def sync_sites(cfg: QPocketConfig, ws: Workspace, records: Sequence[Structure]) -> Tuple[pd.DataFrame, List[str]]:
    """Rediscover the sites; a site whose member cavities are unchanged keeps its folder and results, renamed if needed.

    Args:
        cfg: Run configuration.
        ws: The run.
        records: Every structure of the run.

    Returns:
        The new site table, and the ids of the sites that are new or changed and so hold no per-site results.
    """
    # A crashed sync leaves parked folders: one whose name is free is that old site, the rest are copies.
    for stale in sorted(p for p in ws.sites.glob(f"{PARKED}*") if p.is_dir()):
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
        for folder in _site_folders(ws):
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
    return table, changed


# ── per-site clouds ──────────────────────────────────────────────────────────

class SiteRecord(NamedTuple):
    """One structure at one site.

    Attributes:
        pdb_id: Identifier.
        rmsd_to_ref: RMSD to the reference after superposition.
        aligned_file: Superposed structure.
        cleaned_file: Protein and allowed cofactors only.
        protonated_file: Docking-ready protonated complex.
        protein_state: BOUND_STATE (cavity and ligand), UNBOUND_STATE (cavity only) or FAILED.
        ligand_file: The structure's largest ligand at the site, for cross-docking.
        ligand_id: Its chemical-component ID.
        potential: Residues within reach of the site cloud.
        observed: RES_NUM_CHAIN|Type contacts of the site's ligands, or ``("no_ligand_found",)``.
        actives: (SMILES, ligand name, ligand file) per site ligand of the structure.
    """

    pdb_id: str
    rmsd_to_ref: Optional[float] = None
    aligned_file: Optional[str] = None
    cleaned_file: Optional[str] = None
    protonated_file: Optional[str] = None
    protein_state: Optional[str] = None
    ligand_file: Optional[str] = None
    ligand_id: Optional[str] = None
    potential: Tuple[str, ...] = ()
    observed: Tuple[str, ...] = ()
    actives: Tuple[Tuple[str, str, str], ...] = ()

    def master_row(self) -> Dict[str, Any]:
        """This structure's row in the site's master.csv.

        Returns:
            Column name to value.
        """
        return {"pdb_id": self.pdb_id, "rmsd_to_ref": self.rmsd_to_ref, "aligned_file": self.aligned_file,
                "cleaned_file": self.cleaned_file, "protonated_file": self.protonated_file,
                "ligand_file": self.ligand_file, "ligand_id": self.ligand_id,
                "protein_state": self.protein_state or "FAILED"}

    def to_json(self) -> Dict[str, Any]:
        """Plain form for a sidecar.

        Returns:
            JSON-serialisable dict.
        """
        payload = self._asdict()
        payload.update(potential=list(self.potential), observed=list(self.observed),
                       actives=[list(a) for a in self.actives])
        return payload


def extract_site(cfg: QPocketConfig, ws: Workspace, site: pd.Series, record: Structure,
                 members: Sequence[int], ligand_keys: Set[str]) -> SiteRecord:
    """Write one structure's cloud at one site: its own member cavities' CavFind points, unmodified.

    Args:
        cfg: Run configuration.
        ws: The run.
        site: The site's row of sites.csv.
        record: The structure; it has at least one member cavity at the site.
        members: Its cavity ids in the site.
        ligand_keys: Keys of its ligands assigned to the site.

    Returns:
        The structure's record at the site: BOUND_STATE with ligands, UNBOUND_STATE without.
    """
    site_id = site["site_id"]
    pocket_dir = layout.site_workspace(ws, site_id).pockets / record.pdb_id
    mine = [lig for lig in record.ligands if lig.key in ligand_keys]
    largest = max(mine, key=lambda lig: lig.n_heavy) if mine else None
    contacts = dict(record.contacts)
    observed = tuple(sorted({e for lig in mine for e in contacts.get(lig.key, ()) if "|" in e})) if mine \
        else (NO_LIGAND,)
    state = "BOUND_STATE" if mine else "UNBOUND_STATE"
    with start_action(action_type="extract_site", site_id=site_id, pdb_id=record.pdb_id) as action:
        points = pd.read_csv(record.cavities_file)
        cloud = points.loc[points["cavity"].isin(list(members)), ["x", "y", "z"]].to_numpy(dtype=float)
        if not len(cloud):
            raise ValueError(f"{record.pdb_id}: member cavities {list(members)} not in {record.cavities_file}")
        pocket_dir.mkdir(parents=True, exist_ok=True)
        csv_path = pocket_dir / "binding_site.csv"
        text = pd.DataFrame(cloud, columns=["x", "y", "z"]).to_csv(index=False)
        if not csv_path.is_file() or csv_path.read_text(encoding="utf-8") != text:  # an unchanged cloud keeps its mtime
            csv_path.write_text(text, encoding="utf-8")
        compute_geodesic(csv_path, cfg.distance.geo_graph_neighbor_radius)
        action.add_success_fields(protein_state=state, n_points=len(cloud))
    return SiteRecord(record.pdb_id, record.rmsd_to_ref, record.aligned_file, record.cleaned_file,
                      record.protonated_file, protein_state=state,
                      ligand_file=largest.file if largest else None,
                      ligand_id=largest.resname if largest else None,
                      potential=_potential(record.cleaned_file, cloud), observed=observed,
                      actives=tuple((lig.smiles, lig.resname, lig.file) for lig in mine))


# One (site row, structure, member cavities, ligand keys) per task, so a worker gets only its own inputs.
_Pair = Tuple[pd.Series, Structure, List[int], Set[str]]


def _extract_pair(cfg: QPocketConfig, ws: Workspace, pair: _Pair) -> SiteRecord:
    site, record, members, keys = pair
    return extract_site(cfg, ws, site, record, members, keys)


def _pair_pdb(pair: _Pair) -> str:
    return pair[1].pdb_id


def _pair_label(pair: _Pair) -> str:
    return f"{pair[0]['site_id']}_{pair[1].pdb_id}"


def _failed(record: Structure) -> SiteRecord:
    return SiteRecord(record.pdb_id, record.rmsd_to_ref, record.aligned_file, record.cleaned_file,
                      record.protonated_file, protein_state="FAILED", observed=(NO_LIGAND,))


def extract_all_sites(cfg: QPocketConfig, ws: Workspace, sites: pd.DataFrame,
                      records: Sequence[Structure]) -> None:
    """Cut every member structure's cloud at every site, then collect each site.

    Args:
        cfg: Run configuration.
        ws: The run.
        sites: The site table from :func:`discover_sites` (or a subset of it).
        records: Every structure of the run, in input order.
    """
    membership, ligand_keys = _membership(ws), _site_ligand_keys(ws)
    rows = [row for _, row in sites.iterrows()]
    pairs: List[_Pair] = []
    for row in rows:
        per, keys = membership.get(row["site_id"], {}), ligand_keys.get(row["site_id"], {})
        pairs += [(row, r, per[r.pdb_id], keys.get(r.pdb_id, set())) for r in records if r.pdb_id in per]
    results = parallel.run_map(
        guarded("site", partial(_extract_pair, cfg, ws), _pair_pdb), pairs,
        workers=cfg.compute.workers, backend=cfg.compute.backend, stage="site",
        log_dir=ws.root / parallel.WORKER_LOG_DIR, label=_pair_label)
    for row in rows:
        mine = [(pair[1], res) for pair, res in zip(pairs, results) if pair[0]["site_id"] == row["site_id"]]
        failures = [res for _, res in mine if isinstance(res, Failure)]
        # A pair that raised is still a structure at this site: list it as FAILED.
        site_records = [_failed(rec) if isinstance(res, Failure) else res for rec, res in mine]
        collect_run(layout.site_workspace(ws, row["site_id"]), site_records, failures,
                    compression=cfg.io.compression)
