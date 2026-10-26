"""Site stage: group every ligand of the run into binding sites, then cut each structure's cloud at each site."""
from __future__ import annotations

import hashlib
import json
import operator
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from eliot import log_message, start_action
from scipy.spatial.distance import cdist, pdist

from qpocket.config import QPocketConfig
from qpocket.core.geometry import crop_pocket_by_interactions
from qpocket.core.pocket import interacting_residue_coords
from qpocket.core.result import guarded, partition
from qpocket.core.sites import element_symbol, group_ligands, points_in_box, site_box, site_ids
from qpocket.core.types import Ligand, Structure
from qpocket.io import layout
from qpocket.io.layout import Workspace
from qpocket.pipelines import parallel
from qpocket.pipelines.collect import collect_run
from qpocket.pipelines.distance import compute_geodesic

SITE_COLUMNS = ["site_id", "n_ligands", "n_holo_structures", "centre_x", "centre_y", "centre_z",
                "extent", "resnames", "contact_residues"]
LIGAND_COLUMNS = list(Ligand._fields) + ["site_id"]
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
        coords = [_heavy_coords(lig.file) for lig in ligands]
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


# ── per-site clouds ──────────────────────────────────────────────────────────

class SiteRecord(NamedTuple):
    """One structure at one site.

    Attributes:
        pdb_id: Identifier.
        rmsd_to_ref: RMSD to the reference after superposition.
        aligned_file: Superposed structure.
        cleaned_file: Protein and allowed cofactors only.
        protonated_file: Docking-ready protonated complex.
        protein_state: BOUND_STATE, UNBOUND_STATE, CAVITY_CLOSED, NO_ANY_CAVITY_FOUND or FAILED.
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


def _potential(cleaned_file: Optional[str], cloud: np.ndarray) -> Tuple[str, ...]:
    """Residues with an atom within 4 Å of the cloud, as RES_NUM_CHAIN."""
    if not cleaned_file or not Path(cleaned_file).exists():
        return ()
    atoms = PandasPdb().read_pdb(str(cleaned_file)).df["ATOM"]
    near = atoms[cdist(atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=float), cloud).min(axis=1) <= 4.0]
    # Chain ID included so potential and observed keys intersect in utilisation.
    return tuple(sorted(set(near["residue_name"] + "_" + near["residue_number"].astype(str)
                            + "_" + near["chain_id"])))


def write_contact_atoms(ws: Workspace, site: pd.Series, records: Sequence[Structure]) -> np.ndarray:
    """Aligned-frame atoms of the site's contact residues, from every holo structure of the site.

    Args:
        ws: The run.
        site: Its row of sites.csv.
        records: Every structure of the run.

    Returns:
        ``(n, 3)`` coordinates, also written to ``sites/<id>/contact_atoms.csv``.
    """
    ligands = pd.read_csv(ws.structures / "ligands.csv", dtype=str, keep_default_na=False)
    holo = set(ligands.loc[ligands["site_id"] == site["site_id"], "pdb_id"])
    entries = [f"{res}|" for res in str(site["contact_residues"]).split(";") if res]
    found = [interacting_residue_coords(PandasPdb().read_pdb(str(r.aligned_file)).df["ATOM"], entries)
             for r in records if r.pdb_id in holo and r.aligned_file and entries]
    xyz = np.vstack([c for c in found if c is not None] or [np.zeros((0, 3))])
    pd.DataFrame(xyz, columns=["x", "y", "z"]).to_csv(ws.sites / site["site_id"] / "contact_atoms.csv",
                                                       index=False)
    return xyz


def extract_site(cfg: QPocketConfig, ws: Workspace, site: pd.Series, record: Structure) -> SiteRecord:
    """Cut one structure's cloud at one site and record what lies there.

    Args:
        cfg: Run configuration (site box padding, pocket definition, geodesic radius).
        ws: The run.
        site: The site's row of sites.csv.
        record: The structure.

    Returns:
        The structure's record at the site; a cloud and its geodesics are written only when non-empty.
    """
    site_ws = layout.site_workspace(ws, site["site_id"])
    pocket_dir = site_ws.pockets / record.pdb_id
    ligands = pd.read_csv(ws.structures / "ligands.csv", dtype=str, keep_default_na=False)
    keys = {Ligand(*row).key for row in ligands.loc[(ligands["site_id"] == site["site_id"])
                                                    & (ligands["pdb_id"] == record.pdb_id),
                                                    list(Ligand._fields)].itertuples(index=False)}
    mine = [lig for lig in record.ligands if lig.key in keys]
    largest = max(mine, key=lambda lig: lig.n_heavy) if mine else None
    contacts = dict(record.contacts)
    observed = tuple(sorted({e for lig in mine for e in contacts.get(lig.key, ()) if "|" in e})) if mine \
        else (NO_LIGAND,)
    out = SiteRecord(record.pdb_id, record.rmsd_to_ref, record.aligned_file, record.cleaned_file,
                     record.protonated_file, ligand_file=largest.file if largest else None,
                     ligand_id=largest.resname if largest else None, observed=observed,
                     actives=tuple((lig.smiles, lig.resname, lig.file) for lig in mine))

    with start_action(action_type="extract_site", site_id=site["site_id"], pdb_id=record.pdb_id) as action:
        if record.protein_state in (None, "FAILED"):
            state, cloud = "FAILED", None
        elif not record.cavities_file or not Path(record.cavities_file).exists():
            state, cloud = "NO_ANY_CAVITY_FOUND", None
        else:
            points = pd.read_csv(record.cavities_file)[["x", "y", "z"]].to_numpy(dtype=float)
            cloud = points_in_box(points, *site_box([_heavy_coords(ws.sites / site["site_id"] / "ligands.pdb")],
                                                    cfg.sites.box_padding))
            if cfg.reference.pocket_definition == "Interactions" and len(cloud):
                contact_xyz = pd.read_csv(ws.sites / site["site_id"] / "contact_atoms.csv")[["x", "y", "z"]]
                cloud = crop_pocket_by_interactions(cloud, contact_xyz.to_numpy(dtype=float),
                                                    padding=cfg.reference.interaction_padding) \
                    if len(contact_xyz) else cloud[:0]
            state = ("BOUND_STATE" if mine else "UNBOUND_STATE") if len(cloud) else "CAVITY_CLOSED"
        action.add_success_fields(protein_state=state)
        if state not in ("BOUND_STATE", "UNBOUND_STATE"):
            shutil.rmtree(pocket_dir, ignore_errors=True)  # a regrouped site must not keep a stale cloud
            return out._replace(protein_state=state)
        pocket_dir.mkdir(parents=True, exist_ok=True)
        csv_path = pocket_dir / "binding_site.csv"
        pd.DataFrame(cloud, columns=["x", "y", "z"]).to_csv(csv_path, index=False)
        compute_geodesic(csv_path, cfg.distance.geo_graph_neighbor_radius)
        return out._replace(protein_state=state, potential=_potential(record.cleaned_file, cloud))


def _extract_pair(cfg: QPocketConfig, ws: Workspace, sites: Dict[str, pd.Series],
                  records: Dict[str, Structure], pair: Tuple[str, str]) -> SiteRecord:
    return extract_site(cfg, ws, sites[pair[0]], records[pair[1]])


def _pair_label(pair: Tuple[str, str]) -> str:
    return f"{pair[0]}_{pair[1]}"


def extract_all_sites(cfg: QPocketConfig, ws: Workspace, sites: pd.DataFrame,
                      records: Sequence[Structure]) -> None:
    """Cut every structure's cloud at every site, then collect each site.

    Args:
        cfg: Run configuration.
        ws: The run.
        sites: The site table from :func:`discover_sites`.
        records: Every structure of the run, in input order.
    """
    by_site = {row["site_id"]: row for _, row in sites.iterrows()}
    by_pdb = {r.pdb_id: r for r in records}
    if cfg.reference.pocket_definition == "Interactions":
        for row in by_site.values():
            write_contact_atoms(ws, row, records)
    pairs = [(sid, r.pdb_id) for sid in by_site for r in records]
    results = parallel.run_map(
        guarded("site", partial(_extract_pair, cfg, ws, by_site, by_pdb), operator.itemgetter(1)), pairs,
        workers=cfg.compute.workers, backend=cfg.compute.backend, stage="site",
        log_dir=ws.root / parallel.WORKER_LOG_DIR, label=_pair_label)
    for sid in by_site:
        mine = [res for (s, _), res in zip(pairs, results) if s == sid]
        site_records, failures = partition(mine)
        collect_run(layout.site_workspace(ws, sid), site_records, failures, features=(),
                    compression=cfg.io.compression)
