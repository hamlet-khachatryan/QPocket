"""Site discovery: ligands from every structure grouped into sites and tabulated."""
from pathlib import Path

import numpy as np
import pandas as pd

from qpocket.config import QPocketConfig
from qpocket.core.types import Ligand, Structure
from qpocket.io import layout
from qpocket.pipelines.sites import discover_sites, extract_all_sites, extract_site, read_sites
from tests.site_helpers import FAR, INSIDE, _lig, _structure


def _rec(lig_list, contacts=()):
    return Structure(lig_list[0].pdb_id if lig_list else "P3", "x", ligands=tuple(lig_list),
                     contacts=tuple((lig.key, tuple(contacts)) for lig in lig_list))


def _setup(tmp_path):
    ws = layout.create(str(tmp_path), "run")
    a = _lig(tmp_path, "P1", "AAA", 1, [("C1", "C", 0, 0, 0), ("C2", "C", 3, 0, 0), ("H1", "H", 9, 9, 9)])
    a2 = _lig(tmp_path, "P2", "AAA", 1, [("C1", "C", 1, 0, 0), ("C2", "C", 2, 1, 0)])
    far = _lig(tmp_path, "P2", "BBB", 2, [("C1", "C", 50, 0, 0)])
    recs = [_rec([a], ["TYR_1_A|HBDonor"]), _rec([a2, far], ["no_ligand_found", "ALA_2_A|Hydrophobic"]),
            Structure("P3", "x")]
    return ws, recs


def test_two_sites_from_three_structures(tmp_path):
    ws, recs = _setup(tmp_path)
    sites = discover_sites(QPocketConfig(), ws, recs).set_index("site_id")
    assert list(sites.index) == ["site_01", "site_02"]
    assert sites.loc["site_01", "n_holo_structures"] == 2 and sites.loc["site_01", "n_ligands"] == 2
    assert sites.loc["site_01", "extent"] == 3.0  # the hydrogen at (9,9,9) is ignored
    assert sites.loc["site_01", "contact_residues"] == "ALA_2_A;TYR_1_A"
    assert (ws.sites / "site_01" / "ligands.pdb").exists()
    assert list(pd.read_csv(ws.structures / "ligands.csv").site_id) == ["site_01", "site_01", "site_02"]
    assert read_sites(ws).shape[0] == 2


def test_no_ligands_gives_empty_table(tmp_path):
    ws = layout.create(str(tmp_path), "run")
    sites = discover_sites(QPocketConfig(), ws, [Structure("P1", "x")])
    assert sites.empty and (ws.sites / "sites.csv").read_text().startswith("site_id,")


def test_rerun_with_same_inputs_is_skipped(tmp_path):
    ws, recs = _setup(tmp_path)
    discover_sites(QPocketConfig(), ws, recs)
    stamp = (ws.sites / "sites.csv").stat().st_mtime_ns
    discover_sites(QPocketConfig(), ws, recs)
    assert (ws.sites / "sites.csv").stat().st_mtime_ns == stamp


# ── per-site clouds ──────────────────────────────────────────────────────────

def _site_run(tmp_path, cfg=None):
    cfg = cfg or QPocketConfig()
    ws = layout.create(str(tmp_path), "run")
    lig = _lig(tmp_path, "P1", "LIG", 1, [("C1", "C", 0, 0, 0), ("C2", "C", 3, 0, 0)])._replace(smiles="CC")
    holo = _structure(tmp_path, "P1", [lig])
    sites = discover_sites(cfg, ws, [holo])
    return cfg, ws, sites.iloc[0], holo


def test_holo_and_apo_clouds_cut_by_same_box(tmp_path):
    cfg, ws, site_row, holo = _site_run(tmp_path)
    apo = extract_site(cfg, ws, site_row, _structure(tmp_path, "APO1", state="UNBOUND_STATE"))
    bound = extract_site(cfg, ws, site_row, holo)
    assert apo.protein_state == "UNBOUND_STATE" and apo.observed == ("no_ligand_found",) and not apo.actives
    assert bound.protein_state == "BOUND_STATE" and bound.observed == ("TYR_1_A|HBDonor",)
    assert bound.ligand_id == "LIG" and bound.actives == (("CC", "LIG", holo.ligands[0].file),)
    lo, hi = np.array([-4.0, -4.0, -4.0]), np.array([7.0, 4.0, 4.0])  # ligand span plus box_padding 4
    for pid in ("APO1", "P1"):
        pocket = ws.sites / "site_01" / "pockets" / pid
        pts = pd.read_csv(pocket / "binding_site.csv")[["x", "y", "z"]].to_numpy()
        assert len(pts) == len(INSIDE) and ((pts >= lo) & (pts <= hi)).all()
        assert (pocket / "geospectral_cache.npz").exists()
    assert apo.potential == ("ALA_2_A", "TYR_1_A")  # residues within 4 Å of the cloud


def test_closed_site(tmp_path):
    cfg, ws, site_row, _ = _site_run(tmp_path)
    out = extract_site(cfg, ws, site_row, _structure(tmp_path, "APO2", points=FAR, state="UNBOUND_STATE"))
    assert out.protein_state == "CAVITY_CLOSED"
    assert not (ws.sites / "site_01" / "pockets" / out.pdb_id / "binding_site.csv").exists()


def test_failed_and_cavity_free_structures_still_get_a_record(tmp_path):
    cfg, ws, site_row, _ = _site_run(tmp_path)
    assert extract_site(cfg, ws, site_row, _structure(tmp_path, "F1", state="FAILED")).protein_state == "FAILED"
    none = extract_site(cfg, ws, site_row, _structure(tmp_path, "N1", points=(), state="NO_ANY_CAVITY_FOUND"))
    assert none.protein_state == "NO_ANY_CAVITY_FOUND"


def test_two_ligands_at_one_site_are_two_actives(tmp_path):
    cfg = QPocketConfig()
    ws = layout.create(str(tmp_path), "run")
    small = _lig(tmp_path, "P1", "BTN", 1, [("C1", "C", 0, 0, 0), ("C2", "C", 1, 0, 0)])._replace(smiles="CC")
    big = _lig(tmp_path, "P1", "BTN", 2, [("C1", "C", 2, 0, 0), ("C2", "C", 3, 0, 0),
                                          ("C3", "C", 3, 1, 0)])._replace(smiles="CCC")
    rec = _structure(tmp_path, "P1", [small, big])
    site_row = discover_sites(cfg, ws, [rec]).iloc[0]
    out = extract_site(cfg, ws, site_row, rec)
    assert len(out.actives) == 2 and len({a[2] for a in out.actives}) == 2
    assert out.ligand_file == big.file  # the largest one is the cross-docking pose


def test_extract_all_sites_crops_to_contacts_and_collects_per_site(tmp_path):
    from dataclasses import replace

    base = QPocketConfig()
    cfg = replace(base, reference=replace(base.reference, pocket_definition="Interactions", interaction_padding=2.5),
                  compute=replace(base.compute, workers=1))
    cfg, ws, sites_row, holo = _site_run(tmp_path, cfg)
    apo = _structure(tmp_path, "APO1", state="UNBOUND_STATE")
    failed = _structure(tmp_path, "F1", state="FAILED")
    extract_all_sites(cfg, ws, read_sites(ws), [holo, apo, failed])
    site = ws.sites / "site_01"
    assert pd.read_csv(site / "contact_atoms.csv")[["x", "y", "z"]].values.tolist() == [[6.0, 0.0, 0.0]]
    # only points within 2.5 Å of the TYR atom at (6, 0, 0) survive
    assert pd.read_csv(site / "pockets" / "APO1" / "binding_site.csv")["x"].tolist() == [5.0, 6.0]
    master = pd.read_csv(site / "metadata" / "master.csv")
    assert list(master["pdb_id"]) == ["P1", "APO1", "F1"]
    assert list(master["protein_state"]) == ["BOUND_STATE", "UNBOUND_STATE", "FAILED"]
    assert (site / "screening" / "actives.smi").read_text() == (
        f"SMILES\tPDB_ID\tLigand_Name\tLigand_File\nCC\tP1\tLIG\t{holo.ligands[0].file}\n")


def _interactions_cfg():
    from dataclasses import replace

    base = QPocketConfig()
    return replace(base, reference=replace(base.reference, pocket_definition="Interactions", interaction_padding=2.5),
                   compute=replace(base.compute, workers=1))


def test_a_site_without_contacts_is_not_cropped(tmp_path):
    cfg = _interactions_cfg()
    ws = layout.create(str(tmp_path), "run")
    lig = _lig(tmp_path, "P1", "LIG", 1, [("C1", "C", 0, 0, 0), ("C2", "C", 3, 0, 0)])
    holo = _structure(tmp_path, "P1", [lig], contacts=())
    discover_sites(cfg, ws, [holo])
    apo = _structure(tmp_path, "APO1", state="UNBOUND_STATE")
    extract_all_sites(cfg, ws, read_sites(ws), [holo, apo])
    master = pd.read_csv(ws.sites / "site_01" / "metadata" / "master.csv")
    assert list(master["protein_state"]) == ["BOUND_STATE", "UNBOUND_STATE"]
    assert len(pd.read_csv(ws.sites / "site_01" / "pockets" / "APO1" / "binding_site.csv")) == len(INSIDE)


def test_a_pair_that_raises_is_still_listed_at_the_site(tmp_path):
    cfg, ws, _, holo = _site_run(tmp_path, _interactions_cfg())
    broken = _structure(tmp_path, "BAD", state="UNBOUND_STATE")
    Path(broken.cavities_file).write_text("not,a,cloud\n1,2,3\n")
    extract_all_sites(cfg, ws, read_sites(ws), [holo, broken])
    site = ws.sites / "site_01" / "metadata"
    master = pd.read_csv(site / "master.csv").set_index("pdb_id")
    assert list(master.index) == ["P1", "BAD"] and master.loc["BAD", "protein_state"] == "FAILED"
    assert master.loc["BAD", "rmsd_to_ref"] == 0.5 and master.loc["BAD", "aligned_file"] == broken.aligned_file
    assert list(pd.read_csv(site / "failures.csv")["pdb_id"]) == ["BAD"]
    from qpocket.io import hdf5

    assert list(hdf5.read_interactions(site / "observed_interactions.h5")["BAD"]) == ["no_ligand_found"]
