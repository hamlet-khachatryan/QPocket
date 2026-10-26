"""Cavity-defined sites: CavFind cavities grouped into sites, ligands assigned, misses reported."""
from dataclasses import replace

import numpy as np
import pandas as pd

from qpocket.config import QPocketConfig
from qpocket.io import layout
from qpocket.pipelines.sites import SITE_COLUMNS, discover_sites, extract_all_sites, read_sites
from tests.site_helpers import INSIDE, _lig, _structure


def _cfg(**sites):
    base = QPocketConfig()
    return replace(base, sites=replace(base.sites, **sites),
                   compute=replace(base.compute, workers=1, backend="thread"))


def _run(tmp_path, records, **sites):
    cfg = _cfg(**sites)
    ws = layout.create(str(tmp_path / "work"), "run")
    table = discover_sites(cfg, ws, records)
    extract_all_sites(cfg, ws, table, records)
    return ws, table


def _ligand(tmp_path, pdb_id, x0=0.0, resseq=1, y=0.0, z=0.0):
    atoms = [(f"C{i}", "C", x0 + i, y, z) for i in range(4)]
    return _lig(tmp_path, pdb_id, "LIG", resseq, atoms)._replace(smiles="CCCC")


def _grid(x, y=(0, 4), z=(0, 4)):
    return [(float(a), float(b), float(c)) for a in range(x[0], x[1] + 1)
            for b in range(y[0], y[1] + 1) for c in range(z[0], z[1] + 1)]


def _master(ws, site_id):
    return pd.read_csv(ws.sites / site_id / "metadata" / "master.csv").set_index("pdb_id")


def _misses(ws):
    return pd.read_csv(ws.sites / "cavfind_misses.csv", keep_default_na=False)


def test_holo_cloud_is_the_structures_own_cavity_points_unmodified(tmp_path):
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    ws, table = _run(tmp_path, [holo])
    sid = table.site_id.iloc[0]
    cloud = pd.read_csv(ws.sites / sid / "pockets" / "P1" / "binding_site.csv")
    cav = pd.read_csv(holo.cavities_file)
    expected = cav[cav.cavity == 0][["x", "y", "z"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(cloud[["x", "y", "z"]], expected)
    assert _master(ws, sid).loc["P1", "protein_state"] == "BOUND_STATE"


def test_two_cavities_of_one_structure_in_one_site_are_pooled(tmp_path):
    block = np.array(_grid((0, 6)), float)                    # 175 points per cavity
    rows = [(*p, 0) for p in block] + [(*(p + [0.2, 0, 0]), 1) for p in block]
    path = tmp_path / "P1_two.csv"
    pd.DataFrame(rows, columns=["x", "y", "z", "cavity"]).to_csv(path, index=False)
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])._replace(cavities_file=str(path))
    ws, table = _run(tmp_path, [holo])
    assert len(table) == 1
    cloud = pd.read_csv(ws.sites / table.site_id.iloc[0] / "pockets" / "P1" / "binding_site.csv")
    assert len(cloud) == 2 * len(block)


def test_apo_structure_gets_a_cloud_without_ligand(tmp_path):
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    apo = _structure(tmp_path, "APO1", state="UNBOUND_STATE")
    ws, table = _run(tmp_path, [holo, apo])
    sid = table.site_id.iloc[0]
    assert _master(ws, sid).loc["APO1", "protein_state"] == "UNBOUND_STATE"
    assert (ws.sites / sid / "pockets" / "APO1" / "binding_site.csv").exists()
    assert table.set_index("site_id").loc[sid, ["n_holo", "n_apo"]].tolist() == [1, 1]


def test_ligand_without_cavity_at_its_site_is_a_miss(tmp_path):
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    blind = _structure(tmp_path, "X1", [_ligand(tmp_path, "X1")], points=[(30.0, 30.0, 30.0)])
    ws, table = _run(tmp_path, [holo, blind])
    sid = table.site_id.iloc[0]
    miss = _misses(ws).set_index("pdb_id")
    assert miss.loc["X1", "reason"] == "no_cavity_at_site" and miss.loc["X1", "sites"] == sid
    assert "X1" not in _master(ws, sid).index
    assert not (ws.sites / sid / "pockets" / "X1").exists()
    assert table.set_index("site_id").loc[sid, "n_cavfind_miss"] == 1


def test_ligand_in_no_site_is_a_miss(tmp_path):
    lost = _ligand(tmp_path, "P1", x0=50.0, resseq=2)
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1"), lost])
    ws, _ = _run(tmp_path, [holo])
    miss = _misses(ws)
    assert miss[["ligand_key", "reason", "sites"]].values.tolist() == [[lost.key, "no_site", ""]]


def test_holo_somewhere_is_not_a_miss(tmp_path):
    a = _structure(tmp_path, "A", points=_grid((0, 10)), state="UNBOUND_STATE")
    b = _structure(tmp_path, "B", points=_grid((0, 10)), state="UNBOUND_STATE")
    c = _structure(tmp_path, "C", [_ligand(tmp_path, "C", x0=6.5, y=2, z=2)], points=_grid((6, 16)))
    ws, table = _run(tmp_path, [a, b, c])
    assert len(table) == 2
    lig = pd.read_csv(ws.structures / "ligands.csv", keep_default_na=False)
    assert len(lig.site_ids.iloc[0].split(";")) == 2          # the ligand lies in both boxes
    assert _misses(ws).empty
    states = [_master(ws, sid)["protein_state"].to_dict().get("C") for sid in table.site_id]
    assert sorted(s for s in states if s) == ["BOUND_STATE"]


def test_small_cavities_never_join(tmp_path):
    rec = _structure(tmp_path, "P1", points=[(0.0, 0.0, 0.0)], state="UNBOUND_STATE")  # 8 voxels < 50
    ws, table = _run(tmp_path, [rec])
    assert table.empty
    assert pd.read_csv(ws.sites / "cavities.csv", keep_default_na=False).site_ids.tolist() == [""]


def test_site_kept_by_ligand_below_structure_share(tmp_path):
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    other = _structure(tmp_path, "P2", points=(), state="UNBOUND_STATE")
    _, table = _run(tmp_path, [holo, other], min_structure_fraction=0.9)
    assert len(table) == 1


def test_site_dropped_without_ligand_below_structure_share(tmp_path):
    apo = _structure(tmp_path, "P1", state="UNBOUND_STATE")
    other = _structure(tmp_path, "P2", points=(), state="UNBOUND_STATE")
    _, table = _run(tmp_path, [apo, other], min_structure_fraction=0.9)
    assert table.empty


def test_sites_named_by_rank_and_nearest_residue(tmp_path):
    near = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    far_pts = [(float(x), 30.0, 30.0) for x in range(30, 40)]
    twice = [_structure(tmp_path, f"F{i}", points=INSIDE + far_pts, state="UNBOUND_STATE") for i in range(2)]
    ws, table = _run(tmp_path, [near] + twice)
    # the site near TYR 1 has all three structures, the far one two; GLY 3 sits at (40, 40, 40)
    assert table.site_id.tolist() == ["S1-Y1", "S2-G3"]
    assert (ws.sites / "S1-Y1" / "ligands.pdb").exists()
    assert list(table.columns) == SITE_COLUMNS


def test_structure_without_candidates_reports_misses(tmp_path):
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    none = _structure(tmp_path, "N1", [_ligand(tmp_path, "N1")], points=(), state="NO_ANY_CAVITY_FOUND")
    ws, table = _run(tmp_path, [holo, none])
    assert _misses(ws).set_index("pdb_id").loc["N1", "reason"] == "no_cavity_at_site"
    assert "N1" not in _master(ws, table.site_id.iloc[0]).index


def test_no_candidates_gives_empty_sites(tmp_path):
    rec = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")], points=())
    ws, table = _run(tmp_path, [rec])
    assert table.empty and (ws.sites / "sites.csv").read_text().startswith("site_id,")
    assert _misses(ws).reason.tolist() == ["no_site"]
    assert read_sites(ws).empty


def test_rerun_with_same_inputs_is_skipped(tmp_path):
    cfg = _cfg()
    ws = layout.create(str(tmp_path / "work"), "run")
    recs = [_structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])]
    discover_sites(cfg, ws, recs)
    stamp = (ws.sites / "sites.csv").stat().st_mtime_ns
    discover_sites(cfg, ws, recs)
    assert (ws.sites / "sites.csv").stat().st_mtime_ns == stamp


def test_a_pair_that_raises_is_still_listed_at_the_site(tmp_path):
    cfg = _cfg()
    ws = layout.create(str(tmp_path / "work"), "run")
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    broken = _structure(tmp_path, "BAD", state="UNBOUND_STATE")
    table = discover_sites(cfg, ws, [holo, broken])
    with open(broken.cavities_file, "w") as handle:
        handle.write("not,a,cloud\n1,2,3\n")
    extract_all_sites(cfg, ws, table, [holo, broken])
    sid = table.site_id.iloc[0]
    master = _master(ws, sid)
    assert list(master.index) == ["P1", "BAD"] and master.loc["BAD", "protein_state"] == "FAILED"
    assert list(pd.read_csv(ws.sites / sid / "metadata" / "failures.csv")["pdb_id"]) == ["BAD"]


def test_two_ligands_at_one_site_are_two_actives(tmp_path):
    small = _ligand(tmp_path, "P1", resseq=1)
    big = _lig(tmp_path, "P1", "BTN", 2, [(f"C{i}", "C", 1.0 + i * 0.5, 1.0, 0.0) for i in range(6)])._replace(
        smiles="CCCCCC")
    rec = _structure(tmp_path, "P1", [small, big])
    ws, table = _run(tmp_path, [rec])
    sid = table.site_id.iloc[0]
    assert len((ws.sites / sid / "screening" / "actives.smi").read_text().splitlines()) == 3
    assert _master(ws, sid).loc["P1", "ligand_file"] == big.file
