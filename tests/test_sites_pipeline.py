"""Cavity-defined sites: CavFind cavities grouped into sites, ligands assigned, misses reported."""
from dataclasses import replace

import pandas as pd

from qpocket.config import QPocketConfig
from qpocket.io import layout
from qpocket.pipelines.sites import discover_sites, extract_all_sites
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


def test_apo_structure_gets_a_cloud_without_ligand(tmp_path):
    holo = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    apo = _structure(tmp_path, "APO1", state="UNBOUND_STATE")
    ws, table = _run(tmp_path, [holo, apo])
    sid = table.site_id.iloc[0]
    assert _master(ws, sid).loc["APO1", "protein_state"] == "UNBOUND_STATE"
    assert (ws.sites / sid / "pockets" / "APO1" / "binding_site.csv").exists()
    assert table.set_index("site_id").loc[sid, ["n_holo", "n_apo"]].tolist() == [1, 1]


def test_rerun_with_same_inputs_is_skipped(tmp_path):
    cfg = _cfg()
    ws = layout.create(str(tmp_path / "work"), "run")
    recs = [_structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])]
    discover_sites(cfg, ws, recs)
    stamp = (ws.sites / "sites.csv").stat().st_mtime_ns
    discover_sites(cfg, ws, recs)
    assert (ws.sites / "sites.csv").stat().st_mtime_ns == stamp


def test_two_ligands_at_one_site_are_two_actives(tmp_path):
    small = _ligand(tmp_path, "P1", resseq=1)
    big = _lig(tmp_path, "P1", "BTN", 2, [(f"C{i}", "C", 1.0 + i * 0.5, 1.0, 0.0) for i in range(6)])._replace(
        smiles="CCCCCC")
    rec = _structure(tmp_path, "P1", [small, big])
    ws, table = _run(tmp_path, [rec])
    sid = table.site_id.iloc[0]
    assert len((ws.sites / sid / "screening" / "actives.smi").read_text().splitlines()) == 3
    assert _master(ws, sid).loc["P1", "ligand_file"] == big.file


# --- manual overrides ---------------------------------------------------------

def _two_sites(tmp_path):
    """S1-Y1 (three structures, P1's ligand) and S2-G3 (two structures, no ligand)."""
    far_pts = [(float(x), 30.0, 30.0) for x in range(30, 40)]
    near = _structure(tmp_path, "P1", [_ligand(tmp_path, "P1")])
    twice = [_structure(tmp_path, f"F{i}", points=INSIDE + far_pts, state="UNBOUND_STATE") for i in range(2)]
    return [near] + twice


def _cavities_of(ws, site_id):
    table = pd.read_csv(ws.sites / "cavities.csv", keep_default_na=False)
    return [[r.pdb_id, int(r.cavity)] for r in table.itertuples() if site_id in r.site_ids.split(";")]


def _redo(tmp_path, ws, records, overrides):
    cfg = _cfg(overrides=overrides)
    table = discover_sites(cfg, ws, records)
    extract_all_sites(cfg, ws, table, records)
    return table


def test_a_drop_override_removes_the_site_and_misses_its_ligands(tmp_path):
    records = _two_sites(tmp_path)
    ws, _ = _run(tmp_path, records)
    table = _redo(tmp_path, ws, records, [{"op": "drop", "cavities": _cavities_of(ws, "S1-Y1")}])
    assert table.site_id.tolist() == ["S1-G3"]
    assert _misses(ws).set_index("pdb_id").loc["P1", "reason"] == "site_dropped"


def test_a_merge_override_makes_one_site(tmp_path):
    records = _two_sites(tmp_path)
    ws, _ = _run(tmp_path, records)
    table = _redo(tmp_path, ws, records, [{"op": "merge", "sites": [_cavities_of(ws, "S1-Y1"),
                                                                    _cavities_of(ws, "S2-G3")]}])
    assert len(table) == 1 and table.n_structures.iloc[0] == 3
    import json
    applied = json.loads((ws.sites / "grouping.json").read_text())["applied"]
    assert applied[0]["site_ids"] == table.site_id.tolist() and "groups" not in applied[0]


