"""Site discovery: ligands from every structure grouped into sites and tabulated."""
import pandas as pd

from qpocket.config import QPocketConfig
from qpocket.core.types import Ligand, Structure
from qpocket.io import layout
from qpocket.pipelines.sites import discover_sites, read_sites


def _lig(tmp_path, pdb_id, resname, resseq, atoms):
    path = tmp_path / f"{pdb_id}_{resname}{resseq}.pdb"
    path.write_text("".join(
        f"HETATM{i:5d} {n:<4s} {resname} A{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2s}\n"
        for i, (n, e, x, y, z) in enumerate(atoms, 1)) + "END\n")
    return Ligand(pdb_id, "A", resname, resseq, " ", len(atoms), str(path))


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
